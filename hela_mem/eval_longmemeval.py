"""
LongMemEval Evaluation for Hebbian Memory

Loads encoded Hebbian memory graphs and evaluates on LongMemEval-S questions.
Uses GPT-4o-mini as judge with type-specific prompts (same as LightMem).

Usage:
    python -m hela_mem.eval_longmemeval \
        --data_path /path/to/longmemeval_s.json \
        --mem_dir results/longmemeval_mem_XXXX \
        [--num_items 500] [--start_item 0] [--top_k 20] \
        [--use_consolidation] [--workers 10]
"""

import json
import os
import argparse
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, Optional, Tuple

from .hebbian_memory import HebbianMemoryGraph
from .hebbian_retriever import HebbianRetriever
from .hebbian_knowledge_memory import HebbianKnowledgeMemory
from .utils import (
    gpt_generate_answer_with_rotation,
)


# ========== GPT Judge (from LongMemEval / LightMem) ==========

# Corrupted sample indices from LightMem paper (treated as incorrect)
CORRUPTED_INDICES = {74, 183, 278, 351, 380}


def get_anscheck_prompt(
    task: str, question: str, answer: str, response: str, abstention: bool = False
) -> str:
    """
    Build GPT judge prompt for LongMemEval evaluation.
    Each question type has a tailored evaluation template.
    Directly adapted from LongMemEval official evaluation.
    """
    if not abstention:
        if task in ("single-session-user", "single-session-assistant", "multi-session"):
            template = (
                "I will give you a question, a correct answer, and a response from a model. "
                "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
                "If the response is equivalent to the correct answer or contains all the intermediate "
                "steps to get the correct answer, you should also answer yes. "
                "If the response only contains a subset of the information required by the answer, answer no. "
                "\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\n"
                "Is the model response correct? Answer yes or no only."
            )
        elif task == "temporal-reasoning":
            template = (
                "I will give you a question, a correct answer, and a response from a model. "
                "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
                "If the response is equivalent to the correct answer or contains all the intermediate "
                "steps to get the correct answer, you should also answer yes. "
                "If the response only contains a subset of the information required by the answer, answer no. "
                "In addition, do not penalize off-by-one errors for the number of days. "
                "If the question asks for the number of days/weeks/months, etc., and the model makes "
                "off-by-one errors (e.g., predicting 19 days when the answer is 18), the model's response "
                "is still correct. "
                "\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\n"
                "Is the model response correct? Answer yes or no only."
            )
        elif task == "knowledge-update":
            template = (
                "I will give you a question, a correct answer, and a response from a model. "
                "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
                "If the response contains some previous information along with an updated answer, "
                "the response should be considered as correct as long as the updated answer is the "
                "required answer."
                "\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\n"
                "Is the model response correct? Answer yes or no only."
            )
        elif task == "single-session-preference":
            template = (
                "I will give you a question, a rubric for desired personalized response, and a response "
                "from a model. Please answer yes if the response satisfies the desired response. "
                "Otherwise, answer no. The model does not need to reflect all the points in the rubric. "
                "The response is correct as long as it recalls and utilizes the user's personal "
                "information correctly."
                "\n\nQuestion: {}\n\nRubric: {}\n\nModel Response: {}\n\n"
                "Is the model response correct? Answer yes or no only."
            )
        else:
            # Fallback for unknown types
            template = (
                "I will give you a question, a correct answer, and a response from a model. "
                "Please answer yes if the response contains the correct answer. Otherwise, answer no."
                "\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\n"
                "Is the model response correct? Answer yes or no only."
            )
    else:
        template = (
            "I will give you an unanswerable question, an explanation, and a response from a model. "
            "Please answer yes if the model correctly identifies the question as unanswerable. "
            "The model could say that the information is incomplete, or some other information is "
            "given but the asked information is not."
            "\n\nQuestion: {}\n\nExplanation: {}\n\nModel Response: {}\n\n"
            "Does the model correctly identify the question as unanswerable? Answer yes or no only."
        )

    return template.format(question, answer, response)


def parse_judge_response(response: Optional[str]) -> bool:
    """Parse GPT judge yes/no response into boolean."""
    if response is None:
        return False
    normalized = str(response).strip().lower()
    if not normalized:
        return False
    first_line = normalized.splitlines()[0].strip()
    tokens = first_line.replace(".", "").replace("!", "").replace(":", "").replace(";", "").split()
    if not tokens:
        return False
    head = tokens[0]
    if head in ("yes", "y"):
        return True
    if head in ("no", "n"):
        return False
    if "yes" in first_line:
        return True
    if "no" in first_line:
        return False
    return False


# ========== Consolidation (same as NarrativeQA) ==========

CONSOLIDATION_THRESHOLD = int(os.environ.get("HEBBIAN_CONSOLIDATION_THRESHOLD", "3"))
MAX_CLUSTERS = int(os.environ.get("HEBBIAN_MAX_CLUSTERS", "20"))
MAX_CLUSTER_SIZE = int(os.environ.get("HEBBIAN_MAX_CLUSTER_SIZE", "8"))


def consolidate_memory(
    memory_graph: HebbianMemoryGraph,
    knowledge_memory: HebbianKnowledgeMemory,
) -> int:
    """
    Hebbian-driven consolidation: Episodic -> Semantic Memory.
    1. Build similarity edges to enrich graph structure
    2. Find hub nodes (high-degree)
    3. Collect clusters, use LLM to extract key facts
    4. Store facts in HebbianKnowledgeMemory

    Returns number of facts extracted.
    """
    import numpy as np

    nodes = memory_graph.nodes
    edges = memory_graph.edges

    # Step 0: Build similarity edges
    SIMILARITY_K = 5
    node_ids = list(nodes.keys())
    if len(node_ids) < 3:
        return 0

    embeddings = {}
    for nid in node_ids:
        emb = nodes[nid].get("embedding")
        if emb is not None:
            embeddings[nid] = np.array(emb)

    if len(embeddings) >= 3:
        emb_ids = list(embeddings.keys())
        emb_matrix = np.array([embeddings[nid] for nid in emb_ids])
        norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        normed = emb_matrix / norms
        sim_matrix = normed @ normed.T

        edges_added = 0
        for i, nid in enumerate(emb_ids):
            sims = sim_matrix[i].copy()
            sims[i] = -1
            top_indices = np.argsort(sims)[-SIMILARITY_K:]
            for j in top_indices:
                if sims[j] > 0.5:
                    other_id = emb_ids[j]
                    if other_id not in edges.get(nid, {}):
                        memory_graph.add_edge(nid, other_id, weight=sims[j] * 0.3, bidirectional=True)
                        edges_added += 1
        print(f"    [Consolidation] Built {edges_added} similarity edges")

    # Step 1: Find hub nodes
    hub_candidates = []
    for node_id in nodes:
        degree = len(edges.get(node_id, {}))
        total_weight = sum(edges.get(node_id, {}).values())
        if degree >= CONSOLIDATION_THRESHOLD:
            hub_candidates.append((node_id, degree, total_weight))

    if not hub_candidates:
        return 0

    hub_candidates.sort(key=lambda x: x[1] * x[2], reverse=True)
    hub_candidates = hub_candidates[:MAX_CLUSTERS]

    processed = set()
    total_facts = 0

    for hub_id, degree, total_weight in hub_candidates:
        if hub_id in processed:
            continue

        neighbor_edges = edges.get(hub_id, {})
        sorted_neighbors = sorted(neighbor_edges.items(), key=lambda x: x[1], reverse=True)

        cluster_ids = [hub_id]
        for nid, w in sorted_neighbors[: MAX_CLUSTER_SIZE - 1]:
            if nid not in processed and nid in nodes:
                cluster_ids.append(nid)

        processed.update(cluster_ids)

        cluster_texts = []
        for cid in cluster_ids:
            content = nodes[cid].get("content", "")
            if content:
                cluster_texts.append(content[:1500])

        if len(cluster_texts) < 2:
            continue

        # Step 2: LLM extracts key facts
        combined = "\n\n---\n\n".join(cluster_texts)
        prompt = (
            "You are a knowledge extraction engine.\n"
            "Below are several related conversation passages.\n"
            "Extract the key facts as a list. Each fact should be a single, "
            "self-contained sentence covering people, events, preferences, "
            "dates, or important details.\n"
            "Output one fact per line, prefixed with '- '.\n"
            "Extract 3-8 facts. Output ONLY the fact list.\n\n"
            f"Passages:\n{combined}\n\nFacts:"
        )
        messages = [
            {"role": "system", "content": "Extract key facts from conversation passages."},
            {"role": "user", "content": prompt},
        ]

        try:
            result = gpt_generate_answer_with_rotation(prompt, messages)
            if not result:
                continue
            for line in result.strip().split("\n"):
                line = line.strip()
                if line.startswith("- "):
                    line = line[2:].strip()
                if len(line) > 10 and line.lower() != "none":
                    knowledge_memory.add_knowledge(line)
                    total_facts += 1
        except Exception as e:
            print(f"    [Consolidation] Error: {e}")
            continue

    memory_graph.save()
    knowledge_memory.save()
    print(f"    [Consolidation] Extracted {total_facts} facts -> Semantic Memory")
    return total_facts


# ========== Answer Generation ==========

def build_longmemeval_prompt(
    context_text: str,
    knowledge_text: str,
    profile_text: str,
    assistant_knowledge_text: str,
    question: str,
    question_date: str,
) -> Tuple[str, str]:
    """
    Build system prompt and user prompt for LongMemEval QA.
    Adapted from HebbianRetriever.answer() with conciseness and format
    instructions aligned with the original internal experiment prompt settings.

    Returns:
        (system_prompt, user_prompt)
    """
    system_prompt = (
        "You are a helpful assistant with access to the user's conversation history. "
        "Your task is to answer questions about the user or past conversations "
        "in an extremely concise manner.\n"
        "When the question is: \"What did the charity race raise awareness for?\", "
        "you should not answer in the form of: \"The charity race raised awareness "
        "for mental health.\" Instead, it should be: \"mental health\", as this is "
        "more concise."
    )
    if assistant_knowledge_text:
        system_prompt += f"\n{assistant_knowledge_text}"

    user_prompt = (
        f"<CONTEXT>\n"
        f"Current date: {question_date}\n\n"
        f"Relevant memories from conversation history:\n"
        f"{context_text}\n\n"
    )

    if knowledge_text:
        user_prompt += f"<KNOWLEDGE BASE>\n{knowledge_text}\n\n"

    if profile_text and profile_text != "None":
        user_prompt += f"<CHARACTER TRAITS>\nCharacteristics of the user:\n{profile_text}\n\n"

    user_prompt += (
        f"Question: {question}\n"
        f"Please only provide the content of the answer, without including 'answer:'\n"
        f"For questions that require answering a date or time, strictly follow the "
        f"format \"15 July 2023\" and provide a specific date whenever possible. "
        f"For example, if you need to answer \"last year,\" give the specific year "
        f"rather than just saying \"last year.\" Only provide one year, date, or time, "
        f"without any extra responses.\n"
        f"If the question is about the duration, answer in the form of several years, "
        f"months, or days.\n"
        f"Generate answers primarily composed of concrete entities."
    )

    return system_prompt, user_prompt


def answer_question(
    retriever: HebbianRetriever,
    question: str,
    question_date: str,
    top_k: int = 20,
    knowledge_memory: Optional[HebbianKnowledgeMemory] = None,
    semantic_top_k: int = 5,
    item_id: str = "",
) -> Tuple[str, list]:
    """
    Answer a single LongMemEval question using Hebbian retrieval.

    Returns:
        (answer_text, retrieved_results)
    """
    # 1. Episodic retrieval
    use_reranker = os.environ.get("HEBBIAN_USE_RERANKER", "false").lower() == "true"
    rerank_pool_multiplier = int(os.environ.get("HEBBIAN_RERANK_POOL_MULTIPLIER", "3"))

    retrieve_k = top_k * rerank_pool_multiplier if use_reranker else top_k
    results = retriever.graph.retrieve(question, top_k=retrieve_k)

    if use_reranker and len(results) > top_k:
        from .reranker import rerank_memories
        memories_to_rerank = [{"content": r["node"]["content"], **r} for r in results]
        reranked = rerank_memories(question, memories_to_rerank, top_k=top_k)
        results = [
            {
                "node": r["node"],
                "score": r["score"],
                "base_score": r["base_score"],
                "source": r.get("source", "base"),
            }
            for r in reranked
        ]

    # Build episodic context
    context_blocks = []
    for res in results[:top_k]:
        node = res["node"]
        score = res["score"]
        source_label = "Direct Match" if res.get("base_score", 0) > 0.6 else "Associative Memory"
        block = (
            f"[{source_label} | Relevancy: {score:.2f}]\n"
            f"Time: {node.get('timestamp', 'unknown')}\n"
            f"Content: {node['content']}"
        )
        context_blocks.append(block)

    context_text = "\n\n".join(context_blocks)

    # 2. Semantic retrieval from Knowledge Memory
    knowledge_text = ""
    if knowledge_memory:
        try:
            kb_results = knowledge_memory.search_knowledge(question, top_k=semantic_top_k)
            if kb_results:
                kb_lines = [f"- {kn['knowledge']}" for kn in kb_results]
                knowledge_text = "\n".join(kb_lines)
        except Exception as e:
            print(f"    [{item_id}] Semantic retrieval error: {e}")

    # 3. Get user profile
    profile_text = "None"
    if knowledge_memory and hasattr(knowledge_memory, "get_raw_user_profile"):
        profile_text = knowledge_memory.get_raw_user_profile(item_id) or "None"

    # 4. Get assistant knowledge
    assistant_knowledge_text = ""
    if knowledge_memory and hasattr(knowledge_memory, "get_assistant_knowledge"):
        try:
            ak_list = knowledge_memory.get_assistant_knowledge()
            if ak_list:
                assistant_knowledge_text = "Here are some of your character traits and knowledge:\n"
                for ak_item in ak_list:
                    k_text = ak_item["knowledge"].strip()
                    if k_text:
                        assistant_knowledge_text += f"- {k_text}\n"
        except Exception as e:
            print(f"    [{item_id}] Assistant knowledge error: {e}")

    # 5. Build prompt and generate answer
    system_prompt, user_prompt = build_longmemeval_prompt(
        context_text, knowledge_text, profile_text, assistant_knowledge_text,
        question, question_date,
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    response = gpt_generate_answer_with_rotation(user_prompt, messages)
    return response, results


# ========== Single Item Evaluation ==========

def evaluate_single_item(
    item: Dict[str, Any],
    item_idx: int,
    mem_dir: str,
    top_k: int = 20,
    semantic_top_k: int = 5,
    use_consolidation: bool = False,
    results_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Evaluate a single LongMemEval item.

    Steps:
        1. Load encoded Hebbian memory graph
        2. Optionally run consolidation
        3. Retrieve + generate answer
        4. Judge with GPT-4o-mini
        5. Save per-item result

    Returns:
        Dict with question_id, question_type, correct, generated_answer, etc.
    """
    item_id = item["question_id"]
    question_type = item["question_type"]
    question = item["question"]
    answer = item["answer"]
    question_date = item.get("question_date", "")
    is_abstention = "abs" in item_id

    t_start = time.time()

    # Check for corrupted samples
    if item_idx in CORRUPTED_INDICES:
        print(f"  [{item_idx}] {item_id} - CORRUPTED (skipped, marked incorrect)")
        return {
            "question_id": item_id,
            "question_type": question_type,
            "correct": 0,
            "generated_answer": "[CORRUPTED]",
            "ground_truth": str(answer),
            "is_corrupted": True,
            "eval_time": 0.0,
        }

    # Load memory graph
    mem_path = os.path.join(mem_dir, f"{item_id}_hebbian.json")
    if not os.path.exists(mem_path):
        print(f"  [{item_idx}] {item_id} - Memory file not found: {mem_path}")
        return {
            "question_id": item_id,
            "question_type": question_type,
            "correct": 0,
            "generated_answer": "[NO MEMORY]",
            "ground_truth": str(answer),
            "error": "memory file not found",
            "eval_time": 0.0,
        }

    memory_graph = HebbianMemoryGraph(file_path=mem_path)
    if not memory_graph.nodes:
        print(f"  [{item_idx}] {item_id} - Empty memory graph")
        return {
            "question_id": item_id,
            "question_type": question_type,
            "correct": 0,
            "generated_answer": "[EMPTY GRAPH]",
            "ground_truth": str(answer),
            "error": "empty graph",
            "eval_time": 0.0,
        }

    # Load knowledge memory
    kb_path = os.path.join(mem_dir, f"{item_id}_long_term.json")
    knowledge_memory = HebbianKnowledgeMemory(file_path=kb_path)

    # Optional consolidation
    if use_consolidation:
        consolidate_memory(memory_graph, knowledge_memory)

    # Create retriever
    retriever = HebbianRetriever(memory_graph, profile_memory=knowledge_memory)

    # Generate answer
    try:
        generated_answer, retrieved = answer_question(
            retriever,
            question,
            question_date,
            top_k=top_k,
            knowledge_memory=knowledge_memory,
            semantic_top_k=semantic_top_k,
            item_id=item_id,
        )
    except Exception as e:
        print(f"  [{item_idx}] {item_id} - Answer generation error: {e}")
        generated_answer = ""
        retrieved = []

    # Judge with GPT
    try:
        judge_prompt = get_anscheck_prompt(
            question_type, question, str(answer), generated_answer,
            abstention=is_abstention,
        )
        judge_messages = [{"role": "user", "content": judge_prompt}]
        judge_response = gpt_generate_answer_with_rotation(judge_prompt, judge_messages)
        correct = 1 if parse_judge_response(judge_response) else 0
    except Exception as e:
        print(f"  [{item_idx}] {item_id} - Judge error: {e}")
        correct = 0
        judge_response = ""

    eval_time = time.time() - t_start

    result = {
        "question_id": item_id,
        "question_type": question_type,
        "question": question,
        "ground_truth": str(answer),
        "generated_answer": generated_answer,
        "correct": correct,
        "is_abstention": is_abstention,
        "num_nodes": len(memory_graph.nodes),
        "num_retrieved": len(retrieved),
        "eval_time": eval_time,
    }

    # Save per-item result
    if results_dir:
        result_path = os.path.join(results_dir, f"result_{item_id}.json")
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

    # Save updated graph (Hebbian edges evolved during retrieval)
    memory_graph.save()

    status = "CORRECT" if correct else "WRONG"
    print(f"  [{item_idx}] {item_id} ({question_type}) -> {status} ({eval_time:.1f}s)")

    return result


# ========== Main Evaluation ==========

def eval_longmemeval(
    data_path: str,
    mem_dir: str,
    num_items: Optional[int] = None,
    start_item: int = 0,
    top_k: int = 20,
    semantic_top_k: int = 5,
    use_consolidation: bool = False,
    workers: int = 20,
) -> None:
    """
    Run LongMemEval-S evaluation on encoded Hebbian memories.

    Args:
        data_path: Path to longmemeval_s.json
        mem_dir: Directory with encoded memory files
        num_items: Number of items to evaluate (default: all)
        start_item: Start index
        top_k: Top-K episodic retrieval
        semantic_top_k: Top-K semantic retrieval
        use_consolidation: Whether to run consolidation before eval
        workers: Number of parallel workers
    """
    print("=" * 70)
    print("LongMemEval Hebbian Evaluation")
    print("=" * 70)
    print(f"Data: {data_path}")
    print(f"Memory dir: {mem_dir}")
    print(f"top_k={top_k}, semantic_top_k={semantic_top_k}")
    print(f"Consolidation: {'ON' if use_consolidation else 'OFF'}")
    print(f"Workers: {workers}")
    print(f"Hebbian params: max_flipped={os.environ.get('HEBBIAN_MAX_FLIPPED', '5')}, "
          f"lr={os.environ.get('HEBBIAN_LEARNING_RATE', '0.02')}, "
          f"alpha={os.environ.get('HEBBIAN_ACTIVATION_ALPHA', '0.1')}")
    print("=" * 70)

    # Load dataset
    with open(data_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    print(f"Loaded {len(dataset)} items")

    # Select items
    items = dataset[start_item:]
    if num_items is not None:
        items = items[:num_items]
    print(f"Evaluating items [{start_item}:{start_item + len(items)}]")

    # Create results directory
    results_dir = os.path.join(mem_dir, "eval_results")
    os.makedirs(results_dir, exist_ok=True)

    # Evaluate items in parallel
    all_results = []

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_idx = {
            executor.submit(
                evaluate_single_item,
                item,
                start_item + i,
                mem_dir,
                top_k,
                semantic_top_k,
                use_consolidation,
                results_dir,
            ): start_item + i
            for i, item in enumerate(items)
        }

        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                result = future.result()
                all_results.append(result)
                if len(all_results) % 50 == 0:
                    correct_so_far = sum(r["correct"] for r in all_results)
                    print(f"\n  Progress: {len(all_results)}/{len(items)} | "
                          f"Accuracy: {correct_so_far}/{len(all_results)} "
                          f"({correct_so_far / len(all_results) * 100:.1f}%)\n")
            except Exception as e:
                print(f"  [ERROR] Item {idx}: {e}")
                import traceback
                traceback.print_exc()

    # Sort results by question_id for consistency
    all_results.sort(key=lambda x: x["question_id"])

    # Compute metrics
    total = len(all_results)
    total_correct = sum(r["correct"] for r in all_results)
    overall_accuracy = total_correct / total * 100 if total > 0 else 0.0

    # Per-type metrics
    type_metrics = {}
    for r in all_results:
        qt = r["question_type"]
        if qt not in type_metrics:
            type_metrics[qt] = {"correct": 0, "total": 0}
        type_metrics[qt]["total"] += 1
        type_metrics[qt]["correct"] += r["correct"]

    print(f"\n{'=' * 70}")
    print(f"RESULTS")
    print(f"{'=' * 70}")
    print(f"Total items: {total}")
    print(f"Overall accuracy: {total_correct}/{total} ({overall_accuracy:.2f}%)")
    print(f"\nPer-type accuracy:")
    for qt in sorted(type_metrics.keys()):
        m = type_metrics[qt]
        acc = m["correct"] / m["total"] * 100 if m["total"] > 0 else 0.0
        print(f"  {qt:30s}: {m['correct']:3d}/{m['total']:3d} ({acc:.2f}%)")
    print(f"{'=' * 70}")

    # Save summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "data_path": data_path,
        "mem_dir": mem_dir,
        "total_items": total,
        "total_correct": total_correct,
        "overall_accuracy": overall_accuracy,
        "per_type": {
            qt: {
                "correct": m["correct"],
                "total": m["total"],
                "accuracy": m["correct"] / m["total"] * 100 if m["total"] > 0 else 0.0,
            }
            for qt, m in type_metrics.items()
        },
        "params": {
            "top_k": top_k,
            "semantic_top_k": semantic_top_k,
            "use_consolidation": use_consolidation,
            "max_flipped": os.environ.get("HEBBIAN_MAX_FLIPPED", "5"),
            "learning_rate": os.environ.get("HEBBIAN_LEARNING_RATE", "0.02"),
            "activation_alpha": os.environ.get("HEBBIAN_ACTIVATION_ALPHA", "0.1"),
            "spreading_threshold": os.environ.get("HEBBIAN_SPREADING_THRESHOLD", "0.4"),
            "decay_rate": os.environ.get("HEBBIAN_DECAY_RATE", "0.995"),
            "keyword_weight": os.environ.get("HEBBIAN_KEYWORD_WEIGHT", "0.5"),
            "tau": os.environ.get("HEBBIAN_TAU", "1e7"),
            "use_reranker": os.environ.get("HEBBIAN_USE_RERANKER", "false"),
        },
        "results": all_results,
    }

    summary_path = os.path.join(results_dir, "eval_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Summary saved: {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate LongMemEval-S with Hebbian Memory"
    )
    parser.add_argument(
        "--data_path", type=str, required=True,
        help="Path to longmemeval_s.json",
    )
    parser.add_argument(
        "--mem_dir", type=str, required=True,
        help="Memory directory (output of encode_longmemeval)",
    )
    parser.add_argument("--num_items", type=int, default=None, help="Number of items")
    parser.add_argument("--start_item", type=int, default=0, help="Start index")
    parser.add_argument(
        "--top_k", type=int, default=None,
        help="Top-K episodic retrieval (default: from HEBBIAN_TOP_K env or 20)",
    )
    parser.add_argument(
        "--semantic_top_k", type=int, default=5,
        help="Top-K semantic retrieval (default: 5)",
    )
    parser.add_argument(
        "--use_consolidation", action="store_true",
        help="Run consolidation before evaluation",
    )
    parser.add_argument("--workers", type=int, default=20, help="Parallel workers")

    args = parser.parse_args()

    top_k = args.top_k or int(os.environ.get("HEBBIAN_TOP_K", "20"))

    eval_longmemeval(
        data_path=args.data_path,
        mem_dir=args.mem_dir,
        num_items=args.num_items,
        start_item=args.start_item,
        top_k=top_k,
        semantic_top_k=args.semantic_top_k,
        use_consolidation=args.use_consolidation,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
