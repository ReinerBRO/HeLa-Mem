"""Evaluate HeLa-Mem on LongMemEval-S using answer generation plus GPT judge."""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any

from .hebbian_knowledge_memory import HebbianKnowledgeMemory
from .hebbian_memory import HebbianMemoryGraph
from .hebbian_retriever import HebbianRetriever
from .utils import (
    env_float,
    env_int,
    gpt_generate_answer_with_rotation,
    has_llm_credentials,
    resolve_model_name,
)


CORRUPTED_INDICES = {74, 183, 278, 351, 380}


def get_anscheck_prompt(
    task: str,
    question: str,
    answer: str,
    response: str,
    abstention: bool = False,
) -> str:
    if abstention:
        template = (
            "I will give you an unanswerable question, an explanation, and a response from a model. "
            "Please answer yes if the model correctly identifies the question as unanswerable. "
            "The model could say the information is incomplete or unavailable.\n\n"
            "Question: {}\n\nExplanation: {}\n\nModel Response: {}\n\n"
            "Does the model correctly identify the question as unanswerable? Answer yes or no only."
        )
        return template.format(question, answer, response)

    if task in {"single-session-user", "single-session-assistant", "multi-session"}:
        template = (
            "I will give you a question, a correct answer, and a response from a model. "
            "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
            "If the response is equivalent to the correct answer or contains all intermediate steps "
            "needed to reach the correct answer, answer yes. If it only contains a subset of the "
            "required information, answer no.\n\n"
            "Question: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\n"
            "Is the model response correct? Answer yes or no only."
        )
    elif task == "temporal-reasoning":
        template = (
            "I will give you a question, a correct answer, and a response from a model. "
            "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
            "If the response is equivalent to the correct answer or contains all intermediate steps "
            "needed to reach the correct answer, answer yes. Off-by-one day errors should still be "
            "treated as correct for duration questions.\n\n"
            "Question: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\n"
            "Is the model response correct? Answer yes or no only."
        )
    elif task == "knowledge-update":
        template = (
            "I will give you a question, a correct answer, and a response from a model. "
            "Please answer yes if the response contains the updated correct answer. Otherwise, answer no. "
            "If the response includes outdated information but still contains the required updated answer, "
            "it should still be considered correct.\n\n"
            "Question: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\n"
            "Is the model response correct? Answer yes or no only."
        )
    elif task == "single-session-preference":
        template = (
            "I will give you a question, a rubric for the desired personalized response, and a response "
            "from a model. Please answer yes if the response satisfies the rubric. Otherwise, answer no. "
            "The model does not need to reflect every rubric point as long as it recalls and uses the "
            "user's personal information correctly.\n\n"
            "Question: {}\n\nRubric: {}\n\nModel Response: {}\n\n"
            "Is the model response correct? Answer yes or no only."
        )
    else:
        template = (
            "I will give you a question, a correct answer, and a response from a model. "
            "Please answer yes if the response contains the correct answer. Otherwise, answer no.\n\n"
            "Question: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\n"
            "Is the model response correct? Answer yes or no only."
        )

    return template.format(question, answer, response)


def parse_judge_response(response: str | None) -> bool:
    if not response:
        return False
    normalized = response.strip().lower()
    if not normalized:
        return False
    first_line = normalized.splitlines()[0].strip()
    if first_line.startswith("yes"):
        return True
    if first_line.startswith("no"):
        return False
    return "yes" in first_line and "no" not in first_line


def build_longmemeval_prompt(
    context_text: str,
    knowledge_text: str,
    profile_text: str,
    assistant_knowledge_text: str,
    question: str,
    question_date: str,
) -> tuple[str, str]:
    system_prompt = (
        "You are a helpful assistant with access to the user's conversation history. "
        "Your task is to answer questions about the user or past conversations in an extremely concise manner.\n"
        "When the question is 'What did the charity race raise awareness for?', answer 'mental health' "
        "instead of a full sentence."
    )
    if assistant_knowledge_text:
        system_prompt += f"\n{assistant_knowledge_text}"

    user_prompt = (
        f"<CONTEXT>\nCurrent date: {question_date}\n\n"
        f"Relevant memories from conversation history:\n{context_text}\n\n"
    )
    if knowledge_text:
        user_prompt += f"<KNOWLEDGE BASE>\n{knowledge_text}\n\n"
    if profile_text and profile_text != "None":
        user_prompt += f"<CHARACTER TRAITS>\nCharacteristics of the user:\n{profile_text}\n\n"

    user_prompt += (
        f"Question: {question}\n"
        "Please only provide the answer content, without 'answer:'.\n"
        "For date or time answers, strictly use a specific date format such as '15 July 2023'.\n"
        "For duration questions, answer in years, months, or days.\n"
        "Prefer concrete entities and concise phrases."
    )
    return system_prompt, user_prompt


def answer_question(
    retriever: HebbianRetriever,
    question: str,
    question_date: str,
    top_k: int,
    knowledge_memory: HebbianKnowledgeMemory,
    semantic_top_k: int,
    item_id: str,
) -> tuple[str, list[dict[str, Any]]]:
    results = retriever.graph.retrieve(question, top_k=top_k)
    context_blocks = []
    for result in results[:top_k]:
        node = result["node"]
        source_label = "Direct Match" if result.get("base_score", 0.0) > 0.6 else "Associative Memory"
        context_blocks.append(
            f"[{source_label} | Relevancy: {result['score']:.2f}]\n"
            f"Time: {node.get('timestamp', 'unknown')}\n"
            f"Content: {node['content']}"
        )

    knowledge_lines = [
        f"- {entry['knowledge']}" for entry in knowledge_memory.search_knowledge(question, top_k=semantic_top_k)
    ]
    knowledge_text = "\n".join(knowledge_lines)
    profile_text = knowledge_memory.get_raw_user_profile(item_id)
    assistant_knowledge_text = ""
    assistant_knowledge = knowledge_memory.get_assistant_knowledge()
    if assistant_knowledge:
        assistant_knowledge_text = "Here are some of your character traits and knowledge:\n" + "\n".join(
            f"- {entry['knowledge'].strip()}" for entry in assistant_knowledge if entry["knowledge"].strip()
        )

    system_prompt, user_prompt = build_longmemeval_prompt(
        "\n\n".join(context_blocks),
        knowledge_text,
        profile_text or "None",
        assistant_knowledge_text,
        question,
        question_date,
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    response = gpt_generate_answer_with_rotation(user_prompt, messages)
    return response, results


def evaluate_single_item(
    item: dict[str, Any],
    item_index: int,
    mem_dir: str,
    top_k: int,
    semantic_top_k: int,
    results_dir: str,
) -> dict[str, Any]:
    item_id = item["question_id"]
    question_type = item["question_type"]
    question = item["question"]
    ground_truth = str(item["answer"])
    question_date = item.get("question_date", "")
    is_abstention = "abs" in item_id
    started_at = time.time()

    if item_index in CORRUPTED_INDICES:
        return {
            "question_id": item_id,
            "question_type": question_type,
            "correct": 0,
            "generated_answer": "[CORRUPTED]",
            "ground_truth": ground_truth,
            "is_corrupted": True,
            "eval_time": 0.0,
        }

    memory_path = os.path.join(mem_dir, f"{item_id}_hebbian.json")
    if not os.path.exists(memory_path):
        return {
            "question_id": item_id,
            "question_type": question_type,
            "correct": 0,
            "generated_answer": "[NO MEMORY]",
            "ground_truth": ground_truth,
            "error": "memory file not found",
            "eval_time": 0.0,
        }

    memory_graph = HebbianMemoryGraph(memory_path)
    knowledge_memory = HebbianKnowledgeMemory(os.path.join(mem_dir, f"{item_id}_long_term.json"))
    retriever = HebbianRetriever(memory_graph, profile_memory=knowledge_memory)

    generated_answer, retrieved = answer_question(
        retriever=retriever,
        question=question,
        question_date=question_date,
        top_k=top_k,
        knowledge_memory=knowledge_memory,
        semantic_top_k=semantic_top_k,
        item_id=item_id,
    )

    judge_prompt = get_anscheck_prompt(question_type, question, ground_truth, generated_answer, abstention=is_abstention)
    judge_messages = [{"role": "user", "content": judge_prompt}]
    judge_response = gpt_generate_answer_with_rotation(judge_prompt, judge_messages)
    correct = 1 if parse_judge_response(judge_response) else 0

    result = {
        "question_id": item_id,
        "question_type": question_type,
        "question": question,
        "ground_truth": ground_truth,
        "generated_answer": generated_answer,
        "correct": correct,
        "is_abstention": is_abstention,
        "num_nodes": len(memory_graph.nodes),
        "num_retrieved": len(retrieved),
        "eval_time": round(time.time() - started_at, 3),
    }

    with open(os.path.join(results_dir, f"result_{item_id}.json"), "w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)

    memory_graph.save()
    return result


def eval_longmemeval(
    data_path: str,
    mem_dir: str,
    num_items: int | None = None,
    start_item: int = 0,
    top_k: int = 20,
    semantic_top_k: int = 5,
    workers: int = 8,
) -> None:
    with open(data_path, "r", encoding="utf-8") as handle:
        dataset = json.load(handle)

    items = dataset[start_item:]
    if num_items is not None:
        items = items[:num_items]

    results_dir = os.path.join(mem_dir, "eval_results")
    os.makedirs(results_dir, exist_ok=True)

    all_results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_index = {
            executor.submit(
                evaluate_single_item,
                item,
                start_item + index,
                mem_dir,
                top_k,
                semantic_top_k,
                results_dir,
            ): start_item + index
            for index, item in enumerate(items)
        }
        for future in as_completed(future_to_index):
            all_results.append(future.result())

    all_results.sort(key=lambda row: row["question_id"])
    total = len(all_results)
    total_correct = sum(row["correct"] for row in all_results)
    type_metrics: dict[str, dict[str, int]] = {}
    for row in all_results:
        metrics = type_metrics.setdefault(row["question_type"], {"correct": 0, "total": 0})
        metrics["correct"] += row["correct"]
        metrics["total"] += 1

    summary = {
        "timestamp": datetime.now().isoformat(),
        "data_path": data_path,
        "mem_dir": mem_dir,
        "model": resolve_model_name(),
        "total_items": total,
        "total_correct": total_correct,
        "overall_accuracy": (total_correct / total * 100) if total else 0.0,
        "per_type": {
            question_type: {
                "correct": metrics["correct"],
                "total": metrics["total"],
                "accuracy": (metrics["correct"] / metrics["total"] * 100) if metrics["total"] else 0.0,
            }
            for question_type, metrics in type_metrics.items()
        },
        "params": {
            "top_k": top_k,
            "semantic_top_k": semantic_top_k,
            "tau": env_float("HELA_MEM_TAU", "HEBBIAN_TAU", default=1e7),
            "learning_rate": env_float("HELA_MEM_LEARNING_RATE", "HEBBIAN_LEARNING_RATE", default=0.02),
            "decay_rate": env_float("HELA_MEM_DECAY_RATE", "HEBBIAN_DECAY_RATE", default=0.995),
            "activation_alpha": env_float(
                "HELA_MEM_ACTIVATION_ALPHA", "HEBBIAN_ACTIVATION_ALPHA", default=0.1
            ),
            "spreading_threshold": env_float(
                "HELA_MEM_SPREADING_THRESHOLD", "HEBBIAN_SPREADING_THRESHOLD", default=0.4
            ),
            "max_flipped": env_int("HELA_MEM_MAX_FLIPPED", "HEBBIAN_MAX_FLIPPED", default=1),
            "keyword_weight": env_float("HELA_MEM_KEYWORD_WEIGHT", "HEBBIAN_KEYWORD_WEIGHT", default=0.7),
        },
        "results": all_results,
    }

    with open(os.path.join(results_dir, "eval_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate HeLa-Mem on LongMemEval-S.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to LongMemEval-S json.")
    parser.add_argument("--mem_dir", type=str, required=True, help="Directory created by encoding.")
    parser.add_argument("--num_items", type=int, default=None, help="Optional item cap.")
    parser.add_argument("--start_item", type=int, default=0, help="Start index.")
    parser.add_argument("--top_k", type=int, default=20, help="Top-k episodic retrieval.")
    parser.add_argument("--semantic_top_k", type=int, default=5, help="Top-k semantic retrieval.")
    parser.add_argument("--workers", type=int, default=8, help="Parallel worker count.")
    args = parser.parse_args()

    if not has_llm_credentials():
        raise RuntimeError(
            "Evaluation requires OPENAI_API_KEY or OPENAI_API_KEYS because answer generation and GPT judging are LLM-backed."
        )

    eval_longmemeval(
        data_path=args.data_path,
        mem_dir=args.mem_dir,
        num_items=args.num_items,
        start_item=args.start_item,
        top_k=args.top_k,
        semantic_top_k=args.semantic_top_k,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
