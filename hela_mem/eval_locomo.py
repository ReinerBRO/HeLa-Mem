"""
LoCoMo evaluation for Hebbian Memory.

Loads encoded LoCoMo Hebbian memory graphs, answers each QA pair, and reports
F1 / BLEU-1 metrics by sample and question category.

Usage:
    python -m hela_mem.eval_locomo \
        --data_path /path/to/locomo10.json \
        --mem_dir results/locomo_mem \
        [--top_k 10] [--knowledge_top_k 5] [--workers 5]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

from .encode_locomo import load_locomo_dataset, select_samples
from .hebbian_knowledge_memory import HebbianKnowledgeMemory
from .hebbian_memory import HebbianMemoryGraph
from .hebbian_retriever import HebbianRetriever
from .utils import get_timestamp


CATEGORY_NAMES = {
    1: "Multi-hop",
    2: "Temporal",
    3: "Open-domain",
    4: "Single-hop",
    5: "Adversarial",
}


def normalize_text(text: Any) -> str:
    """SQuAD/GAM-style normalization used by the original LoCoMo evaluation."""
    value = "" if text is None else str(text)
    value = value.lower().strip()
    value = re.sub(r"[^\w\s]", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    value = re.sub(r"(^|\s)(a|an|the)(\s|$)", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def tokens(text: Any) -> List[str]:
    return normalize_text(text).split()


def calculate_f1(prediction: Any, reference: Any) -> float:
    """Token F1 with token counts, matching the GAM-style LoCoMo metric."""
    pred_tokens = tokens(prediction)
    ref_tokens = tokens(reference)
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0

    pred_counts = Counter(pred_tokens)
    ref_counts = Counter(ref_tokens)
    overlap = sum(min(pred_counts[token], ref_counts[token]) for token in pred_counts)
    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def calculate_bleu1(prediction: Any, reference: Any) -> float:
    """BLEU-1 with clipped unigram precision and brevity penalty."""
    pred_tokens = tokens(prediction)
    ref_tokens = tokens(reference)
    if not pred_tokens or not ref_tokens:
        return 0.0

    pred_counts = Counter(pred_tokens)
    ref_counts = Counter(ref_tokens)
    clipped = sum(min(pred_counts[token], ref_counts[token]) for token in pred_counts)
    precision = clipped / len(pred_tokens)
    brevity_penalty = 1.0 if len(pred_tokens) >= len(ref_tokens) else math.exp(1 - len(ref_tokens) / len(pred_tokens))
    return brevity_penalty * precision


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _category_key(category: Any) -> int:
    try:
        return int(category)
    except (TypeError, ValueError):
        return 0


def _answer_for_qa(
    retriever: HebbianRetriever,
    qa: Dict[str, Any],
    speaker_a: str,
    speaker_b: str,
    top_k: int,
    knowledge_top_k: int,
) -> Dict[str, Any]:
    question = str(qa.get("question", ""))
    original_answer = qa.get("answer", "")
    if original_answer in ("", None):
        original_answer = qa.get("adversarial_answer", "")

    t_start = time.time()
    try:
        system_answer, retrieved_context = retriever.answer(
            question,
            speaker_a,
            speaker_b,
            top_k=top_k,
            knowledge_top_k=knowledge_top_k,
        )
    except Exception as exc:
        print(f"    [QA ERROR] {question[:80]}: {exc}")
        system_answer = ""
        retrieved_context = []

    f1 = calculate_f1(system_answer, original_answer)
    bleu1 = calculate_bleu1(system_answer, original_answer)

    return {
        "question": question,
        "system_answer": system_answer,
        "original_answer": str(original_answer),
        "category": qa.get("category"),
        "category_name": CATEGORY_NAMES.get(_category_key(qa.get("category")), "Unknown"),
        "evidence": qa.get("evidence", []),
        "retrieved_context": [
            item.get("node", {}).get("content", "")
            for item in retrieved_context
            if isinstance(item, dict)
        ],
        "f1": f1,
        "bleu1": bleu1,
        "eval_time": time.time() - t_start,
        "timestamp": get_timestamp(),
    }


def _mean(values: Sequence[float]) -> float:
    return statistics.mean(values) if values else 0.0


def _std(values: Sequence[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def summarize_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate LoCoMo metrics overall, by category, and by sample."""
    f1_scores = [float(row.get("f1", 0.0)) for row in results]
    bleu1_scores = [float(row.get("bleu1", 0.0)) for row in results]

    per_category: Dict[str, Dict[str, Any]] = {}
    category_groups: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for row in results:
        category_groups[_category_key(row.get("category"))].append(row)

    for category, rows in sorted(category_groups.items()):
        cat_f1 = [float(row.get("f1", 0.0)) for row in rows]
        cat_bleu = [float(row.get("bleu1", 0.0)) for row in rows]
        per_category[str(category)] = {
            "name": CATEGORY_NAMES.get(category, "Unknown"),
            "count": len(rows),
            "f1": _mean(cat_f1),
            "f1_std": _std(cat_f1),
            "bleu1": _mean(cat_bleu),
            "bleu1_std": _std(cat_bleu),
        }

    per_sample: Dict[str, Dict[str, Any]] = {}
    sample_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in results:
        sample_groups[str(row.get("sample_id", "unknown"))].append(row)

    for sample_id, rows in sorted(sample_groups.items()):
        sample_f1 = [float(row.get("f1", 0.0)) for row in rows]
        sample_bleu = [float(row.get("bleu1", 0.0)) for row in rows]
        per_sample[sample_id] = {
            "count": len(rows),
            "f1": _mean(sample_f1),
            "bleu1": _mean(sample_bleu),
        }

    return {
        "total_qa": len(results),
        "overall": {
            "f1": _mean(f1_scores),
            "f1_std": _std(f1_scores),
            "bleu1": _mean(bleu1_scores),
            "bleu1_std": _std(bleu1_scores),
        },
        "per_category": per_category,
        "per_sample": per_sample,
    }


def evaluate_single_sample(
    sample: Dict[str, Any],
    mem_dir: str,
    results_dir: str,
    top_k: int = 10,
    knowledge_top_k: int = 5,
    max_qa: Optional[int] = None,
) -> Dict[str, Any]:
    """Evaluate all selected QA pairs for one LoCoMo sample."""
    sample_id = str(sample.get("sample_id", "unknown_sample"))
    conversation = sample["conversation"]
    speaker_a = conversation["speaker_a"]
    speaker_b = conversation["speaker_b"]

    mem_path = os.path.join(mem_dir, f"{sample_id}_hebbian.json")
    if not os.path.exists(mem_path):
        raise FileNotFoundError(f"Memory file not found for {sample_id}: {mem_path}")

    memory_graph = HebbianMemoryGraph(file_path=mem_path)
    knowledge_memory = HebbianKnowledgeMemory(
        file_path=os.path.join(mem_dir, f"{sample_id}_long_term.json")
    )
    retriever = HebbianRetriever(
        memory_graph,
        profile_memory=knowledge_memory,
        use_planner=_env_flag("HEBBIAN_USE_PLANNER"),
        use_investigator=_env_flag("HEBBIAN_USE_INVESTIGATOR"),
        use_critic=_env_flag("HEBBIAN_USE_CRITIC"),
        use_surgeon=_env_flag("HEBBIAN_USE_SURGEON"),
        use_architect=_env_flag("HEBBIAN_USE_ARCHITECT"),
        use_hippocampus=_env_flag("HEBBIAN_USE_HIPPOCAMPUS"),
        use_extra_prompt=_env_flag("HEBBIAN_USE_EXTRA_PROMPT"),
    )

    qa_pairs = list(sample.get("qa", []))
    if max_qa is not None:
        qa_pairs = qa_pairs[:max_qa]

    print(f"  [{sample_id}] Evaluating {len(qa_pairs)} QA pairs")
    qa_results: List[Dict[str, Any]] = []

    for qa_idx, qa in enumerate(qa_pairs):
        result = _answer_for_qa(
            retriever, qa, speaker_a, speaker_b, top_k, knowledge_top_k
        )
        result["sample_id"] = sample_id
        result["qa_index"] = qa_idx
        qa_results.append(result)

        if (qa_idx + 1) % 25 == 0:
            sample_summary = summarize_results(qa_results)
            print(
                f"    [{sample_id}] {qa_idx + 1}/{len(qa_pairs)} | "
                f"F1={sample_summary['overall']['f1'] * 100:.2f} | "
                f"BLEU-1={sample_summary['overall']['bleu1'] * 100:.2f}"
            )

    memory_graph.save()

    sample_summary = summarize_results(qa_results)
    sample_output = {
        "sample_id": sample_id,
        "speaker_a": speaker_a,
        "speaker_b": speaker_b,
        "top_k": top_k,
        "knowledge_top_k": knowledge_top_k,
        "summary": sample_summary,
        "results": qa_results,
    }
    result_path = os.path.join(results_dir, f"result_{sample_id}.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(sample_output, f, ensure_ascii=False, indent=2)

    print(
        f"  [{sample_id}] Done | "
        f"F1={sample_summary['overall']['f1'] * 100:.2f} | "
        f"BLEU-1={sample_summary['overall']['bleu1'] * 100:.2f}"
    )
    return sample_output


def eval_locomo(
    data_path: str,
    mem_dir: str,
    output_dir: Optional[str] = None,
    num_samples: Optional[int] = None,
    start_sample: int = 0,
    sample_ids: Optional[Sequence[str]] = None,
    top_k: int = 10,
    knowledge_top_k: int = 5,
    workers: int = 5,
    max_qa_per_sample: Optional[int] = None,
) -> Dict[str, Any]:
    """Run LoCoMo evaluation on encoded Hebbian memories."""
    print("=" * 70)
    print("LoCoMo Hebbian Evaluation")
    print("=" * 70)
    print(f"Data: {data_path}")
    print(f"Memory dir: {mem_dir}")
    print(f"top_k={top_k}, knowledge_top_k={knowledge_top_k}")
    print(f"Workers: {workers}")
    print(f"Hebbian params: max_flipped={os.environ.get('HEBBIAN_MAX_FLIPPED', '5')}, "
          f"lr={os.environ.get('HEBBIAN_LEARNING_RATE', '0.02')}, "
          f"alpha={os.environ.get('HEBBIAN_ACTIVATION_ALPHA', '0.1')}")
    print("=" * 70)

    dataset = load_locomo_dataset(data_path)
    samples = select_samples(dataset, start_sample, num_samples, sample_ids)
    print(f"Loaded {len(dataset)} samples; evaluating {len(samples)}")

    results_dir = output_dir or os.path.join(mem_dir, "eval_results")
    os.makedirs(results_dir, exist_ok=True)

    sample_outputs: List[Dict[str, Any]] = []
    failed: List[Dict[str, str]] = []

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_sample = {
            executor.submit(
                evaluate_single_sample,
                sample,
                mem_dir,
                results_dir,
                top_k,
                knowledge_top_k,
                max_qa_per_sample,
            ): str(sample.get("sample_id", f"sample_{i}"))
            for i, sample in enumerate(samples)
        }

        for future in as_completed(future_to_sample):
            sample_id = future_to_sample[future]
            try:
                sample_outputs.append(future.result())
            except Exception as exc:
                print(f"  [ERROR] {sample_id}: {exc}")
                import traceback
                traceback.print_exc()
                failed.append({"sample_id": sample_id, "error": str(exc)})

    all_results: List[Dict[str, Any]] = []
    for sample_output in sample_outputs:
        all_results.extend(sample_output.get("results", []))

    summary_metrics = summarize_results(all_results)
    summary = {
        "timestamp": datetime.now().isoformat(),
        "data_path": data_path,
        "mem_dir": mem_dir,
        "results_dir": results_dir,
        "failed": failed,
        "params": {
            "top_k": top_k,
            "knowledge_top_k": knowledge_top_k,
            "max_flipped": os.environ.get("HEBBIAN_MAX_FLIPPED", "5"),
            "learning_rate": os.environ.get("HEBBIAN_LEARNING_RATE", "0.02"),
            "activation_alpha": os.environ.get("HEBBIAN_ACTIVATION_ALPHA", "0.1"),
            "spreading_threshold": os.environ.get("HEBBIAN_SPREADING_THRESHOLD", "0.6"),
            "decay_rate": os.environ.get("HEBBIAN_DECAY_RATE", "0.995"),
            "keyword_weight": os.environ.get("HEBBIAN_KEYWORD_WEIGHT", "0.5"),
            "tau": os.environ.get("HEBBIAN_TAU", "5184000"),
            "use_extra_prompt": os.environ.get("HEBBIAN_USE_EXTRA_PROMPT", "false"),
        },
        "metrics": summary_metrics,
        "results": sorted(
            all_results,
            key=lambda row: (str(row.get("sample_id", "")), int(row.get("qa_index", 0))),
        ),
    }

    summary_path = os.path.join(results_dir, "eval_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    overall = summary_metrics["overall"]
    print(f"\n{'=' * 70}")
    print("RESULTS")
    print(f"{'=' * 70}")
    print(f"Total QA: {summary_metrics['total_qa']}")
    print(f"Overall F1:     {overall['f1'] * 100:.2f}%")
    print(f"Overall BLEU-1: {overall['bleu1'] * 100:.2f}%")
    print("\nPer-category:")
    for category, metrics in summary_metrics["per_category"].items():
        print(
            f"  {category} {metrics['name']:<12s} "
            f"count={metrics['count']:4d} | "
            f"F1={metrics['f1'] * 100:6.2f}% | "
            f"BLEU-1={metrics['bleu1'] * 100:6.2f}%"
        )
    if failed:
        print(f"\nFailed samples: {failed}")
    print(f"Summary saved: {summary_path}")
    print(f"{'=' * 70}")

    return summary


def _parse_sample_ids(value: Optional[str]) -> Optional[List[str]]:
    if not value:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate LoCoMo with Hebbian Memory")
    parser.add_argument("--data_path", type=str, required=True, help="Path to locomo10.json")
    parser.add_argument("--mem_dir", type=str, required=True, help="Memory directory from encode_locomo")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for eval results")
    parser.add_argument("--num_samples", type=int, default=None, help="Number of samples")
    parser.add_argument("--start_sample", type=int, default=0, help="Start sample index")
    parser.add_argument(
        "--sample_ids",
        type=str,
        default=None,
        help="Comma-separated sample ids to evaluate, e.g. conv-26,conv-30",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=None,
        help="Top-K episodic retrieval (default: HEBBIAN_TOP_K or 10)",
    )
    parser.add_argument(
        "--knowledge_top_k",
        type=int,
        default=None,
        help="Top-K semantic retrieval (default: HEBBIAN_KNOWLEDGE_TOP_K or 5)",
    )
    parser.add_argument("--workers", type=int, default=5, help="Parallel sample workers")
    parser.add_argument(
        "--max_qa_per_sample",
        type=int,
        default=None,
        help="Limit QA pairs per sample for sanity checks",
    )
    args = parser.parse_args()

    top_k = args.top_k if args.top_k is not None else int(os.environ.get("HEBBIAN_TOP_K", "10"))
    knowledge_top_k = (
        args.knowledge_top_k
        if args.knowledge_top_k is not None
        else int(os.environ.get("HEBBIAN_KNOWLEDGE_TOP_K", "5"))
    )

    eval_locomo(
        data_path=args.data_path,
        mem_dir=args.mem_dir,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        start_sample=args.start_sample,
        sample_ids=_parse_sample_ids(args.sample_ids),
        top_k=top_k,
        knowledge_top_k=knowledge_top_k,
        workers=args.workers,
        max_qa_per_sample=args.max_qa_per_sample,
    )


if __name__ == "__main__":
    main()
