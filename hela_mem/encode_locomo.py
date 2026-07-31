"""
LoCoMo encoding for Hebbian Memory.

Encodes LoCoMo conversations into one Hebbian memory graph per sample.

Usage:
    python -m hela_mem.encode_locomo \
        --data_path /path/to/locomo10.json \
        --output_dir results/locomo_mem \
        [--num_samples 10] [--start_sample 0] [--workers 5] \
        [--skip_knowledge]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

from .hebbian_knowledge_memory import HebbianKnowledgeMemory
from .hebbian_memory import HebbianMemoryGraph
from .hebbian_retriever import HebbianRetriever
from .profile_utils import OpenAIClient, gpt_personality_analysis, gpt_update_profile
from .utils import get_timestamp, load_api_keys


BUFFER_SIZE = int(os.environ.get("HEBBIAN_KNOWLEDGE_BUFFER_SIZE", "10"))
_ENCODE_BASE_URL = os.environ.get("OPENAI_BASE_URL")


def load_locomo_dataset(data_path: str) -> List[Dict[str, Any]]:
    """Load LoCoMo JSON in either list or {'samples': [...]} format."""
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data
    if isinstance(data, dict) and isinstance(data.get("samples"), list):
        return data["samples"]
    raise ValueError("Unrecognized LoCoMo JSON shape. Expected a list or {'samples': [...]}.")


def _session_sort_key(key: str) -> int:
    match = re.fullmatch(r"session_(\d+)", key)
    return int(match.group(1)) if match else 10**9


def _dialog_text(dialog: Dict[str, Any]) -> str:
    text = str(dialog.get("text", "")).strip()
    caption = str(dialog.get("blip_caption", "") or "").strip()
    if caption:
        text = f"{text} (image description: {caption})" if text else f"(image description: {caption})"
    return text


def parse_conversation(sample: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Parse a LoCoMo sample into user/assistant turns.

    LoCoMo stores speaker names in conversation['speaker_a'] and conversation['speaker_b'].
    The original HeLa-Mem LoCoMo scripts treat speaker_a as the user and pair each
    speaker_a utterance with the following non-speaker_a utterance.
    """
    conversation = sample["conversation"]
    speaker_a = conversation["speaker_a"]
    turns: List[Dict[str, str]] = []

    session_keys = sorted(
        [
            key for key in conversation
            if key.startswith("session_") and not key.endswith("_date_time")
        ],
        key=_session_sort_key,
    )

    for session_key in session_keys:
        timestamp = conversation.get(f"{session_key}_date_time") or get_timestamp()
        pending_user: List[str] = []

        for dialog in conversation.get(session_key, []):
            speaker = dialog.get("speaker", "")
            text = _dialog_text(dialog)
            if not text:
                continue

            if speaker == speaker_a:
                pending_user.append(text)
            elif pending_user:
                turns.append({
                    "user_text": pending_user.pop(0),
                    "ai_text": text,
                    "timestamp": timestamp,
                })

    return turns


def _iter_clean_lines(text: str) -> List[str]:
    lines = []
    for line in str(text or "").splitlines():
        clean = line.strip().lstrip("- ").strip()
        if clean and clean.lower() != "none" and not clean.startswith("【"):
            lines.append(clean)
    return lines


def process_incremental_buffer(
    buffer: List[Dict[str, str]],
    knowledge_memory: HebbianKnowledgeMemory,
    sample_id: str,
    client: Any,
) -> None:
    """Extract profile, user facts, and assistant knowledge from a batch of turns."""
    if not buffer:
        return

    dialogs_for_analysis = [
        {
            "user_input": turn["user_text"],
            "agent_response": turn["ai_text"],
            "timestamp": turn["timestamp"],
        }
        for turn in buffer
    ]

    try:
        result = gpt_personality_analysis(dialogs_for_analysis, client)

        new_profile = result["profile"]
        old_profile = knowledge_memory.get_raw_user_profile(sample_id)
        updated_profile = (
            gpt_update_profile(old_profile, new_profile, client)
            if old_profile else new_profile
        )
        knowledge_memory.update_user_profile(sample_id, updated_profile)

        for fact in _iter_clean_lines(result.get("private", "")):
            knowledge_memory.add_knowledge(fact)

        for fact in _iter_clean_lines(result.get("assistant_knowledge", "")):
            knowledge_memory.add_assistant_knowledge(fact)
    except Exception as exc:
        print(f"  [{sample_id}] Knowledge extraction error: {exc}")


def encode_single_sample(
    sample: Dict[str, Any],
    output_dir: str,
    client: Any,
    skip_knowledge: bool = False,
) -> Dict[str, Any]:
    """Encode one LoCoMo sample into Hebbian graph and semantic memory files."""
    sample_id = str(sample.get("sample_id") or f"sample_{hash(json.dumps(sample, sort_keys=True)) & 0xffff:x}")
    t_start = time.time()

    mem_path = os.path.join(output_dir, f"{sample_id}_hebbian.json")
    kb_path = os.path.join(output_dir, f"{sample_id}_long_term.json")

    memory_graph = HebbianMemoryGraph(file_path=mem_path)
    knowledge_memory = HebbianKnowledgeMemory(file_path=kb_path)
    retriever = HebbianRetriever(memory_graph, profile_memory=knowledge_memory)

    turns = parse_conversation(sample)
    session_count = len([
        key for key in sample.get("conversation", {})
        if key.startswith("session_") and not key.endswith("_date_time")
    ])
    print(f"  [{sample_id}] {len(turns)} turns from {session_count} sessions")

    incremental_buffer: List[Dict[str, str]] = []
    for turn in turns:
        retriever.process_conversation_turn(
            turn["user_text"], turn["ai_text"], timestamp=turn["timestamp"]
        )

        if not skip_knowledge:
            incremental_buffer.append(turn)
            if len(incremental_buffer) >= BUFFER_SIZE:
                process_incremental_buffer(
                    incremental_buffer, knowledge_memory, sample_id, client
                )
                incremental_buffer = []

    if incremental_buffer and not skip_knowledge:
        process_incremental_buffer(
            incremental_buffer, knowledge_memory, sample_id, client
        )

    memory_graph.save()
    knowledge_memory.save()

    encoding_time = time.time() - t_start
    print(f"  [{sample_id}] Done: {len(memory_graph.nodes)} nodes, {encoding_time:.1f}s")

    return {
        "sample_id": sample_id,
        "num_nodes": len(memory_graph.nodes),
        "num_turns": len(turns),
        "num_qa": len(sample.get("qa", [])),
        "encoding_time": encoding_time,
    }


def select_samples(
    dataset: Sequence[Dict[str, Any]],
    start_sample: int = 0,
    num_samples: Optional[int] = None,
    sample_ids: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    """Select samples by ids or contiguous slice."""
    if sample_ids:
        wanted = {str(item) for item in sample_ids}
        selected = [
            sample for sample in dataset
            if str(sample.get("sample_id", "")) in wanted
        ]
        missing = wanted - {str(sample.get("sample_id", "")) for sample in selected}
        if missing:
            raise ValueError(f"Unknown sample_id(s): {', '.join(sorted(missing))}")
        return selected

    selected = list(dataset[start_sample:])
    if num_samples is not None:
        selected = selected[:num_samples]
    return selected


def encode_locomo(
    data_path: str,
    output_dir: str,
    num_samples: Optional[int] = None,
    start_sample: int = 0,
    sample_ids: Optional[Sequence[str]] = None,
    workers: int = 5,
    skip_knowledge: bool = False,
) -> str:
    """Encode LoCoMo conversations into Hebbian memory graphs."""
    print("=" * 70)
    print("LoCoMo Hebbian Encoding")
    print("=" * 70)
    print(f"Data: {data_path}")
    print(f"Output: {output_dir}")
    print(f"Workers: {workers}")
    print(f"Knowledge extraction: {'OFF' if skip_knowledge else 'ON (buffer=' + str(BUFFER_SIZE) + ')'}")
    print(f"Hebbian params: lr={os.environ.get('HEBBIAN_LEARNING_RATE', '0.02')}, "
          f"decay={os.environ.get('HEBBIAN_DECAY_RATE', '0.995')}, "
          f"alpha={os.environ.get('HEBBIAN_ACTIVATION_ALPHA', '0.1')}")
    print("=" * 70)

    dataset = load_locomo_dataset(data_path)
    samples = select_samples(dataset, start_sample, num_samples, sample_ids)
    print(f"Loaded {len(dataset)} samples; encoding {len(samples)}")

    os.makedirs(output_dir, exist_ok=True)

    clients: List[OpenAIClient] = []
    if not skip_knowledge:
        keys = load_api_keys()
        if not keys:
            raise RuntimeError(
                "Knowledge extraction requires OPENAI_API_KEY or OPENAI_API_KEYS. "
                "Pass --skip_knowledge to build only episodic graphs."
            )
        clients = [OpenAIClient(api_key=key, base_url=_ENCODE_BASE_URL) for key in keys]
        print(f"Initialized {len(clients)} API clients for key rotation")

    results: List[Dict[str, Any]] = []
    failed: List[str] = []

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_sample = {
            executor.submit(
                encode_single_sample,
                sample,
                output_dir,
                clients[i % len(clients)] if clients else None,
                skip_knowledge,
            ): str(sample.get("sample_id", f"sample_{i}"))
            for i, sample in enumerate(samples)
        }

        for future in as_completed(future_to_sample):
            sample_id = future_to_sample[future]
            try:
                results.append(future.result())
            except Exception as exc:
                print(f"  [ERROR] {sample_id}: {exc}")
                import traceback
                traceback.print_exc()
                failed.append(sample_id)

    summary = {
        "timestamp": datetime.now().isoformat(),
        "data_path": data_path,
        "output_dir": output_dir,
        "total_samples": len(samples),
        "encoded": len(results),
        "failed": len(failed),
        "failed_ids": failed,
        "skip_knowledge": skip_knowledge,
        "buffer_size": BUFFER_SIZE,
        "params": {
            "learning_rate": os.environ.get("HEBBIAN_LEARNING_RATE", "0.02"),
            "decay_rate": os.environ.get("HEBBIAN_DECAY_RATE", "0.995"),
            "activation_alpha": os.environ.get("HEBBIAN_ACTIVATION_ALPHA", "0.1"),
            "spreading_threshold": os.environ.get("HEBBIAN_SPREADING_THRESHOLD", "0.6"),
            "max_flipped": os.environ.get("HEBBIAN_MAX_FLIPPED", "5"),
            "tau": os.environ.get("HEBBIAN_TAU", "5184000"),
        },
        "results": sorted(results, key=lambda row: row["sample_id"]),
    }

    summary_path = os.path.join(output_dir, "encode_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n{'=' * 70}")
    print(f"Encoding complete: {len(results)}/{len(samples)} samples")
    if failed:
        print(f"Failed: {failed}")
    print(f"Total nodes: {sum(row['num_nodes'] for row in results)}")
    print(f"Summary: {summary_path}")
    print(f"{'=' * 70}")

    return output_dir


def _parse_sample_ids(value: Optional[str]) -> Optional[List[str]]:
    if not value:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Encode LoCoMo into Hebbian Memory")
    parser.add_argument("--data_path", type=str, required=True, help="Path to locomo10.json")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    parser.add_argument("--num_samples", type=int, default=None, help="Number of samples")
    parser.add_argument("--start_sample", type=int, default=0, help="Start sample index")
    parser.add_argument(
        "--sample_ids",
        type=str,
        default=None,
        help="Comma-separated sample ids to encode, e.g. conv-26,conv-30",
    )
    parser.add_argument("--workers", type=int, default=5, help="Parallel sample workers")
    parser.add_argument(
        "--skip_knowledge",
        action="store_true",
        help="Skip incremental profile and semantic knowledge extraction",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    if output_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"results/locomo_mem_{ts}"

    encode_locomo(
        data_path=args.data_path,
        output_dir=output_dir,
        num_samples=args.num_samples,
        start_sample=args.start_sample,
        sample_ids=_parse_sample_ids(args.sample_ids),
        workers=args.workers,
        skip_knowledge=args.skip_knowledge,
    )


if __name__ == "__main__":
    main()
