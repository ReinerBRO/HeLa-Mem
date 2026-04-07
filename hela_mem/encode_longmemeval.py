"""
LongMemEval Encoding for Hebbian Memory

Encodes LongMemEval-S haystack sessions into Hebbian memory graphs.
Each of the 500 items gets its own HebbianMemoryGraph + HebbianKnowledgeMemory.

Usage:
    python -m hela_mem.encode_longmemeval \
        --data_path /path/to/longmemeval_s.json \
        --output_dir results/longmemeval_mem \
        [--num_items 500] [--start_item 0] [--workers 5] \
        [--skip_knowledge]
"""

import json
import os
import argparse
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

from .hebbian_memory import HebbianMemoryGraph
from .hebbian_retriever import HebbianRetriever
from .hebbian_knowledge_memory import HebbianKnowledgeMemory
from .profile_utils import OpenAIClient, gpt_personality_analysis, gpt_update_profile
from .utils import get_timestamp, load_api_keys

HAS_KNOWLEDGE_UTILS = True
_ENCODE_BASE_URL = os.environ.get("OPENAI_BASE_URL")

# Incremental knowledge extraction buffer size
BUFFER_SIZE = int(os.environ.get("HEBBIAN_KNOWLEDGE_BUFFER_SIZE", "10"))


def parse_sessions(item: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Parse LongMemEval item's haystack_sessions into a flat list of
    (user_text, ai_text, timestamp) turns.

    LongMemEval format:
        haystack_sessions: list of sessions, each session is a list of
            [{role: "user", content: ...}, {role: "assistant", content: ...}]
        haystack_dates: list of date strings, one per session
    """
    turns = []
    sessions = item["haystack_sessions"]
    dates = item["haystack_dates"]

    for session_idx, session in enumerate(sessions):
        timestamp = dates[session_idx] if session_idx < len(dates) else get_timestamp()

        # Skip leading non-user messages
        start = 0
        while start < len(session) and session[start]["role"] != "user":
            start += 1

        # Pair user/assistant turns
        i = start
        while i + 1 < len(session):
            if session[i]["role"] == "user" and session[i + 1]["role"] == "assistant":
                turns.append({
                    "user_text": session[i]["content"],
                    "ai_text": session[i + 1]["content"],
                    "timestamp": timestamp,
                })
                i += 2
            else:
                i += 1

    return turns


def process_incremental_buffer(
    buffer: List[Dict[str, str]],
    knowledge_memory: HebbianKnowledgeMemory,
    item_id: str,
    client: Any,
) -> None:
    """
    Process a batch of dialog turns for incremental knowledge extraction.
    Extracts user profile, user data, and assistant knowledge.
    """
    if not buffer or not HAS_KNOWLEDGE_UTILS:
        return

    dialogs_for_analysis = [
        {
            "user_input": t["user_text"],
            "agent_response": t["ai_text"],
            "timestamp": t["timestamp"],
        }
        for t in buffer
    ]

    try:
        result = gpt_personality_analysis(dialogs_for_analysis, client)
        new_profile = result["profile"]
        new_private = result["private"]
        assistant_knowledge = result["assistant_knowledge"]

        # 1. Update user profile (merge with existing)
        old_profile = knowledge_memory.get_raw_user_profile(item_id)
        if old_profile:
            updated_profile = gpt_update_profile(old_profile, new_profile, client)
        else:
            updated_profile = new_profile
        knowledge_memory.update_user_profile(item_id, updated_profile)

        # 2. Add user knowledge facts
        if new_private and new_private.strip().lower() not in ("none", "- none", ""):
            for line in new_private.split("\n"):
                fact = line.strip().lstrip("- ").strip()
                if fact and fact.lower() != "none" and not fact.startswith("【"):
                    knowledge_memory.add_knowledge(fact)

        # 3. Add assistant knowledge
        if assistant_knowledge and assistant_knowledge.strip().lower() != "none":
            for line in assistant_knowledge.split("\n"):
                ak = line.strip().lstrip("- ").strip()
                if ak and ak.lower() != "none" and not ak.startswith("【"):
                    knowledge_memory.add_assistant_knowledge(ak)

    except Exception as e:
        print(f"  [{item_id}] Knowledge extraction error: {e}")


def encode_single_item(
    item: Dict[str, Any],
    output_dir: str,
    client: Any,
    skip_knowledge: bool = False,
) -> Dict[str, Any]:
    """
    Encode a single LongMemEval item into Hebbian memory.

    Steps:
        1. Parse haystack sessions into user/assistant turns
        2. Add each turn to HebbianMemoryGraph via process_conversation_turn
        3. Incremental knowledge extraction every BUFFER_SIZE turns
        4. Save graph and knowledge memory

    Returns:
        Dict with item_id, num_nodes, num_turns, encoding_time
    """
    item_id = item["question_id"]
    t_start = time.time()

    # Initialize Hebbian Memory Graph
    mem_path = os.path.join(output_dir, f"{item_id}_hebbian.json")
    memory_graph = HebbianMemoryGraph(file_path=mem_path)

    # Initialize Knowledge Memory
    kb_path = os.path.join(output_dir, f"{item_id}_long_term.json")
    knowledge_memory = HebbianKnowledgeMemory(file_path=kb_path)

    # Create retriever (for process_conversation_turn)
    retriever = HebbianRetriever(memory_graph, profile_memory=knowledge_memory)

    # Parse sessions into turns
    turns = parse_sessions(item)
    print(f"  [{item_id}] {len(turns)} turns from {len(item['haystack_sessions'])} sessions")

    # Encode turns
    incremental_buffer = []
    for turn in turns:
        retriever.process_conversation_turn(
            turn["user_text"], turn["ai_text"], timestamp=turn["timestamp"]
        )

        # Accumulate for knowledge extraction
        if not skip_knowledge and HAS_KNOWLEDGE_UTILS:
            incremental_buffer.append(turn)
            if len(incremental_buffer) >= BUFFER_SIZE:
                process_incremental_buffer(
                    incremental_buffer, knowledge_memory, item_id, client
                )
                incremental_buffer = []

    # Process remaining buffer
    if incremental_buffer and not skip_knowledge and HAS_KNOWLEDGE_UTILS:
        process_incremental_buffer(
            incremental_buffer, knowledge_memory, item_id, client
        )

    # Save
    memory_graph.save()
    knowledge_memory.save()

    encoding_time = time.time() - t_start
    print(f"  [{item_id}] Done: {len(memory_graph.nodes)} nodes, {encoding_time:.1f}s")

    return {
        "item_id": item_id,
        "num_nodes": len(memory_graph.nodes),
        "num_turns": len(turns),
        "encoding_time": encoding_time,
    }


def encode_longmemeval(
    data_path: str,
    output_dir: str,
    num_items: Optional[int] = None,
    start_item: int = 0,
    workers: int = 10,
    skip_knowledge: bool = False,
) -> str:
    """
    Encode LongMemEval-S dataset into Hebbian memory graphs.

    Args:
        data_path: Path to longmemeval_s.json
        output_dir: Directory to save memory files
        num_items: Number of items to encode (default: all)
        start_item: Start index
        workers: Number of parallel workers
        skip_knowledge: If True, skip incremental knowledge extraction

    Returns:
        output_dir path
    """
    print("=" * 70)
    print("LongMemEval Hebbian Encoding")
    print("=" * 70)
    print(f"Data: {data_path}")
    print(f"Output: {output_dir}")
    print(f"Workers: {workers}")
    print(f"Knowledge extraction: {'OFF' if skip_knowledge else 'ON (buffer=' + str(BUFFER_SIZE) + ')'}")
    print(f"Hebbian params: lr={os.environ.get('HEBBIAN_LEARNING_RATE', '0.02')}, "
          f"decay={os.environ.get('HEBBIAN_DECAY_RATE', '0.995')}, "
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
    print(f"Encoding items [{start_item}:{start_item + len(items)}]")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Initialize OpenAI clients for knowledge extraction (one per API key for rotation)
    clients = []
    if not skip_knowledge and HAS_KNOWLEDGE_UTILS:
        keys = load_api_keys()
        if not keys:
            raise RuntimeError(
                "Knowledge extraction requires OPENAI_API_KEY or OPENAI_API_KEYS."
            )
        for key in keys:
            clients.append(OpenAIClient(api_key=key, base_url=_ENCODE_BASE_URL))
        print(f"Initialized {len(clients)} API clients for key rotation")

    # Encode items in parallel
    results = []
    failed = []

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_item = {
            executor.submit(
                encode_single_item, item, output_dir,
                clients[i % len(clients)] if clients else None,
                skip_knowledge,
            ): item["question_id"]
            for i, item in enumerate(items)
        }

        for future in as_completed(future_to_item):
            item_id = future_to_item[future]
            try:
                result = future.result()
                results.append(result)
                if len(results) % 50 == 0:
                    print(f"\nProgress: {len(results)}/{len(items)} items encoded\n")
            except Exception as e:
                print(f"  [ERROR] {item_id}: {e}")
                import traceback
                traceback.print_exc()
                failed.append(item_id)

    # Save encoding summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "data_path": data_path,
        "output_dir": output_dir,
        "total_items": len(items),
        "encoded": len(results),
        "failed": len(failed),
        "failed_ids": failed,
        "skip_knowledge": skip_knowledge,
        "buffer_size": BUFFER_SIZE,
        "params": {
            "learning_rate": os.environ.get("HEBBIAN_LEARNING_RATE", "0.02"),
            "decay_rate": os.environ.get("HEBBIAN_DECAY_RATE", "0.995"),
            "activation_alpha": os.environ.get("HEBBIAN_ACTIVATION_ALPHA", "0.1"),
            "spreading_threshold": os.environ.get("HEBBIAN_SPREADING_THRESHOLD", "0.4"),
            "max_flipped": os.environ.get("HEBBIAN_MAX_FLIPPED", "5"),
            "tau": os.environ.get("HEBBIAN_TAU", "1e7"),
        },
        "results": sorted(results, key=lambda x: x["item_id"]),
    }

    summary_path = os.path.join(output_dir, "encode_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n{'=' * 70}")
    print(f"Encoding complete: {len(results)}/{len(items)} items")
    if failed:
        print(f"Failed: {failed}")
    total_nodes = sum(r["num_nodes"] for r in results)
    total_time = sum(r["encoding_time"] for r in results)
    print(f"Total nodes: {total_nodes}, Total time: {total_time:.1f}s")
    print(f"Summary: {summary_path}")
    print(f"{'=' * 70}")

    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Encode LongMemEval-S into Hebbian Memory"
    )
    parser.add_argument(
        "--data_path", type=str, required=True,
        help="Path to longmemeval_s.json",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Output directory (default: auto-generated)",
    )
    parser.add_argument("--num_items", type=int, default=None, help="Number of items")
    parser.add_argument("--start_item", type=int, default=0, help="Start index")
    parser.add_argument("--workers", type=int, default=10, help="Parallel workers")
    parser.add_argument(
        "--skip_knowledge", action="store_true",
        help="Skip incremental knowledge extraction",
    )

    args = parser.parse_args()

    output_dir = args.output_dir
    if output_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"hebbian/results/longmemeval_mem_{ts}"

    encode_longmemeval(
        data_path=args.data_path,
        output_dir=output_dir,
        num_items=args.num_items,
        start_item=args.start_item,
        workers=args.workers,
        skip_knowledge=args.skip_knowledge,
    )


if __name__ == "__main__":
    main()
