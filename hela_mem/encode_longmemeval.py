"""Encode LongMemEval-S haystack sessions into HeLa-Mem Hebbian memory graphs."""

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
from .profile_extraction import gpt_personality_analysis, gpt_update_profile
from .utils import (
    create_chat_client,
    env_float,
    env_int,
    get_timestamp,
    has_llm_credentials,
    load_api_keys,
    resolve_model_name,
)


BUFFER_SIZE = env_int("HELA_MEM_KNOWLEDGE_BUFFER_SIZE", "HEBBIAN_KNOWLEDGE_BUFFER_SIZE", default=10)


def parse_sessions(item: dict[str, Any]) -> list[dict[str, str]]:
    turns: list[dict[str, str]] = []
    sessions = item["haystack_sessions"]
    dates = item["haystack_dates"]

    for session_index, session in enumerate(sessions):
        timestamp = dates[session_index] if session_index < len(dates) else get_timestamp()
        start_index = 0
        while start_index < len(session) and session[start_index].get("role") != "user":
            start_index += 1

        index = start_index
        while index + 1 < len(session):
            current = session[index]
            nxt = session[index + 1]
            if current.get("role") == "user" and nxt.get("role") == "assistant":
                turns.append(
                    {
                        "user_text": current.get("content", ""),
                        "ai_text": nxt.get("content", ""),
                        "timestamp": timestamp,
                    }
                )
                index += 2
            else:
                index += 1

    return turns


def process_incremental_buffer(
    buffer: list[dict[str, str]],
    knowledge_memory: HebbianKnowledgeMemory,
    item_id: str,
    client,
) -> None:
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

    result = gpt_personality_analysis(dialogs_for_analysis, client)
    new_profile = result["profile"]
    new_private = result["private"]
    assistant_knowledge = result["assistant_knowledge"]

    old_profile = knowledge_memory.get_raw_user_profile(item_id)
    updated_profile = gpt_update_profile(old_profile, new_profile, client) if old_profile else new_profile
    knowledge_memory.update_user_profile(item_id, updated_profile)

    if new_private and new_private.strip().lower() not in {"none", "- none", ""}:
        for line in new_private.splitlines():
            fact = line.strip().lstrip("- ").strip()
            if fact and fact.lower() != "none" and not fact.startswith("【"):
                knowledge_memory.add_knowledge(fact)

    if assistant_knowledge and assistant_knowledge.strip().lower() != "none":
        for line in assistant_knowledge.splitlines():
            fact = line.strip().lstrip("- ").strip()
            if fact and fact.lower() != "none" and not fact.startswith("【"):
                knowledge_memory.add_assistant_knowledge(fact)


def encode_single_item(
    item: dict[str, Any],
    output_dir: str,
    client=None,
    skip_knowledge: bool = False,
) -> dict[str, Any]:
    item_id = item["question_id"]
    started_at = time.time()

    memory_path = os.path.join(output_dir, f"{item_id}_hebbian.json")
    knowledge_path = os.path.join(output_dir, f"{item_id}_long_term.json")

    memory_graph = HebbianMemoryGraph(memory_path)
    knowledge_memory = HebbianKnowledgeMemory(knowledge_path)
    retriever = HebbianRetriever(memory_graph, profile_memory=knowledge_memory)

    turns = parse_sessions(item)
    incremental_buffer: list[dict[str, str]] = []

    for turn in turns:
        retriever.process_conversation_turn(turn["user_text"], turn["ai_text"], timestamp=turn["timestamp"])
        if not skip_knowledge and client is not None:
            incremental_buffer.append(turn)
            if len(incremental_buffer) >= BUFFER_SIZE:
                process_incremental_buffer(incremental_buffer, knowledge_memory, item_id, client)
                incremental_buffer = []

    if incremental_buffer and not skip_knowledge and client is not None:
        process_incremental_buffer(incremental_buffer, knowledge_memory, item_id, client)

    memory_graph.save()
    knowledge_memory.save()

    return {
        "item_id": item_id,
        "num_nodes": len(memory_graph.nodes),
        "num_turns": len(turns),
        "encoding_time": round(time.time() - started_at, 3),
    }


def encode_longmemeval(
    data_path: str,
    output_dir: str,
    num_items: int | None = None,
    start_item: int = 0,
    workers: int = 8,
    skip_knowledge: bool = False,
) -> str:
    with open(data_path, "r", encoding="utf-8") as handle:
        dataset = json.load(handle)

    items = dataset[start_item:]
    if num_items is not None:
        items = items[:num_items]

    os.makedirs(output_dir, exist_ok=True)

    clients = []
    if not skip_knowledge:
        if not has_llm_credentials():
            raise RuntimeError(
                "Knowledge extraction requires OPENAI_API_KEY or OPENAI_API_KEYS. "
                "Use --skip_knowledge for local smoke encoding."
            )
        for api_key in load_api_keys():
            clients.append(create_chat_client(api_key=api_key))

    results: list[dict[str, Any]] = []
    failed: list[str] = []

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_item = {
            executor.submit(
                encode_single_item,
                item,
                output_dir,
                clients[index % len(clients)] if clients else None,
                skip_knowledge,
            ): item["question_id"]
            for index, item in enumerate(items)
        }

        for future in as_completed(future_to_item):
            item_id = future_to_item[future]
            try:
                results.append(future.result())
            except Exception:
                failed.append(item_id)

    results.sort(key=lambda row: row["item_id"])
    summary = {
        "timestamp": datetime.now().isoformat(),
        "data_path": data_path,
        "output_dir": output_dir,
        "model": resolve_model_name(),
        "total_items": len(items),
        "encoded": len(results),
        "failed": len(failed),
        "failed_ids": failed,
        "skip_knowledge": skip_knowledge,
        "buffer_size": BUFFER_SIZE,
        "params": {
            "tau": env_float("HELA_MEM_TAU", "HEBBIAN_TAU", default=1e7),
            "learning_rate": env_float("HELA_MEM_LEARNING_RATE", "HEBBIAN_LEARNING_RATE", default=0.02),
            "decay_rate": env_float("HELA_MEM_DECAY_RATE", "HEBBIAN_DECAY_RATE", default=0.995),
            "activation_alpha": env_float(
                "HELA_MEM_ACTIVATION_ALPHA", "HEBBIAN_ACTIVATION_ALPHA", default=0.1
            ),
            "spreading_threshold": env_float(
                "HELA_MEM_SPREADING_THRESHOLD", "HEBBIAN_SPREADING_THRESHOLD", default=0.4
            ),
            "max_flipped": env_int("HELA_MEM_MAX_FLIPPED", "HEBBIAN_MAX_FLIPPED", default=5),
        },
        "results": results,
    }

    with open(os.path.join(output_dir, "encode_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Encode LongMemEval-S into HeLa-Mem Hebbian memory.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to LongMemEval-S json.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory. Defaults to results/longmemeval_mem_<timestamp>.",
    )
    parser.add_argument("--num_items", type=int, default=None, help="Optional item cap.")
    parser.add_argument("--start_item", type=int, default=0, help="Start index.")
    parser.add_argument("--workers", type=int, default=8, help="Parallel worker count.")
    parser.add_argument(
        "--skip_knowledge",
        action="store_true",
        help="Skip profile and knowledge extraction. Useful for local smoke tests.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join(
        "results", f"longmemeval_mem_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )

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
