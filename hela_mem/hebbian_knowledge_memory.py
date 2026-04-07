"""Long-term knowledge memory used by the LongMemEval release."""

from __future__ import annotations

import json
import os

from .hebbian_memory import HebbianMemoryGraph
from .utils import env_int, get_timestamp, get_embedding, normalize_vector


class HebbianKnowledgeMemory:
    def __init__(self, file_path: str, max_capacity: int = 100) -> None:
        self.file_path = file_path
        self.max_capacity = max_capacity
        self.user_profiles: dict[str, dict] = {}
        self.assistant_knowledge: list[dict] = []
        self.kb_max_flipped = env_int("HELA_MEM_KB_MAX_FLIPPED", "HEBBIAN_KB_MAX_FLIPPED", default=5)

        self.hebbian_kb_path = self.file_path.replace(".json", "_kb_graph.json")
        self.knowledge_graph = HebbianMemoryGraph(self.hebbian_kb_path)

        if os.path.exists(self.file_path):
            self.load()

    def update_user_profile(self, user_id: str, new_data: str, merge: bool = False) -> None:
        if merge and user_id in self.user_profiles:
            current = self.user_profiles[user_id]["data"]
            if isinstance(current, str) and isinstance(new_data, str):
                new_data = f"{current}\n\n--- Updated ---\n{new_data}"

        self.user_profiles[user_id] = {
            "data": new_data,
            "last_updated": get_timestamp(),
        }
        self.save()

    def get_raw_user_profile(self, user_id: str) -> str:
        return self.user_profiles.get(user_id, {}).get("data", "")

    def get_user_profile(self, user_id: str) -> dict:
        return self.user_profiles.get(user_id, {})

    def add_assistant_knowledge(self, knowledge_text: str) -> None:
        cleaned = knowledge_text.strip()
        if not cleaned or cleaned.lower() in {"none", "- none", "- none."}:
            return

        entry = {
            "knowledge": cleaned,
            "timestamp": get_timestamp(),
            "knowledge_embedding": normalize_vector(get_embedding(cleaned)).tolist(),
        }
        self.assistant_knowledge.append(entry)
        if len(self.assistant_knowledge) > self.max_capacity:
            self.assistant_knowledge = self.assistant_knowledge[-self.max_capacity :]
        self.save()

    def get_assistant_knowledge(self) -> list[dict]:
        return self.assistant_knowledge

    def add_knowledge(self, knowledge_text: str) -> None:
        cleaned = knowledge_text.strip()
        if not cleaned or cleaned.lower() in {"none", "- none", "- none."}:
            return
        self.knowledge_graph.add_memory(cleaned, role="system", metadata={"type": "fact"})
        self.knowledge_graph.save()
        self.save()

    def search_knowledge(self, query: str, top_k: int = 5) -> list[dict]:
        if not self.knowledge_graph.nodes:
            return []
        results = self.knowledge_graph.retrieve(query, top_k=top_k, override_max_flipped=self.kb_max_flipped)
        return [
            {
                "knowledge": result["node"]["content"],
                "timestamp": result["node"]["timestamp"],
                "knowledge_embedding": result["node"]["embedding"],
                "score": result["score"],
            }
            for result in results
        ]

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
        with open(self.file_path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "user_profiles": self.user_profiles,
                    "assistant_knowledge": self.assistant_knowledge,
                },
                handle,
                ensure_ascii=False,
                indent=2,
            )
        self.knowledge_graph.save()

    def load(self) -> None:
        with open(self.file_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        self.user_profiles = data.get("user_profiles", {})
        self.assistant_knowledge = data.get("assistant_knowledge", [])
