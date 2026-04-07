"""Minimal retriever wrapper for LongMemEval encoding."""

from __future__ import annotations

from .hebbian_knowledge_memory import HebbianKnowledgeMemory
from .hebbian_memory import HebbianMemoryGraph


class HebbianRetriever:
    def __init__(self, memory_graph: HebbianMemoryGraph, profile_memory: HebbianKnowledgeMemory | None = None) -> None:
        self.graph = memory_graph
        self.profile_memory = profile_memory

    def process_conversation_turn(self, user_input: str, agent_response: str, timestamp: str | None = None) -> None:
        content = f"User: {user_input}\nAI: {agent_response}"
        self.graph.add_memory(content, role="interaction", timestamp=timestamp)
