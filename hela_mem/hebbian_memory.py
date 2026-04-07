"""Hebbian episodic memory graph used for LongMemEval experiments."""

from __future__ import annotations

import json
import os
from collections import defaultdict

import numpy as np

from .utils import (
    compute_time_decay,
    env_bool,
    env_float,
    env_int,
    get_embedding,
    get_timestamp,
    llm_extract_keywords,
    normalize_vector,
)


class HebbianMemoryGraph:
    def __init__(self, file_path: str, embedding_dim: int = 384) -> None:
        self.file_path = file_path
        self.embedding_dim = embedding_dim
        self.nodes: dict[str, dict] = {}
        self.edges: defaultdict[str, defaultdict[str, float]] = defaultdict(lambda: defaultdict(float))

        self.learning_rate = env_float("HELA_MEM_LEARNING_RATE", "HEBBIAN_LEARNING_RATE", default=0.02)
        self.decay_rate = env_float("HELA_MEM_DECAY_RATE", "HEBBIAN_DECAY_RATE", default=0.995)
        self.activation_alpha = env_float(
            "HELA_MEM_ACTIVATION_ALPHA", "HEBBIAN_ACTIVATION_ALPHA", default=0.1
        )
        self.spreading_threshold = env_float(
            "HELA_MEM_SPREADING_THRESHOLD", "HEBBIAN_SPREADING_THRESHOLD", default=0.4
        )
        self.max_flipped = env_int("HELA_MEM_MAX_FLIPPED", "HEBBIAN_MAX_FLIPPED", default=5)

        if os.path.exists(self.file_path):
            self.load()

    def add_memory(
        self,
        content: str,
        role: str = "interaction",
        embedding: np.ndarray | None = None,
        metadata: dict | None = None,
        timestamp: str | None = None,
    ) -> str:
        node_id = str(len(self.nodes))
        if embedding is None:
            embedding = get_embedding(content)

        if timestamp is None:
            timestamp = get_timestamp()

        keywords = list(llm_extract_keywords(content))
        node = {
            "id": node_id,
            "content": content,
            "role": role,
            "embedding": normalize_vector(embedding).tolist(),
            "timestamp": timestamp,
            "keywords": keywords,
            "metadata": metadata or {},
        }
        self.nodes[node_id] = node

        if len(self.nodes) > 1:
            previous_id = str(len(self.nodes) - 2)
            self.add_edge(previous_id, node_id, weight=0.5, bidirectional=True)

        return node_id

    def add_edge(self, source_id: str, target_id: str, weight: float = 0.1, bidirectional: bool = True) -> None:
        if source_id == target_id:
            return
        self.edges[source_id][target_id] = min(1.0, self.edges[source_id][target_id] + weight)
        if bidirectional:
            self.edges[target_id][source_id] = min(1.0, self.edges[target_id][source_id] + weight)

    def retrieve(self, query: str, top_k: int = 20, override_max_flipped: int | None = None) -> list[dict]:
        if not self.nodes:
            return []

        use_time_decay = env_bool("HELA_MEM_USE_TIME_DECAY", "HEBBIAN_USE_TIME_DECAY", default=True)
        use_keyword_match = env_bool("HELA_MEM_USE_KEYWORD_MATCH", "HEBBIAN_USE_KEYWORD_MATCH", default=True)
        keyword_weight = env_float("HELA_MEM_KEYWORD_WEIGHT", "HEBBIAN_KEYWORD_WEIGHT", default=0.5)

        query_vec = normalize_vector(get_embedding(query))
        query_keywords = llm_extract_keywords(query)

        node_ids = list(self.nodes.keys())
        node_matrix = np.array([self.nodes[node_id]["embedding"] for node_id in node_ids], dtype=np.float32)
        base_activations = np.dot(node_matrix, query_vec)
        base_activations = (base_activations + 1.0) / 2.0

        current_time = get_timestamp()
        enhanced_activations = np.zeros_like(base_activations)
        for index, node_id in enumerate(node_ids):
            node = self.nodes[node_id]

            time_factor = (
                compute_time_decay(node["timestamp"], current_time=current_time) if use_time_decay else 1.0
            )

            keyword_score = 0.0
            if use_keyword_match and query_keywords:
                node_keywords = set(node.get("keywords", []))
                if node_keywords:
                    overlap = query_keywords & node_keywords
                    keyword_score = len(overlap) / max(len(query_keywords), 1)

            combined = base_activations[index] + keyword_weight * keyword_score
            enhanced_activations[index] = combined * time_factor

        final_scores = enhanced_activations.copy()
        id_to_index = {node_id: index for index, node_id in enumerate(node_ids)}

        for index, score in enumerate(enhanced_activations):
            if score <= self.spreading_threshold:
                continue
            source_id = node_ids[index]
            for target_id, weight in self.edges.get(source_id, {}).items():
                target_index = id_to_index.get(target_id)
                if target_index is None:
                    continue
                final_scores[target_index] += score * weight * self.activation_alpha

        base_ranking = np.argsort(enhanced_activations)[::-1]
        spreading_ranking = np.argsort(final_scores)[::-1]
        base_indices = list(base_ranking[:top_k])
        base_index_set = set(base_indices)

        effective_max_flipped = override_max_flipped if override_max_flipped is not None else self.max_flipped
        if self.activation_alpha <= 0:
            effective_max_flipped = 0

        flipped_indices: list[int] = []
        if effective_max_flipped > 0:
            for candidate in spreading_ranking[:top_k]:
                if candidate not in base_index_set:
                    flipped_indices.append(candidate)
                    if len(flipped_indices) >= effective_max_flipped:
                        break

        results: list[dict] = []
        retrieved_ids: list[str] = []
        for index in base_indices:
            node_id = node_ids[index]
            results.append(
                {
                    "node": self.nodes[node_id],
                    "score": float(final_scores[index]),
                    "base_score": float(base_activations[index]),
                    "source": "base",
                }
            )
            retrieved_ids.append(node_id)

        for index in flipped_indices:
            node_id = node_ids[index]
            results.append(
                {
                    "node": self.nodes[node_id],
                    "score": float(final_scores[index]),
                    "base_score": float(base_activations[index]),
                    "source": "hebbian",
                }
            )
            retrieved_ids.append(node_id)

        self.reinforce_memory_cluster(retrieved_ids)
        return results

    def reinforce_memory_cluster(self, node_ids: list[str]) -> None:
        for left_index in range(len(node_ids)):
            for right_index in range(left_index + 1, len(node_ids)):
                self.add_edge(node_ids[left_index], node_ids[right_index], weight=self.learning_rate, bidirectional=True)

    def global_decay(self) -> None:
        to_remove: list[tuple[str, str]] = []
        for source_id, neighbors in self.edges.items():
            for target_id in list(neighbors.keys()):
                neighbors[target_id] *= self.decay_rate
                if neighbors[target_id] < 0.01:
                    to_remove.append((source_id, target_id))

        for source_id, target_id in to_remove:
            del self.edges[source_id][target_id]

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
        data = {
            "nodes": self.nodes,
            "edges": {source_id: dict(targets) for source_id, targets in self.edges.items()},
        }
        with open(self.file_path, "w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2)

    def load(self) -> None:
        with open(self.file_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)

        self.nodes = data.get("nodes", {})
        self.edges = defaultdict(lambda: defaultdict(float))
        for source_id, targets in data.get("edges", {}).items():
            self.edges[source_id] = defaultdict(float, {target_id: float(weight) for target_id, weight in targets.items()})
