import json
import numpy as np
import os
from collections import defaultdict
from .utils import get_timestamp, get_embedding, normalize_vector, compute_time_decay, llm_extract_keywords

class HebbianMemoryGraph:
    def __init__(self, file_path, embedding_dim=384):
        self.file_path = file_path
        self.embedding_dim = embedding_dim
        
        # Nodes: {node_id: {content, embedding, timestamp, type, metadata}}
        self.nodes = {}
        
        # Edges: {source_id: {target_id: weight}} (Adjacency List)
        # This represents the "Synaptic Strength"
        self.edges = defaultdict(lambda: defaultdict(float))
        
        # Hebbian Parameters
        # [NEW] Read from env vars or use defaults
        self.decay_rate = float(os.environ.get("HEBBIAN_DECAY_RATE", 0.995))
        self.learning_rate = float(os.environ.get("HEBBIAN_LEARNING_RATE", 0.01))
        self.activation_alpha = float(os.environ.get("HEBBIAN_ACTIVATION_ALPHA", 0.1))
        self.spreading_threshold = float(os.environ.get("HEBBIAN_SPREADING_THRESHOLD", 0.4))
        self.max_flipped = int(os.environ.get("HEBBIAN_MAX_FLIPPED", 5))  # [NEW] Standalone hebbian bonus
        
        print(f"[Hebbian] Initialized with: LR={self.learning_rate}, Decay={self.decay_rate}, Alpha={self.activation_alpha}, Threshold={self.spreading_threshold}, MaxFlipped={self.max_flipped}")
        
        self.load()

    # [ORIGINAL]
    # def add_memory(self, content, role="user", embedding=None, metadata=None, timestamp=None):
    #     """
    #     Add a new memory node.
    #     Auto-link to the immediately preceding memory (Temporal Association).
    #     """
    #     node_id = str(len(self.nodes))
    #     
    #     if embedding is None:
    #         embedding = get_embedding(content)
    #     
    #     norm_embedding = normalize_vector(embedding).tolist()
    #     
    #     if timestamp is None:
    #         timestamp = get_timestamp()
    #         
    #     node = {
    #         "id": node_id,
    #         "content": content,
    #         "role": role,
    #         "embedding": norm_embedding,
    #         "timestamp": timestamp,
    #         "metadata": metadata or {}
    #     }
    #     
    #     # 1. Add Node
    #     self.nodes[node_id] = node
    #     
    #     # 2. Create Temporal Edge (Link to previous node)
    #     # "Neurons that happen together, wire together"
    #     if len(self.nodes) > 1:
    #         prev_id = str(len(self.nodes) - 2)
    #         # Initial temporal weight is moderate
    #         self.add_edge(prev_id, node_id, weight=0.5, bidirectional=True)
    #         
    #     return node_id

    def add_memory(self, content, role="user", embedding=None, metadata=None, timestamp=None):
        """
        [NEW] Add a new memory node.
        Improvements:
        1. Keyword Extraction: Extracts keywords from content using LLM and stores them in metadata.
        """
        node_id = str(len(self.nodes))
        
        if embedding is None:
            embedding = get_embedding(content)
        
        norm_embedding = normalize_vector(embedding).tolist()
        
        if timestamp is None:
            timestamp = get_timestamp()

        # [NEW] Extract keywords
        try:
            keywords = list(llm_extract_keywords(content))
        except Exception as e:
            print(f"Error extracting keywords: {e}")
            keywords = []
            
        node = {
            "id": node_id,
            "content": content,
            "role": role,
            "embedding": norm_embedding,
            "timestamp": timestamp,
            "keywords": keywords, # [NEW] Store keywords
            "metadata": metadata or {}
        }
        
        # 1. Add Node
        self.nodes[node_id] = node
        
        # 2. Create Temporal Edge (Link to previous node)
        # "Neurons that happen together, wire together"
        if len(self.nodes) > 1:
            prev_id = str(len(self.nodes) - 2)
            # Initial temporal weight is moderate
            self.add_edge(prev_id, node_id, weight=0.5, bidirectional=True)
            
        return node_id

    def add_edge(self, u, v, weight=0.1, bidirectional=True):
        """Add or update a synaptic connection"""
        if u == v: return
        self.edges[u][v] = min(1.0, self.edges[u][v] + weight)
        if bidirectional:
            self.edges[v][u] = min(1.0, self.edges[v][u] + weight)

    # [ORIGINAL]
    # def retrieve(self, query, top_k=5):
    #     """
    #     Hebbian Retrieval: Vector Similarity + Spreading Activation
    #     """
    #     if not self.nodes:
    #         return []
    #
    #     # 1. Base Activation (Vector Similarity)
    #     query_vec = normalize_vector(get_embedding(query))
    #     
    #     # Matrix operation for speed
    #     node_ids = list(self.nodes.keys())
    #     node_matrix = np.array([self.nodes[nid]["embedding"] for nid in node_ids])
    #     
    #     # Cosine similarity (-1 to 1)
    #     # shape: (N,)
    #     base_activations = np.dot(node_matrix, query_vec)
    #     
    #     # Normalize to 0-1 for easier propagation logic (optional but good)
    #     base_activations = (base_activations + 1) / 2.0 
    #
    #     # 2. Spreading Activation (The Hebbian Part)
    #     # A_final = A_base + alpha * (A_base * W)
    #     # To be efficient, we only spread from highly activated nodes
    #     
    #     final_scores = base_activations.copy()
    #     
    #     # Build a quick lookup for index
    #     id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
    #     
    #     # Spread loop
    #     for i, score in enumerate(base_activations):
    #         # Only spread if the node is relevant enough (Thresholding)
    #         if score > 0.6: 
    #             source_id = node_ids[i]
    #             neighbors = self.edges.get(source_id, {})
    #             
    #             for target_id, weight in neighbors.items():
    #                 if target_id in id_to_idx:
    #                     target_idx = id_to_idx[target_id]
    #                     # The boost depends on:
    #                     # - Source relevance (score)
    #                     # - Connection strength (weight)
    #                     # - Alpha parameter
    #                     boost = score * weight * self.activation_alpha
    #                     final_scores[target_idx] += boost
    #
    #     # 3. Ranking
    #     # Get Top-K indices
    #     top_indices = np.argsort(final_scores)[::-1][:top_k]
    #     
    #     results = []
    #     retrieved_ids = []
    #     
    #     for idx in top_indices:
    #         nid = node_ids[idx]
    #         node = self.nodes[nid]
    #         score = final_scores[idx]
    #         
    #         # Add time decay factor to the final presentation order (but not retrieval logic itself)
    #         # to prefer recent memories if scores are tied
    #         recency = compute_time_decay(node["timestamp"])
    #         
    #         results.append({
    #             "node": node,
    #             "score": float(score),
    #             "base_score": float(base_activations[idx]),
    #             "recency": recency
    #         })
    #         retrieved_ids.append(nid)
    #         
    #     # 4. Hebbian Learning (Reinforcement)
    #     # "Neurons that fire together (are retrieved together), wire together"
    #     # We reinforce connections between all pairs in the Top-K results
    #     self.reinforce_memory_cluster(retrieved_ids)
    #         
    #     return results

    def retrieve(self, query, top_k=5, override_max_flipped=None):
        """
        [NEW] Hebbian Retrieval: Vector Similarity + Spreading Activation + Time Decay + Keyword Matching
        Improvements:
        1. Time Decay: Older memories are penalized.
        2. Keyword Matching: Nodes with matching keywords get a boost.
        """
        if not self.nodes:
            return []

        # [OLD] Query Augmentation (commented out - reverted to keyword matching)
        # use_query_augmentation = os.environ.get("HEBBIAN_USE_QUERY_AUGMENTATION", "true").lower() == "true"
        # augmented_query = query
        # query_entities = set()
        # if use_query_augmentation:
        #     try:
        #         query_entities = llm_extract_keywords(query)
        #         if query_entities:
        #             entities_str = ", ".join(query_entities)
        #             augmented_query = f"{query} [Key Entities: {entities_str}]"
        #     except:
        #         pass

        # Extract query keywords
        try:
            query_keywords = llm_extract_keywords(query)
        except:
            query_keywords = set()

        # 1. Base Activation (Vector Similarity)
        query_vec = normalize_vector(get_embedding(query))
        
        # Matrix operation for speed
        node_ids = list(self.nodes.keys())
        node_matrix = np.array([self.nodes[nid]["embedding"] for nid in node_ids])
        
        # Cosine similarity (-1 to 1)
        # shape: (N,)
        base_activations = np.dot(node_matrix, query_vec)
        
        # Normalize to 0-1 for easier propagation logic (optional but good)
        base_activations = (base_activations + 1) / 2.0 

        # Apply Time Decay and Keyword Matching to Base Activation
        current_time = get_timestamp()
        enhanced_activations = np.zeros_like(base_activations)
        
        # Check if time decay and keyword matching are enabled
        use_time_decay = os.environ.get("HEBBIAN_USE_TIME_DECAY", "true").lower() == "true"
        use_keyword_match = os.environ.get("HEBBIAN_USE_KEYWORD_MATCH", "true").lower() == "true"
        
        for i, nid in enumerate(node_ids):
            node = self.nodes[nid]
            
            # Time Decay (can be disabled)
            if use_time_decay:
                time_decay = compute_time_decay(node["timestamp"], current_time)
            else:
                time_decay = 1.0  # No decay
            
            # Keyword Score (can be disabled)
            keyword_score = 0.0
            if use_keyword_match:
                node_keywords = set(node.get("keywords", []))
                if query_keywords and node_keywords:
                    overlap = query_keywords & node_keywords
                    # Jaccard-like or simple overlap ratio
                    keyword_score = len(overlap) / len(query_keywords)
            
            # Combined Base Score
            # Formula: Time_Decay * (Vector_Sim + Alpha * Keyword_Score)
            # We use a keyword_weight (configurable via env)
            keyword_weight = float(os.environ.get("HEBBIAN_KEYWORD_WEIGHT", 0.5))
            
            # Base activation is already 0-1. Keyword score is 0-1.
            # We sum them and then apply decay.
            combined_score = base_activations[i] + (keyword_weight * keyword_score)
            
            # Apply decay
            enhanced_activations[i] = combined_score * time_decay

        # 2. Spreading Activation (The Hebbian Part)
        # A_final = A_base + alpha * (A_base * W)
        # To be efficient, we only spread from highly activated nodes
        
        # Use enhanced_activations as the starting point for spreading
        final_scores = enhanced_activations.copy()
        
        # Build a quick lookup for index
        id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
        
        # Spread loop
        for i, score in enumerate(enhanced_activations):
            # Only spread if the node is relevant enough (Thresholding)
            # Threshold might need adjustment since scores are now decayed
            if score > self.spreading_threshold: # [MODIFIED] Use configurable threshold
                source_id = node_ids[i]
                neighbors = self.edges.get(source_id, {})
                
                for target_id, weight in neighbors.items():
                    if target_id in id_to_idx:
                        target_idx = id_to_idx[target_id]
                        # The boost depends on:
                        # - Source relevance (score)
                        # - Connection strength (weight)
                        # - Alpha parameter
                        boost = score * weight * self.activation_alpha
                        final_scores[target_idx] += boost

        # 3. Dual-Pathway Ranking: Base (top_k) + Hebbian Bonus (max_flipped)
        base_ranking = np.argsort(enhanced_activations)[::-1]
        spreading_ranking = np.argsort(final_scores)[::-1]
        
        # Base = top_k (pure semantic), Hebbian = extra bonus
        base_count = top_k
        # Use override if provided (for KB independence)
        effective_max_flipped = override_max_flipped if override_max_flipped is not None else self.max_flipped
        max_flipped = effective_max_flipped if self.activation_alpha > 0 else 0
            
        # Base pathway: FIXED Top-K from base ranking
        base_indices = list(base_ranking[:base_count])
        base_indices_set = set(base_indices)
        
        # What would be in Top-K without spreading (for comparison)
        top_indices_no_spreading = set(base_ranking[:top_k])
        
        # Spreading pathway: Find flipped entries (not in base top-K, not in base selection)
        # ONLY look within the natural Top-K of spreading ranking!
        flipped_indices = []
        if max_flipped > 0:
            spreading_top_k = spreading_ranking[:top_k]
            for idx in spreading_top_k:
                if idx not in top_indices_no_spreading and idx not in base_indices_set:
                    flipped_indices.append(idx)
                    if len(flipped_indices) >= max_flipped:
                        break
        
        # Combine: Base + Flipped (no fill-up)
        top_indices = base_indices + flipped_indices
        
        results = []
        retrieved_ids = []
        spreading_flipped_count = len(flipped_indices)
        
        # Build results with source marking
        for idx in base_indices:
            nid = node_ids[idx]
            node = self.nodes[nid]
            score = final_scores[idx]
            recency = compute_time_decay(node["timestamp"], current_time)
            
            results.append({
                "node": node,
                "score": float(score),
                "base_score": float(base_activations[idx]),
                "flipped_by_spreading": False,
                "source": "base",  # [NEW] Mark as base
                "recency": recency
            })
            retrieved_ids.append(nid)
        
        for idx in flipped_indices:
            nid = node_ids[idx]
            node = self.nodes[nid]
            score = final_scores[idx]
            recency = compute_time_decay(node["timestamp"], current_time)
            
            results.append({
                "node": node,
                "score": float(score),
                "base_score": float(base_activations[idx]),
                "flipped_by_spreading": True,
                "source": "hebbian",  # [NEW] Mark as hebbian flipped
                "recency": recency
            })
            retrieved_ids.append(nid)
        
        if self.activation_alpha > 0 and spreading_flipped_count > 0:
            print(f"  [Spreading] {spreading_flipped_count}/{max_flipped} flipped added.")
            
        # 4. Hebbian Learning
        self.reinforce_memory_cluster(retrieved_ids)
            
        return results

    def reinforce_memory_cluster(self, node_ids):
        """
        Strengthen connections between simultaneously retrieved memories.
        This builds the 'Associative Chunk Graph'.
        """
        # Simple O(N^2) loop for small K (K=5 -> 25 iters, trivial)
        for i in range(len(node_ids)):
            for j in range(i + 1, len(node_ids)):
                u, v = node_ids[i], node_ids[j]
                self.add_edge(u, v, weight=self.learning_rate, bidirectional=True)

    def global_decay(self):
        """
        Apply global decay to all edge weights.
        Prune weak connections.
        """
        to_remove = []
        for u in self.edges:
            for v in self.edges[u]:
                self.edges[u][v] *= self.decay_rate
                if self.edges[u][v] < 0.01: # Pruning threshold
                    to_remove.append((u, v))
        
        for u, v in to_remove:
            del self.edges[u][v]
    
    def adaptive_forgetting(self, min_edge_weight=0.1, min_age_days=30, dry_run=True):
        """
        [NEW] Hebbian-guided Adaptive Forgetting
        
        Identifies memory nodes that are candidates for forgetting based on:
        1. Low total edge weight (low connectivity = low importance)
        2. Never accessed (access_count = 0)
        3. Old age (> min_age_days)
        
        Args:
            min_edge_weight: Minimum total edge weight to keep
            min_age_days: Minimum age in days to consider for forgetting
            dry_run: If True, only report candidates without deleting
            
        Returns:
            List of node_ids marked for forgetting
        """
        from datetime import datetime
        
        # Check if adaptive forgetting is enabled
        use_adaptive_forgetting = os.environ.get("HEBBIAN_USE_ADAPTIVE_FORGETTING", "false").lower() == "true"
        if not use_adaptive_forgetting:
            return []
        
        forgetting_threshold = float(os.environ.get("HEBBIAN_FORGETTING_EDGE_THRESHOLD", str(min_edge_weight)))
        forgetting_age_days = int(os.environ.get("HEBBIAN_FORGETTING_AGE_DAYS", str(min_age_days)))
        
        candidates = []
        current_time = datetime.now()
        
        for node_id, node in self.nodes.items():
            # Calculate total edge weight for this node
            total_edge_weight = sum(self.edges.get(node_id, {}).values())
            
            # Get access count (default 0 if not tracked)
            access_count = node.get("access_count", 0)
            
            # Calculate age in days
            try:
                # Parse timestamp like "1:56 pm on 8 May, 2023"
                ts_str = node.get("timestamp", "")
                # Try multiple formats
                age_days = 999  # Default to old if can't parse
                for fmt in ["%I:%M %p on %d %B, %Y", "%Y-%m-%d %H:%M:%S", "%d %B %Y"]:
                    try:
                        ts = datetime.strptime(ts_str, fmt)
                        age_days = (current_time - ts).days
                        break
                    except:
                        continue
            except:
                age_days = 0  # Don't forget if can't determine age
            
            # Check forgetting conditions
            should_forget = (
                total_edge_weight < forgetting_threshold and
                access_count == 0 and
                age_days > forgetting_age_days
            )
            
            if should_forget:
                candidates.append({
                    "node_id": node_id,
                    "edge_weight": total_edge_weight,
                    "access_count": access_count,
                    "age_days": age_days,
                    "content_preview": node.get("content", "")[:50]
                })
        
        if candidates:
            print(f"[Adaptive Forgetting] Found {len(candidates)} candidates for forgetting")
            
            if not dry_run:
                for c in candidates:
                    nid = c["node_id"]
                    # Remove node
                    if nid in self.nodes:
                        del self.nodes[nid]
                    # Remove all edges involving this node
                    if nid in self.edges:
                        del self.edges[nid]
                    for u in list(self.edges.keys()):
                        if nid in self.edges[u]:
                            del self.edges[u][nid]
                print(f"[Adaptive Forgetting] Deleted {len(candidates)} memories")
        
        return candidates
            
    def save(self):
        # Convert defaultdict to dict for JSON serialization
        serializable_edges = {k: dict(v) for k, v in self.edges.items()}
        
        data = {
            "nodes": self.nodes,
            "edges": serializable_edges
        }
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self):
        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.nodes = data.get("nodes", {})
                    raw_edges = data.get("edges", {})
                    # Restore defaultdict structure
                    for u, neighbors in raw_edges.items():
                        for v, w in neighbors.items():
                            self.edges[u][v] = w
            except Exception as e:
                print(f"Error loading memory: {e}")

