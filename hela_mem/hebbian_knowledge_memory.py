import json
import os
import sys
from .hebbian_memory import HebbianMemoryGraph
from .utils import get_timestamp, get_embedding, normalize_vector

class HebbianKnowledgeMemory:
    def __init__(self, file_path="long_term_hebbian.json", max_capacity=100):
        self.file_path = file_path
        self.max_capacity = max_capacity
        self.user_profiles = {}
        self.assistant_knowledge = [] # Keep assistant knowledge simple for now
        self.fallback_kb = []  # [NEW] Fallback KB from long_term.json
        
        # [NEW] Independent KB max_flipped parameter
        self.kb_max_flipped = int(os.environ.get("HEBBIAN_KB_MAX_FLIPPED", 5))
        
        # Initialize Hebbian Graph for Knowledge Base
        self.hebbian_kb_path = file_path.replace(".json", "_kb_graph.json")
        self.knowledge_graph = HebbianMemoryGraph(self.hebbian_kb_path)
        
        self.load()

    def update_user_profile(self, user_id, new_data, merge=False):
        """
        Update user profile (Standard implementation)
        """
        if merge and user_id in self.user_profiles:
            current_data = self.user_profiles[user_id]["data"]
            if isinstance(current_data, str) and isinstance(new_data, str):
                updated_data = f"{current_data}\n\n--- Updated ---\n{new_data}"
            else:
                updated_data = new_data
        else:
            updated_data = new_data
        
        self.user_profiles[user_id] = {
            "data": updated_data,
            "last_updated": get_timestamp()
        }
        print("HebbianKB: Updated user profile.")
        self.save()

    def add_assistant_knowledge(self, knowledge_text):
        """
        Add assistant knowledge (Standard implementation)
        """
        if not knowledge_text or knowledge_text.strip() in ["", "- None", "- None."]:
            return
            
        vec = get_embedding(knowledge_text)
        vec = normalize_vector(vec).tolist()
        entry = {
            "knowledge": knowledge_text,
            "timestamp": get_timestamp(),
            "knowledge_embedding": vec
        }
        self.assistant_knowledge.append(entry)
        if len(self.assistant_knowledge) > self.max_capacity:
            self.assistant_knowledge = self.assistant_knowledge[-self.max_capacity:]
        print("HebbianKB: Added assistant knowledge.")
        self.save()

    def get_assistant_knowledge(self):
        return self.assistant_knowledge

    def get_raw_user_profile(self, user_id):
        return self.user_profiles.get(user_id, {}).get("data", "")
    
    def get_user_profile(self, user_id):
        return self.user_profiles.get(user_id, {})

    def add_knowledge(self, knowledge_text):
        """
        Add knowledge to Hebbian Graph
        """
        if not knowledge_text or knowledge_text.strip() in ["", "- None", "- None."]:
            return

        # Add to graph
        # We use 'system' role and 'fact' type to distinguish
        self.knowledge_graph.add_memory(
            content=knowledge_text,
            role="system",
            metadata={"type": "fact"}
        )
        print("HebbianKB: Added knowledge node to graph.")
        self.knowledge_graph.save()
        self.save() # Save other data

    def get_knowledge(self):
        # Return all nodes content as a list (for compatibility if needed)
        # This might be expensive if graph is huge
        return [{"knowledge": node["content"]} for node in self.knowledge_graph.nodes.values()]

    def search_knowledge(self, query, threshold=0.1, top_k=10):
        """
        Retrieve knowledge using Hebbian Spreading Activation.
        Falls back to vector similarity search if no graph nodes exist.
        """
        # Check if we have graph nodes
        if len(self.knowledge_graph.nodes) > 0:
            # 1. Retrieve from Graph with KB-specific max_flipped
            hebbian_results = self.knowledge_graph.retrieve(query, top_k=top_k, override_max_flipped=self.kb_max_flipped)
            
            # 2. Format results to match expected interface
            results = []
            for res in hebbian_results:
                node = res["node"]
                entry = {
                    "knowledge": node["content"],
                    "timestamp": node["timestamp"],
                    "knowledge_embedding": node["embedding"],
                    "score": res["score"]
                }
                results.append(entry)
            
            print(f"HebbianKB: Retrieved {len(results)} knowledge entries from graph.")
            return results
        
        # [FALLBACK] Use vector similarity on fallback_kb
        if len(self.fallback_kb) == 0:
            print(f"HebbianKB: Retrieved 0 knowledge entries (no data).")
            return []
        
        # Compute query embedding
        query_vec = normalize_vector(get_embedding(query))
        
        # Score all KB entries
        scored = []
        for entry in self.fallback_kb:
            if "knowledge_embedding" in entry and entry["knowledge_embedding"]:
                kb_vec = normalize_vector(entry["knowledge_embedding"])
                score = float(query_vec @ kb_vec)
                scored.append((score, entry))
        
        # Sort by score and take top_k
        scored.sort(key=lambda x: -x[0])
        results = []
        for score, entry in scored[:top_k]:
            results.append({
                "knowledge": entry.get("knowledge", ""),
                "timestamp": entry.get("timestamp", ""),
                "knowledge_embedding": entry.get("knowledge_embedding", []),
                "score": score
            })
        
        print(f"HebbianKB: Retrieved {len(results)} knowledge entries (fallback mode).")
        return results

    def save(self):
        data = {
            "user_profiles": self.user_profiles,
            "assistant_knowledge": self.assistant_knowledge
        }
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        # Graph saves itself in add_memory, but good to be safe
        self.knowledge_graph.save()
        print("HebbianKB: Saved successfully.")

    def load(self):
        try:
            if os.path.exists(self.file_path):
                with open(self.file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.user_profiles = data.get("user_profiles", {})
                    self.assistant_knowledge = data.get("assistant_knowledge", [])
                    # [NEW] Load fallback KB from knowledge_base field
                    self.fallback_kb = data.get("knowledge_base", [])
                    if self.fallback_kb:
                        print(f"HebbianKB: Loaded {len(self.fallback_kb)} KB entries from fallback.")
            
            # Graph loads itself in __init__
            print("HebbianKB: Loaded successfully.")
        except Exception as e:
            print(f"HebbianKB: Error loading data: {e}")
            self.user_profiles = {}
            self.assistant_knowledge = []
            self.fallback_kb = []
