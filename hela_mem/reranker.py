"""
Reranker module for HeLa-Mem.
Uses Cross-Encoder to re-rank retrieved passages for higher precision.
"""

import os
from typing import List, Tuple, Dict
from functools import lru_cache

# Global reranker instance (lazy loaded)
_reranker_instance = None

def get_reranker():
    """Lazy load reranker model (singleton pattern for efficiency)."""
    global _reranker_instance
    if _reranker_instance is None:
        try:
            from sentence_transformers import CrossEncoder
            model_name = os.environ.get('HEBBIAN_RERANKER_MODEL', 'cross-encoder/ms-marco-MiniLM-L-6-v2')
            print(f"[Reranker] Loading model: {model_name}")
            # Force CPU to avoid meta tensor issues
            _reranker_instance = CrossEncoder(model_name, device='cpu')
            print(f"[Reranker] Model loaded successfully")
        except ImportError:
            print("[Reranker] Warning: sentence-transformers not installed. Using BM25 fallback.")
            _reranker_instance = "FALLBACK"
        except Exception as e:
            # Catch PyTorch meta tensor and other loading errors
            print(f"[Reranker] Warning: Failed to load CrossEncoder: {e}")
            print("[Reranker] Using BM25 fallback reranking.")
            _reranker_instance = "FALLBACK"
    
    # Return None if fallback, so callers use BM25
    if _reranker_instance == "FALLBACK":
        return None
    return _reranker_instance


def rerank_passages(query: str, passages: List[str], top_k: int = 5) -> List[str]:
    """
    Re-rank passages using Cross-Encoder.
    
    Args:
        query: The question/query string
        passages: List of passage strings to rerank
        top_k: Number of top passages to return
    
    Returns:
        List of top-k passages sorted by relevance
    """
    if not passages:
        return []
    
    reranker = get_reranker()
    
    if reranker is None:
        # Fallback: return original order
        return passages[:top_k]
    
    # Create query-passage pairs
    pairs = [(query, passage) for passage in passages]
    
    # Get scores from cross-encoder
    scores = reranker.predict(pairs)
    
    # Sort by score (descending)
    scored_passages = list(zip(passages, scores))
    scored_passages.sort(key=lambda x: x[1], reverse=True)
    
    # Return top-k
    return [p for p, s in scored_passages[:top_k]]


def rerank_memories(query: str, memories: List[Dict], top_k: int = 5) -> List[Dict]:
    """
    Re-rank memory dicts using Cross-Encoder.
    
    Args:
        query: The question/query string
        memories: List of memory dicts with 'content' field
        top_k: Number of top memories to return
    
    Returns:
        List of top-k memory dicts sorted by relevance
    """
    if not memories:
        return []
    
    reranker = get_reranker()
    
    if reranker is None:
        # Fallback: use BM25-like keyword matching
        passages = [m.get('content', '') for m in memories]
        scores = [bm25_score(query, p) for p in passages]
        scored_memories = list(zip(memories, scores))
        scored_memories.sort(key=lambda x: x[1], reverse=True)
        return [m for m, s in scored_memories[:top_k]]
    
    # Extract content for scoring
    passages = [m.get('content', '') for m in memories]
    pairs = [(query, p) for p in passages]
    
    # Get scores
    scores = reranker.predict(pairs)
    
    # Sort by score
    scored_memories = list(zip(memories, scores))
    scored_memories.sort(key=lambda x: x[1], reverse=True)
    
    return [m for m, s in scored_memories[:top_k]]


def bm25_score(query: str, passage: str) -> float:
    """
    Simple BM25-like keyword matching score.
    Fallback when cross-encoder is not available.
    """
    import re
    
    # Tokenize
    query_tokens = set(re.findall(r'\w+', query.lower()))
    passage_tokens = re.findall(r'\w+', passage.lower())
    passage_set = set(passage_tokens)
    
    # Count matches
    matches = query_tokens & passage_set
    
    if len(query_tokens) == 0:
        return 0.0
    
    # Simple TF score
    score = len(matches) / len(query_tokens)
    
    return score


def hybrid_rerank(query: str, passages: List[str], 
                  embedding_scores: List[float] = None,
                  top_k: int = 5,
                  alpha: float = 0.7) -> List[str]:
    """
    Hybrid reranking combining cross-encoder and embedding scores.
    
    Args:
        query: Query string
        passages: List of passages
        embedding_scores: Original embedding similarity scores
        top_k: Number to return
        alpha: Weight for cross-encoder score (1-alpha for embedding)
    
    Returns:
        Reranked top-k passages
    """
    if not passages:
        return []
    
    reranker = get_reranker()
    
    # Get cross-encoder scores
    if reranker is not None:
        pairs = [(query, p) for p in passages]
        ce_scores = reranker.predict(pairs)
        # Normalize to [0, 1]
        ce_min, ce_max = min(ce_scores), max(ce_scores)
        if ce_max > ce_min:
            ce_scores = [(s - ce_min) / (ce_max - ce_min) for s in ce_scores]
        else:
            ce_scores = [0.5] * len(ce_scores)
    else:
        # Fallback to BM25
        ce_scores = [bm25_score(query, p) for p in passages]
    
    # Use provided embedding scores or default to 0.5
    if embedding_scores is None:
        embedding_scores = [0.5] * len(passages)
    else:
        # Normalize embedding scores
        emb_min, emb_max = min(embedding_scores), max(embedding_scores)
        if emb_max > emb_min:
            embedding_scores = [(s - emb_min) / (emb_max - emb_min) for s in embedding_scores]
    
    # Combine scores
    final_scores = [
        alpha * ce + (1 - alpha) * emb
        for ce, emb in zip(ce_scores, embedding_scores)
    ]
    
    # Sort and return
    scored = list(zip(passages, final_scores))
    scored.sort(key=lambda x: x[1], reverse=True)
    
    return [p for p, s in scored[:top_k]]
