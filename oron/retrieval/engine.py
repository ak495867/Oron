import numpy as np
from typing import List, Dict, Any, Optional

class RetrievalEngine:
    """
    Hybrid retrieval engine combining vector similarity, 
    graph traversal, and MMR re-ranking with biological decay.
    """
    def __init__(self, mmr_lambda: float = 0.5):
        self.mmr_lambda = mmr_lambda

    def _cosine_sim(self, v1: np.ndarray, v2: np.ndarray) -> float:
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(v1, v2) / (norm1 * norm2))

    def fuse_and_rerank(
        self, 
        query_embedding: List[float], 
        candidates: List[Dict[str, Any]], 
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Maximal Marginal Relevance re-ranking across heterogeneous stores.
        Expects candidates to have: 'content', 'embedding', 'salience', 'type'.
        """
        if not candidates:
            return []
            
        indices = list(range(len(candidates)))
        selected_indices: List[int] = []
        
        q_emb = np.array(query_embedding)
        c_embs = [np.array(c["embedding"]) for c in candidates]
        
        while len(selected_indices) < min(top_k, len(candidates)):
            best_score = -float('inf')
            best_idx = -1
            
            for idx in indices:
                if idx in selected_indices:
                    continue
                    
                # Base Relevance: Cosine Similarity
                relevance = self._cosine_sim(q_emb, c_embs[idx])
                
                # Biologically-inspired modification: multiply relevance by Salience
                # Salience accounts for time decay, confidence, and usage.
                salience = candidates[idx].get("salience", 1.0)
                relevance_score = relevance * salience
                
                # Diversity Penalty
                if not selected_indices:
                    diversity = 0.0
                else:
                    diversity = max([
                        self._cosine_sim(c_embs[idx], c_embs[s_idx])
                        for s_idx in selected_indices
                    ])
                
                # MMR Score
                score = self.mmr_lambda * relevance_score - (1 - self.mmr_lambda) * diversity
                
                if score > best_score:
                    best_score = score
                    best_idx = idx
            
            if best_idx != -1:
                selected_indices.append(best_idx)
            else:
                break
                
        return [candidates[i] for i in selected_indices]
