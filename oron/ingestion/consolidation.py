from typing import List, Dict, Any, Optional
import numpy as np
from ..store.episodic import EpisodicStore
from ..store.semantic import SemanticStore
from ..adapters.base import BaseAdapter
import json

class MemoryConsolidator:
    """
    Consolidates episodic memories into semantic facts.
    Identifies repeating patterns and promotes them to the Semantic Store.
    """
    def __init__(
        self, 
        episodic_store: EpisodicStore, 
        semantic_store: SemanticStore, 
        adapter: BaseAdapter,
        repetition_threshold: int = 3
    ):
        self.episodic_store = episodic_store
        self.semantic_store = semantic_store
        self.adapter = adapter
        self.repetition_threshold = repetition_threshold

    def consolidate(self, user_id: str) -> int:
        """
        Runs the consolidation process for a user.
        Returns the number of new semantic facts promoted.
        """
        # 1. Fetch all episodic memories for the user
        # (This is a simplified version; in production, we'd use a rolling window)
        results = self.episodic_store.collection.get(
            where={"user_id": user_id},
            include=["documents", "embeddings", "metadatas"]
        )
        
        if not results["documents"] or len(results["documents"]) < self.repetition_threshold:
            return 0

        documents = results["documents"]
        embeddings = results["embeddings"]
        
        # 2. Identify clusters of similar memories
        # Naive clustering using cosine similarity
        clusters: List[List[int]] = []
        visited = set()
        
        for i in range(len(documents)):
            if i in visited: continue
            
            cluster = [i]
            visited.add(i)
            
            for j in range(i + 1, len(documents)):
                if j in visited: continue
                
                # Cosine similarity
                sim = np.dot(embeddings[i], embeddings[j]) / (np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j]))
                if sim > 0.85: # High similarity threshold
                    cluster.append(j)
                    visited.add(j)
            
            if len(cluster) >= self.repetition_threshold:
                clusters.append(cluster)

        promoted_count = 0
        
        # 3. For each cluster, use LLM to synthesize a Semantic Fact
        for cluster in clusters:
            cluster_docs = [documents[idx] for idx in cluster]
            
            synthesis_prompt = (
                "The following messages all convey a similar repeating theme or fact. "
                "Synthesize them into a single, high-confidence semantic fact (Subject-Relation-Object) "
                "or a core user preference.\n\n"
                f"Messages:\n" + "\n".join([f"- {d}" for d in cluster_docs]) +
                "\n\nReturn ONLY a JSON object: {\"facts\": [{\"subject\": str, \"relation\": str, \"object\": str}]}"
            )
            
            try:
                response = self.adapter.chat(synthesis_prompt, memories=[], model="llama-3.1-8b-instant")
                
                # Parse JSON
                json_str = response.strip()
                if "```json" in json_str:
                    json_str = json_str.split("```json")[1].split("```")[0].strip()
                data = json.loads(json_str)
                
                for fact in data.get("facts", []):
                    self.semantic_store.add_fact(
                        user_id, 
                        fact["subject"], 
                        fact["relation"], 
                        fact["object"],
                        metadata={"type": "consolidated", "source_count": len(cluster)}
                    )
                    promoted_count += 1
            except Exception as e:
                print(f"Error synthesizing cluster: {e}")

        return promoted_count
