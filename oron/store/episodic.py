import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Optional
import time


class EpisodicStore:
    """
    Episodic memory store using ChromaDB for vector-based
    similarity search of time-indexed events.
    """

    def __init__(self, path: str = "./chroma_db"):
        self.client = chromadb.PersistentClient(path=path)
        self.collection = self.client.get_or_create_collection(
            name="episodic_memories", metadata={"hnsw:space": "cosine"}
        )

    def add(
        self,
        user_id: str,
        content: str,
        embeddings: List[float],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        memory_id = f"{user_id}_{int(time.time() * 1000)}"

        meta = metadata or {}
        meta.update({"user_id": user_id, "timestamp": time.time(), "type": "episodic"})

        self.collection.add(
            ids=[memory_id],
            embeddings=[embeddings],
            documents=[content],
            metadatas=[meta],
        )
        return memory_id

    def search(
        self, user_id: str, query_embeddings: List[float], limit: int = 5
    ) -> List[Dict[str, Any]]:
        results = self.collection.query(
            query_embeddings=[query_embeddings],
            n_results=limit,
            where={"user_id": user_id},
        )

        memories = []
        if results["ids"]:
            for i in range(len(results["ids"][0])):
                memories.append(
                    {
                        "id": results["ids"][0][i],
                        "content": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "distance": (
                            results["distances"][0][i] if results["distances"] else None
                        ),
                    }
                )
        return memories

    def delete(self, user_id: str, memory_id: Optional[str] = None) -> None:
        if memory_id:
            self.collection.delete(ids=[memory_id])
        else:
            self.collection.delete(where={"user_id": user_id})
