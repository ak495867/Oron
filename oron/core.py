import functools
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Optional, Callable
from .store.episodic import EpisodicStore
from .store.semantic import SemanticStore
from .store.procedural import ProceduralStore
from .ingestion.embedder import Embedder
from .ingestion.scorer import ImportanceScorer
from .ingestion.extractor import KGExtractor
from .ingestion.decay import DecayManager
from .retrieval.engine import RetrievalEngine
from .ingestion.consolidation import MemoryConsolidator
from .brain.processor import BrainProcessor
from .utils.visualizer import GraphVisualizer
import time

_oron_instances: Dict[str, "Oron"] = {}


def remember(user_id: str):
    """
    Decorator to automatically wrap a function with Oron.
    Retrieves context before call and remembers prompt after call.
    """

    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if user_id not in _oron_instances:
                _oron_instances[user_id] = Oron(user_id=user_id)
            mem = _oron_instances[user_id]
            prompt = args[0] if args else kwargs.get("prompt", "")
            recall_results = mem._recall_raw(str(prompt))
            memories = recall_results.get("context", [])
            procedural = recall_results.get("procedural", [])

            context = "\n".join([f"- {m}" for m in memories])
            system_rules = "\n".join([f"- {p}" for p in procedural])

            augmented_prompt = f"System Rules:\n{system_rules}\n\nBackground:\n{context}\n\nUser: {prompt}"
            result = func(augmented_prompt, *args[1:], **kwargs)
            mem.remember(str(prompt))
            return result

        return wrapper

    return decorator


class Oron:
    """
    Main API for Oron. Orchestrates ingestion, storage, and retrieval.
    v0.2 supports Async ingestion and Content/Intent split.
    """

    def __init__(
        self,
        user_id: str,
        db_dir: str = "./oron_data",
        model_name: str = "all-MiniLM-L6-v2",
        use_brain: bool = False,
        adapter: Optional[Any] = None,
    ):
        self.user_id = user_id
        self.embedder = Embedder(model_name=model_name)
        self.scorer = ImportanceScorer()
        self.extractor = KGExtractor()
        self.decay = DecayManager()
        self.episodic_store = EpisodicStore(path=f"{db_dir}/episodic")
        self.semantic_store = SemanticStore(path=f"{db_dir}/semantic.json")
        self.procedural_store = ProceduralStore(db_path=f"{db_dir}/procedural.db")
        self.retrieval_engine = RetrievalEngine()
        self.use_brain = use_brain
        self.brain = BrainProcessor(adapter) if use_brain and adapter else None
        self.adapter = adapter
        self.consolidator = (
            MemoryConsolidator(self.episodic_store, self.semantic_store, adapter)
            if adapter
            else None
        )
        self.visualizer = GraphVisualizer(self.semantic_store)
        self.executor = ThreadPoolExecutor(max_workers=4)

    def inspect(self):
        self.visualizer.render(self.user_id)

    def consolidate(self) -> int:
        if self.consolidator:
            return self.consolidator.consolidate(self.user_id)
        return 0

    async def aremember(self, text: str) -> None:
        """
        Async ingestion. Non-blocking.
        """
        if self.use_brain and self.brain:
            analysis = await self.brain.aanalyze(text)
            if (
                analysis.get("is_injection", False)
                or analysis.get("importance", 0.0) < 0.3
            ):
                return

            # 1. Episodic
            attributed_text = f"[USER_CLAIM] {text}"
            embedding = self.embedder.embed_text(attributed_text)
            self.episodic_store.add(self.user_id, attributed_text, embedding)

            # 2. Semantic
            for fact in analysis.get("facts", []):
                subj = fact["subject"]
                if subj.lower() in ["i", "me", "my", "user", "person"]:
                    subj = "user"
                self.semantic_store.add_fact(
                    self.user_id,
                    f"[CLAIM] {subj}",
                    fact["relation"],
                    fact["object"],
                    metadata={
                        "source": "user_claim",
                        "permanence": analysis.get("permanence"),
                    },
                )

            # 3. Procedural
            for pref in analysis.get("preferences", []):
                self.procedural_store.set(self.user_id, pref["key"], pref["value"])
                self.semantic_store.add_fact(
                    self.user_id, "user", pref["key"], str(pref["value"])
                )
        else:
            # Fallback to rule-based sync
            self._remember_sync(text)

    def _remember_sync(self, text: str) -> None:
        """Internal sync implementation of rule-based ingestion"""
        if not self.scorer.is_important(text, threshold=0.3):
            return
        embedding = self.embedder.embed_text(text)
        self.episodic_store.add(self.user_id, text, embedding)
        facts = self.extractor.extract_facts(text)
        for subj, rel, obj in facts:
            self.semantic_store.add_fact(self.user_id, subj, rel, obj)
        text_lower = text.lower()
        if "my name is" in text_lower or "i am called" in text_lower:
            words = text.split()
            for i, word in enumerate(words):
                if word.lower() in ["is", "called"] and i + 1 < len(words):
                    name = words[i + 1].strip(".,!?")
                    self.procedural_store.set(self.user_id, "user_name", name)
                    self.semantic_store.add_fact(self.user_id, "user", "name", name)
                    break

    def remember(self, text: str) -> None:
        """
        Store a new memory. v0.2 offloads to background thread to prevent blocking.
        """
        if self.use_brain:

            def run_async():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(self.aremember(text))
                loop.close()

            self.executor.submit(run_async)
        else:
            self._remember_sync(text)

    async def achat(
        self, prompt: str, adapter: Optional[Any] = None, **kwargs: Any
    ) -> str:
        memories = self.recall(prompt)
        active_adapter = adapter or self.adapter
        if active_adapter:
            response = await active_adapter.achat(prompt, memories, **kwargs)
        else:
            context = "\n".join([f"- {m}" for m in memories])
            response = f"[Simulated response]\nContext: {context}"
        asyncio.create_task(self.aremember(prompt))
        return response

    def chat(self, prompt: str, adapter: Optional[Any] = None, **kwargs: Any) -> str:
        memories = self.recall(prompt)
        active_adapter = adapter or self.adapter
        if active_adapter:
            response = active_adapter.chat(prompt, memories, **kwargs)
        else:
            context = "\n".join([f"- {m}" for m in memories])
            response = f"[Simulated response]\nContext: {context}"
        self.remember(prompt)
        return response

    def _recall_raw(self, query: str, limit: int = 5) -> Dict[str, List[str]]:
        """
        Internal method that fetches raw candidates, applies decay,
        and fuses via MMR. Returns structured memories (context vs procedural).
        """
        query_emb = self.embedder.embed_text(query)
        candidates = []

        # 1. Fetch Episodic
        episodic_results = self.episodic_store.search(
            self.user_id, query_emb, limit=limit * 2
        )
        for r in episodic_results:
            emb = self.embedder.embed_text(r["content"])
            timestamp = r["metadata"].get("timestamp", time.time())
            importance = 0.5
            salience = self.decay.calculate_episodic_salience(timestamp, importance)

            candidates.append(
                {
                    "type": "episodic",
                    "content": r["content"],
                    "embedding": emb,
                    "salience": salience,
                }
            )

        # 2. Fetch Semantic
        try:
            query_doc = self.extractor.nlp(query)
            entities = [ent.text for ent in query_doc.ents] + [
                token.text for token in query_doc if token.pos_ in ["NOUN", "PROPN"]
            ]
        except ImportError:
            entities = query.split()  # fallback: treat every word as a potential entity

        seen_facts = set()
        for entity in entities:
            if len(entity) < 3:
                continue
            facts = self.semantic_store.get_related(self.user_id, entity.lower())
            for fact in facts:
                s, r, o = fact["subject"], fact["relation"], fact["object"]
                fact_str = f"Fact: {s} {r} {o}"

                if fact_str not in seen_facts:
                    seen_facts.add(fact_str)
                    emb = self.embedder.embed_text(fact_str)
                    salience = self.decay.calculate_semantic_salience(
                        fact.get("confidence", 1)
                    )
                    candidates.append(
                        {
                            "type": "semantic",
                            "content": fact_str,
                            "embedding": emb,
                            "salience": salience,
                        }
                    )

        # 3. Fetch Procedural
        all_procedural = self.procedural_store.get_all(self.user_id)
        for k, v in all_procedural.items():
            rule_str = f"Preference/Rule ({k}): {v}"
            emb = self.embedder.embed_text(rule_str)
            salience = self.decay.calculate_procedural_salience(1, time.time())
            candidates.append(
                {
                    "type": "procedural",
                    "content": rule_str,
                    "embedding": emb,
                    "salience": salience,
                }
            )

        # Also always include User Identity as a top-level procedural fact
        user_name = self.procedural_store.get(self.user_id, "user_name")
        if user_name:
            candidates.append(
                {
                    "type": "procedural",
                    "content": f"User Identity: The user's name is {user_name}.",
                    "embedding": query_emb,
                    "salience": 1.0,
                }
            )

        # 4. Fuse & MMR Re-rank
        ranked_candidates = self.retrieval_engine.fuse_and_rerank(
            query_emb, candidates, top_k=limit + 2
        )

        # 5. Separate outputs
        context_memories = []
        procedural_memories = []
        for c in ranked_candidates:
            if c["type"] == "procedural":
                procedural_memories.append(c["content"])
            else:
                context_memories.append(c["content"])

        return {"context": context_memories, "procedural": procedural_memories}

    def recall(self, query: str, limit: int = 5) -> List[str]:
        """
        Public backwards-compatible recall. Returns flat list of context.
        """
        results = self._recall_raw(query, limit)
        return results["procedural"] + results["context"]
