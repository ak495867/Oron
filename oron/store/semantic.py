import networkx as nx
from typing import List, Dict, Any, Tuple, Optional
import os
import json
import time

class SemanticStore:
    """
    Semantic memory store using NetworkX for knowledge graph 
    storage of structured facts and relations.
    """
    def __init__(self, path: str = "oron_semantic.json"):
        self.path = path
        self.graph = nx.DiGraph()
        self._load()

    def _load(self) -> None:
        if os.path.exists(self.path):
            try:
                with open(self.path, "r") as f:
                    data = json.load(f)
                    self.graph = nx.node_link_graph(data)
            except Exception:
                self.graph = nx.DiGraph()

    def _save(self) -> None:
        data = nx.node_link_data(self.graph)
        with open(self.path, "w") as f:
            json.dump(data, f)

    def add_fact(
        self, 
        user_id: str, 
        subject: str, 
        relation: str, 
        object_: str, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        s_node = f"{user_id}:{subject.lower()}"
        o_node = f"{user_id}:{object_.lower()}"
        
        if not self.graph.has_node(s_node):
            self.graph.add_node(s_node, label=subject, user_id=user_id)
        if not self.graph.has_node(o_node):
            self.graph.add_node(o_node, label=object_, user_id=user_id)
        
        meta = metadata or {}
        new_conf = meta.get("confidence", 1)

        # 1. Check for Exact Match (Update Confidence)
        if self.graph.has_edge(s_node, o_node):
            data = self.graph[s_node][o_node]
            if data.get("relation") == relation:
                # Increment confidence of exact same fact
                data["confidence"] = data.get("confidence", 1) + new_conf
                data["last_seen"] = time.time()
                self._save()
                return

            # Define relations that should trigger a strict overwrite
    MUTUALLY_EXCLUSIVE_RELATIONS = {"diet", "name", "birthplace", "age", "current_location"}

            # 2. Check for Contradictions (Same S-R, Different O)
    edge_to_remove = None
    if relation.lower() in MUTUALLY_EXCLUSIVE_RELATIONS:
        for _, existing_obj, data in list(self.graph.out_edges(s_node, data=True)):
            if data.get("relation") == relation:
                existing_conf = data.get("confidence", 1)

                if existing_conf > 2 and new_conf <= 1:
                            # Existing fact is heavily confirmed, ignore the one-off contradiction
                    return
                else:
                            # Overwrite the old state with the new one
                    edge_to_remove = existing_obj
                    break

        if edge_to_remove:
            self.graph.remove_edge(s_node, edge_to_remove)
        
        # 3. Add the New Fact
        self.graph.add_edge(
            s_node, 
            o_node, 
            relation=relation, 
            confidence=new_conf,
            last_seen=time.time(),
            metadata=meta
        )
        self._save()

    def get_related(self, user_id: str, entity: str) -> List[Dict[str, Any]]:
        """Returns facts with confidence data."""
        node = f"{user_id}:{entity.lower()}"
        facts = []
        if node in self.graph:
            # Outgoing edges
            for _, target, data in self.graph.out_edges(node, data=True):
                target_label = self.graph.nodes[target].get("label", target)
                facts.append({
                    "subject": entity,
                    "relation": data["relation"],
                    "object": target_label,
                    "confidence": data.get("confidence", 1)
                })
            # Incoming edges
            for source, _, data in self.graph.in_edges(node, data=True):
                source_label = self.graph.nodes[source].get("label", source)
                facts.append({
                    "subject": source_label,
                    "relation": data["relation"],
                    "object": entity,
                    "confidence": data.get("confidence", 1)
                })
        return facts

    def delete_entity(self, user_id: str, entity: str) -> None:
        node = f"{user_id}:{entity.lower()}"
        if node in self.graph:
            self.graph.remove_node(node)
            self._save()
