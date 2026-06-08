from rich.tree import Tree
from rich.console import Console
from rich.panel import Panel
from ..store.semantic import SemanticStore

class GraphVisualizer:
    """
    CLI-based visualizer for the Oron Knowledge Graph.
    Uses 'rich' to render a beautiful tree view of entities and relations.
    """
    def __init__(self, semantic_store: SemanticStore):
        self.store = semantic_store
        self.console = Console()

    def render(self, user_id: str):
        """
        Renders the Knowledge Graph for a specific user as a tree.
        """
        root = Tree(f"[bold cyan]Oron Knowledge Graph[/bold cyan] (User: [yellow]{user_id}[/yellow])")
        
        # We need to find nodes belonging to this user
        # In our SemanticStore, nodes are prefixed with "user_id:"
        user_prefix = f"{user_id}:"
        
        # Group facts by subject
        subjects = {}
        for node in self.store.graph.nodes:
            if node.startswith(user_prefix):
                label = self.store.graph.nodes[node].get("label", node.replace(user_prefix, ""))
                subjects[node] = label

        if not subjects:
            self.console.print("[red]No semantic facts found for this user.[/red]")
            return

        for node, label in subjects.items():
            subject_tree = root.add(f"[bold green]{label}[/bold green]")
            
            # Outgoing edges
            for _, target, data in self.store.graph.out_edges(node, data=True):
                target_label = self.store.graph.nodes[target].get("label", target.replace(user_prefix, ""))
                rel = data.get("relation", "relates_to")
                conf = data.get("confidence", 1)
                
                # Color code based on confidence
                conf_color = "green" if conf > 2 else "white"
                subject_tree.add(f"-> [blue]{rel}[/blue] -> [bold]{target_label}[/bold] ([{conf_color}]conf: {conf}[/{conf_color}])")

        self.console.print("\n")
        self.console.print(Panel(root, border_style="bright_blue", expand=False))
        self.console.print("\n")
