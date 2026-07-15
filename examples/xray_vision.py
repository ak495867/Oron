import asyncio
import os
import shutil
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from oron import Oron
from oron.adapters.groq import GroqAdapter

console = Console()


async def run_xray():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        console.print("[red]GROQ_API_KEY not set.[/red]")
        return

    db_dir = "./xray_data"
    if os.path.exists(db_dir):
        shutil.rmtree(db_dir)

    adapter = GroqAdapter(api_key=api_key)
    mem = Oron(user_id="xray_user", db_dir=db_dir, use_brain=True, adapter=adapter)

    console.print(
        Panel.fit(
            "[bold magenta]Oron X-Ray Vision[/bold magenta]\nProving this is a true Cognitive Engine, not a wrapper.",
            border_style="magenta",
        )
    )

    # Turn 1: Ingestion
    msg1 = "I absolutely hate writing Java, but I love Python. My name is Vroski."
    console.print(f"\n[bold cyan]User:[/bold cyan] {msg1}")

    # Manually trigger Brain to show what it does
    console.print("[yellow]--- 1. BRAIN INGESTION (LLM Cognitive Pass) ---[/yellow]")
    analysis = await mem.brain.aanalyze(msg1)
    console.print(analysis)

    # Actually ingest it
    await mem.aremember(msg1)

    # Turn 2: Recall & Math
    msg2 = "What do I think about Java and what is my name?"
    console.print(f"\n[bold cyan]User:[/bold cyan] {msg2}")

    console.print("[yellow]--- 2. HYBRID SEARCH & BIOLOGICAL DECAY ---[/yellow]")

    # Intercept the raw recall to show the math
    query_emb = mem.embedder.embed_text(msg2)
    raw_candidates = []

    # Fake a fetch to show the raw hits
    ep_hits = mem.episodic_store.search("xray_user", query_emb, limit=5)
    for hit in ep_hits:
        salience = mem.decay.calculate_episodic_salience(
            hit["metadata"]["timestamp"], 0.5
        )
        raw_candidates.append(
            {
                "type": "episodic",
                "content": hit["content"],
                "salience": salience,
                "embedding": mem.embedder.embed_text(hit["content"]),
            }
        )

    sem_hits = mem.semantic_store.get_related(
        "xray_user", "vroski"
    ) + mem.semantic_store.get_related("xray_user", "java")
    for hit in sem_hits:
        salience = mem.decay.calculate_semantic_salience(hit["confidence"])
        raw_candidates.append(
            {
                "type": "semantic",
                "content": f"Fact: {hit['subject']} {hit['relation']} {hit['object']}",
                "salience": salience,
                "embedding": mem.embedder.embed_text(
                    f"Fact: {hit['subject']} {hit['relation']} {hit['object']}"
                ),
            }
        )

    table = Table(title="Raw Candidates Before MMR Fusion")
    table.add_column("Type", style="cyan")
    table.add_column("Salience (Decay Math)", justify="right", style="green")
    table.add_column("Content")

    for c in raw_candidates:
        table.add_row(c["type"], f"{c['salience']:.4f}", c["content"])
    console.print(table)

    console.print("[yellow]--- 3. MMR FUSION (Diversity + Relevance) ---[/yellow]")
    reranked = mem.retrieval_engine.fuse_and_rerank(query_emb, raw_candidates, top_k=5)

    mmr_table = Table(title="Final Context Window (After MMR)")
    mmr_table.add_column("Rank", justify="center")
    mmr_table.add_column("Type", style="cyan")
    mmr_table.add_column("Content")

    for i, c in enumerate(reranked):
        mmr_table.add_row(str(i + 1), c["type"], c["content"])
    console.print(mmr_table)

    console.print("[yellow]--- 4. FINAL LLM GENERATION ---[/yellow]")
    # Use the actual engine
    response = await mem.achat(msg2)
    console.print(f"[bold green]AI:[/bold green] {response}")


if __name__ == "__main__":
    asyncio.run(run_xray())
