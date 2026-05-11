#!/usr/bin/env python3
"""
cli.py — run TrustRAG from the command line without starting the API server.

Usage:
    python cli.py --query "What is RAG?" --urls https://example.com https://other.com
    python cli.py --query "..." --urls https://... --save-index ./my_index
"""
from __future__ import annotations
import argparse
import json
import sys

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

from trustrag.core.pipeline import TrustRAGPipeline
from trustrag.core.vector_store import VectorStore

console = Console()


def main() -> None:
    parser = argparse.ArgumentParser(description="TrustRAG CLI")
    parser.add_argument("--query", "-q", required=True, help="Question to answer")
    parser.add_argument("--urls", "-u", nargs="+", required=True, help="URLs to retrieve from")
    parser.add_argument("--save-index", help="Save FAISS index to this directory after query")
    parser.add_argument("--load-index", help="Load existing FAISS index from this directory")
    parser.add_argument("--json", action="store_true", help="Output raw JSON instead of rich display")
    args = parser.parse_args()

    # Set up pipeline
    if args.load_index:
        console.print(f"[dim]Loading index from {args.load_index}…[/]")
        store = VectorStore.load(args.load_index)
        pipeline = TrustRAGPipeline(store=store)
    else:
        pipeline = TrustRAGPipeline()

    console.print(f"[bold cyan]TrustRAG[/] running query…\n")

    result = pipeline.query(args.query, args.urls)

    if args.json:
        print(json.dumps(result.__dict__, indent=2))
        return

    # --- Rich display ---

    # Answer panel
    trust_color = "green" if result.is_trusted else "yellow"
    flagged_label = " ⚠ FLAGGED" if result.flagged else ""
    title = (
        f"Answer  ·  confidence {result.confidence:.0%}  "
        f"[{trust_color}]{'✓ trusted' if result.is_trusted else '✗ not trusted'}[/]{flagged_label}"
    )
    console.print(Panel(result.answer, title=title, border_style=trust_color))

    # References
    if result.references:
        table = Table(title="References", box=box.SIMPLE, show_header=True)
        table.add_column("ID", style="bold", width=4)
        table.add_column("URL", style="cyan")
        table.add_column("Snippet", style="dim")
        for ref in result.references:
            table.add_row(f"[{ref['id']}]", ref["url"], ref["snippet"][:80] + "…")
        console.print(table)

    # Ungrounded claims
    if result.ungrounded_claims:
        console.print("\n[yellow]⚠ Ungrounded claims:[/]")
        for c in result.ungrounded_claims:
            console.print(f"  • {c}")

    # Stats
    console.print(
        f"\n[dim]loops: {result.loops_used}  |  "
        f"latency: {result.latency_ms:.0f} ms  |  "
        f"sources indexed: {len(result.sources_indexed)}[/]"
    )

    if args.save_index:
        pipeline.store.save(args.save_index)
        console.print(f"[dim]Index saved to {args.save_index}[/]")


if __name__ == "__main__":
    main()