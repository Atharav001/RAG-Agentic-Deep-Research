#!/usr/bin/env python3
"""
Main entry point — run the full pipeline from a fresh clone.

Usage:
    python run.py scrape       # Step 1: Scrape arXiv papers
    python run.py parse        # Step 2: Parse PDFs and chunk text
    python run.py index        # Step 3: Build retrieval indices
    python run.py agent        # Step 4: Run agent on a single question (interactive)
    python run.py ablation     # Step 5: Run all ablation configs on eval set
    python run.py evaluate     # Step 6: Evaluate predictions
    python run.py all          # Run steps 1-6 end-to-end
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def cmd_scrape():
    from src.scraper.arxiv_scraper import main as scrape_main
    scrape_main()


def cmd_parse():
    from src.indexer.pdf_parser import process_all_papers
    process_all_papers()


def cmd_index():
    from src.indexer.build_index import build_all
    build_all()


def cmd_agent():
    """Interactive: run the full agent on a single question."""
    from src.agent.research_agent import ResearchAgent, CONFIGS
    agent = ResearchAgent(CONFIGS["full_agent"])
    print("\nAgentic Deep Research — Interactive Mode")
    print("Type a research question (or 'quit' to exit):\n")
    while True:
        question = input("> ").strip()
        if question.lower() in ("quit", "exit", "q"):
            break
        if not question:
            continue
        answer, trace = agent.run(question)
        print(f"\n{'='*60}")
        print("ANSWER:")
        print(f"{'='*60}")
        print(answer)
        print(f"\nCited papers: {trace.cited_arxiv_ids}")
        print(f"Tool calls: {trace.total_tool_calls}, Latency: {trace.latency_seconds:.1f}s\n")


def cmd_ablation():
    from src.evaluation.run_ablation import run_all_ablations
    run_all_ablations()


def cmd_evaluate():
    from src.evaluation.run_ablation import print_ablation_table
    from src.evaluation.evaluator import evaluate_config
    import json
    from src.config import PREDICTIONS_DIR

    configs = ["full_agent", "baseline", "no_planner", "no_reranker",
               "no_reflector", "no_hybrid", "no_citation_verifier"]
    results = {}
    for c in configs:
        pred_path = PREDICTIONS_DIR / f"{c}.jsonl"
        if pred_path.exists():
            results[c] = evaluate_config(c)
    print_ablation_table(results)


def cmd_all():
    print("\n" + "=" * 60)
    print("FULL PIPELINE — Agentic Deep Research")
    print("=" * 60)

    print("\n>>> STEP 1/5: Scraping arXiv papers...")
    cmd_scrape()

    print("\n>>> STEP 2/5: Parsing PDFs and chunking text...")
    cmd_parse()

    print("\n>>> STEP 3/5: Building retrieval indices...")
    cmd_index()

    print("\n>>> STEP 4/5: Running ablation study...")
    cmd_ablation()

    print("\n>>> STEP 5/5: Evaluating predictions...")
    cmd_evaluate()

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)


COMMANDS = {
    "scrape": cmd_scrape,
    "parse": cmd_parse,
    "index": cmd_index,
    "agent": cmd_agent,
    "ablation": cmd_ablation,
    "evaluate": cmd_evaluate,
    "all": cmd_all,
}


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print(__doc__)
        print(f"Available commands: {', '.join(COMMANDS.keys())}")
        sys.exit(1)

    command = sys.argv[1]
    COMMANDS[command]()


if __name__ == "__main__":
    main()
