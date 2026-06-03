#!/usr/bin/env python3
"""
Parallel Ablation Runner — runs all configs on all questions simultaneously.

Uses ThreadPoolExecutor to process 4 questions in parallel.
Threads are ideal here: Ollama runs as a background server, so threads
wait on HTTP responses without duplicating the 4GB model into RAM.

Usage:
    ollama serve          # (in separate terminal)
    python run_parallel.py
"""

import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

sys.path.insert(0, str(Path(__file__).parent))

from src.agent.research_agent import ResearchAgent, CONFIGS
from src.config import EVAL_DIR, PREDICTIONS_DIR

# Thread-safe lock for printing and result collection
_print_lock = Lock()
_results_lock = Lock()


def extract_cited_papers(answer: str) -> list[str]:
    """Extract clean arXiv IDs from answer text using regex."""
    all_citations = re.findall(r'\[([^]]+)\]', answer)
    clean_ids = set()
    for c in all_citations:
        match = re.search(r'(\d{4}\.\d{4,5})', c)
        if match:
            clean_ids.add(match.group(1))
    return sorted(clean_ids)


def process_question(args: tuple) -> dict:
    """
    Process a single (config_name, config_obj, question) tuple.
    Returns the formatted prediction dict with only the 3 submission fields.
    """
    config_name, config_obj, question = args
    q_id = question["id"]
    q_type = question.get("type", "factoid")

    try:
        agent = ResearchAgent(config_obj)
        answer, trace = agent.run(question["question"], q_type=q_type)
        cited_papers = extract_cited_papers(answer)
    except Exception as e:
        with _print_lock:
            print(f"[ERROR] {q_id} - {config_name}: {e}")
        answer = ""
        cited_papers = []

    result = {
        "id": q_id,
        "answer": answer,
        "cited_papers": cited_papers,
    }

    with _print_lock:
        print(f"[FINISHED] {q_id} - {config_name}")

    return config_name, result


def load_questions() -> list[dict]:
    """Load evaluation questions."""
    questions_path = EVAL_DIR / "questions.jsonl"
    if not questions_path.exists():
        print(f"Error: {questions_path} not found")
        return []
    questions = []
    with open(questions_path) as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))
    return questions


def main():
    questions = load_questions()
    if not questions:
        return

    # Build task list: every config × every question
    tasks = []
    for config_name, config_obj in CONFIGS.items():
        for q in questions:
            tasks.append((config_name, config_obj, q))

    total = len(tasks)
    print(f"Running {total} tasks ({len(CONFIGS)} configs × {len(questions)} questions) with 4 workers...")

    start_time = time.time()

    # Collect results grouped by config_name
    all_results: dict[str, list[dict]] = {name: [] for name in CONFIGS}

    with ThreadPoolExecutor(max_workers=4) as executor:
        future_to_task = {
            executor.submit(process_question, task): task
            for task in tasks
        }

        completed = 0
        for future in as_completed(future_to_task):
            completed += 1
            try:
                config_name, result = future.result()
                with _results_lock:
                    all_results[config_name].append(result)
            except Exception as e:
                task = future_to_task[future]
                with _print_lock:
                    print(f"[FATAL] {task[2]['id']} - {task[0]}: {e}")

            if completed % 10 == 0 or completed == total:
                elapsed = time.time() - start_time
                with _print_lock:
                    print(f"  Progress: {completed}/{total} ({elapsed:.0f}s elapsed)")

    # Write predictions per config
    for config_name, results in all_results.items():
        if not results:
            continue
        # Sort by question ID to maintain consistent order
        results.sort(key=lambda r: r["id"])
        output_path = PREDICTIONS_DIR / f"{config_name}.jsonl"
        with open(output_path, "w") as f:
            for pred in results:
                submission_format = {
                    "id": pred.get("id"),
                    "answer": pred.get("answer"),
                    "cited_papers": pred.get("cited_papers", []),
                }
                f.write(json.dumps(submission_format) + "\n")
        print(f"Saved {len(results)} predictions to {output_path}")

    elapsed = time.time() - start_time
    print(f"\nDone. Total time: {elapsed:.0f}s ({elapsed/60:.1f}min)")


if __name__ == "__main__":
    main()
