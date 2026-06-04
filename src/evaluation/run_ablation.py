"""
Ablation Runner — runs all agent configurations on the eval set
and produces predictions/<config>.jsonl files.
"""

import json
import re
from pathlib import Path
from typing import List

from pydantic import BaseModel, Field, field_validator

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.agent.research_agent import ResearchAgent, CONFIGS, AgentConfig
from src.evaluation.evaluator import evaluate_config
from src.config import EVAL_DIR, PREDICTIONS_DIR


class Prediction(BaseModel):
    """Schema-validated prediction record for submission."""
    id: str = Field(..., description="Question ID (e.g., 'q01')")
    answer: str = Field(..., description="Generated answer text")
    cited_papers: List[str] = Field(
        default_factory=list,
        description="List of arXiv IDs cited in the answer"
    )

    @field_validator("cited_papers")
    @classmethod
    def validate_arxiv_ids(cls, v: List[str]) -> List[str]:
        """Ensure all cited papers match the arXiv ID format (YYMM.NNNNN)."""
        pattern = re.compile(r"^\d{4}\.\d{4,5}$")
        cleaned = [pid for pid in v if pattern.match(pid)]
        return sorted(set(cleaned))

    def to_jsonl(self) -> str:
        return self.model_dump_json()


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


def extract_cited_papers(answer: str) -> list[str]:
    """Extract clean arXiv IDs from answer text using regex."""
    all_citations = re.findall(r'\[([^]]+)\]', answer)
    clean_ids = set()
    for c in all_citations:
        match = re.search(r'(\d{4}\.\d{4,5})', c)
        if match:
            clean_ids.add(match.group(1))
    return sorted(clean_ids)


def run_config(config_name: str, questions: list[dict]) -> list[dict]:
    """Run a single config on all questions and save predictions."""
    config = CONFIGS[config_name]
    print(f"\n{'#'*60}")
    print(f"Running: {config.describe()}")
    print(f"{'#'*60}")

    agent = ResearchAgent(config)
    predictions = []

    for i, q in enumerate(questions):
        print(f"\n--- Question {i+1}/{len(questions)} (ID: {q['id']}) ---")
        q_type = q.get("type", "factoid")
        answer, trace = agent.run(q["question"], q_type=q_type)

        cited_papers = extract_cited_papers(answer)

        submission = Prediction(
            id=q["id"],
            answer=answer,
            cited_papers=cited_papers,
        )

        prediction = {
            "id": q["id"],
            "question_id": q["id"],
            "question": q["question"],
            "answer": answer,
            "cited_papers": cited_papers,
            "cited_arxiv_ids": cited_papers,
            "latency": trace.latency_seconds,
            "tool_calls": trace.total_tool_calls,
            "sub_questions": trace.sub_questions,
            "retrieval_rounds": trace.retrieval_rounds,
            "reflections": trace.reflections,
            "evidence_texts": [
                f"[{e['arxiv_id']}] {e['text'][:300]}"
                for e in trace.all_evidence
            ],
        }

        predictions.append(prediction)

    # Save predictions via Pydantic validation
    output_path = PREDICTIONS_DIR / f"{config_name}.jsonl"
    with open(output_path, "w") as f:
        for pred in predictions:
            submission = Prediction(
                id=pred["id"],
                answer=pred["answer"],
                cited_papers=pred["cited_papers"],
            )
            f.write(submission.to_jsonl() + "\n")
    print(f"\nPredictions saved to {output_path}")

    return predictions


def run_all_ablations():
    """Run all configurations and evaluate."""
    questions = load_questions()
    if not questions:
        return

    print(f"Loaded {len(questions)} evaluation questions")

    all_results = {}
    config_names = ["full_agent", "baseline", "no_planner", "no_reranker",
                    "no_reflector", "no_hybrid", "no_citation_verifier", "no_compressor"]

    for config_name in config_names:
        run_config(config_name, questions)
        results = evaluate_config(config_name)
        all_results[config_name] = results

    # Print ablation table
    print_ablation_table(all_results)

    # Save ablation table
    table_path = PREDICTIONS_DIR / "ablation_results.json"
    with open(table_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nFull ablation results saved to {table_path}")


def print_ablation_table(results: dict):
    """Print a formatted ablation table."""
    print(f"\n{'='*100}")
    print("ABLATION TABLE")
    print(f"{'='*100}")
    header = f"{'Config':<25} {'Accuracy':>8} {'Faithful':>9} {'Cit.P':>7} {'Cit.R':>7} {'Cit.F1':>7} {'Latency':>8} {'Calls':>6}"
    print(header)
    print("-" * 100)

    for config_name, res in results.items():
        if "error" in res:
            print(f"{config_name:<25} ERROR: {res['error']}")
            continue
        row = (
            f"{config_name:<25} "
            f"{res.get('avg_accuracy', 0):>8.2f} "
            f"{res.get('avg_faithfulness', 0):>9.2f} "
            f"{res.get('avg_citation_precision', 0):>7.4f} "
            f"{res.get('avg_citation_recall', 0):>7.4f} "
            f"{res.get('avg_citation_f1', 0):>7.4f} "
            f"{res.get('avg_latency', 0):>7.1f}s "
            f"{res.get('avg_tool_calls', 0):>6.1f}"
        )
        print(row)

    print(f"{'='*100}")


if __name__ == "__main__":
    run_all_ablations()
