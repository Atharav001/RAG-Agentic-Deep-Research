"""
Evaluation Pipeline — LLM-as-judge + citation metrics.

Metrics:
- Answer accuracy (LLM-as-judge, 1-5 scale)
- Faithfulness (LLM-as-judge, 1-5 scale)
- Citation precision (|predicted ∩ gold| / |predicted|)
- Citation recall (|predicted ∩ gold| / |gold|)
- Latency (seconds)
- Tool-call count
"""

import json
import re
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.agent.llm_client import call_llm
from src.config import EVAL_DIR, PREDICTIONS_DIR


def judge_accuracy(question: str, answer: str, reference_answer: str = "") -> dict:
    """Use LLM-as-judge to score answer accuracy (1-5)."""
    ref_section = ""
    if reference_answer:
        ref_section = f"\nReference answer (for comparison):\n{reference_answer}\n"

    prompt = f"""You are an expert judge evaluating the quality of a research answer.

Question: {question}
{ref_section}
Answer to evaluate:
{answer}

Score the answer on a scale of 1-5:
1 = Completely wrong or irrelevant
2 = Partially addresses the question with significant errors
3 = Addresses the question but missing key details or has minor errors
4 = Good answer with minor omissions
5 = Excellent, comprehensive, and accurate

Return ONLY a JSON object: {{"score": <int>, "reasoning": "<brief explanation>"}}"""

    response = call_llm(prompt, provider="ollama", temperature=0.1)
    try:
        match = re.search(r"\{.*\}", response, re.DOTALL)
        if match:
            return json.loads(match.group())
    except (json.JSONDecodeError, AttributeError):
        pass
    return {"score": 3, "reasoning": "Could not parse judge response"}


def judge_faithfulness(question: str, answer: str, evidence_texts: list[str]) -> dict:
    """Use LLM-as-judge to score faithfulness to retrieved evidence (1-5)."""
    evidence_str = "\n\n".join(f"[Evidence {i+1}]: {t[:400]}" for i, t in enumerate(evidence_texts[:10]))

    prompt = f"""You are an expert judge evaluating whether a research answer is faithful to its source evidence.

Question: {question}

Source Evidence:
{evidence_str}

Answer:
{answer}

Score faithfulness on a scale of 1-5:
1 = Answer contains mostly fabricated claims not in evidence
2 = Answer mixes supported and unsupported claims
3 = Most claims are supported but some lack evidence
4 = Nearly all claims are well-supported by evidence
5 = Every claim is directly supported by the provided evidence

Return ONLY a JSON object: {{"score": <int>, "reasoning": "<brief explanation>"}}"""

    response = call_llm(prompt, provider="ollama", temperature=0.1)
    try:
        match = re.search(r"\{.*\}", response, re.DOTALL)
        if match:
            return json.loads(match.group())
    except (json.JSONDecodeError, AttributeError):
        pass
    return {"score": 3, "reasoning": "Could not parse judge response"}


def citation_precision_recall(predicted_ids: list[str], gold_ids: list[str]) -> dict:
    """Compute citation precision and recall via exact set overlap."""
    predicted_set = set(predicted_ids)
    gold_set = set(gold_ids)

    if not predicted_set:
        precision = 0.0
    else:
        precision = len(predicted_set & gold_set) / len(predicted_set)

    if not gold_set:
        recall = 1.0
    else:
        recall = len(predicted_set & gold_set) / len(gold_set)

    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "predicted_ids": sorted(predicted_ids),
        "gold_ids": sorted(gold_ids),
    }


def evaluate_single(prediction: dict, gold: dict) -> dict:
    """Evaluate a single prediction against gold standard."""
    question = gold["question"]
    answer = prediction["answer"]
    predicted_ids = prediction.get("cited_arxiv_ids", [])
    gold_ids = gold.get("must_cite_arxiv_ids", [])
    reference_answer = gold.get("reference_answer", "")

    # LLM-as-judge: accuracy
    accuracy = judge_accuracy(question, answer, reference_answer)

    # LLM-as-judge: faithfulness
    evidence_texts = prediction.get("evidence_texts", [])
    faithfulness = judge_faithfulness(question, answer, evidence_texts)

    # Citation metrics
    citation_metrics = citation_precision_recall(predicted_ids, gold_ids)

    return {
        "question_id": gold.get("id", ""),
        "question": question,
        "accuracy_score": accuracy["score"],
        "accuracy_reasoning": accuracy["reasoning"],
        "faithfulness_score": faithfulness["score"],
        "faithfulness_reasoning": faithfulness["reasoning"],
        "citation_precision": citation_metrics["precision"],
        "citation_recall": citation_metrics["recall"],
        "citation_f1": citation_metrics["f1"],
        "latency": prediction.get("latency", 0),
        "tool_calls": prediction.get("tool_calls", 0),
    }


def evaluate_config(config_name: str, gold_path: Path | None = None) -> dict:
    """Evaluate all predictions for a given config against gold standard."""
    pred_path = PREDICTIONS_DIR / f"{config_name}.jsonl"
    if not pred_path.exists():
        print(f"Predictions not found: {pred_path}")
        return {}

    if gold_path is None:
        gold_path = EVAL_DIR / "questions.jsonl"

    # Load predictions and gold
    predictions = []
    with open(pred_path) as f:
        for line in f:
            if line.strip():
                predictions.append(json.loads(line))

    gold_questions = []
    with open(gold_path) as f:
        for line in f:
            if line.strip():
                gold_questions.append(json.loads(line))

    # Match by question_id
    gold_by_id = {g["id"]: g for g in gold_questions}

    results = []
    for pred in predictions:
        qid = pred.get("question_id", "")
        if qid in gold_by_id:
            result = evaluate_single(pred, gold_by_id[qid])
            results.append(result)
            print(f"  Q{qid}: accuracy={result['accuracy_score']}, "
                  f"faith={result['faithfulness_score']}, "
                  f"cit_p={result['citation_precision']:.2f}, "
                  f"cit_r={result['citation_recall']:.2f}")

    # Aggregate
    if not results:
        return {"config": config_name, "error": "No matched predictions"}

    avg = lambda key: round(sum(r[key] for r in results) / len(results), 4)

    summary = {
        "config": config_name,
        "num_questions": len(results),
        "avg_accuracy": avg("accuracy_score"),
        "avg_faithfulness": avg("faithfulness_score"),
        "avg_citation_precision": avg("citation_precision"),
        "avg_citation_recall": avg("citation_recall"),
        "avg_citation_f1": avg("citation_f1"),
        "avg_latency": avg("latency"),
        "avg_tool_calls": avg("tool_calls"),
        "per_question": results,
    }

    # Save results
    results_path = PREDICTIONS_DIR / f"{config_name}_eval.json"
    with open(results_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {results_path}")

    return summary
