# Submission Format

Each configuration produces a `predictions/<config>.jsonl` file.

## Line format (one JSON object per line):

```json
{
  "question_id": "q01",
  "question": "The original question text",
  "answer": "The agent's answer with inline citations [XXXX.XXXXX]",
  "cited_arxiv_ids": ["2210.03629", "2310.11511"],
  "latency": 12.5,
  "tool_calls": 8
}
```

## Required prediction files:

- `predictions/full_agent.jsonl` — all components enabled
- `predictions/baseline.jsonl` — single-shot retrieval + one LLM call
- `predictions/no_planner.jsonl` — planner disabled
- `predictions/no_reranker.jsonl` — reranker disabled
- `predictions/no_reflector.jsonl` — reflector disabled
- `predictions/no_hybrid.jsonl` — BM25 disabled (semantic only)
- `predictions/no_citation_verifier.jsonl` — citation verifier disabled

## Evaluation metrics:

- **Answer accuracy** (1-5, LLM-as-judge)
- **Faithfulness** (1-5, LLM-as-judge)
- **Citation precision** (|predicted ∩ gold| / |predicted|)
- **Citation recall** (|predicted ∩ gold| / |gold|)
- **Latency** (seconds)
- **Tool-call count**
