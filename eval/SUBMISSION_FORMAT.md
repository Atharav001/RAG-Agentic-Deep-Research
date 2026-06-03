<div align="center">

# Submission Format

</div>

Your system must produce a single predictions.jsonl file in this exact format. One line per question, matching the IDs in questions.jsonl.

Example line (pretty-printed here for readability; in the actual file each prediction must be a single line of JSON):

```json
{
  "id": "q01",
  "answer": "<your system's answer>",
  "cited_papers": ["2504.19413", "2502.12110"]
}
```

## Fields

- **id**: must match an id in questions.jsonl. All 30 IDs must appear, no duplicates.
- **answer**: your system's natural-language answer. Plain text. Inline citations of the form [arxiv_id] are encouraged but optional; the cited_papers field is the authoritative citation list for grading.
- **cited_papers**: flat list of arXiv IDs (no version suffix, no URL, e.g. "2504.19413") that your system used as evidence for this answer. Order is not significant. Duplicates will be deduplicated by the grader.

## Length guidance

- **factoid** answers: 1 to 3 sentences.
- **comparative** answers: 100 to 300 words.
- **survey** answers: 250 to 600 words.

## Hard rules

- Do NOT see or modify ground-truth files. The grader holds the hidden groundtruth_private.jsonl and will run scoring after submission.
- Cite only papers that are actually present in your indexed corpus. Citing a paper your retriever never returned will hurt your citation precision.
- The grader will use an LLM-as-judge for answer accuracy and faithfulness, and exact-set overlap of cited_papers against the hidden must-cite set for citation precision and recall. Your reported numbers in the report must match what the grader computes within a small tolerance.

## Where it lives

Place predictions.jsonl at the top of your repo. Provide one such file for each configuration you ran (full agent, baseline, and each ablation), under `predictions/<config_name>.jsonl`.
