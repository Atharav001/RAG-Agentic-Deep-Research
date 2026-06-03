<div align="center">

# Deep Research Agent

</div>

An autonomous deep research agent that answers complex questions over a corpus of academic papers (arXiv: cs.CL, cs.AI, cs.LG, 2024–2026). Given a research question, the system decomposes it into sub-questions, retrieves relevant passages via hybrid search (semantic + BM25), reranks with a cross-encoder, compresses context using an LLM, reflects on evidence sufficiency, and synthesizes a cited answer. The reflector dynamically refines search queries when evidence is insufficient, looping until the agent has enough material. Each component (planner, reflector, compressor, citation verifier) can be toggled on or off for systematic ablation studies across 7 configurations. Predictions are output as strict JSONL with `{"id", "answer", "cited_papers"}` per the grading schema.

---

## Overview

| Field | Detail |
|---|---|
| Corpus | ~400 arXiv papers (cs.CL, cs.AI, cs.LG, 2024–2026) |
| Retrieval | Hybrid (BM25 + FAISS dense) + cross-encoder reranking |
| Agent Loop | Plan → Retrieve → Compress → Reflect → Synthesize → Verify |
| LLM Backend | Ollama (gemma3:4b, local) |
| Output Format | `{"id": "q01", "answer": "...", "cited_papers": ["..."]}` |
| Evaluation Questions | 30 (factoid, comparative, survey) |
| Ablation Configs | 7 (full_agent, baseline, no_planner, no_reranker, no_reflector, no_hybrid, no_citation_verifier) |

---

## Quick Start

```bash
pip install -r requirements.txt
ollama pull gemma3:4b
ollama serve
python run_parallel.py
```

---

## License

MIT
