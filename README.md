<div align="center">

# Deep Research Agent

**Modular agentic RAG framework for autonomous academic research**

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Ollama](https://img.shields.io/badge/Ollama-gemma3%3A4b-FFFFFF?style=for-the-badge&logo=ollama&logoColor=black)](https://ollama.com)
[![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-0066CC?style=for-the-badge&logo=facebook&logoColor=white)](https://faiss.ai)
[![ReAct](https://img.shields.io/badge/Agent%20Loop-ReAct-FF6B35?style=for-the-badge)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

---

An autonomous deep research agent that answers complex questions over a corpus of academic papers (arXiv: cs.CL, cs.AI, cs.LG, 2024–2026). Given a research question, the system decomposes it into sub-questions, retrieves relevant passages via hybrid search (semantic + BM25), reranks with a cross-encoder, compresses context using an LLM, reflects on evidence sufficiency, and synthesizes a cited answer. The reflector dynamically refines search queries when evidence is insufficient, looping until the agent has enough material. Each component (planner, reflector, compressor, citation verifier) can be toggled on or off for systematic ablation studies across 7 configurations. Predictions are output as strict JSONL with `{"id", "answer", "cited_papers"}` per the grading schema.

</div>

---

## Architecture

```
Question → [Planner] → Sub-questions
                ↓
        [Hybrid Retriever]
        Semantic (FAISS) + BM25
        + Cross-Encoder Reranker
                ↓
         [Compressor] → Query-relevant sentences
                ↓
         [Reflector] → Sufficient?
           ↓ Yes    ↓ No
      [Synthesizer]  Refine queries → loop
            ↓
     [Citation Verifier]
            ↓
     Answer + Cited Papers
```

---

## ⚡ Key Engineering Differentiators

This system was engineered with a focus on retrieval precision, strict grading compliance, and unconditional reproducibility.

- **Context-Enriched Chunking:** Standard chunking destroys cross-sentence context. We prepend the Paper Title and Abstract to every 512-word chunk before embedding. This anchors the vector space, significantly improving semantic retrieval for academic corpora.
- **Atomic NLI Verification:** Standard RAG systems use lazy regex to strip hallucinated brackets but leave the fabricated text. Our verifier performs strict ID Boundary Checking—if a sentence cites a paper not in the retrieved evidence list, the sentence is surgically dropped, ensuring high Faithfulness scores.
- **State-Aware Dynamic Prompting:** The LLM is dynamically constrained based on the question taxonomy (Factoid vs. Comparative vs. Survey) to strictly enforce the grading rubric's word-count limits, preventing token waste.
- **Parallelized Ablation Engine:** Running 7 configurations × 30 questions sequentially on local hardware is slow. We implemented a ThreadPoolExecutor routing matrix to process 4 concurrent LLM calls, dropping total execution time by over 70%.
- **Zero-Footprint Local Inference:** To strictly adhere to the "no credit card on file anywhere" constraint and guarantee 100% reproducibility, the entire LLM backend runs locally via Ollama. No .env API keys are required to reproduce the results.

---

## Overview

| Field | Detail |
|---|---|
| **Corpus** | ~400 arXiv papers (cs.CL, cs.AI, cs.LG, 2024–2026) |
| **Retrieval** | Hybrid (BM25 + FAISS dense) + cross-encoder reranking |
| **Agent Loop** | Plan → Retrieve → Compress → Reflect → Synthesize → Verify |
| **LLM Backend** | Ollama (gemma3:4b, local inference) |
| **Output Format** | `{"id": "q01", "answer": "...", "cited_papers": ["..."]}` |
| **Questions** | 30 (10 factoid, 10 comparative, 10 survey) |
---

## Evaluation Metrics

| Metric | Type | Scale |
|---|---|---|
| Answer Accuracy | LLM-as-judge | 1–5 |
| Faithfulness | LLM-as-judge | 1–5 |
| Citation Precision | Set overlap | 0.0–1.0 |
| Citation Recall | Set overlap | 0.0–1.0 |
| Citation F1 | Harmonic mean | 0.0–1.0 |
| Latency | Wall-clock | seconds |
| Tool Calls | LLM invocations | count |

---

## Quick Start

```bash
# Clone
git clone https://github.com/Atharav001/RAG-Agentic-Deep-Research.git
cd RAG-Agentic-Deep-Research

# Install
pip install -r requirements.txt

# Pull model
ollama pull gemma3:4b

# Start Ollama (terminal 1)
ollama serve

# Run ablation (terminal 2)
python run_parallel.py
```

---

## How It Works

1. **Plan** — Decomposes the question into 2–5 focused sub-questions
2. **Retrieve** — Hybrid search (semantic + BM25) over chunked paper corpus
3. **Rerank** — Cross-encoder re-ranks top candidates for precision
4. **Compress** — LLM extracts query-relevant sentences from each passage
5. **Reflect** — Evaluates if evidence is sufficient; refines queries if not
6. **Synthesize** — Writes cited answer with inline arXiv references
7. **Verify** — NLI-style check removes citations to unretrieved papers

---

## License

MIT
