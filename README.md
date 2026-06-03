<div align="center">

# Deep Research Agent

**A modular, ablation-ready framework for agentic deep research over academic papers**

[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Framework](https://img.shields.io/badge/built%20with-ReAct-red)](#)
[![RAG](https://img.shields.io/badge/RAG-hybrid%20%2B%20reranker-orange)](#)

</div>

---

## Overview

This system implements an **autonomous deep research agent** that answers complex questions over a corpus of academic papers (arXiv: cs.CL, cs.AI, cs.LG, 2024–2026). It follows a ReAct-style loop with **toggleable components** for systematic ablation studies.

The agent plans, retrieves, reflects, compresses, synthesizes, and verifies — all driven by local or cloud LLMs — and produces structured predictions with cited evidence.

---

## Architecture

```
                     ┌─────────────────────────┐
                     │      Question Input      │
                     └───────────┬─────────────┘
                                 ▼
                     ┌─────────────────────────┐
                     │   PLAN (Planner)         │
                     │   Decompose → sub-Q's    │
                     └───────────┬─────────────┘
                                 ▼
                     ┌─────────────────────────┐
          ┌─────────►│   RETRIEVE (Hybrid)      │
          │          │   Semantic + BM25        │
          │          │   + Reranker (CrossEnc)  │
          │          └───────────┬─────────────┘
          │                      ▼
          │          ┌─────────────────────────┐
          │          │   COMPRESS (LLM)         │
          │          │   Extract 1-2 sentences  │
          │          └───────────┬─────────────┘
          │                      ▼
          │          ┌─────────────────────────┐
          │          │   REFLECT (Reflector)    │
          │          │   Evidence sufficient?   │
          │          └───────────┬─────────────┘
          │                      │
          │        ┌─────────────┴─────────────┐
          │        ▼                           ▼
          │   [No — refine queries]      [Yes — continue]
          │        │                           │
          └────────┘                           ▼
                                     ┌─────────────────────────┐
                                     │   SYNTHESIZE             │
                                     │   Write cited answer     │
                                     └───────────┬─────────────┘
                                                 ▼
                                     ┌─────────────────────────┐
                                     │   VERIFY (Citation NLI)  │
                                     │   Strip invalid cit's    │
                                     └───────────┬─────────────┘
                                                 ▼
                                     ┌─────────────────────────┐
                                     │   Final Answer +         │
                                     │   Cited Papers           │
                                     └─────────────────────────┘
```

### Components

| Component | Toggle | Function |
|---|---|---|
| **Planner** | `use_planner` | Decomposes question into 2–5 focused sub-questions |
| **Hybrid Retriever** | `use_semantic` / `use_bm25` | Dense + sparse retrieval over chunked paper corpus |
| **Reranker** | `use_reranker` | Cross-encoder re-ranks top candidates |
| **Compressor** | `use_compressor` | LLM extracts 1–2 sentences relevant to query |
| **Reflector** | `use_reflector` | Evaluates evidence sufficiency, refines search queries |
| **Synthesizer** | always on | Writes answer with inline citations from evidence |
| **Citation Verifier** | `use_citation_verifier` | NLI-style removal of citations to unretrieved papers |

---

## Features

- **8 ablation configurations** — isolate each component's contribution
- **Hybrid retrieval** — BM25 (sparse) + semantic (dense) + cross-encoder reranker
- **LLM-based compression** — per-chunk extraction of query-relevant sentences
- **Dynamic retrieval** — reflector refines queries when evidence is insufficient
- **Smart citation verification** — NLI boundary checking removes hallucinated citations
- **Question-type-aware synthesis** — length constraints per type (factoid / comparative / survey)
- **Regex citation extraction** — robust arXiv ID parsing from answer text
- **Local + Cloud LLM support** — Ollama (local) or Nvidia NIM (cloud)

---

## Installation

```bash
# Clone
git clone https://github.com/Atharav001/RAG-Agentic-Deep-Research.git
cd RAG-Agentic-Deep-Research

# Environment
python3 -m venv venv
source venv/bin/activate

# Dependencies
pip install -r requirements.txt

# LLM provider
## Option A: Local (Ollama)
ollama pull gemma3:4b
ollama serve

## Option B: Cloud (Nvidia NIM)
cp .env.example .env
# Edit .env with your NVIDIA_NIM_API_KEY
```

---

## Usage

```bash
# Step 1: Scrape papers from arXiv
python run.py scrape

# Step 2: Parse PDFs and chunk text
python run.py parse

# Step 3: Build retrieval indices
python run.py index

# Step 4: Interactive single question
python run.py agent

# Step 5: Run ablation study (all 8 configs)
python run.py ablation

# Step 6: Evaluate predictions
python run.py evaluate

# Full pipeline (steps 1–6)
python run.py all
```

### Output

Predictions are saved to `predictions/<config>.jsonl` in the following format:

```json
{
  "id": "q01",
  "answer": "The ReAct framework combines reasoning with action... [2505.15182]",
  "cited_papers": ["2505.15182", "2604.24320"]
}
```

### Validation

```bash
python validate.py
```

---

## Ablation Configurations

| Config | Planner | Reflector | Citation Verifier | Compressor | Reranker | BM25 |
|---|---|---|---|---|---|---|
| `full_agent` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `baseline` | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ |
| `no_planner` | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `no_reranker` | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ |
| `no_reflector` | ✅ | ❌ | ✅ | ✅ | ✅ | ✅ |
| `no_hybrid` | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| `no_citation_verifier` | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ |
| `no_compressor` | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ |

---

## Project Structure

```
├── eval/
│   ├── questions.jsonl          # 30 evaluation questions
│   └── SUBMISSION_FORMAT.md     # Official submission spec
├── predictions/                 # Generated prediction files
├── src/
│   ├── agent/
│   │   ├── llm_client.py        # LLM provider abstraction (Ollama / Nvidia)
│   │   ├── components.py        # Planner, Reflector, Synthesizer, Verifier
│   │   └── research_agent.py    # Agent loop + configs
│   ├── evaluation/
│   │   ├── evaluator.py         # LLM-as-judge + citation metrics
│   │   └── run_ablation.py      # Ablation runner
│   ├── indexer/
│   │   ├── retriever.py         # Hybrid retriever (BM25 + dense)
│   │   ├── pdf_parser.py        # PDF chunking
│   │   └── build_index.py       # Index construction
│   ├── scraper/
│   │   └── arxiv_scraper.py     # arXiv API client
│   └── config.py                # Central configuration
├── data/                        # Papers, chunks, indices (gitignored)
├── run.py                       # CLI entry point
├── validate.py                  # Pre-submission validation
├── requirements.txt
└── README.md
```

---

## Evaluation Metrics

- **Answer Accuracy** (1–5) — LLM-as-judge
- **Faithfulness** (1–5) — LLM-as-judge
- **Citation Precision** — `|predicted ∩ gold| / |predicted|`
- **Citation Recall** — `|predicted ∩ gold| / |gold|`
- **Citation F1** — Harmonic mean of precision & recall
- **Latency** — Wall-clock seconds per question
- **Tool Calls** — Number of LLM invocations per question

---

## LLM Providers

| Provider | Type | Setup |
|---|---|---|
| **Ollama** (default) | Local | `ollama pull gemma3:4b` + `ollama serve` |
| **Nvidia NIM** | Cloud API | Set `NVIDIA_NIM_API_KEY` in `.env` |

---

## Citation

If you use this framework, please cite:

```bibtex
@misc{deep-research-agent-2026,
  author = {Atharav Narang},
  title = {Deep Research Agent: Modular Ablation Framework for Agentic Deep Research},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/Atharav001/RAG-Agentic-Deep-Research}
}
```

---

<div align="center">
Built with Python, sentence-transformers, BM25, and ReAct-style prompting.
</div>
