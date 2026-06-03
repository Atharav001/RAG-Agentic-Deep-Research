# Agentic Deep Research — Setup & Next Steps

## What's Already Done

| Step | Status | Details |
|------|--------|---------|
| Scrape arXiv papers | DONE | 374 papers filtered, 439 PDFs downloaded |
| Parse PDFs & chunk text | DONE | 13,656 chunks extracted |
| Build retrieval index | DONE | FAISS (20MB) + BM25 (44MB) + metadata (41MB) |
| Agent code | DONE | ReAct-style loop with 5 toggleable components |
| Evaluation framework | DONE | LLM-as-judge + citation metrics |
| Eval questions | DONE | 30 questions with gold standard citations |
| Bug fixes | DONE | Evidence tracking, evidence_texts, gold standard |

## What's Left To Do

### Step 1: Set Up Environment (5 min)

```bash
# Navigate to project
cd atharav_project

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Get a Free Gemini API Key (2 min)

1. Go to https://aistudio.google.com/apikey
2. Sign in with Google account (no credit card needed)
3. Click "Create API Key"
4. Copy the key

```bash
# Create .env file with your key
echo "GEMINI_API_KEY=your_key_here" > .env
```

### Step 3: Test the LLM Connection (1 min)

```bash
source venv/bin/activate
python -c "
from src.agent.llm_client import call_llm
print(call_llm('Say hello in one word.'))
"
```

If this prints a response, the API key works.

### Step 4: Test the Retriever (2 min)

```bash
python -c "
from src.indexer.retriever import HybridRetriever
r = HybridRetriever(use_semantic=True, use_bm25=True, use_reranker=True)
results = r.retrieve('What is ReAct?', top_k=3)
for x in results:
    print(f'[{x[\"arxiv_id\"]}] {x[\"text\"][:100]}...')
print('Retriever OK!')
"
```

### Step 5: Test the Agent on One Question (5 min)

```bash
python -c "
from src.agent.research_agent import ResearchAgent, CONFIGS
agent = ResearchAgent(CONFIGS['full_agent'])
answer, trace = agent.run('What is the ReAct framework?')
print('\n=== ANSWER ===')
print(answer)
print(f'\nCited: {trace.cited_arxiv_ids}')
print(f'Tool calls: {trace.total_tool_calls}, Latency: {trace.latency_seconds:.1f}s')
"
```

### Step 6: Run Full Ablation Study (~30-60 min)

This runs ALL 7 configurations on ALL 30 evaluation questions:

```bash
python run.py ablation
```

**Configurations run:**
1. `full_agent` — all components enabled
2. `baseline` — single-shot retrieval + one LLM call (no loop)
3. `no_planner` — planner disabled
4. `no_reranker` — reranker disabled
5. `no_reflector` — no reflection loop
6. `no_hybrid` — semantic only, no BM25
7. `no_citation_verifier` — citation verifier disabled

**Output:** 7 files in `predictions/` folder (one `.jsonl` per config)

### Step 7: Run Evaluation (~15 min)

```bash
python run.py evaluate
```

This scores each prediction using LLM-as-judge and citation metrics, then prints the ablation table.

**Metrics reported:**
- Answer accuracy (1-5, LLM-as-judge)
- Faithfulness (1-5, LLM-as-judge)
- Citation precision and recall (exact set overlap)
- Latency (seconds)
- Tool-call count

### Step 8: Interactive Mode (Optional)

Ask any question to the full agent:

```bash
python run.py agent
```

Type a question and get a cited research answer with trace.

---

## Project Structure

```
atharav_project/
├── run.py                    # Main CLI: python run.py [scrape|parse|index|agent|ablation|evaluate|all]
├── requirements.txt          # Python dependencies
├── .env                      # YOUR API KEY (create this!)
├── .env.example              # Template
├── eval/
│   ├── questions.jsonl       # 30 eval questions with gold standard
│   └── SUBMISSION_FORMAT.md  # Output format spec
├── predictions/              # Output: one .jsonl per config (created by ablation)
├── data/
│   ├── papers/               # Downloaded PDFs (~440 files)
│   ├── metadata/             # Paper metadata JSON
│   ├── chunks/               # Chunked text (13,656 chunks)
│   └── index/                # FAISS + BM25 indices
├── src/
│   ├── config.py             # Central configuration
│   ├── scraper/
│   │   └── arxiv_scraper.py  # arXiv API scraper
│   ├── indexer/
│   │   ├── pdf_parser.py     # PDF text extraction + chunking
│   │   ├── build_index.py    # FAISS + BM25 index builder
│   │   └── retriever.py      # Hybrid retriever (toggleable components)
│   ├── agent/
│   │   ├── llm_client.py     # Gemini API wrapper
│   │   ├── components.py     # Planner, Reflector, Synthesizer, Citation Verifier
│   │   └── research_agent.py # ReAct-style agent loop
│   └── evaluation/
│       ├── evaluator.py      # LLM-as-judge + citation metrics
│       └── run_ablation.py   # Runs all configs, produces ablation table
└── INSTRUCTIONS.md           # This file
```

## Quick Reference Commands

| Command | What it does |
|---------|-------------|
| `python run.py scrape` | Scrape papers from arXiv (already done) |
| `python run.py parse` | Parse PDFs into chunks (already done) |
| `python run.py index` | Build retrieval indices (already done) |
| `python run.py agent` | Interactive agent mode |
| `python run.py ablation` | Run all 7 ablation configs on 30 questions |
| `python run.py evaluate` | Score predictions and print ablation table |
| `python run.py all` | Run entire pipeline end-to-end |

## Troubleshooting

- **"GEMINI_API_KEY not set"** → Create `.env` file with your key (Step 2)
- **"papers_metadata.json not found"** → Run `python run.py scrape` first
- **"faiss_index.bin not found"** → Run `python run.py index` first
- **Rate limit errors from Gemini** → Wait a minute and retry; free tier has limits
- **Import errors** → Make sure you activated the venv: `source venv/bin/activate`

## Reference Papers (from the assignment)

| Paper | arXiv ID | Role in this project |
|-------|----------|---------------------|
| Deep Research Agents Survey | 2506.18096 | Architecture design reference |
| ReAct | 2210.03629 | Agent loop pattern |
| Self-RAG | 2310.11511 | Reflector component design |
| Reflexion | 2303.11366 | Reflection loop design |
| LLM-as-Judge | 2306.05685 | Evaluation methodology |
| RAGAS | 2309.15217 | RAG evaluation metrics |

## Deliverables Checklist

- [ ] Steps 1-7 completed
- [ ] `predictions/` folder has 7 `.jsonl` files
- [ ] Ablation table shows results for all configs
- [ ] Write 4-6 page technical report (see assignment PDF)
- [ ] Push to GitHub repository
- [ ] (Optional) Build browser demo with trace view
