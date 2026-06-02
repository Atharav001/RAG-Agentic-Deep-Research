"""Central configuration for the Agentic Deep Research system."""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PAPERS_DIR = DATA_DIR / "papers"
METADATA_DIR = DATA_DIR / "metadata"
CHUNKS_DIR = DATA_DIR / "chunks"
INDEX_DIR = DATA_DIR / "index"
EVAL_DIR = PROJECT_ROOT / "eval"
PREDICTIONS_DIR = PROJECT_ROOT / "predictions"

# Ensure directories exist
for d in [PAPERS_DIR, METADATA_DIR, CHUNKS_DIR, INDEX_DIR, EVAL_DIR, PREDICTIONS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# arXiv scraper settings
ARXIV_API_URL = "http://export.arxiv.org/api/query"
ARXIV_CATEGORIES = ["cs.CL", "cs.AI", "cs.LG"]
ARXIV_DATE_START = "2024-01-01"
ARXIV_DATE_END = "2026-04-30"
ARXIV_KEYWORDS = [
    "LLM agent",
    "language model agent",
    "autonomous agent",
    "agentic RAG",
    "retrieval augmented generation agent",
    "tool use",
    "tool-use",
    "agent memory",
    "agent benchmark",
    "computer-use agent",
    "web agent",
    "code agent",
    "multi-agent",
    "agent planning",
    "agent reasoning",
    "ReAct",
    "function calling",
    "agentic",
    "agent framework",
]
ARXIV_MAX_RESULTS = 800  # fetch extra, filter down to 400-700
ARXIV_BATCH_SIZE = 100  # results per API call
ARXIV_RATE_LIMIT_SECONDS = 3  # be polite to arXiv

# PDF parsing settings
CHUNK_SIZE = 512  # tokens per chunk
CHUNK_OVERLAP = 64  # token overlap between chunks

# Embedding settings
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIMENSION = 384
CHUNK_PREFIX_ENABLED = True

# Reranker settings
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANKER_TOP_K = 10  # rerank top-k results

# Retrieval settings
RETRIEVAL_TOP_K = 20  # initial retrieval count
FINAL_TOP_K = 5  # passages to pass to LLM after reranking
BM25_WEIGHT = 0.3  # weight for BM25 in hybrid retrieval
SEMANTIC_WEIGHT = 0.7  # weight for semantic in hybrid retrieval

# Agent settings
MAX_REFLECTION_ROUNDS = 3  # max retrieval-reflection loops
MAX_SUB_QUESTIONS = 5  # max planner decomposition

# LLM settings
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
LLM_TEMPERATURE = 0.1
LLM_MAX_TOKENS = 4096

# Evaluation settings
JUDGE_MODEL = LLM_MODEL
EVAL_QUESTIONS_PATH = EVAL_DIR / "questions.jsonl"
