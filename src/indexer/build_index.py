"""
Retrieval Index Builder — creates FAISS vector index + BM25 lexical index.

Components:
- Semantic: sentence-transformers embeddings → FAISS
- Lexical: BM25 (rank_bm25)
- Reranker: cross-encoder for reranking top results
"""

import json
import pickle
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import (
    CHUNKS_DIR,
    INDEX_DIR,
    EMBEDDING_MODEL,
    EMBEDDING_DIMENSION,
    RERANKER_MODEL,
)


def load_chunks() -> list[dict]:
    """Load processed chunks."""
    chunks_path = CHUNKS_DIR / "all_chunks.json"
    if not chunks_path.exists():
        print("Error: all_chunks.json not found. Run pdf_parser first.")
        return []
    with open(chunks_path) as f:
        return json.load(f)


def build_faiss_index(chunks: list[dict]):
    """Build FAISS index from chunk embeddings."""
    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    texts = [c["text"] for c in chunks]
    print(f"Encoding {len(texts)} chunks...")
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        batch_size=64,
        normalize_embeddings=True,
    )
    embeddings = np.array(embeddings, dtype="float32")

    # Build FAISS index (Inner Product since embeddings are normalized = cosine similarity)
    index = faiss.IndexFlatIP(EMBEDDING_DIMENSION)
    index.add(embeddings)

    # Save
    faiss.write_index(index, str(INDEX_DIR / "faiss_index.bin"))
    print(f"FAISS index saved ({index.ntotal} vectors)")
    return index


def build_bm25_index(chunks: list[dict]):
    """Build BM25 lexical index."""
    print("Building BM25 index...")
    tokenized_corpus = [c["text"].lower().split() for c in chunks]
    bm25 = BM25Okapi(tokenized_corpus)

    with open(INDEX_DIR / "bm25_index.pkl", "wb") as f:
        pickle.dump(bm25, f)
    print("BM25 index saved")
    return bm25


def save_chunk_metadata(chunks: list[dict]):
    """Save chunk metadata (everything except embeddings) for retrieval lookup."""
    metadata_path = INDEX_DIR / "chunk_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(chunks, f)
    print(f"Chunk metadata saved ({len(chunks)} chunks)")


def build_all():
    """Build all retrieval indices."""
    print("=" * 60)
    print("Building Retrieval Index")
    print("=" * 60)

    chunks = load_chunks()
    if not chunks:
        return

    print(f"\nLoaded {len(chunks)} chunks from {len(set(c['arxiv_id'] for c in chunks))} papers\n")

    # Build FAISS
    print("[1/3] Building FAISS semantic index...")
    build_faiss_index(chunks)

    # Build BM25
    print("\n[2/3] Building BM25 lexical index...")
    build_bm25_index(chunks)

    # Save metadata
    print("\n[3/3] Saving chunk metadata...")
    save_chunk_metadata(chunks)

    print("\nAll indices built successfully!")


if __name__ == "__main__":
    build_all()
