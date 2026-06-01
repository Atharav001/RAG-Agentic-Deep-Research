"""
Hybrid Retriever — combines FAISS semantic search + BM25 lexical search + cross-encoder reranking.

Each component can be toggled on/off for ablation studies.
"""

import json
import pickle
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import (
    INDEX_DIR,
    EMBEDDING_MODEL,
    RERANKER_MODEL,
    RETRIEVAL_TOP_K,
    FINAL_TOP_K,
    RERANKER_TOP_K,
    BM25_WEIGHT,
    SEMANTIC_WEIGHT,
)


class HybridRetriever:
    """Hybrid retriever with toggleable components for ablation."""

    def __init__(
        self,
        use_semantic: bool = True,
        use_bm25: bool = True,
        use_reranker: bool = True,
    ):
        self.use_semantic = use_semantic
        self.use_bm25 = use_bm25
        self.use_reranker = use_reranker

        # Load chunk metadata
        with open(INDEX_DIR / "chunk_metadata.json") as f:
            self.chunks = json.load(f)

        # Load FAISS index + embedding model
        if self.use_semantic:
            self.faiss_index = faiss.read_index(str(INDEX_DIR / "faiss_index.bin"))
            self.embed_model = SentenceTransformer(EMBEDDING_MODEL)

        # Load BM25 index
        if self.use_bm25:
            with open(INDEX_DIR / "bm25_index.pkl", "rb") as f:
                self.bm25 = pickle.load(f)

        # Load reranker
        if self.use_reranker:
            self.reranker = CrossEncoder(RERANKER_MODEL)

    def semantic_search(self, query: str, top_k: int = RETRIEVAL_TOP_K) -> list[tuple[int, float]]:
        """Return (chunk_index, score) pairs from FAISS."""
        query_embedding = self.embed_model.encode(
            [query], normalize_embeddings=True
        ).astype("float32")
        scores, indices = self.faiss_index.search(query_embedding, top_k)
        return [(int(idx), float(score)) for idx, score in zip(indices[0], scores[0]) if idx >= 0]

    def bm25_search(self, query: str, top_k: int = RETRIEVAL_TOP_K) -> list[tuple[int, float]]:
        """Return (chunk_index, score) pairs from BM25."""
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(int(idx), float(scores[idx])) for idx in top_indices if scores[idx] > 0]

    def hybrid_search(self, query: str, top_k: int = RETRIEVAL_TOP_K) -> list[tuple[int, float]]:
        """Combine semantic + BM25 with reciprocal rank fusion."""
        score_map: dict[int, float] = {}

        if self.use_semantic:
            semantic_results = self.semantic_search(query, top_k)
            for rank, (idx, _score) in enumerate(semantic_results):
                rrf_score = SEMANTIC_WEIGHT / (rank + 60)  # RRF with k=60
                score_map[idx] = score_map.get(idx, 0) + rrf_score

        if self.use_bm25:
            bm25_results = self.bm25_search(query, top_k)
            for rank, (idx, _score) in enumerate(bm25_results):
                rrf_score = BM25_WEIGHT / (rank + 60)
                score_map[idx] = score_map.get(idx, 0) + rrf_score

        # Sort by combined score
        ranked = sorted(score_map.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    def rerank(self, query: str, candidates: list[tuple[int, float]], top_k: int = FINAL_TOP_K) -> list[tuple[int, float]]:
        """Rerank candidates using cross-encoder."""
        if not candidates:
            return []

        pairs = [(query, self.chunks[idx]["text"]) for idx, _ in candidates[:RERANKER_TOP_K]]
        rerank_scores = self.reranker.predict(pairs)

        reranked = [
            (candidates[i][0], float(rerank_scores[i]))
            for i in range(len(pairs))
        ]
        reranked.sort(key=lambda x: x[1], reverse=True)
        return reranked[:top_k]

    def retrieve(self, query: str, top_k: int = FINAL_TOP_K) -> list[dict]:
        """Full retrieval pipeline: search → (optional rerank) → return chunks."""
        # Step 1: Search
        if self.use_semantic and self.use_bm25:
            candidates = self.hybrid_search(query)
        elif self.use_semantic:
            candidates = self.semantic_search(query)
        elif self.use_bm25:
            candidates = self.bm25_search(query)
        else:
            return []

        # Step 2: Rerank (optional)
        if self.use_reranker and candidates:
            candidates = self.rerank(query, candidates, top_k)
        else:
            candidates = candidates[:top_k]

        # Step 3: Return full chunk data
        results = []
        for idx, score in candidates:
            chunk = self.chunks[idx].copy()
            chunk["retrieval_score"] = score
            results.append(chunk)

        return results
