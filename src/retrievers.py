"""Retrieval methods over a list of Chunks. All expose `search(query, k) -> list[(chunk_index, score)]`.

  bm25          lexical, strong on exact terms (error codes, CLI flags, limits)
  dense         bi-encoder embeddings + cosine similarity, strong on paraphrase
  hybrid        reciprocal rank fusion (RRF) of bm25 + dense: robust without score calibration
  hybrid+rerank hybrid candidates re-scored by a cross-encoder that reads query and passage together

Vectors live in a plain NumPy matrix: with a few thousand chunks an exact dot product takes milliseconds,
so an ANN index (FAISS, a vector DB) would add operational weight and no accuracy.
"""
import hashlib
import os
import re

import numpy as np

from src.chunking import Chunk

DENSE_MODEL = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "  # bge convention, queries only
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
STOP = set("a an the of to in on for is are was were be been can could does do how what which when where why "
           "with and or by at as it its i my this that from if there so than then".split())

METHODS = ["bm25", "dense", "hybrid", "hybrid+rerank"]


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9][a-z0-9_.\-]*", text.lower()) if t not in STOP]


class BM25Retriever:
    def __init__(self, chunks: list[Chunk]):
        from rank_bm25 import BM25Okapi
        self.bm25 = BM25Okapi([tokenize(c.retrieval_text) for c in chunks])

    def search(self, query: str, k: int = 10):
        scores = self.bm25.get_scores(tokenize(query))
        top = np.argsort(-scores)[:k]
        return [(int(i), float(scores[i])) for i in top]


class DenseRetriever:
    def __init__(self, chunks: list[Chunk], model_name: str = DENSE_MODEL, cache_dir: str = ".cache"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self.prefix = QUERY_PREFIX if "bge" in model_name else ""
        texts = [c.retrieval_text for c in chunks]
        key = hashlib.sha1((model_name + "\n".join(texts)).encode("utf-8")).hexdigest()[:16]
        path = os.path.join(cache_dir, f"emb_{key}.npy")
        if os.path.exists(path):
            self.matrix = np.load(path)
        else:
            self.matrix = self.model.encode(texts, batch_size=32, normalize_embeddings=True,
                                            show_progress_bar=False).astype("float32")
            os.makedirs(cache_dir, exist_ok=True)
            np.save(path, self.matrix)

    def search(self, query: str, k: int = 10):
        q = self.model.encode([self.prefix + query], normalize_embeddings=True).astype("float32")[0]
        scores = self.matrix @ q
        top = np.argsort(-scores)[:k]
        return [(int(i), float(scores[i])) for i in top]


def rrf(rankings: list[list[int]], k: int = 60) -> list[tuple[int, float]]:
    """Reciprocal rank fusion: score(d) = sum over rankers of 1 / (k + rank). Rank-based, so no score scaling."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking, start=1):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: -kv[1])


class HybridRetriever:
    def __init__(self, bm25, dense, candidates: int = 50):
        self.bm25, self.dense, self.candidates = bm25, dense, candidates

    def search(self, query: str, k: int = 10):
        a = [i for i, _ in self.bm25.search(query, self.candidates)]
        b = [i for i, _ in self.dense.search(query, self.candidates)]
        return rrf([a, b])[:k]


class Reranked:
    def __init__(self, base, chunks: list[Chunk], model_name: str = RERANK_MODEL, pool: int = 20):
        from sentence_transformers import CrossEncoder
        self.base, self.chunks, self.pool = base, chunks, pool
        self.model = CrossEncoder(model_name)

    def search(self, query: str, k: int = 10):
        cand = [i for i, _ in self.base.search(query, self.pool)]
        scores = self.model.predict([(query, self.chunks[i].retrieval_text) for i in cand])
        order = np.argsort(-scores)[:k]
        return [(cand[j], float(scores[j])) for j in order]


def build_retrievers(chunks: list[Chunk], methods: list[str] | None = None, cache_dir: str = ".cache") -> dict:
    """Build only what the requested methods need (the dense model and reranker are the slow parts)."""
    methods = methods or METHODS
    out = {}
    need_dense = any(m in methods for m in ("dense", "hybrid", "hybrid+rerank"))
    bm25 = BM25Retriever(chunks)
    dense = DenseRetriever(chunks, cache_dir=cache_dir) if need_dense else None
    hybrid = HybridRetriever(bm25, dense) if dense else None
    if "bm25" in methods:
        out["bm25"] = bm25
    if "dense" in methods:
        out["dense"] = dense
    if "hybrid" in methods:
        out["hybrid"] = hybrid
    if "hybrid+rerank" in methods:
        out["hybrid+rerank"] = Reranked(hybrid, chunks)
    return out
