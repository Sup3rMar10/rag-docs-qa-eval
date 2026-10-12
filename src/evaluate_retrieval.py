"""Compare retrieval methods x chunk sizes on the answerable questions.

A retrieved chunk is a HIT if it contains any of the question's verbatim evidence strings.
Metrics: Hit@k (is a gold chunk in the top k?) and MRR@10, plus the average context size retrieved at k=5
so chunk sizes are compared fairly (bigger chunks = more text per hit).
"""
import argparse
import json
import os
import time

from src.chunking import load_corpus, normalize
from src.evalset import load_questions
from src.retrievers import METHODS, build_retrievers

KS = [1, 3, 5, 10]


def hit_rank(retrieved: list[int], chunk_norm: list[str], evidence: list[str]):
    """1-based rank of the first chunk containing evidence, or None."""
    needles = [normalize(e) for e in evidence]
    for rank, idx in enumerate(retrieved, start=1):
        if any(n in chunk_norm[idx] for n in needles):
            return rank
    return None


def evaluate(chunks, retriever, questions, max_k=10):
    chunk_norm = [normalize(c.text) for c in chunks]
    rows = []
    for q in questions:
        t0 = time.time()
        retrieved = [i for i, _ in retriever.search(q["question"], max_k)]
        rows.append({"id": q["id"], "difficulty": q["difficulty"], "category": q["category"],
                     "rank": hit_rank(retrieved, chunk_norm, q["evidence"]),
                     "ctx_words_at5": sum(chunks[i].words for i in retrieved[:5]),
                     "latency_ms": (time.time() - t0) * 1000})
    return rows


def summarize(rows):
    n = len(rows)
    out = {"n": n, "mrr@10": sum(1 / r["rank"] for r in rows if r["rank"]) / n,
           "avg_ctx_words@5": sum(r["ctx_words_at5"] for r in rows) / n,
           "avg_latency_ms": sum(r["latency_ms"] for r in rows) / n}
    for k in KS:
        out[f"hit@{k}"] = sum(1 for r in rows if r["rank"] and r["rank"] <= k) / n
    out["by_difficulty"] = {}
    for d in ("easy", "medium", "hard"):
        sub = [r for r in rows if r["difficulty"] == d]
        if sub:
            hits5 = sum(1 for r in sub if r["rank"] and r["rank"] <= 5)
            out["by_difficulty"][d] = {"n": len(sub), "hit@5": hits5 / len(sub)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/corpus")
    ap.add_argument("--questions", default="data/eval_questions.jsonl")
    ap.add_argument("--chunk-sizes", type=int, nargs="+", default=[128, 256])
    ap.add_argument("--methods", nargs="+", default=METHODS)
    ap.add_argument("--out", default="results/retrieval.json")
    a = ap.parse_args()

    questions = [q for q in load_questions(a.questions) if q["type"] == "answerable"]
    results = {}
    for size in a.chunk_sizes:
        chunks = load_corpus(a.corpus, size)
        print(f"chunk size {size}: {len(chunks)} chunks", flush=True)
        retrievers = build_retrievers(chunks, a.methods)
        for method, retriever in retrievers.items():
            rows = evaluate(chunks, retriever, questions)
            s = summarize(rows)
            results[f"{method}@{size}"] = {"method": method, "chunk_size": size, "n_chunks": len(chunks),
                                          **s, "per_question": rows}
            print(f"  {method:14s} hit@1={s['hit@1']:.2f} hit@3={s['hit@3']:.2f} hit@5={s['hit@5']:.2f} "
                  f"hit@10={s['hit@10']:.2f} mrr={s['mrr@10']:.3f} ({s['avg_latency_ms']:.0f} ms/query)", flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(results, f, indent=1)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
