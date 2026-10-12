"""End-to-end answer quality: closed-book vs RAG, on answerable AND unanswerable questions.

Reported per run
  answerable:    answer_accuracy, citation_grounded_rate, false_abstention_rate
  unanswerable:  abstention_rate (higher is better), hallucination_rate = 1 - abstention
  decomposition: answer_accuracy when the evidence WAS in the prompt vs when retrieval missed it,
                 which separates retrieval failures from generation failures.
"""
import argparse
import json
import os
import time

from src.chunking import load_corpus
from src.evalset import load_questions
from src.rag import DEFAULT_PROMPT, RAG_SYSTEMS, LocalGenerator, closed_book_messages, rag_messages
from src.retrievers import build_retrievers
from src.scoring import answer_correct, citation_grounded, evidence_in_context, is_abstention


def rate(flags):
    flags = list(flags)
    return sum(flags) / len(flags) if flags else None


def summarize(rows):
    ans = [r for r in rows if r["type"] == "answerable"]
    una = [r for r in rows if r["type"] == "unanswerable"]
    with_ev = [r for r in ans if r["evidence_in_context"]]
    without = [r for r in ans if not r["evidence_in_context"]]
    return {
        "n_answerable": len(ans), "n_unanswerable": len(una),
        "answer_accuracy": rate(r["correct"] for r in ans),
        "citation_grounded_rate": rate(r["citation_grounded"] for r in ans),
        "false_abstention_rate": rate(r["abstained"] for r in ans),
        "abstention_rate_on_unanswerable": rate(r["abstained"] for r in una),
        "hallucination_rate_on_unanswerable": rate(not r["abstained"] for r in una),
        "n_evidence_in_context": len(with_ev),
        "accuracy_when_evidence_in_context": rate(r["correct"] for r in with_ev),
        "accuracy_when_retrieval_missed": rate(r["correct"] for r in without),
        "avg_seconds_per_question": sum(r["seconds"] for r in rows) / len(rows),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--mode", choices=["rag", "closed-book"], default="rag")
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--method", default="hybrid+rerank")
    ap.add_argument("--chunk-size", type=int, default=128)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--prompt", choices=sorted(RAG_SYSTEMS), default=DEFAULT_PROMPT)
    ap.add_argument("--n", type=int, default=0, help="only the first N questions (0 = all)")
    ap.add_argument("--corpus", default="data/corpus")
    ap.add_argument("--questions", default="data/eval_questions.jsonl")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    questions = load_questions(a.questions)
    questions = questions[:a.n] if a.n else questions
    gen = LocalGenerator(a.model)
    chunks, retriever = None, None
    if a.mode == "rag":
        chunks = load_corpus(a.corpus, a.chunk_size)
        retriever = build_retrievers(chunks, [a.method])[a.method]

    rows = []
    for i, q in enumerate(questions, start=1):
        t0 = time.time()
        if a.mode == "rag":
            hits = retriever.search(q["question"], a.k)
            passages = [chunks[j] for j, _ in hits]
            answer = gen.generate(rag_messages(q["question"], passages, a.prompt))
            ctx = [p.text for p in passages]
            ctx_docs = [p.doc for p in passages]
        else:
            answer, ctx, ctx_docs = gen.generate(closed_book_messages(q["question"])), [], []
        is_ans = q["type"] == "answerable"
        rows.append({
            "id": q["id"], "type": q["type"], "difficulty": q["difficulty"], "question": q["question"],
            "answer": answer, "context_docs": ctx_docs,
            "evidence_in_context": evidence_in_context(ctx, q["evidence"]) if is_ans else False,
            "correct": answer_correct(answer, q["answer_keys"]) if is_ans else False,
            "citation_grounded": citation_grounded(answer, ctx, q["evidence"]) if is_ans else False,
            "abstained": is_abstention(answer), "seconds": round(time.time() - t0, 1),
        })
        print(f"[{a.name}] {i}/{len(questions)} {rows[-1]['seconds']}s", flush=True)

    summary = summarize(rows)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, f"generation.{a.name}.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump({"name": a.name, "mode": a.mode, "model": a.model, "method": a.method,
                   "chunk_size": a.chunk_size, "k": a.k, "prompt": a.prompt, "summary": summary, "rows": rows}, f, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
