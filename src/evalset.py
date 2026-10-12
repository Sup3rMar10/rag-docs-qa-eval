"""Evaluation question set: loading and validation.

Each question is a JSON object:
  id, question, type ("answerable" | "unanswerable"), category, difficulty ("easy" | "medium" | "hard"),
  evidence      : list of verbatim sentences/phrases from the (cleaned) docs that contain the answer.
                  A retrieved chunk is a HIT if it contains any evidence string. Empty for unanswerable.
  answer_keys   : list of groups; the answer is correct if, for EVERY group, at least one string in the
                  group appears in the generated answer (case-insensitive). Empty for unanswerable.
  reference_answer : short human-readable answer (not used for scoring).
"""
import json
import os
import re
import sys

from src.chunking import clean_markdown, normalize

REQUIRED = {"id", "question", "type", "category", "difficulty", "evidence", "answer_keys", "reference_answer"}


def load_questions(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def corpus_text(corpus_dir: str) -> dict[str, str]:
    out = {}
    for name in sorted(os.listdir(corpus_dir)):
        if name.endswith(".md"):
            with open(os.path.join(corpus_dir, name), encoding="utf-8") as f:
                out[name] = normalize(clean_markdown(f.read()))
    return out


def evidence_docs(evidence: str, docs: dict[str, str]) -> list[str]:
    needle = normalize(evidence)
    return [name for name, text in docs.items() if needle in text]


def validate(questions: list[dict], corpus_dir: str) -> list[str]:
    """Return a list of problems (empty = valid)."""
    docs = corpus_text(corpus_dir)
    problems, seen_ids, seen_q = [], set(), set()
    for q in questions:
        qid = q.get("id", "?")
        missing = REQUIRED - set(q)
        if missing:
            problems.append(f"{qid}: missing fields {sorted(missing)}")
            continue
        if qid in seen_ids:
            problems.append(f"{qid}: duplicate id")
        seen_ids.add(qid)
        if q["question"].strip().lower() in seen_q:
            problems.append(f"{qid}: duplicate question")
        seen_q.add(q["question"].strip().lower())
        if q["type"] not in {"answerable", "unanswerable"}:
            problems.append(f"{qid}: bad type {q['type']!r}")
        if q["difficulty"] not in {"easy", "medium", "hard"}:
            problems.append(f"{qid}: bad difficulty {q['difficulty']!r}")
        if q["type"] == "answerable":
            if not q["evidence"]:
                problems.append(f"{qid}: answerable question needs evidence")
            if not q["answer_keys"] or not all(isinstance(g, list) and g for g in q["answer_keys"]):
                problems.append(f"{qid}: answer_keys must be a non-empty list of non-empty lists")
            for ev in q["evidence"]:
                if len(ev.split()) > 60:
                    problems.append(f"{qid}: evidence too long ({len(ev.split())} words): {ev[:50]}...")
                if not evidence_docs(ev, docs):
                    problems.append(f"{qid}: evidence not found verbatim in corpus: {ev[:80]!r}")
            for group in q["answer_keys"]:
                for key in group:
                    if not any(normalize(key) in normalize(ev) for ev in q["evidence"]):
                        problems.append(f"{qid}: answer key {key!r} does not appear in any evidence string")
        elif q["evidence"] or q["answer_keys"]:
            problems.append(f"{qid}: unanswerable question must have empty evidence and answer_keys")
    return problems


def lexical_overlap(q: dict) -> float:
    """Share of the question's content words that also appear in its evidence (high = easy for BM25)."""
    stop = set("a an the of to in on for is are can does do how what which when where why with and or by at as it its "
               "i my be this that from".split())
    qw = {w for w in re.findall(r"[a-z0-9]+", q["question"].lower()) if w not in stop}
    ew = set(re.findall(r"[a-z0-9]+", " ".join(q["evidence"]).lower()))
    return len(qw & ew) / len(qw) if qw else 0.0


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "data/eval_questions.jsonl"
    corpus = sys.argv[2] if len(sys.argv) > 2 else "data/corpus"
    qs = load_questions(path)
    probs = validate(qs, corpus)
    ans = [q for q in qs if q["type"] == "answerable"]
    print(f"{len(qs)} questions ({len(ans)} answerable, {len(qs) - len(ans)} unanswerable)")
    if ans:
        print(f"mean question/evidence lexical overlap: {sum(map(lexical_overlap, ans)) / len(ans):.2f}")
    for p in probs:
        print("PROBLEM:", p)
    sys.exit(1 if probs else 0)
