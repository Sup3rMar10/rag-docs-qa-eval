"""Fast unit tests: no model downloads, no corpus needed."""
from fastapi.testclient import TestClient

from src.api import create_app
from src.chunking import chunk_document, clean_markdown, normalize
from src.evalset import lexical_overlap, validate
from src.evaluate_retrieval import hit_rank, summarize
from src.rag import ABSTAIN_TEXT, format_context, rag_messages
from src.retrievers import BM25Retriever, rrf
from src.scoring import answer_correct, citation_grounded, evidence_in_context, is_abstention

DOC = """---
title: Scaling apps
description: x
---

# Scaling apps

[!INCLUDE [shared](includes/x.md)]

Container Apps scales to **zero** replicas when idle. See [the guide](guide.md) for more.

> [!NOTE]
> The default cooldown is 300 seconds.

## HTTP rules

An HTTP rule scales on concurrent requests. The default is `10` concurrent requests per replica.

## Tiny

Short.

## Jobs

Jobs run to completion. A manual job starts on demand and a scheduled job runs on a cron expression.
"""


# ---------------------------------------------------------------- chunking
def test_clean_markdown_strips_noise():
    out = clean_markdown(DOC)
    assert "title:" not in out and "INCLUDE" not in out and "[!NOTE]" not in out
    assert "the guide" in out and "guide.md" not in out        # link text kept, url dropped
    assert "**" not in out and "`" not in out
    assert "The default cooldown is 300 seconds." in out         # blockquote marker removed, text kept


def test_chunks_keep_heading_path_title_and_ids():
    chunks = chunk_document("scale.md", DOC, max_words=30, min_words=0)
    assert all(c.title == "Scaling apps" for c in chunks)
    assert any(c.heading.endswith("HTTP rules") for c in chunks)
    assert [c.id for c in chunks] == [f"scale.md#{i}" for i in range(len(chunks))]


def test_chunks_respect_budget_and_lose_no_text():
    chunks = chunk_document("scale.md", DOC, max_words=25, min_words=0)
    assert all(c.words <= 40 for c in chunks)
    joined = normalize(" ".join(c.text for c in chunks))
    for sentence in ["scales to zero replicas when idle", "default is 10 concurrent requests per replica",
                     "scheduled job runs on a cron expression"]:
        assert sentence in joined


def test_small_sections_are_merged():
    split = chunk_document("scale.md", DOC, max_words=200, min_words=0)
    merged = chunk_document("scale.md", DOC, max_words=200, min_words=40)
    assert len(merged) < len(split)
    assert any("Short." in c.text and "Jobs run to completion" in c.text for c in merged)


def test_retrieval_text_includes_title_and_heading():
    c = chunk_document("scale.md", DOC, max_words=200, min_words=0)[1]
    assert c.retrieval_text.startswith("Scaling apps")


# ---------------------------------------------------------------- retrieval
def test_bm25_finds_the_right_chunk_and_rrf_fuses():
    chunks = chunk_document("scale.md", DOC, max_words=30, min_words=0)
    bm25 = BM25Retriever(chunks)
    top = bm25.search("how many concurrent requests per replica for an HTTP rule", 1)[0][0]
    assert "concurrent requests" in chunks[top].text
    fused = rrf([[1, 3, 2], [1, 9, 3]])   # 1 ranked first by both; 3 beats 9 on combined rank
    assert [i for i, _ in fused][:2] == [1, 3] and {i for i, _ in fused} == {1, 2, 3, 9}


def test_hit_rank_and_summary():
    chunk_norm = ["alpha beta", "the default is 10 requests", "gamma"]
    assert hit_rank([0, 1, 2], chunk_norm, ["default is 10 requests"]) == 2
    assert hit_rank([0, 2], chunk_norm, ["default is 10 requests"]) is None
    rows = [{"rank": 1, "difficulty": "easy", "ctx_words_at5": 10, "latency_ms": 1.0},
            {"rank": None, "difficulty": "hard", "ctx_words_at5": 20, "latency_ms": 3.0},
            {"rank": 4, "difficulty": "hard", "ctx_words_at5": 30, "latency_ms": 2.0}]
    s = summarize(rows)
    assert s["hit@1"] == 1 / 3 and s["hit@5"] == 2 / 3 and abs(s["mrr@10"] - (1 + 0.25) / 3) < 1e-9


# ---------------------------------------------------------------- scoring
def test_answer_correct_requires_every_group():
    keys = [["4 minutes", "240 seconds"], ["http"]]
    assert answer_correct("It is 240 seconds for HTTP.", keys)
    assert not answer_correct("It is 240 seconds.", keys)
    assert not answer_correct("anything", [])


def test_abstention_detection():
    assert is_abstention(ABSTAIN_TEXT)
    assert is_abstention("The passages do not contain that. I don’t know.")
    assert not is_abstention("Set the replica retry limit to at least 1 [2].")


def test_citation_grounding():
    ctx = ["unrelated text", "set the retry limit to at least 1"]
    assert citation_grounded("Use at least 1 [2]", ctx, ["retry limit to at least 1"])
    assert not citation_grounded("Use at least 1 [1]", ctx, ["retry limit to at least 1"])   # cites wrong passage
    assert not citation_grounded("Use at least 1 [7]", ctx, ["retry limit to at least 1"])   # nonexistent passage
    assert not citation_grounded("Use at least 1", ctx, ["retry limit to at least 1"])       # no citation
    assert evidence_in_context(ctx, ["retry limit to at least 1"])


# ---------------------------------------------------------------- prompts + eval-set validation
def test_prompt_numbers_passages_and_forces_abstain_phrase():
    chunks = chunk_document("scale.md", DOC, max_words=30, min_words=0)
    msgs = rag_messages("q?", chunks[:2])
    assert "[1]" in msgs[1]["content"] and "[2]" in msgs[1]["content"]
    assert ABSTAIN_TEXT in msgs[0]["content"]
    assert format_context(chunks[:1]).startswith("[1] (Scaling apps")


def test_evalset_validator_catches_bad_evidence(tmp_path):
    (tmp_path / "a.md").write_text(DOC, encoding="utf-8")
    base = {"id": "q1", "question": "What is the default concurrency?", "type": "answerable", "category": "scaling",
            "difficulty": "easy", "reference_answer": "10"}
    good = {**base, "evidence": ["The default is 10 concurrent requests per replica."], "answer_keys": [["10"]]}
    bad_ev = {**base, "id": "q2", "question": "other?", "evidence": ["This sentence is not in the docs."],
              "answer_keys": [["10"]]}
    bad_key = {**base, "id": "q3", "question": "another?", "evidence": good["evidence"], "answer_keys": [["999"]]}
    una_bad = {**base, "id": "q4", "question": "unanswerable?", "type": "unanswerable", "evidence": ["x"],
               "answer_keys": []}
    probs = validate([good, bad_ev, bad_key, una_bad], str(tmp_path))
    assert not [p for p in probs if p.startswith("q1")]
    assert any(p.startswith("q2") and "not found verbatim" in p for p in probs)
    assert any(p.startswith("q3") and "answer key" in p for p in probs)
    assert any(p.startswith("q4") for p in probs)
    assert 0 <= lexical_overlap(good) <= 1


# ---------------------------------------------------------------- API (stub retriever + generator)
class StubRetriever:
    def search(self, query, k=10):
        return [(0, 1.0), (1, 0.5)][:k]


class StubGenerator:
    def generate(self, messages, max_new_tokens=140):
        assert "Question:" in messages[1]["content"]
        return "Scales to zero when idle [1]."


def test_api_returns_answer_with_sources():
    chunks = chunk_document("scale.md", DOC, max_words=30, min_words=0)
    client = TestClient(create_app(StubRetriever(), chunks, StubGenerator()))
    assert client.get("/health").json()["chunks"] == len(chunks)
    r = client.post("/ask", json={"question": "When does it scale to zero?", "k": 2})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"].endswith("[1].") and len(body["sources"]) == 2
    assert body["sources"][0]["n"] == 1 and body["sources"][0]["doc"] == "scale.md"


def test_api_validates_input_and_handles_failures():
    chunks = chunk_document("scale.md", DOC, max_words=30, min_words=0)

    class Boom:
        def generate(self, *a, **k):
            raise RuntimeError("model crashed")

    client = TestClient(create_app(StubRetriever(), chunks, Boom()))
    assert client.post("/ask", json={"question": "hi"}).status_code == 422          # too short
    assert client.post("/ask", json={"question": "valid question", "k": 99}).status_code == 422
    r = client.post("/ask", json={"question": "valid question here"})
    assert r.status_code == 502 and "RuntimeError" in r.json()["detail"]
