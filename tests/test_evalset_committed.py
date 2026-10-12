"""The committed question set must stay consistent with the pinned corpus (run after scripts/fetch_corpus.py)."""
import os

import pytest

from src.evalset import load_questions, validate

CORPUS, QUESTIONS = "data/corpus", "data/eval_questions.jsonl"


@pytest.mark.skipif(not os.path.isdir(CORPUS), reason="corpus not fetched (python scripts/fetch_corpus.py)")
def test_committed_questions_are_valid_against_corpus():
    qs = load_questions(QUESTIONS)
    assert validate(qs, CORPUS) == []
    assert sum(q["type"] == "answerable" for q in qs) == 40
    assert sum(q["type"] == "unanswerable" for q in qs) == 8
