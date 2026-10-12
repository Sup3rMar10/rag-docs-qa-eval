"""Deterministic answer scoring (no LLM judge): cheap, reproducible, and easy to audit.

Trade-off, stated plainly: substring checks on short factual answers are strict about wording and cannot
judge free-form reasoning. They are used here because every question has a specific, checkable answer.
"""
import re

from src.chunking import normalize

ABSTAIN_PATTERNS = [
    r"\bi don'?t know\b", r"\bi do not know\b", r"\bnot (?:mentioned|stated|specified|provided|covered)\b",
    r"\bcannot (?:be )?(?:determined|found)\b", r"\bno information\b", r"\bnot contain\b",
    r"\bdoes not (?:say|mention|specify|state|provide)\b",
]


def _flat(text: str) -> str:
    return normalize(text.replace("’", "'"))


def answer_correct(answer: str, answer_keys: list[list[str]]) -> bool:
    """Every group must have at least one of its strings in the answer."""
    a = _flat(answer)
    return bool(answer_keys) and all(any(_flat(k) in a for k in group) for group in answer_keys)


def is_abstention(answer: str) -> bool:
    a = _flat(answer)
    return any(re.search(p, a) for p in ABSTAIN_PATTERNS)


def cited_numbers(answer: str) -> list[int]:
    return [int(n) for n in re.findall(r"\[(\d{1,2})\]", answer)]


def citation_grounded(answer: str, context_texts: list[str], evidence: list[str]) -> bool:
    """True if the answer cites a passage number that exists AND that passage contains evidence."""
    needles = [normalize(e) for e in evidence]
    for n in cited_numbers(answer):
        if 1 <= n <= len(context_texts) and any(nd in normalize(context_texts[n - 1]) for nd in needles):
            return True
    return False


def evidence_in_context(context_texts: list[str], evidence: list[str]) -> bool:
    needles = [normalize(e) for e in evidence]
    return any(nd in normalize(t) for t in context_texts for nd in needles)
