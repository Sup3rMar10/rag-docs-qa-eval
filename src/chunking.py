"""Markdown cleaning and structure-aware chunking.

Chunks follow the document's own structure: split on headings, then pack whole paragraphs up to a word
budget. A paragraph is only split when it alone exceeds the budget (long lists / code blocks), and then on
line boundaries. Every chunk remembers its document and heading path so answers can cite sources.
"""
import os
import re
from dataclasses import dataclass

FRONT_MATTER = re.compile(r"\A---\r?\n.*?\r?\n---\r?\n", re.S)
INCLUDE = re.compile(r"^\s*\[!INCLUDE[^\n]*$", re.M | re.I)
DIRECTIVE = re.compile(r"^\s*:::[^\n]*$", re.M)            # ::: zone, ::: image, :::row::: ...
HTML_COMMENT = re.compile(r"<!--.*?-->", re.S)
IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
LINK = re.compile(r"\[([^\]]+)\]\([^)]*\)")
CALLOUT = re.compile(r"^\s*>\s*\[!(NOTE|TIP|WARNING|IMPORTANT|CAUTION)\]\s*$", re.M | re.I)
HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
EMPHASIS = re.compile(r"(\*\*|__|`)")


def clean_markdown(text: str) -> str:
    text = text.replace("\r\n", "\n")
    text = FRONT_MATTER.sub("", text)
    text = HTML_COMMENT.sub("", text)
    text = INCLUDE.sub("", text)
    text = DIRECTIVE.sub("", text)
    text = CALLOUT.sub("", text)
    text = re.sub(r"^\s*>\s?", "", text, flags=re.M)        # blockquote markers
    text = IMAGE.sub("", text)
    text = LINK.sub(r"\1", text)
    text = EMPHASIS.sub("", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def normalize(text: str) -> str:
    """Canonical form for matching evidence strings against chunk text."""
    return re.sub(r"\s+", " ", EMPHASIS.sub("", text)).strip().lower()


def title_of(raw: str, fallback: str) -> str:
    m = re.search(r"^title:\s*(.+)$", raw, re.M)
    return m.group(1).strip().strip("'\"") if m else fallback


@dataclass
class Chunk:
    id: str
    doc: str
    title: str
    heading: str
    text: str

    @property
    def words(self) -> int:
        return len(self.text.split())

    @property
    def retrieval_text(self) -> str:
        """What gets indexed: body plus its title/heading, which carries a lot of topical signal."""
        return f"{self.title} > {self.heading}\n{self.text}" if self.heading else f"{self.title}\n{self.text}"


def _sections(body: str):
    """Yield (heading_path, text) pairs, one per heading-delimited section."""
    path, buf = [], []
    for line in body.split("\n"):
        m = HEADING.match(line) if not line.startswith("    ") else None
        if m and not _in_code_fence(buf):
            if "".join(buf).strip():
                yield " > ".join(path), "\n".join(buf).strip()
            level, title = len(m.group(1)), m.group(2)
            path = path[:level - 1] + [title]
            buf = []
        else:
            buf.append(line)
    if "".join(buf).strip():
        yield " > ".join(path), "\n".join(buf).strip()


def _in_code_fence(lines: list[str]) -> bool:
    return sum(1 for ln in lines if ln.strip().startswith("```")) % 2 == 1


def _split_long(paragraph: str, max_words: int) -> list[str]:
    parts, cur, count = [], [], 0
    for line in paragraph.split("\n"):
        w = len(line.split())
        if cur and count + w > max_words:
            parts.append("\n".join(cur))
            cur, count = [], 0
        cur.append(line)
        count += w
    if cur:
        parts.append("\n".join(cur))
    return parts


def _merge_small(chunks: list[Chunk], doc: str, max_words: int, min_words: int) -> list[Chunk]:
    """Fold chunks under `min_words` into their successor so a lone heading + one line is not its own chunk."""
    merged: list[Chunk] = []
    for c in chunks:
        prev = merged[-1] if merged else None
        if prev and prev.words < min_words and prev.words + c.words <= max_words:
            label = c.heading.split(" > ")[-1] if c.heading else ""
            parts = [prev.text, label, c.text] if label else [prev.text, c.text]
            prev.text = "\n\n".join(parts)
        else:
            merged.append(c)
    for i, c in enumerate(merged):
        c.id = f"{doc}#{i}"
    return merged


def chunk_document(doc: str, raw: str, max_words: int = 200, min_words: int = 40) -> list[Chunk]:
    title = title_of(raw, doc)
    body = clean_markdown(raw)
    chunks: list[Chunk] = []
    for heading, text in _sections(body):
        paragraphs = []
        for p in re.split(r"\n\s*\n", text):
            p = p.strip()
            if p:
                paragraphs.extend(_split_long(p, max_words) if len(p.split()) > max_words else [p])
        cur, count = [], 0
        for p in paragraphs + [None]:
            w = len(p.split()) if p else 0
            if p is None or (cur and count + w > max_words):
                chunks.append(Chunk(f"{doc}#{len(chunks)}", doc, title, heading, "\n\n".join(cur)))
                cur, count = [], 0
            if p is not None:
                cur.append(p)
                count += w
    return _merge_small([c for c in chunks if c.text.strip()], doc, max_words, min_words)


def load_corpus(corpus_dir: str, max_words: int = 200, min_words: int = 40) -> list[Chunk]:
    chunks = []
    for name in sorted(os.listdir(corpus_dir)):
        if not name.endswith(".md"):
            continue
        with open(os.path.join(corpus_dir, name), encoding="utf-8") as f:
            chunks.extend(chunk_document(name, f.read(), max_words, min_words))
    return chunks
