"""
scraper.py — fetch a URL, extract clean text, chunk it into overlapping windows.
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field

import httpx
from bs4 import BeautifulSoup

from trustrag.config import settings


@dataclass
class Chunk:
    text: str
    source_url: str
    chunk_index: int
    char_start: int
    char_end: int
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; TrustRAG/1.0; +https://github.com/trustrag)"
    )
}


def fetch_url(url: str, timeout: int = 15) -> str:
    """Return the visible text of a web page."""
    with httpx.Client(follow_redirects=True, timeout=timeout, headers=HEADERS) as client:
        resp = client.get(url)
        resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "lxml")

    # Remove boilerplate
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
        tag.decompose()

    # Prefer article / main body if available
    body = soup.find("article") or soup.find("main") or soup.body or soup
    raw = body.get_text(separator="\n")

    # Normalise whitespace
    lines = [ln.strip() for ln in raw.splitlines()]
    text = "\n".join(ln for ln in lines if ln)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_text(text: str, url: str) -> list[Chunk]:
    """
    Sliding-window character chunker.
    Returns a list of Chunk objects with source metadata.
    """
    size = settings.chunk_size
    overlap = settings.chunk_overlap
    chunks: list[Chunk] = []
    start = 0
    idx = 0

    while start < len(text):
        end = min(start + size, len(text))
        # Try to end at a sentence boundary within ±50 chars
        if end < len(text):
            boundary = _find_sentence_boundary(text, end, window=50)
            if boundary:
                end = boundary

        chunk_text_val = text[start:end].strip()
        if len(chunk_text_val) > 30:   # skip tiny trailing fragments
            chunks.append(
                Chunk(
                    text=chunk_text_val,
                    source_url=url,
                    chunk_index=idx,
                    char_start=start,
                    char_end=end,
                )
            )
            idx += 1

        start = end - overlap if end < len(text) else len(text)

    return chunks


def _find_sentence_boundary(text: str, pos: int, window: int = 50) -> int | None:
    """Return position of nearest sentence-end (.!?) near `pos`."""
    segment = text[max(0, pos - window): pos + window]
    for pattern in (r"[.!?]\s", r"\n\n"):
        for m in re.finditer(pattern, segment):
            abs_pos = pos - window + m.end()
            if 0 < abs_pos < len(text):
                return abs_pos
    return None