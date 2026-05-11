"""
reranker.py — cross-encoder reranking of retrieved chunks.

Uses ms-marco-MiniLM (open-source, ~80 MB) which was trained specifically
for passage reranking and works well zero-shot.
"""
from __future__ import annotations
from typing import Optional

from sentence_transformers import CrossEncoder

from trustrag.config import settings
from trustrag.utils.scraper import Chunk


_reranker: Optional[CrossEncoder] = None


def _get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(settings.reranker_model, max_length=512)
    return _reranker


def rerank(
    query: str,
    candidates: list[tuple[Chunk, float]],
    top_k: Optional[int] = None,
) -> list[tuple[Chunk, float]]:
    """
    Re-score candidates with cross-encoder, return top_k sorted by new score.

    `candidates` is a list of (Chunk, bi-encoder-score) pairs from VectorStore.search().
    Returns (Chunk, cross-encoder-score) pairs.
    """
    if not candidates:
        return []

    top_k = top_k or settings.top_k_rerank
    reranker = _get_reranker()

    pairs = [(query, chunk.text) for chunk, _ in candidates]
    scores = reranker.predict(pairs)   # returns numpy array of floats

    ranked = sorted(
        zip([c for c, _ in candidates], scores),
        key=lambda x: x[1],
        reverse=True,
    )

    return [(chunk, float(score)) for chunk, score in ranked[:top_k]]