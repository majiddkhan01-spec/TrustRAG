"""
vector_store.py — FAISS-backed store with sentence-transformers embeddings.
Supports per-session in-memory indexes (stateless API) or a persistent file store.
"""
from __future__ import annotations
import hashlib
import pickle
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from trustrag.config import settings
from trustrag.utils.scraper import Chunk


_embed_model: Optional[SentenceTransformer] = None


def _get_embed_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(settings.embed_model)
    return _embed_model


# ---------------------------------------------------------------------------
# VectorStore
# ---------------------------------------------------------------------------

class VectorStore:
    """
    Lightweight FAISS index wrapping a list of Chunks.
    One instance = one session / one document set.
    """

    def __init__(self) -> None:
        self.chunks: list[Chunk] = []
        self.index: Optional[faiss.Index] = None
        self._dim: Optional[int] = None

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def add_chunks(self, chunks: list[Chunk]) -> None:
        """Embed and index a list of Chunk objects."""
        if not chunks:
            return

        model = _get_embed_model()
        texts = [c.text for c in chunks]
        embeddings = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
        embeddings = np.array(embeddings, dtype="float32")

        if self.index is None:
            self._dim = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(self._dim)   # Inner-product = cosine for L2-normalised

        self.index.add(embeddings)
        self.chunks.extend(chunks)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def search(self, query: str, k: Optional[int] = None) -> list[tuple[Chunk, float]]:
        """
        Return top-k (chunk, score) pairs.
        Adaptive: if fewer chunks exist than k, returns all.
        """
        if self.index is None or len(self.chunks) == 0:
            return []

        k = k or settings.top_k_retrieve
        k = min(k, len(self.chunks))

        model = _get_embed_model()
        q_vec = model.encode([query], normalize_embeddings=True)
        q_vec = np.array(q_vec, dtype="float32")

        scores, indices = self.index.search(q_vec, k)

        results: list[tuple[Chunk, float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:
                results.append((self.chunks[idx], float(score)))

        return results

    # ------------------------------------------------------------------
    # Persistence (optional)
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path / "index.faiss"))
        with open(path / "chunks.pkl", "wb") as f:
            pickle.dump(self.chunks, f)

    @classmethod
    def load(cls, path: str | Path) -> "VectorStore":
        path = Path(path)
        store = cls()
        store.index = faiss.read_index(str(path / "index.faiss"))
        with open(path / "chunks.pkl", "rb") as f:
            store.chunks = pickle.load(f)
        store._dim = store.index.d
        return store


# ---------------------------------------------------------------------------
# URL cache: avoid re-fetching + re-embedding the same URL in one run
# ---------------------------------------------------------------------------

_url_hash_cache: set[str] = set()


def url_fingerprint(url: str) -> str:
    return hashlib.md5(url.encode()).hexdigest()


def already_indexed(url: str) -> bool:
    return url_fingerprint(url) in _url_hash_cache


def mark_indexed(url: str) -> None:
    _url_hash_cache.add(url_fingerprint(url))