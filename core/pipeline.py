"""
pipeline.py — the TrustRAG agentic loop.

Flow:
  1. Fetch + chunk each URL
  2. Embed + index into FAISS
  3. Retrieve top-k candidates
  4. Cross-encoder rerank
  5. Gemini Flash generation
  6. NLI hallucination detection + confidence scoring
  7. If confidence < threshold → corrective loop (widen k, stronger instruction)
  8. Citation grounding
  9. Return structured result
"""
from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Optional

from trustrag.config import settings
from trustrag.core.citation import CitedAnswer, attach_citations
from trustrag.core.llm import generate
from trustrag.core.reranker import rerank
from trustrag.core.trust_scorer import TrustResult, score_answer
from trustrag.core.vector_store import VectorStore, already_indexed, mark_indexed
from trustrag.utils.scraper import Chunk, chunk_text, fetch_url


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class TrustRAGResult:
    query: str
    answer: str                          # cited answer text
    references: list[dict]               # [{id, url, chunk_index, snippet}]
    confidence: float
    is_trusted: bool
    flagged: bool
    ungrounded_claims: list[str]
    loops_used: int
    sources_indexed: list[str]
    latency_ms: float
    raw_answer: str = ""                 # last raw answer before citation


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class TrustRAGPipeline:
    """
    Stateless pipeline. Pass a fresh VectorStore for each session,
    or reuse one across calls to accumulate documents.
    """

    def __init__(self, store: Optional[VectorStore] = None) -> None:
        self.store = store or VectorStore()

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest_urls(self, urls: list[str]) -> list[str]:
        """
        Fetch, chunk, and index a list of URLs.
        Skips URLs already indexed in this session.
        Returns list of successfully indexed URLs.
        """
        indexed: list[str] = []
        for url in urls:
            if already_indexed(url):
                continue
            try:
                text = fetch_url(url)
                chunks = chunk_text(text, url)
                self.store.add_chunks(chunks)
                mark_indexed(url)
                indexed.append(url)
            except Exception as exc:
                print(f"[WARN] Failed to fetch {url}: {exc}")
        return indexed

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(self, query: str, urls: list[str]) -> TrustRAGResult:
        t0 = time.time()

        # 1. Ingest
        sources_indexed = self.ingest_urls(urls)

        # 2. Adaptive retrieval
        k = settings.top_k_retrieve
        candidates = self.store.search(query, k=k)

        # 3. Rerank
        context_chunks = rerank(query, candidates)

        # 4–7. Generation + trust loop
        answer = ""
        trust: Optional[TrustResult] = None
        loops = 0
        extra_instruction = ""

        while loops <= settings.max_corrective_loops:
            # Generate
            answer = generate(query, context_chunks, extra_instruction)

            # Score
            trust = score_answer(answer, context_chunks)
            loops += 1

            if trust.is_trusted or loops > settings.max_corrective_loops:
                break

            # Corrective: widen retrieval and add instruction
            k = min(k + 10, len(self.store.chunks))
            candidates = self.store.search(query, k=k)
            context_chunks = rerank(query, candidates, top_k=min(settings.top_k_rerank + 2, k))
            extra_instruction = (
                "The previous answer contained ungrounded claims. "
                "Be very conservative—only state what is explicitly in the context. "
                f"Ungrounded claims to avoid: {trust.ungrounded_claims}"
            )

        # 8. Citation grounding
        cited: CitedAnswer = attach_citations(
            answer,
            trust.claim_verdicts if trust else [],
            context_chunks,
        )

        latency = (time.time() - t0) * 1000

        return TrustRAGResult(
            query=query,
            answer=cited.text,
            references=cited.references,
            confidence=round(trust.confidence if trust else 0.0, 4),
            is_trusted=trust.is_trusted if trust else False,
            flagged=trust.flagged if trust else True,
            ungrounded_claims=trust.ungrounded_claims if trust else [],
            loops_used=loops,
            sources_indexed=sources_indexed,
            latency_ms=round(latency, 1),
            raw_answer=answer,
        )