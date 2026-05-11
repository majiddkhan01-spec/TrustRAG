"""
trust_scorer.py — hallucination detection and confidence scoring.

Strategy:
  1. Split the generated answer into individual claims (sentences).
  2. For each claim, run NLI cross-encoder against the top retrieved chunks.
  3. A claim is "grounded" if at least one chunk gives high ENTAILMENT score.
  4. Overall confidence = fraction of grounded claims, weighted by claim length.

Model: cross-encoder/nli-deberta-v3-small (~180 MB) — accurate, free, offline.
Label order for this model: contradiction=0, entailment=1, neutral=2.
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from sentence_transformers import CrossEncoder

from trustrag.config import settings
from trustrag.utils.scraper import Chunk


_nli_model: Optional[CrossEncoder] = None

# DeBERTa NLI label indices (check model card if you swap models)
_LABEL_CONTRADICTION = 0
_LABEL_ENTAILMENT = 1
_LABEL_NEUTRAL = 2


def _get_nli() -> CrossEncoder:
    global _nli_model
    if _nli_model is None:
        _nli_model = CrossEncoder(
            settings.nli_model,
            max_length=512,
            default_activation_function=None,
        )
    return _nli_model


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ClaimVerdict:
    claim: str
    is_grounded: bool
    entailment_score: float
    contradiction_score: float
    best_source_url: Optional[str] = None
    best_chunk_index: Optional[int] = None


@dataclass
class TrustResult:
    confidence: float                        # 0.0 – 1.0
    is_trusted: bool                         # confidence >= threshold
    claim_verdicts: list[ClaimVerdict] = field(default_factory=list)
    ungrounded_claims: list[str] = field(default_factory=list)
    flagged: bool = False                    # True if any strong contradictions found


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def score_answer(
    answer: str,
    context_chunks: list[tuple[Chunk, float]],
) -> TrustResult:
    """
    Evaluate the answer against retrieved context.
    Returns a TrustResult with per-claim verdicts and overall confidence.
    """
    claims = _split_into_claims(answer)
    if not claims or not context_chunks:
        return TrustResult(confidence=0.0, is_trusted=False, flagged=True)

    nli = _get_nli()
    verdicts: list[ClaimVerdict] = []

    for claim in claims:
        best_entailment = 0.0
        best_contradiction = 0.0
        best_url: Optional[str] = None
        best_idx: Optional[int] = None

        # Test each context chunk
        pairs = [(chunk.text, claim) for chunk, _ in context_chunks]
        raw_scores = nli.predict(pairs, apply_softmax=True)

        for (chunk, _), scores in zip(context_chunks, raw_scores):
            entail_score = float(scores[_LABEL_ENTAILMENT])
            contra_score = float(scores[_LABEL_CONTRADICTION])

            if entail_score > best_entailment:
                best_entailment = entail_score
                best_contradiction = contra_score
                best_url = chunk.source_url
                best_idx = chunk.chunk_index

        is_grounded = best_entailment >= (1 - settings.hallucination_threshold)
        verdicts.append(
            ClaimVerdict(
                claim=claim,
                is_grounded=is_grounded,
                entailment_score=best_entailment,
                contradiction_score=best_contradiction,
                best_source_url=best_url,
                best_chunk_index=best_idx,
            )
        )

    # Weight by claim length (longer claims matter more)
    weights = np.array([len(v.claim) for v in verdicts], dtype=float)
    weights /= weights.sum()
    grounded_flags = np.array([v.is_grounded for v in verdicts], dtype=float)
    confidence = float(np.dot(grounded_flags, weights))

    ungrounded = [v.claim for v in verdicts if not v.is_grounded]
    flagged = any(v.contradiction_score > 0.6 for v in verdicts)

    return TrustResult(
        confidence=confidence,
        is_trusted=confidence >= settings.confidence_threshold,
        claim_verdicts=verdicts,
        ungrounded_claims=ungrounded,
        flagged=flagged,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _split_into_claims(text: str) -> list[str]:
    """Split answer into individual sentences / claims."""
    # Split on . ! ? followed by space or newline, keep minimum length
    raw = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in raw if len(s.strip()) > 15]