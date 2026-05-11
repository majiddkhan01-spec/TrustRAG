"""
citation.py — attach inline citations to the generated answer.

For each sentence in the answer, find the best-matching grounded chunk
and append a [n] marker. Returns answer with inline citations + a reference list.
"""
from __future__ import annotations
import re
from dataclasses import dataclass

from trustrag.core.trust_scorer import ClaimVerdict
from trustrag.utils.scraper import Chunk


@dataclass
class CitedAnswer:
    text: str                          # answer with [n] inline markers
    references: list[dict]             # [{id, url, chunk_index, snippet}]


def attach_citations(
    answer: str,
    claim_verdicts: list[ClaimVerdict],
    context_chunks: list[tuple[Chunk, float]],
) -> CitedAnswer:
    """
    Rewrites the answer inserting [n] citation markers after each grounded claim.
    Ungrounded claims get a [?] marker instead.
    """
    # Build a url → reference id map
    url_to_ref: dict[str, int] = {}
    references: list[dict] = []

    def _get_ref_id(url: str, chunk: Chunk) -> int:
        if url not in url_to_ref:
            ref_id = len(references) + 1
            url_to_ref[url] = ref_id
            references.append({
                "id": ref_id,
                "url": url,
                "chunk_index": chunk.chunk_index,
                "snippet": chunk.text[:180].replace("\n", " ") + "…",
            })
        return url_to_ref[url]

    # Build a lookup: claim text → verdict
    verdict_map: dict[str, ClaimVerdict] = {v.claim: v for v in claim_verdicts}

    # Walk sentences in the answer, append citation markers
    sentences = re.split(r"(?<=[.!?])\s+", answer.strip())
    cited_sentences: list[str] = []

    for sent in sentences:
        verdict = verdict_map.get(sent.strip())
        if verdict and verdict.is_grounded and verdict.best_source_url:
            # Find the actual chunk object for snippet
            chunk = _find_chunk(
                context_chunks, verdict.best_source_url, verdict.best_chunk_index
            )
            if chunk:
                ref_id = _get_ref_id(verdict.best_source_url, chunk)
                cited_sentences.append(f"{sent} [{ref_id}]")
            else:
                cited_sentences.append(f"{sent} [?]")
        elif verdict and not verdict.is_grounded:
            cited_sentences.append(f"{sent} [?]")
        else:
            cited_sentences.append(sent)

    return CitedAnswer(
        text=" ".join(cited_sentences),
        references=references,
    )


def _find_chunk(
    context_chunks: list[tuple[Chunk, float]],
    url: str,
    chunk_index: int | None,
) -> Chunk | None:
    for chunk, _ in context_chunks:
        if chunk.source_url == url and (chunk_index is None or chunk.chunk_index == chunk_index):
            return chunk
    return None