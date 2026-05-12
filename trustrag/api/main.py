"""
main.py — TrustRAG FastAPI server.

Endpoints:
  POST /query      — full agentic RAG query
  POST /ingest     — pre-index URLs without querying
  GET  /health     — liveness check
  GET  /docs       — Swagger UI (auto-generated)
"""
from __future__ import annotations
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl, Field

from trustrag.core.pipeline import TrustRAGPipeline, TrustRAGResult
from trustrag.core.vector_store import VectorStore


# ---------------------------------------------------------------------------
# App lifecycle: one shared pipeline per server process
# ---------------------------------------------------------------------------

_pipeline: TrustRAGPipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pipeline
    _pipeline = TrustRAGPipeline(store=VectorStore())
    yield
    _pipeline = None


app = FastAPI(
    title="TrustRAG",
    description=(
        "Self-Verifying Agentic Retrieval-Augmented Generation. "
        "Features: adaptive retrieval · reranking · hallucination detection · "
        "confidence scoring · citation grounding · corrective loop."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3, description="The question to answer")
    urls: list[HttpUrl] = Field(..., min_length=1, description="Web pages to retrieve from")

    model_config = {"json_schema_extra": {
        "example": {
            "query": "What are the main risks of transformer-based LLMs?",
            "urls": ["https://en.wikipedia.org/wiki/Large_language_model"]
        }
    }}


class IngestRequest(BaseModel):
    urls: list[HttpUrl] = Field(..., min_length=1)


class IngestResponse(BaseModel):
    indexed: list[str]
    message: str


class QueryResponse(BaseModel):
    query: str
    answer: str
    confidence: float
    is_trusted: bool
    flagged: bool
    loops_used: int
    ungrounded_claims: list[str]
    references: list[dict]
    sources_indexed: list[str]
    latency_ms: float


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok", "pipeline_ready": _pipeline is not None}


@app.post("/ingest", response_model=IngestResponse)
async def ingest(req: IngestRequest):
    if _pipeline is None:
        raise HTTPException(503, "Pipeline not initialised")
    url_strings = [str(u) for u in req.urls]
    indexed = _pipeline.ingest_urls(url_strings)
    return IngestResponse(
        indexed=indexed,
        message=f"Indexed {len(indexed)} new URL(s). {len(url_strings) - len(indexed)} skipped (already cached).",
    )


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    if _pipeline is None:
        raise HTTPException(503, "Pipeline not initialised")
    url_strings = [str(u) for u in req.urls]
    try:
        result: TrustRAGResult = _pipeline.query(req.query, url_strings)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, f"Pipeline error: {e}")

    return QueryResponse(
        query=result.query,
        answer=result.answer,
        confidence=result.confidence,
        is_trusted=result.is_trusted,
        flagged=result.flagged,
        loops_used=result.loops_used,
        ungrounded_claims=result.ungrounded_claims,
        references=result.references,
        sources_indexed=result.sources_indexed,
        latency_ms=result.latency_ms,
    )