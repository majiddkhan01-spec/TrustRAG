# TrustRAG

**A Self-Verifying Agentic Retrieval-Augmented Generation System**

## Features

| Feature | Implementation |
|---|---|
| Adaptive retrieval | FAISS inner-product search, k widens on corrective loop |
| Reranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Hallucination detection | `cross-encoder/nli-deberta-v3-small` NLI per-claim check |
| Confidence scoring | Weighted fraction of grounded claims |
| Citation grounding | Inline [n] markers + reference list with source URLs |
| Corrective loop | Up to N retries with widened context + stronger instruction |
| LLM backbone | Gemini Flash (`gemini-1.5-flash`) |
| Embeddings | `all-MiniLM-L6-v2` via sentence-transformers (free, local) |

---

## Setup

### 1. Install dependencies

```bash
cd trustrag
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY
```

Get a free Gemini API key at: https://aistudio.google.com/app/apikey

### 3. Run

**Option A — CLI (quickest for testing)**

```bash
python cli.py \
  --query "What are the risks of large language models?" \
  --urls https://en.wikipedia.org/wiki/Large_language_model
```

**Option B — FastAPI server**

```bash
uvicorn trustrag.api.main:app --reload --port 8000
```

Then open http://localhost:8000/docs for the interactive Swagger UI.

**Example API call:**

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is retrieval-augmented generation?",
    "urls": ["https://en.wikipedia.org/wiki/Retrieval-augmented_generation"]
  }'
```

---

## Response structure

```json
{
  "query": "...",
  "answer": "RAG combines retrieval with generation [1]. ...",
  "confidence": 0.87,
  "is_trusted": true,
  "flagged": false,
  "loops_used": 1,
  "ungrounded_claims": [],
  "references": [
    {
      "id": 1,
      "url": "https://...",
      "chunk_index": 3,
      "snippet": "Retrieval-augmented generation (RAG) is a technique..."
    }
  ],
  "sources_indexed": ["https://..."],
  "latency_ms": 3200.0
}
```

| Field | Meaning |
|---|---|
| `confidence` | 0–1, fraction of claims grounded in source text |
| `is_trusted` | `true` if confidence ≥ threshold (default 0.60) |
| `flagged` | `true` if any strong contradictions were detected |
| `loops_used` | How many corrective iterations were needed |
| `ungrounded_claims` | Sentences the NLI model couldn't verify |

---

## Architecture

```
User query + URLs
      │
      ▼
Fetch + chunk (beautifulsoup4 + sliding window)
      │
      ▼
Embed + FAISS index (sentence-transformers all-MiniLM-L6-v2)
      │
      ▼
Cross-encoder rerank (ms-marco-MiniLM-L-6-v2)
      │
      ▼
Gemini Flash generation
      │
      ▼
NLI hallucination check (nli-deberta-v3-small) → confidence score
      │
   trusted? ──no──▶ widen k, stronger prompt ──▶ (loop)
      │yes
      ▼
Citation grounding → [n] markers + references
      │
      ▼
Structured response
```

---

## Tuning

Edit `config.py` or set env vars in `.env`:

| Variable | Default | Effect |
|---|---|---|
| `CONFIDENCE_THRESHOLD` | 0.60 | Lower = more permissive |
| `HALLUCINATION_THRESHOLD` | 0.35 | Lower = stricter NLI entailment required |
| `MAX_CORRECTIVE_LOOPS` | 2 | More loops = more accurate, slower |
| `TOP_K_RETRIEVE` | 20 | More candidates = better recall |
| `TOP_K_RERANK` | 6 | More context = richer answers (but slower NLI) |
| `CHUNK_SIZE` | 512 | Smaller = more precise citations |

---

## Persisting indexes

```bash
# Save after first run
python cli.py --query "..." --urls https://... --save-index ./my_index

# Load on next run (skips re-fetching)
python cli.py --query "..." --urls https://... --load-index ./my_index
```

---

## Cost estimate (per query)

| Component | Cost |
|---|---|
| Gemini Flash | ~$0.00007 per query (1k tokens in/out) |
| Embeddings | Free (local) |
| Reranker | Free (local) |
| NLI model | Free (local) |
| **Total** | **< $0.001 per query** |