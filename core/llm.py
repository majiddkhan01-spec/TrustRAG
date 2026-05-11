"""
llm.py — Groq generation wrapper.
Uses Groq's OpenAI-compatible API endpoint.
Free tier: 30 RPM, no expiry, no credit card needed.
Get your key at: https://console.groq.com/keys
"""
from __future__ import annotations
import httpx

from trustrag.config import settings
from trustrag.utils.scraper import Chunk


SYSTEM_PROMPT = """\
You are TrustRAG, a precise research assistant.
Rules:
1. Answer ONLY using the provided context passages.
2. Write in full, grammatically complete sentences.
3. If a statement is not supported by the context, write [UNCERTAIN] at the end of that sentence.
4. Do NOT invent facts, statistics, or quotes.
5. Keep your answer focused and concise (3-6 sentences unless the question requires more).
"""


def generate(
    query: str,
    context_chunks: list[tuple[Chunk, float]],
    extra_instruction: str = "",
) -> str:
    if not settings.groq_api_key:
        raise ValueError(
            "GROQ_API_KEY not set. Add it to your .env file:\n"
            "  GROQ_API_KEY=your_key_here\n"
            "Get a free key at: https://console.groq.com/keys"
        )

    # Format context
    context_parts = []
    for i, (chunk, score) in enumerate(context_chunks, 1):
        context_parts.append(
            f"[Context {i}] (source: {chunk.source_url})\n{chunk.text}"
        )
    context_block = "\n\n".join(context_parts)

    user_prompt = (
        f"Context passages:\n{context_block}\n\n"
        f"{'Additional instruction: ' + extra_instruction + chr(10) if extra_instruction else ''}"
        f"Question: {query}\n\nAnswer:"
    )

    payload = {
        "model": settings.groq_model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": 1024,
        "temperature": 0.1,
    }

    with httpx.Client(timeout=60) as client:
        resp = client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {settings.groq_api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json()

    return data["choices"][0]["message"]["content"].strip()