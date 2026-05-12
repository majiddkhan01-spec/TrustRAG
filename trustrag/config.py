from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # --- LLM: Groq ---
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"  # free, fast, excellent quality

    # --- Local models ---
    embed_model: str = "all-MiniLM-L6-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    nli_model: str = "cross-encoder/nli-deberta-v3-small"

    # --- Retrieval ---
    chunk_size: int = 512
    chunk_overlap: int = 64
    top_k_retrieve: int = 20
    top_k_rerank: int = 6

    # --- Trust thresholds ---
    confidence_threshold: float = 0.60
    hallucination_threshold: float = 0.35
    max_corrective_loops: int = 2

    class Config:
        env_file = ".env"


settings = Settings()