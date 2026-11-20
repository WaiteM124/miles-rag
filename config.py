"""Configuration loaded from environment variables."""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
QDRANT_URL: str = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION: str = os.getenv("QDRANT_COLLECTION", "miles_chunks")
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Retrieval fusion weights (must sum to 1.0)
BM25_WEIGHT: float = 0.4
DENSE_WEIGHT: float = 0.6
TOP_K: int = 8

# Cross-encoder re-ranker applied after RRF fusion
RERANKER_MODEL: str = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

# Minimum RRF score for a chunk to pass through to generation.
# Chunks below this threshold are discarded; if *all* chunks are below the
# threshold the pipeline returns a structured "no relevant context" refusal
# rather than passing empty context to the LLM.
MIN_SCORE_THRESHOLD: float = float(os.getenv("MIN_SCORE_THRESHOLD", "0.004"))
