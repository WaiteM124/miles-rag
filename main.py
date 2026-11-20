"""FastAPI serving layer for the RAG assistant."""

from __future__ import annotations

import hashlib
import os
import time
from collections import OrderedDict
from collections.abc import Iterator
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from generation import answer, answer_stream, is_refusal
from retrieval import Chunk, HybridRetriever

app = FastAPI(title="miles-rag")
retriever = HybridRetriever()

CACHE_TTL_SECONDS = 24 * 3600
CACHE_MAX_SIZE = 1000
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "")


class LRUCache:
    """Simple LRU cache with TTL and size bound."""

    def __init__(self, max_size: int = CACHE_MAX_SIZE, ttl: int = CACHE_TTL_SECONDS) -> None:
        self._cache: OrderedDict[str, tuple[Any, float]] = OrderedDict()
        self._max_size = max_size
        self._ttl = ttl

    def get(self, key: str) -> Any | None:
        if key not in self._cache:
            return None
        value, timestamp = self._cache[key]
        if time.time() - timestamp > self._ttl:
            del self._cache[key]
            return None
        self._cache.move_to_end(key)
        return value

    def set(self, key: str, value: Any) -> None:
        if key in self._cache:
            del self._cache[key]
        self._cache[key] = (value, time.time())
        while len(self._cache) > self._max_size:
            self._cache.popitem(last=False)

    def clear(self) -> int:
        n = len(self._cache)
        self._cache.clear()
        return n

    def __len__(self) -> int:
        return len(self._cache)


_cache = LRUCache()


class Query(BaseModel):
    question: str


class Source(BaseModel):
    content: str
    source: str


class Answer(BaseModel):
    answer: str
    refusal: bool
    sources: list[Source]


def verify_admin_token(x_admin_token: str = Header(default="")) -> str:
    if not ADMIN_TOKEN:
        raise HTTPException(status_code=503, detail="Admin endpoint not configured")
    if x_admin_token != ADMIN_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid admin token")
    return x_admin_token


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/health/cache/clear")
def clear_cache(_: str = Depends(verify_admin_token)) -> dict[str, int]:
    """Eval harness must call this before re-running questions. Requires ADMIN_TOKEN."""
    n = _cache.clear()
    return {"cleared": n}


@app.post("/query", response_model=Answer)
def query(q: Query) -> Answer:
    """Retrieve-then-generate with LRU caching.

    Returns a structured refusal (``refusal=True``) when no retrieved chunks
    clear the similarity threshold — the LLM is never called in that case.
    """
    key = hashlib.sha256(q.question.encode()).hexdigest()
    cached = _cache.get(key)
    if cached is not None:
        text, sources_data = cached
        return Answer(
            answer=text,
            refusal=is_refusal(text),
            sources=[Source(**s) for s in sources_data],
        )

    chunks: list[Chunk] = retriever.search(q.question)
    text = answer(q.question, chunks)
    sources = [Source(content=c.content, source=c.source) for c in chunks]
    _cache.set(key, (text, [s.model_dump() for s in sources]))
    return Answer(answer=text, refusal=is_refusal(text), sources=sources)


@app.post("/query/stream")
def query_stream(q: Query) -> StreamingResponse:
    """Stream the answer token-by-token.

    Tokens arrive via ``text/plain`` chunked transfer encoding.  The streaming
    endpoint bypasses the LRU cache (each call hits the retriever and LLM
    directly) and therefore always returns a fresh response.

    When no chunks clear the similarity threshold the full refusal string is
    returned as a single chunk without calling the LLM.

    Example (curl)::

        curl -X POST http://localhost:8000/query/stream \\
          -H "Content-Type: application/json" \\
          -d '{"question": "How do I post a manual journal?"}' \\
          --no-buffer
    """
    chunks: list[Chunk] = retriever.search(q.question)

    def token_gen() -> Iterator[str]:
        yield from answer_stream(q.question, chunks)

    return StreamingResponse(token_gen(), media_type="text/plain; charset=utf-8")
