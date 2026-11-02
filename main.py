"""FastAPI serving layer for the RAG assistant."""
import hashlib
import time

from fastapi import FastAPI
from pydantic import BaseModel

from generation import answer, is_refusal
from retrieval import Chunk, HybridRetriever

app = FastAPI(title="alex-rag-poc")
retriever = HybridRetriever()

_cache: dict[str, tuple[str, list[dict], float]] = {}
CACHE_TTL_SECONDS = 24 * 3600  # known footgun: stale cache pollutes eval runs; clear before tests


class Query(BaseModel):
    question: str


class Source(BaseModel):
    content: str
    source: str


class Answer(BaseModel):
    answer: str
    refusal: bool
    sources: list[Source]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/health/cache/clear")
def clear_cache():
    """Eval harness must call this before re-running questions."""
    n = len(_cache)
    _cache.clear()
    return {"cleared": n}


@app.post("/query", response_model=Answer)
def query(q: Query):
    key = hashlib.sha256(q.question.encode()).hexdigest()
    hit = _cache.get(key)
    if hit and time.time() - hit[2] < CACHE_TTL_SECONDS:
        return Answer(answer=hit[0], refusal=is_refusal(hit[0]), sources=[Source(**s) for s in hit[1]])

    chunks: list[Chunk] = retriever.search(q.question)
    text = answer(q.question, chunks)
    sources = [Source(content=c.content, source=c.source) for c in chunks]
    _cache[key] = (text, [s.model_dump() for s in sources], time.time())
    return Answer(answer=text, refusal=is_refusal(text), sources=sources)
