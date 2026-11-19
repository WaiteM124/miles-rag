"""Hybrid retrieval: BM25 + dense vectors fused with weighted RRF."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from config import (
    BM25_WEIGHT,
    DENSE_WEIGHT,
    EMBEDDING_MODEL,
    QDRANT_COLLECTION,
    QDRANT_URL,
    TOP_K,
)

if TYPE_CHECKING:
    from qdrant_client.models import Record


@dataclass
class Chunk:
    content: str
    source: str
    score: float


_WORD_PATTERN = re.compile(r"\b\w+\b", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Simple word tokenizer: lowercased word boundaries, handles punctuation."""
    return _WORD_PATTERN.findall(text.lower())


def weighted_rrf(
    ranked_lists: list[list[str]], weights: list[float], k: int = 60
) -> dict[str, float]:
    """Reciprocal rank fusion with per-list weights. Chunk IDs must be unique."""
    fused: dict[str, float] = {}
    for ids, weight in zip(ranked_lists, weights, strict=False):
        for rank, cid in enumerate(ids, start=1):
            fused[cid] = fused.get(cid, 0.0) + weight / (k + rank)
    return fused


class HybridRetriever:
    def __init__(self, qdrant_client: QdrantClient | None = None) -> None:
        self.client = qdrant_client or QdrantClient(url=QDRANT_URL)
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)
        self._bm25: BM25Okapi | None = None
        self._bm25_ids: list[str] = []
        self._bm25_version: int = 0

    def refresh_bm25_index(self) -> None:
        """Rebuild the BM25 index from the current collection state."""
        all_points: list[Record] = []
        offset = None
        while True:
            points, offset = self.client.scroll(
                QDRANT_COLLECTION,
                limit=1000,
                offset=offset,
                with_payload=True,
            )
            all_points.extend(points)
            if offset is None:
                break

        if not all_points:
            self._bm25 = None
            self._bm25_ids = []
            return

        corpus = [tokenize(p.payload["content"]) for p in all_points]
        self._bm25_ids = [str(p.id) for p in all_points]
        self._bm25 = BM25Okapi(corpus)
        self._bm25_version += 1

    def _ensure_bm25_index(self) -> None:
        if self._bm25 is None:
            self.refresh_bm25_index()

    def search(self, query: str, top_k: int = TOP_K) -> list[Chunk]:
        self._ensure_bm25_index()

        if self._bm25 is None or not self._bm25_ids:
            return []

        query_tokens = tokenize(query)
        bm25_scores = self._bm25.get_scores(query_tokens)
        bm25_ranked = [
            cid
            for cid, _ in sorted(
                zip(self._bm25_ids, bm25_scores, strict=False), key=lambda x: x[1], reverse=True
            )
        ][: top_k * 2]

        vec = self.embedder.encode(query).tolist()
        dense_hits = self.client.query_points(
            collection_name=QDRANT_COLLECTION, query=vec, limit=top_k * 2
        ).points
        dense_ranked = [str(h.id) for h in dense_hits]

        fused = weighted_rrf(
            [bm25_ranked, dense_ranked],
            weights=[BM25_WEIGHT, DENSE_WEIGHT],
        )
        top_ids = sorted(fused, key=fused.get, reverse=True)[:top_k]

        records = self.client.retrieve(QDRANT_COLLECTION, ids=top_ids, with_payload=True)
        id_to_record = {str(r.id): r for r in records}
        return [
            Chunk(
                content=id_to_record[cid].payload["content"],
                source=id_to_record[cid].payload["source"],
                score=fused[cid],
            )
            for cid in top_ids
            if cid in id_to_record
        ]
