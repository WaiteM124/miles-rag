"""Hybrid retrieval: BM25 + dense vectors fused with weighted RRF."""
from dataclasses import dataclass

from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from config import (
    QDRANT_URL, QDRANT_COLLECTION, EMBEDDING_MODEL,
    BM25_WEIGHT, DENSE_WEIGHT, TOP_K,
)


@dataclass
class Chunk:
    content: str
    source: str
    score: float


def weighted_rrf(ranked_lists: list[list[str]], weights: list[float], k: int = 60) -> dict[str, float]:
    """Reciprocal rank fusion with per-list weights. Chunk IDs must be unique."""
    fused: dict[str, float] = {}
    for ids, weight in zip(ranked_lists, weights):
        for rank, cid in enumerate(ids, start=1):
            fused[cid] = fused.get(cid, 0.0) + weight / (k + rank)
    return fused


class HybridRetriever:
    def __init__(self):
        self.client = QdrantClient(url=QDRANT_URL)
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)
        self._bm25 = None
        self._bm25_ids = None

    def _load_corpus(self):
        if self._bm25 is not None:
            return
        points, _ = self.client.scroll(QDRANT_COLLECTION, limit=10_000, with_payload=True)
        corpus = [p.payload["content"].split() for p in points]
        self._bm25_ids = [str(p.id) for p in points]
        self._bm25 = BM25Okapi(corpus)

    def search(self, query: str, top_k: int = TOP_K) -> list[Chunk]:
        self._load_corpus()

        bm25_scores = self._bm25.get_scores(query.split())
        bm25_ranked = [cid for cid, _ in sorted(
            zip(self._bm25_ids, bm25_scores), key=lambda x: x[1], reverse=True)][:top_k * 2]

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
        return [
            Chunk(content=r.payload["content"], source=r.payload["source"], score=fused[str(r.id)])
            for r in records
        ]
