"""Hybrid retrieval: BM25 + dense vectors fused with weighted RRF, then cross-encoder re-ranked.

Retrieval pipeline (all stages observable via structured stderr logging):

  1. BM25 sparse search  — keyword overlap over the full corpus
  2. Dense vector search — cosine similarity via Qdrant
  3. Weighted RRF fusion — merges the two ranked lists with configurable weights
  4. Threshold filter    — discards chunks below MIN_SCORE_THRESHOLD (returns [] on no match)
  5. Cross-encoder re-rank — re-scores surviving candidates with a fine-tuned cross-encoder
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

from config import (
    BM25_WEIGHT,
    DENSE_WEIGHT,
    EMBEDDING_MODEL,
    MIN_SCORE_THRESHOLD,
    QDRANT_COLLECTION,
    QDRANT_URL,
    RERANKER_MODEL,
    TOP_K,
)

if TYPE_CHECKING:
    from qdrant_client.models import Record

logger = logging.getLogger(__name__)

_WORD_PATTERN = re.compile(r"\b\w+\b", re.UNICODE)


@dataclass
class Chunk:
    """A retrieved document chunk with provenance and scoring metadata."""

    content: str
    source: str
    score: float  # RRF fusion score (pre-reranker)
    chunk_id: str = field(default="")
    reranker_score: float | None = field(default=None)  # cross-encoder logit


def tokenize(text: str) -> list[str]:
    """Simple word tokenizer: lowercased word boundaries, handles punctuation."""
    return _WORD_PATTERN.findall(text.lower())


def weighted_rrf(
    ranked_lists: list[list[str]], weights: list[float], k: int = 60
) -> dict[str, float]:
    """Reciprocal rank fusion with per-list weights.

    Each chunk ID's score is the sum of weight / (k + rank) across all lists
    it appears in.  Higher score = better combined rank.
    """
    fused: dict[str, float] = {}
    for ids, weight in zip(ranked_lists, weights, strict=False):
        for rank, cid in enumerate(ids, start=1):
            fused[cid] = fused.get(cid, 0.0) + weight / (k + rank)
    return fused


class HybridRetriever:
    """Two-stage retriever: hybrid BM25+dense → RRF fusion → cross-encoder re-rank.

    The cross-encoder model is lazy-loaded on the first ``search()`` call so
    that importing this module or creating the retriever object is fast (useful
    for tests and cold-start scenarios).
    """

    def __init__(self, qdrant_client: QdrantClient | None = None) -> None:
        self.client = qdrant_client or QdrantClient(url=QDRANT_URL)
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)
        self._reranker: CrossEncoder | None = None
        self._bm25: BM25Okapi | None = None
        self._bm25_ids: list[str] = []
        self._bm25_version: int = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_reranker(self) -> CrossEncoder:
        """Lazy-load the cross-encoder model on first use."""
        if self._reranker is None:
            logger.info("retrieval stage=reranker_init model=%s", RERANKER_MODEL)
            self._reranker = CrossEncoder(RERANKER_MODEL)
        return self._reranker

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

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int = TOP_K) -> list[Chunk]:
        """Retrieve and re-rank the top-k most relevant chunks for *query*.

        Returns an empty list when no chunks meet the minimum similarity
        threshold — callers must handle this as a "no context" signal rather
        than passing an empty list to the LLM.

        Observable side-effects: one structured log line per pipeline stage
        is written to stderr via the standard ``logging`` module.  Enable
        with ``logging.basicConfig(level=logging.INFO, stream=sys.stderr)``
        or equivalent.
        """
        t_start = time.perf_counter()
        self._ensure_bm25_index()

        if self._bm25 is None or not self._bm25_ids:
            logger.info(
                "retrieval stage=empty query=%r reason=no_bm25_index latency_ms=0.0",
                query[:80],
            )
            return []

        # ── Stage 1: BM25 sparse retrieval ──────────────────────────────
        t0 = time.perf_counter()
        query_tokens = tokenize(query)
        bm25_scores = self._bm25.get_scores(query_tokens)
        bm25_ranked = [
            cid
            for cid, _ in sorted(
                zip(self._bm25_ids, bm25_scores, strict=False),
                key=lambda x: x[1],
                reverse=True,
            )
        ][: top_k * 2]
        t_bm25_ms = (time.perf_counter() - t0) * 1000

        logger.info(
            "retrieval stage=bm25 query=%r candidates=%d latency_ms=%.1f",
            query[:80],
            len(bm25_ranked),
            t_bm25_ms,
        )

        # ── Stage 2: Dense vector retrieval ─────────────────────────────
        t0 = time.perf_counter()
        vec = self.embedder.encode(query).tolist()
        dense_hits = self.client.query_points(
            collection_name=QDRANT_COLLECTION, query=vec, limit=top_k * 2
        ).points
        dense_ranked = [str(h.id) for h in dense_hits]
        t_dense_ms = (time.perf_counter() - t0) * 1000

        logger.info(
            "retrieval stage=dense query=%r candidates=%d latency_ms=%.1f",
            query[:80],
            len(dense_ranked),
            t_dense_ms,
        )

        # ── Stage 3: Weighted RRF fusion ─────────────────────────────────
        t0 = time.perf_counter()
        fused = weighted_rrf(
            [bm25_ranked, dense_ranked],
            weights=[BM25_WEIGHT, DENSE_WEIGHT],
        )
        top_ids = sorted(fused, key=fused.get, reverse=True)[:top_k]
        t_rrf_ms = (time.perf_counter() - t0) * 1000

        logger.info(
            "retrieval stage=rrf_fusion query=%r candidates=%d "
            "bm25_ms=%.1f dense_ms=%.1f rrf_ms=%.1f",
            query[:80],
            len(top_ids),
            t_bm25_ms,
            t_dense_ms,
            t_rrf_ms,
        )

        # ── Fetch payloads ───────────────────────────────────────────────
        records = self.client.retrieve(QDRANT_COLLECTION, ids=top_ids, with_payload=True)
        id_to_record = {str(r.id): r for r in records}
        candidates = [
            Chunk(
                content=id_to_record[cid].payload["content"],
                source=id_to_record[cid].payload["source"],
                score=fused[cid],
                chunk_id=cid,
            )
            for cid in top_ids
            if cid in id_to_record
        ]

        # ── Stage 4: Threshold filter ────────────────────────────────────
        above_threshold = [c for c in candidates if c.score >= MIN_SCORE_THRESHOLD]
        if not above_threshold:
            logger.info(
                "retrieval stage=threshold_filter query=%r "
                "threshold=%.4f max_score=%.4f result=empty",
                query[:80],
                MIN_SCORE_THRESHOLD,
                max((c.score for c in candidates), default=0.0),
            )
            return []

        logger.info(
            "retrieval stage=threshold_filter query=%r threshold=%.4f passed=%d/%d",
            query[:80],
            MIN_SCORE_THRESHOLD,
            len(above_threshold),
            len(candidates),
        )

        # ── Stage 5: Cross-encoder re-ranking ────────────────────────────
        t0 = time.perf_counter()
        reranker = self._get_reranker()
        pairs = [[query, c.content] for c in above_threshold]
        reranker_scores: list[float] = reranker.predict(pairs).tolist()
        rerank_ms = (time.perf_counter() - t0) * 1000

        for chunk, rs in zip(above_threshold, reranker_scores, strict=False):
            chunk.reranker_score = float(rs)

        reranked = sorted(above_threshold, key=lambda c: c.reranker_score or 0.0, reverse=True)[
            :top_k
        ]

        t_total_ms = (time.perf_counter() - t_start) * 1000
        for rank, chunk in enumerate(reranked, start=1):
            logger.info(
                "retrieval stage=final rank=%d chunk_id=%.8s source=%s "
                "rrf_score=%.4f reranker_score=%.4f rerank_ms=%.1f total_ms=%.1f",
                rank,
                chunk.chunk_id,
                chunk.source,
                chunk.score,
                chunk.reranker_score or 0.0,
                rerank_ms,
                t_total_ms,
            )

        return reranked
