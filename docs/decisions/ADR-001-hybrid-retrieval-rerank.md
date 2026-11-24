# ADR-001: Hybrid BM25 + dense retrieval with RRF and cross-encoder re-rank

**Status:** Accepted  
**Date:** 2026-11-20  
**Provenance:** Reconstructed from code (`retrieval.py`, `config.py`) and README — original decision write-up was not checked in.

---

## Context

The assistant must retrieve relevant chunks from a small ERP/FP&A markdown corpus for grounded generation. Exact terminology (account codes, field names) and paraphrased natural-language questions both appear in the golden set. A single sparse or dense ranker alone is a weak fit.

## Decision

Use a five-stage serving path in `HybridRetriever.search()`:

1. BM25 sparse (`rank_bm25.BM25Okapi`)
2. Dense vectors (`sentence-transformers/all-MiniLM-L6-v2` + Qdrant)
3. Weighted reciprocal rank fusion (`BM25_WEIGHT=0.4`, `DENSE_WEIGHT=0.6`)
4. Minimum RRF score threshold (see ADR-002)
5. Cross-encoder re-rank (`cross-encoder/ms-marco-MiniLM-L-6-v2`) on survivors

Emit structured per-stage logs (latency, candidate counts, scores). Lazy-load the cross-encoder on first use.

## Alternatives rejected

| Option | Why not |
|---|---|
| Dense-only / BM25-only | Weaker recall on mixed keyword vs paraphrase queries (documented intent in README) |
| Score-sum / min-max fusion | RRF avoids brittle score-scale calibration between BM25 and cosine |
| Re-rank entire corpus with cross-encoder | Too expensive; re-rank only the small post-RRF set |
| Learned fusion weights | Overkill for a tiny PoC corpus; fixed weights stay inspectable and pinned |

## Consequences

- Changing weights, `TOP_K`, models, or dropping a stage requires offline retrieval metrics (`eval_retrieval.py` / `eval.py`).
- BM25 index is in-memory and rebuilt via `refresh_bm25_index()` after ingestion.
- Cold start pays for embedding model load; cross-encoder cost is deferred until first search.
- Perfect hit-rate on the tiny sample corpus is expected; metrics will drop on a larger corpus.
