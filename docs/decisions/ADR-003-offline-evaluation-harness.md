# ADR-003: Offline-first evaluation harness

**Status:** Accepted  
**Date:** 2026-11-20  
**Provenance:** Reconstructed from code (`eval.py`, `eval_retrieval.py`, `run_ragas.py`, `custom_judges.py`) and README — original decision write-up was not checked in.

---

## Context

RAG quality must be measurable without always spending Anthropic tokens or requiring a long-lived API server. Contributors and CI need a default path that is deterministic and runnable offline, plus an optional fuller LLM-judged path for portfolio metrics.

## Decision

Provide three layers:

1. **Offline retrieval / faithfulness proxies** — `eval_retrieval.py` and `eval.py` (default mode) use in-memory Qdrant, the sample corpus, and keyword-overlap relevance; **no API key**.
2. **Optional answer mode** — `eval.py --answer` generates with Claude when a key is present.
3. **Full LLM judges** — `run_ragas.py` + `custom_judges.py` (extractability, refusal clarity) against a running API and `golden_sample.jsonl`.

Unit tests under `tests/` stay mocked and form the default PR CI gate. The offline retrieval eval is the dedicated **eval-gate** (scheduled / manual) because model downloads make it too heavy for every PR.

## Alternatives rejected

| Option | Why not |
|---|---|
| RAGAS-only evaluation | Requires live LLM + server; blocks offline contributor loops |
| Embedding eval into every PR CI job | Deterministic but not cheap (downloads embedding + cross-encoder weights) |
| No golden set / metrics in-repo | Undermines the portfolio claim of evaluation-driven development |

## Consequences

- Documented README metrics must come from these scripts (or be marked not yet run).
- Keyword-overlap relevance is a proxy — good for regression on this tiny corpus, not a substitute for human labels at scale.
- Changing retrieval defaults should trigger an offline eval-gate run before merge when practical.
- LLM judge scores remain optional and must not be fabricated in docs.
