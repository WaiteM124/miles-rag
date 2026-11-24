# ADR-002: Threshold-gated structured refusal before generation

**Status:** Accepted  
**Date:** 2026-11-20  
**Provenance:** Reconstructed from code (`retrieval.py`, `generation.py`, `main.py`) and README — original decision write-up was not checked in.

---

## Context

Out-of-scope and low-similarity queries (e.g. Salesforce integration against an ERP journal corpus) must not produce fluent hallucinations. Passing empty or irrelevant context into Claude is unsafe and wastes tokens.

## Decision

After RRF fusion, discard chunks with score `< MIN_SCORE_THRESHOLD` (default `0.004`, env-configurable). If none remain, `search()` returns `[]`. `answer` / `answer_stream` return `NO_CONTEXT_RESPONSE` immediately **without** calling Anthropic. The system prompt also requires `REFUSAL:` prefixes for not-in-documentation and ambiguous cases. API responses expose `refusal: bool` via `is_refusal()`.

## Alternatives rejected

| Option | Why not |
|---|---|
| Always call the LLM and rely on the system prompt | Still allows soft hedges; costs tokens; harder to detect programmatically |
| Soft “I’m not sure” prose without a prefix | Breaks `is_refusal()` and custom refusal-clarity judges |
| Hard-coded topic denylist | Does not generalise; threshold uses retrieval confidence instead |

## Consequences

- Threshold tuning is a product decision: too high → false refusals; too low → weak context reaches the LLM.
- Streaming and non-streaming paths share the empty-chunks short-circuit.
- Eval suites must include out-of-scope / ambiguous golden items (present in `eval.py` and `golden_sample.jsonl`).
- Cache stores refusal text like any other answer; eval harness clears cache via admin token before re-runs.
