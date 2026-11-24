## Summary

<!-- What changed and why. State assumptions and out-of-scope items. -->

## Checklist

- [ ] **Principles respected** — grounded answers only; fail closed below `MIN_SCORE_THRESHOLD` (no LLM call on empty context); hybrid retrieval changes backed by offline metrics; LLM output not trusted without eval/tests ([AGENTS.md](../AGENTS.md))
- [ ] **Tests added or updated** — known-answer cases where retrieval, chunking, refusal, or API contracts change; existing tests not weakened or deleted to pass
- [ ] **Tests and lint passing** — `ruff check` / `make lint` and `pytest tests/` (CI green)
- [ ] **No secrets** — no API keys, `ADMIN_TOKEN` values, credentials, or private keys in code or this PR
- [ ] **Docs updated** — README / ADRs / `AGENTS.md` / `CONTRIBUTING.md` updated if behaviour or workflow changed

## Test plan

<!-- Commands run locally, or "rely on CI". For retrieval-path changes, note offline eval_retrieval.py / eval.py results or the scheduled eval-gate run. -->
