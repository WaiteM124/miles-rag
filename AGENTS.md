# AGENTS.md — Miles RAG contributor & agent guide

Single source of truth for humans and AI agents (including Cursor) working in this repository.

**Product:** Miles RAG — a proof-of-concept retrieval-augmented generation (RAG) assistant. Hybrid search, grounded generation with structured refusals, streaming answers, and an offline evaluation harness over a small synthetic ERP/FP&A corpus.

**Owner:** Miles Waite (AI systems architect). Learning / portfolio piece with production-aware patterns (threshold-gated refusal, observability, eval harness). Not a multi-tenant production deployment.

---

## Repository map

| Path | Role |
|---|---|
| `config.py` | Env-loaded settings: API key, Qdrant, embedding/reranker models, RRF weights, `MIN_SCORE_THRESHOLD` |
| `ingestion.py` | Chunk (overlap), embed, upsert to Qdrant with content-hash stable IDs |
| `retrieval.py` | Hybrid BM25 + dense → weighted RRF → threshold filter → cross-encoder re-rank; structured stage logs |
| `generation.py` | Grounded Claude prompts, `REFUSAL:` protocol, `answer` / `answer_stream` |
| `main.py` | FastAPI: `/health`, `/query` (LRU cache), `/query/stream`, admin cache clear |
| `demo.py` | One-command in-memory demo (streaming + retrieval logs) |
| `eval.py` | Offline eval harness (recall@K, faithfulness proxy); optional `--answer` needs API key |
| `eval_retrieval.py` | Offline retrieval metrics (hit rate, MRR) on `golden_sample.jsonl` |
| `run_ragas.py` / `custom_judges.py` | Full RAGAS + extractability / refusal-clarity judges (API + running server) |
| `golden_sample.jsonl` | 35 hand-written evaluation Q&A pairs |
| `data/docs/` | Sample ERP/FP&A markdown corpus |
| `results/` | Checked-in / generated evaluation output |
| `tests/` | Offline unit tests (mocked; no live Qdrant/LLM required for assertions) |
| `docker-compose.yml` | Optional local Qdrant |
| `.github/workflows/ci.yml` | PR CI: ruff + pytest; scheduled/manual offline eval-gate |

Stack: **Python 3.11+** (CI uses 3.12), FastAPI, Pydantic v2, Qdrant, sentence-transformers, rank-bm25, Anthropic Claude, pytest, ruff. Demo and offline eval use **in-memory Qdrant** (no Docker). Live API + RAGAS need `ANTHROPIC_API_KEY` and usually Docker Qdrant.

---

## Product principles (non-negotiable)

1. **Grounded answers only.** Answer from retrieved context. If the answer is missing, ambiguous, or below the similarity threshold, emit an explicit `REFUSAL:` (or the deterministic `NO_CONTEXT_RESPONSE`). Never invent identifiers, amounts, or steps.
2. **Fail closed on weak retrieval.** When no chunk clears `MIN_SCORE_THRESHOLD`, return an empty candidate list and a structured refusal — **do not call the LLM** with empty or irrelevant context.
3. **Hybrid retrieval stays measurable.** BM25 + dense + weighted RRF + threshold + cross-encoder re-rank is the serving path. Prefer structured per-stage latency/score logs. Do not remove a stage or change fusion/threshold defaults without offline retrieval metrics.
4. **Evaluate before trusting LLM output.** Retrieval and answer quality are measured (`eval.py`, `eval_retrieval.py`, RAGAS, custom judges). Treat model output as untrusted until eval or tests cover the behaviour. Do not invent metrics in docs without measured evidence.

---

## Engineering standards

- **Typed code.** Type hints on public functions and methods. Pydantic models at HTTP request/response boundaries (`main.py`).
- **Tests as the spec.** Prefer known-answer / golden cases (RRF fusion, chunk IDs, refusal detection, chunking). Cover happy path, empty input, and out-of-scope / refusal behaviour where relevant.
- **Green before PR.** `ruff check` (and `make lint` / CI equivalents) plus `pytest tests/` must pass locally. CI must stay green.
- **Small, focused PRs.** One concern per PR; clear title and description; state assumptions and out-of-scope items.
- **No secrets in git.** Use `.env` (gitignored) or a secret manager. Never commit API keys, `ADMIN_TOKEN` values, credentials, or private keys. Prefer `.env.example` for documentation. Do not put secrets in git remotes.
- **Pin versions.** Dependencies are pinned in `requirements.txt` / `pyproject.toml`. Model names (`EMBEDDING_MODEL`, `RERANKER_MODEL`, Claude model id in `generation.py` / judges) stay pinned; do not silently bump models in a feature PR.
- **Structured logging.** Retrieval uses the standard `logging` module with explicit field-like messages (`stage=`, `latency_ms=`, scores, IDs). Do not log secrets or full API keys.
- **AI / LLM outputs evaluated and never trusted blindly.** Prefer offline retrieval gates for retrieval changes; use RAGAS / custom judges when answer quality changes and an API key is available.
- **Docs track behaviour.** Update README / ADRs / this file when behaviour, contracts, or run instructions change.

### Definition of done

- [ ] Principles above respected (grounding, fail-closed threshold, measurable retrieval, evaluated LLM output)
- [ ] Types and Pydantic contracts in place at boundaries
- [ ] Tests added or updated; known-answer cases where logic changes
- [ ] Lint and tests pass (`make lint` / `make test` or CI equivalents)
- [ ] No secrets introduced
- [ ] Docs updated if behaviour or workflow changed

---

## How to run and test

### Setup

```bash
pip install -r requirements.txt   # includes pinned ruff and pytest
cp .env.example .env              # set ANTHROPIC_API_KEY (and optional ADMIN_TOKEN) for live LLM / demo
```

Optional persistent Qdrant: `docker compose up -d`.

### Run

```bash
make demo                         # in-memory Qdrant; requires ANTHROPIC_API_KEY
# or full path:
python ingestion.py               # needs Qdrant at QDRANT_URL
uvicorn main:app --reload         # http://localhost:8000  docs: /docs  health: /health
```

### Test and lint

```bash
make lint                         # ruff check + ruff format --check
make test                         # pytest tests/ -v
# Offline retrieval eval (downloads embedding/reranker models; no API key):
python eval_retrieval.py --output results/retrieval_metrics.json
python eval.py                    # offline recall@K; add --answer for LLM faithfulness
```

RAGAS / custom judges need a running API, `ANTHROPIC_API_KEY`, and usually `ADMIN_TOKEN` for cache clear — see README.

---

## Workflow for agents

1. **Read before writing.** Inspect this file, neighbouring modules, and tests. Prefer matching existing patterns over inventing new ones.
2. **Plan large changes.** For multi-file or cross-cutting work, outline the approach, files touched, and risks before coding. Keep scope to the assigned task.
3. **Do not weaken tests.** Never delete, skip, or loosen assertions solely to make CI pass. Fix the code or add a justified, documented test change.
4. **State assumptions in the PR.** Call out env expectations (API keys, Qdrant), concurrent work, and follow-ups.
5. **Docs/config-only vs product code.** If the task says docs/config only, do not change application behaviour.
6. **Cursor rules.** Follow `.cursor/rules/*.mdc`; they summarise this file for scoped globs. When in conflict, **this file wins**.

---

## Related docs

- [CONTRIBUTING.md](CONTRIBUTING.md) — short human setup and PR conventions
- [README.md](README.md) — product overview, feature deep-dives, and quick start
- [docs/decisions/](docs/decisions/) — architecture decision records
