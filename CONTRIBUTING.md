# Contributing

Short human guide. Agents and detailed product rules: see [AGENTS.md](AGENTS.md).

## Setup

**Prerequisites:** Python 3.11+ (3.12 recommended; CI uses 3.12), pip; optional Docker (Compose for Qdrant).

```bash
pip install -r requirements.txt   # includes pinned ruff and pytest
cp .env.example .env              # set ANTHROPIC_API_KEY for demo / live generation
```

Optional local Qdrant:

```bash
docker compose up -d
```

### Pre-commit hooks

Enable the repo hooks (ruff + gitleaks) once per clone:

```bash
pip install pre-commit
pre-commit install
```

Hooks run automatically on `git commit`. Run against all files with `pre-commit run --all-files`.

## Running

```bash
make demo                         # in-memory demo; needs ANTHROPIC_API_KEY
python ingestion.py               # ingest sample docs into Qdrant
uvicorn main:app --reload         # API on http://localhost:8000
```

## Tests and lint

```bash
make lint                         # ruff check + format --check (ruff==0.8.2)
make test                         # pytest tests/ -v
```

Offline retrieval eval (deterministic, no API key; downloads embedding/reranker models — not part of default PR CI):

```bash
python eval_retrieval.py --output results/retrieval_metrics.json
python eval.py
```

## Branch and PR conventions

- Branch from `main` with a short, descriptive name.
- Keep PRs **small and focused** on one concern.
- Fill in the pull request template checklist.
- State assumptions, env needs, and follow-ups in the description.
- Do not commit secrets; use env vars or a secret manager.
- Do not delete or weaken tests solely to make CI pass.

## Definition of done

- Product principles in `AGENTS.md` respected (grounded answers, fail-closed threshold, measurable retrieval, evaluated LLM output)
- Tests added/updated and passing; lint clean
- No secrets in the change
- Docs updated when behaviour or workflow changes
- CI green on the PR
