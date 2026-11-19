"""Run the RAGAS baseline against a golden dataset, then the custom judges.

Usage:
    python run_ragas.py golden_sample.jsonl results/baseline.json

Clears the serving cache first: re-tests must not return stale answers.
Requires ADMIN_TOKEN env var to clear the cache.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import httpx
from datasets import Dataset
from ragas import evaluate as ragas_evaluate
from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

from custom_judges import judge_extractability, judge_refusal_clarity

API = os.getenv("RAG_API_URL", "http://localhost:8000")
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "")


def run(golden_path: str, out_path: str) -> None:
    headers = {"X-Admin-Token": ADMIN_TOKEN} if ADMIN_TOKEN else {}
    if headers:
        httpx.post(f"{API}/api/health/cache/clear", headers=headers)

    rows = [json.loads(line) for line in open(golden_path, encoding="utf-8")]
    records = []
    for row in rows:
        r = httpx.post(f"{API}/query", json={"question": row["question"]}).json()
        records.append(
            {
                "question": row["question"],
                "answer": r["answer"],
                "contexts": [s["content"] for s in r["sources"]],
                "reference": row["reference"],
                "category": row.get("category", "clear_factual"),
            }
        )

    ds = Dataset.from_list(records)
    ragas_result = ragas_evaluate(
        ds,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )

    custom = []
    for rec in records:
        ext = judge_extractability(rec["question"], rec["reference"], rec["answer"])
        ref = judge_refusal_clarity(rec["question"], rec["answer"])
        custom.append(
            {
                "question": rec["question"],
                "category": rec["category"],
                "extractability": ext["score"],
                "refusal_clarity": ref["score"],
            }
        )

    out = {
        "ragas": {m.name: round(v, 2) for m, v in ragas_result.items()},
        "custom": custom,
    }

    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out["ragas"], indent=2))


if __name__ == "__main__":
    run(
        sys.argv[1] if len(sys.argv) > 1 else "golden_sample.jsonl",
        sys.argv[2] if len(sys.argv) > 2 else "results/baseline.json",
    )
