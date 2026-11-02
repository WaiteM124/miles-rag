"""Run the RAGAS baseline against a golden dataset, then the custom judges.

Usage:
    python -m eval.run_ragas eval/golden_sample.jsonl results/baseline.json

Clears the serving cache first: re-tests must not return stale answers.
"""
import json
import sys

import httpx
from datasets import Dataset
from ragas import evaluate as ragas_evaluate
from ragas.metrics import (
    faithfulness, answer_relevancy, context_precision, context_recall,
)

from custom_judges import judge_extractability, judge_refusal_clarity

API = "http://localhost:8000"


def run(golden_path: str, out_path: str):
    httpx.post(f"{API}/api/health/cache/clear")

    rows = [json.loads(l) for l in open(golden_path, encoding="utf-8")]
    records = []
    for row in rows:
        r = httpx.post(f"{API}/query", json={"question": row["question"]}).json()
        records.append({
            "question": row["question"],
            "answer": r["answer"],
            "contexts": [s["content"] for s in r["sources"]],
            "reference": row["reference"],
            "category": row.get("category", "clear_factual"),
        })

    ds = Dataset.from_list(records)
    ragas_result = ragas_evaluate(
        ds,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )

    custom = []
    for rec in records:
        ext = judge_extractability(rec["question"], rec["reference"], rec["answer"])
        ref = judge_refusal_clarity(rec["question"], rec["answer"])
        custom.append({"question": rec["question"], "category": rec["category"],
                       "extractability": ext["score"], "refusal_clarity": ref["score"]})

    out = {
        "ragas": {m.name: round(v, 2) for m, v in ragas_result.items()},
        "custom": custom,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out["ragas"], indent=2))


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "eval/golden_sample.jsonl",
        sys.argv[2] if len(sys.argv) > 2 else "results/baseline.json")
