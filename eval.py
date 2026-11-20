#!/usr/bin/env python3
"""Comprehensive evaluation harness for the RAG pipeline.

Measures two offline metrics that require no LLM calls:

  Retrieval Recall@K
      For each question, did any of the top-K retrieved chunks contain content
      relevant to the reference answer?  Relevance is judged by keyword overlap
      (same approach as eval_retrieval.py).

  Answer Faithfulness
      Fraction of reference-answer keywords that appear in the generated
      answer.  Computed via keyword overlap when ``--answer`` is omitted
      (offline mode); skipped when no ANTHROPIC_API_KEY is available.

Usage::

    # Offline mode — no API key needed, skips LLM generation
    python eval.py

    # Full mode — generates answers and measures faithfulness
    python eval.py --answer

    # Custom top-k
    python eval.py --top-k 5

The script uses an in-memory Qdrant instance and the sample corpus from
``data/docs/``, so no external services are required.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import textwrap
from dataclasses import dataclass, field

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL
from ingestion import ingest
from retrieval import HybridRetriever, tokenize

# ---------------------------------------------------------------------------
# Hardcoded evaluation QA set
# ---------------------------------------------------------------------------
# Each entry has:
#   question   — the natural-language query
#   reference  — gold-standard answer text (used for relevance + faithfulness)
#   category   — 'answerable' | 'out_of_scope' (retriever should return empty)

EVAL_QA: list[dict[str, str]] = [
    {
        "question": "How do I post a manual journal?",
        "reference": (
            "Open Journals, select New Journal, enter date and narrative, "
            "add balancing lines, validate, then post."
        ),
        "category": "answerable",
    },
    {
        "question": "What depreciation profiles are available for fixed assets?",
        "reference": "Straight line over 3, 5, or 10 years, and reducing balance at 25 percent.",
        "category": "answerable",
    },
    {
        "question": "How do I upload a budget?",
        "reference": (
            "Open Planning, Budgets, Upload; choose scenario and CSV; validate; confirm."
        ),
        "category": "answerable",
    },
    {
        "question": "What fields are required when registering a fixed asset?",
        "reference": "Asset code, purchase date, cost, and depreciation profile.",
        "category": "answerable",
    },
    {
        "question": "What is the format for budget CSV files?",
        "reference": (
            "Three columns: account code, period (YYYY-MM), and amount. Amounts are signed."
        ),
        "category": "answerable",
    },
    {
        "question": "What happens if a journal does not balance?",
        "reference": (
            "Validation fails and the journal cannot be posted until debits equal credits."
        ),
        "category": "answerable",
    },
    {
        "question": "What budget scenarios are available?",
        "reference": "Base, Stretch, or Forecast.",
        "category": "answerable",
    },
    {
        "question": "How do I correct a posted journal?",
        "reference": "Create a reversing journal that references the original journal number.",
        "category": "answerable",
    },
    {
        "question": "How do I configure the payroll integration?",
        "reference": "Not covered in the documentation.",
        "category": "out_of_scope",
    },
    {
        "question": "How do I connect to Salesforce?",
        "reference": "Not covered in the documentation.",
        "category": "out_of_scope",
    },
]

# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

_STOPWORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "is",
        "are",
        "to",
        "and",
        "or",
        "of",
        "in",
        "for",
        "on",
        "with",
        "then",
        "from",
        "that",
        "this",
        "it",
        "be",
        "as",
        "at",
        "by",
        "do",
        "i",
        "how",
        "what",
        "when",
        "where",
    }
)


def _keywords(text: str) -> frozenset[str]:
    return frozenset(tokenize(text)) - _STOPWORDS


def chunk_matches_reference(chunk_content: str, reference: str) -> bool:
    """True if the chunk contains enough reference keywords to be relevant."""
    if "not covered" in reference.lower():
        return False
    ref_kw = _keywords(reference)
    if not ref_kw:
        return False
    chunk_kw = _keywords(chunk_content)
    overlap = chunk_kw & ref_kw
    threshold = min(3, max(1, int(len(ref_kw) * 0.4)))
    return len(overlap) >= threshold


def faithfulness_score(answer_text: str, reference: str) -> float:
    """Keyword-overlap faithfulness: fraction of reference keywords in the answer.

    Returns a float in [0.0, 1.0].  A score ≥ 0.5 is considered faithful.
    """
    ref_kw = _keywords(reference)
    if not ref_kw:
        return 1.0  # vacuously faithful
    ans_kw = _keywords(answer_text)
    return len(ans_kw & ref_kw) / len(ref_kw)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class EvalResult:
    question: str
    category: str
    recall_hit: bool | None  # None for out_of_scope
    num_chunks: int
    first_relevant_rank: int | None
    faithfulness: float | None = field(default=None)  # None when LLM not run
    answer_snippet: str = field(default="")


# ---------------------------------------------------------------------------
# Main evaluation logic
# ---------------------------------------------------------------------------


def run_eval(
    retriever: HybridRetriever,
    qa_set: list[dict[str, str]],
    top_k: int,
    run_generation: bool,
) -> list[EvalResult]:
    """Evaluate *retriever* (and optionally the full pipeline) on *qa_set*."""
    # Import generation lazily so the script is importable without an API key
    # when run_generation is False.
    answer_fn = None
    if run_generation:
        from generation import answer as _answer

        answer_fn = _answer

    results: list[EvalResult] = []
    for item in qa_set:
        q = item["question"]
        ref = item["reference"]
        cat = item["category"]

        chunks = retriever.search(q, top_k=top_k)

        # ── Retrieval recall ──────────────────────────────────────────────
        recall_hit: bool | None = None
        first_rank: int | None = None
        if cat == "answerable":
            recall_hit = False
            for rank, chunk in enumerate(chunks, start=1):
                if chunk_matches_reference(chunk.content, ref):
                    recall_hit = True
                    if first_rank is None:
                        first_rank = rank

        # ── Faithfulness (optional) ───────────────────────────────────────
        faith: float | None = None
        snippet = ""
        if answer_fn is not None and cat == "answerable":
            try:
                generated = answer_fn(q, chunks)
                faith = faithfulness_score(generated, ref)
                snippet = generated[:120].replace("\n", " ")
            except Exception as exc:
                snippet = f"[ERROR: {exc}]"

        results.append(
            EvalResult(
                question=q,
                category=cat,
                recall_hit=recall_hit,
                num_chunks=len(chunks),
                first_relevant_rank=first_rank,
                faithfulness=faith,
                answer_snippet=snippet,
            )
        )

    return results


# ---------------------------------------------------------------------------
# Pretty-print results table
# ---------------------------------------------------------------------------


def _truncate(text: str, width: int) -> str:
    return text if len(text) <= width else text[: width - 1] + "…"


def print_results_table(results: list[EvalResult], top_k: int) -> None:
    """Print a human-readable results table to stdout."""
    answerable = [r for r in results if r.category == "answerable"]
    out_of_scope = [r for r in results if r.category == "out_of_scope"]

    # ── Per-question detail ───────────────────────────────────────────────────
    col_q = 48
    col_hit = 10
    col_faith = 11
    col_rank = 6

    sep = "─" * (col_q + col_hit + col_faith + col_rank + 9)
    header = (
        f"{'Question':<{col_q}}  {'Recall@K':<{col_hit}}  "
        f"{'Faithful':<{col_faith}}  {'Rank':<{col_rank}}"
    )
    print()
    print("┌" + sep + "┐")
    print(f"│ {header} │")
    print("├" + sep + "┤")

    for r in results:
        if r.category == "answerable":
            hit_str = "✓ hit" if r.recall_hit else "✗ miss"
            faith_str = f"{r.faithfulness:.0%}" if r.faithfulness is not None else "—"
            rank_str = str(r.first_relevant_rank) if r.first_relevant_rank else "—"
        else:
            hit_str = "out-of-scope"
            faith_str = "—"
            rank_str = "—"

        q_short = _truncate(r.question, col_q)
        print(
            f"│ {q_short:<{col_q}}  {hit_str:<{col_hit}}  "
            f"{faith_str:<{col_faith}}  {rank_str:<{col_rank}} │"
        )

    print("└" + sep + "┘")

    # ── Aggregate metrics ─────────────────────────────────────────────────────
    if answerable:
        hits = sum(1 for r in answerable if r.recall_hit)
        recall_at_k = hits / len(answerable)
        mrr_values = [
            1.0 / r.first_relevant_rank if r.first_relevant_rank else 0.0 for r in answerable
        ]
        mrr = sum(mrr_values) / len(mrr_values)

        faith_values = [r.faithfulness for r in answerable if r.faithfulness is not None]
        mean_faith = sum(faith_values) / len(faith_values) if faith_values else None

        print()
        print("AGGREGATE METRICS")
        print(
            f"  Recall@{top_k}:          {recall_at_k:.1%}  ({hits}/{len(answerable)} answerable)"
        )
        print(f"  MRR:                {mrr:.3f}")
        if mean_faith is not None:
            print(f"  Mean Faithfulness:  {mean_faith:.1%}  (keyword overlap vs. reference)")
        else:
            print("  Mean Faithfulness:  — (run with --answer to compute)")
        print(f"  Out-of-scope Qs:    {len(out_of_scope)}")

    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=textwrap.dedent(
            """\
            Offline RAG evaluation harness.

            Metrics computed:
              Recall@K    — did a relevant chunk appear in the top-K results?
              Faithfulness — fraction of reference keywords in the generated answer
                             (only with --answer; requires ANTHROPIC_API_KEY)
            """
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=8,
        metavar="K",
        help="number of chunks to retrieve per question (default: 8)",
    )
    p.add_argument(
        "--answer",
        action="store_true",
        help="call the LLM and compute answer faithfulness (requires ANTHROPIC_API_KEY)",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="emit retrieval INFO logs to stderr",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    # Suppress retrieval logs during evaluation unless verbose is requested
    log_level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(level=log_level, stream=sys.stderr)

    if args.answer and not os.getenv("ANTHROPIC_API_KEY"):
        print("ERROR: --answer requires ANTHROPIC_API_KEY to be set.", file=sys.stderr)
        return 1

    print("Initializing in-memory Qdrant...")
    client = QdrantClient(":memory:")
    embedder = SentenceTransformer(EMBEDDING_MODEL)

    print("Ingesting sample corpus...")
    n_chunks = ingest(client=client, embedder=embedder)
    print(f"Ingested {n_chunks} chunks from data/docs/\n")

    retriever = HybridRetriever(qdrant_client=client)

    mode = "retrieval + LLM faithfulness" if args.answer else "retrieval only (no API key needed)"
    print(f"Running eval ({mode}, top_k={args.top_k}, {len(EVAL_QA)} questions)...")

    results = run_eval(
        retriever=retriever,
        qa_set=EVAL_QA,
        top_k=args.top_k,
        run_generation=args.answer,
    )

    print_results_table(results, top_k=args.top_k)
    return 0


if __name__ == "__main__":
    sys.exit(main())
