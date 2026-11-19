#!/usr/bin/env python3
"""Evaluate retrieval quality on the golden dataset.

Computes offline metrics that don't require LLM calls:
- Hit Rate @ K: fraction of questions where any retrieved chunk contains relevant content
- Mean Reciprocal Rank (MRR): average of 1/rank for first relevant chunk

Usage:
    python eval_retrieval.py [--output results/retrieval_metrics.json]

This script uses an in-memory Qdrant instance, so no external services are required.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL
from ingestion import ingest
from retrieval import HybridRetriever, tokenize


def load_golden_set(path: str = "golden_sample.jsonl") -> list[dict]:
    """Load the golden evaluation dataset."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def content_matches_reference(chunk_content: str, reference: str) -> bool:
    """Check if a chunk contains content relevant to the reference answer.

    Uses keyword overlap to determine relevance.
    """
    if "not covered" in reference.lower() or "too broad" in reference.lower():
        return False

    chunk_tokens = set(tokenize(chunk_content))
    reference_tokens = set(tokenize(reference))

    stopwords = {
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
    }
    keywords = reference_tokens - stopwords

    if len(keywords) == 0:
        return False

    overlap = chunk_tokens & keywords
    return len(overlap) >= min(3, max(1, int(len(keywords) * 0.4)))


def evaluate_retrieval(
    retriever: HybridRetriever,
    golden_set: list[dict],
    top_k: int = 8,
) -> dict:
    """Evaluate retrieval performance on the golden dataset."""
    results = []
    hit_count = 0
    reciprocal_ranks = []
    total_answerable = 0

    for item in golden_set:
        question = item["question"]
        reference = item["reference"]
        category = item.get("category", "clear_factual")

        is_answerable = category == "clear_factual"
        if is_answerable:
            total_answerable += 1

        chunks = retriever.search(question, top_k=top_k)

        hit = False
        first_relevant_rank = None

        for rank, chunk in enumerate(chunks, start=1):
            if content_matches_reference(chunk.content, reference):
                hit = True
                if first_relevant_rank is None:
                    first_relevant_rank = rank

        result = {
            "question": question,
            "category": category,
            "num_chunks": len(chunks),
            "hit": hit if is_answerable else None,
            "first_relevant_rank": first_relevant_rank,
            "top_sources": [c.source for c in chunks[:3]],
        }
        results.append(result)

        if is_answerable:
            if hit:
                hit_count += 1
            if first_relevant_rank:
                reciprocal_ranks.append(1.0 / first_relevant_rank)
            else:
                reciprocal_ranks.append(0.0)

    hit_rate = hit_count / total_answerable if total_answerable > 0 else 0
    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0

    return {
        "metrics": {
            "hit_rate_at_8": round(hit_rate, 3),
            "mrr": round(mrr, 3),
            "top_k": top_k,
            "total_questions": len(golden_set),
            "answerable_questions": total_answerable,
            "hits": hit_count,
        },
        "per_question": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval quality")
    parser.add_argument(
        "--golden",
        default="golden_sample.jsonl",
        help="Path to golden dataset",
    )
    parser.add_argument(
        "--output",
        default="results/retrieval_metrics.json",
        help="Output path for results",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=8,
        help="Number of chunks to retrieve",
    )
    args = parser.parse_args()

    print("Initializing in-memory Qdrant...")
    client = QdrantClient(":memory:")
    embedder = SentenceTransformer(EMBEDDING_MODEL)

    print("Ingesting sample corpus...")
    n_chunks = ingest(client=client, embedder=embedder)
    print(f"Ingested {n_chunks} chunks")

    print("\nLoading golden dataset...")
    golden_set = load_golden_set(args.golden)
    print(f"Loaded {len(golden_set)} questions")

    print("\nInitializing hybrid retriever...")
    retriever = HybridRetriever(qdrant_client=client)

    print(f"\nEvaluating retrieval (top_k={args.top_k})...")
    results = evaluate_retrieval(retriever, golden_set, top_k=args.top_k)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 50)
    print("RETRIEVAL METRICS")
    print("=" * 50)
    metrics = results["metrics"]
    print(f"Hit Rate @ {metrics['top_k']}:  {metrics['hit_rate_at_8']:.1%}")
    print(f"MRR:                {metrics['mrr']:.3f}")
    print(f"Answerable Qs:      {metrics['answerable_questions']}")
    print(f"Hits:               {metrics['hits']}")
    print("=" * 50)
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
