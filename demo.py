#!/usr/bin/env python3
"""One-command demo: ingests sample corpus and runs sample queries.

Uses in-memory Qdrant by default (no Docker required).
Set QDRANT_URL environment variable to use an external Qdrant instance.
"""

from __future__ import annotations

import os
import sys

from dotenv import load_dotenv

load_dotenv()

SAMPLE_QUESTIONS = [
    "How do I post a manual journal?",
    "What depreciation profiles are available for fixed assets?",
    "How do I upload a budget?",
]


def check_api_key() -> bool:
    """Check if Anthropic API key is configured."""
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY environment variable is required.")
        print()
        print("Get an API key from https://console.anthropic.com/")
        print("Then run: export ANTHROPIC_API_KEY='your-key-here'")
        print("Or create a .env file with: ANTHROPIC_API_KEY=your-key-here")
        return False
    return True


def main() -> int:
    """Run the full demo."""
    print("=" * 60)
    print("MILES RAG DEMO")
    print("=" * 60)
    print()

    if not check_api_key():
        return 1

    from qdrant_client import QdrantClient
    from sentence_transformers import SentenceTransformer

    from config import EMBEDDING_MODEL, QDRANT_URL
    from generation import answer
    from ingestion import ingest
    from retrieval import HybridRetriever

    use_external = os.getenv("QDRANT_URL") and "localhost" not in QDRANT_URL
    if use_external:
        print(f"Using external Qdrant at {QDRANT_URL}")
        client = QdrantClient(url=QDRANT_URL)
    else:
        print("Using in-memory Qdrant (no Docker required)")
        client = QdrantClient(":memory:")

    print("Loading embedding model...")
    embedder = SentenceTransformer(EMBEDDING_MODEL)

    print("Ingesting sample documents...")
    n = ingest(client=client, embedder=embedder)
    print(f"Ingested {n} chunks from data/docs/")

    print("\nInitializing hybrid retriever...")
    retriever = HybridRetriever(qdrant_client=client)

    print("\n" + "=" * 60)
    print("SAMPLE QUERIES")
    print("=" * 60)

    for i, question in enumerate(SAMPLE_QUESTIONS, 1):
        print(f"\n--- Question {i} ---")
        print(f"Q: {question}")
        print()

        chunks = retriever.search(question)
        if not chunks:
            print("No relevant documents found.")
            continue

        print(f"Retrieved {len(chunks)} chunks from:")
        for c in chunks[:3]:
            print(f"  - {c.source} (score: {c.score:.4f})")

        print()
        try:
            response = answer(question, chunks)
            print(f"A: {response}")
        except Exception as e:
            print(f"Error generating answer: {e}")
        print()

    print("=" * 60)
    print("Demo complete!")
    print()
    print("To run the full API server:")
    print("  1. Start Qdrant: docker compose up -d")
    print("  2. Ingest docs:  python ingestion.py")
    print("  3. Start API:    uvicorn main:app --reload")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
