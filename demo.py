#!/usr/bin/env python3
"""One-command demo: ingests sample corpus and runs sample queries with streaming output.

Uses in-memory Qdrant by default (no Docker required).
Set QDRANT_URL environment variable to use an external Qdrant instance.

The demo showcases all five pipeline improvements:
  1. Hybrid retrieval (BM25 + dense + RRF + cross-encoder re-rank)
  2. Failure-mode refusal when no context is found
  3. Streaming token-by-token output
  4. Observable retrieval logs written to stderr
  5. Eval harness available separately via: python eval.py
"""

from __future__ import annotations

import logging
import os
import sys

from dotenv import load_dotenv

load_dotenv()

# ── Observability: emit retrieval logs to stderr ──────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    stream=sys.stderr,
    format="%(asctime)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)

SAMPLE_QUESTIONS = [
    "How do I post a manual journal?",
    "What depreciation profiles are available for fixed assets?",
    "How do I upload a budget?",
    # This question intentionally has no matching context — demonstrates the
    # structured refusal failure-mode response instead of hallucination.
    "How do I configure the Salesforce integration?",
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
    from generation import answer_stream
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

    print("\nInitializing hybrid retriever (lazy-loads cross-encoder on first query)...")
    retriever = HybridRetriever(qdrant_client=client)

    print("\n" + "=" * 60)
    print("SAMPLE QUERIES  (retrieval logs → stderr, answer → stdout)")
    print("=" * 60)

    for i, question in enumerate(SAMPLE_QUESTIONS, 1):
        print(f"\n--- Question {i} ---")
        print(f"Q: {question}")
        print()

        chunks = retriever.search(question)

        if not chunks:
            print("⚠  No chunks cleared the similarity threshold.")
            print("   Pipeline returns a structured refusal without calling the LLM.")
            print()

        print("A (streaming): ", end="", flush=True)
        try:
            for token in answer_stream(question, chunks):
                sys.stdout.write(token)
                sys.stdout.flush()
            print()  # final newline after streamed tokens
        except Exception as e:
            print(f"\nError generating answer: {e}")

        if chunks:
            print()
            print(f"  Retrieved {len(chunks)} chunks (re-ranked by cross-encoder):")
            for c in chunks[:3]:
                reranker_info = (
                    f"reranker={c.reranker_score:.2f}" if c.reranker_score is not None else ""
                )
                print(f"    · {c.source}  rrf={c.score:.4f}  {reranker_info}")
        print()

    print("=" * 60)
    print("Demo complete!")
    print()
    print("Other entry points:")
    print("  Offline eval:   python eval.py")
    print("  Retrieval eval: python eval_retrieval.py")
    print("  API server:     uvicorn main:app --reload")
    print("  Streaming curl: curl -X POST http://localhost:8000/query/stream \\")
    print('                    -H "Content-Type: application/json" \\')
    print('                    -d \'{"question": "How do I post a journal?"}\' --no-buffer')
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
