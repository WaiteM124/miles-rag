"""Grounded generation with strict citation, explicit refusal, and streaming support."""

from __future__ import annotations

from collections.abc import Iterator

import anthropic

from config import ANTHROPIC_API_KEY
from retrieval import Chunk

SYSTEM_PROMPT = """You are a knowledge assistant grounded exclusively in the provided context.
Rules:
1. Answer only from the context below. If the answer is not there, respond exactly with: REFUSAL: NOT IN DOCUMENTATION
2. Cite the source document for every claim.
3. If the question is ambiguous or too broad to answer precisely, respond exactly with: REFUSAL: AMBIGUOUS QUESTION
4. Never invent identifiers, amounts, or steps.
"""

REFUSAL_PREFIX = "REFUSAL:"

# Returned verbatim when the retriever finds no chunks above the similarity
# threshold.  Starts with REFUSAL: so is_refusal() catches it automatically.
NO_CONTEXT_RESPONSE = (
    "REFUSAL: NO RELEVANT CONTEXT FOUND — the query did not match any document "
    "chunks above the minimum similarity threshold. Please rephrase your question "
    "or verify that the relevant documents have been ingested."
)

_client: anthropic.Anthropic | None = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    return _client


def is_refusal(text: str) -> bool:
    return text.strip().upper().startswith(REFUSAL_PREFIX)


def build_prompt(query: str, chunks: list[Chunk]) -> str:
    context = "\n\n".join(f"[Source: {c.source}]\n{c.content}" for c in chunks)
    return f"Context:\n{context}\n\nQuestion: {query}"


def answer(query: str, chunks: list[Chunk], model: str = "claude-sonnet-4-5") -> str:
    """Generate a grounded answer.

    Returns ``NO_CONTEXT_RESPONSE`` immediately (without calling the LLM) when
    *chunks* is empty — this is the structured failure-mode response emitted
    when no retrieved chunks cleared the similarity threshold.
    """
    if not chunks:
        return NO_CONTEXT_RESPONSE

    client = _get_client()
    msg = client.messages.create(
        model=model,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": build_prompt(query, chunks)}],
    )
    return msg.content[0].text


def answer_stream(
    query: str, chunks: list[Chunk], model: str = "claude-sonnet-4-5"
) -> Iterator[str]:
    """Stream the answer token-by-token using the Anthropic streaming API.

    Yields string tokens as they arrive from the model.  When *chunks* is
    empty the full ``NO_CONTEXT_RESPONSE`` string is yielded in a single
    iteration without making any API call.

    Example usage::

        for token in answer_stream(query, chunks):
            sys.stdout.write(token)
            sys.stdout.flush()
        print()  # final newline
    """
    if not chunks:
        yield NO_CONTEXT_RESPONSE
        return

    client = _get_client()
    with client.messages.stream(
        model=model,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": build_prompt(query, chunks)}],
    ) as stream:
        yield from stream.text_stream
