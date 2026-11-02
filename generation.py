"""Grounded generation with strict citation and explicit refusal."""
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


def is_refusal(text: str) -> bool:
    return text.strip().upper().startswith(REFUSAL_PREFIX)


def build_prompt(query: str, chunks: list[Chunk]) -> str:
    context = "\n\n".join(
        f"[Source: {c.source}]\n{c.content}" for c in chunks
    )
    return f"Context:\n{context}\n\nQuestion: {query}"


def answer(query: str, chunks: list[Chunk], model: str = "claude-sonnet-4-5") -> str:
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    msg = client.messages.create(
        model=model,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": build_prompt(query, chunks)}],
    )
    return msg.content[0].text
