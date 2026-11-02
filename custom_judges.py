"""Custom LLM-as-judge evaluators for machine-consumable output.

Metrics the standard RAGAS set does not capture: extractability (can a
downstream model parse a specific fact out of the response) and refusal
clarity (is non-answer detectable programmatically).
"""
import json
import anthropic

from config import ANTHROPIC_API_KEY

JUDGE_MODEL = "claude-sonnet-4-5"

EXTRACTABILITY_PROMPT = """You are evaluating whether a specific fact can be reliably extracted from an AI response by a downstream program.

Question: {question}
Reference answer: {reference}
AI response: {response}

Can a program parse the reference fact out of the response without ambiguity?
Respond with JSON only: {{"score": 1 or 0, "reason": "one sentence"}}"""

REFUSAL_CLARITY_PROMPT = """You are evaluating whether an AI system communicated "I cannot answer" in a form a calling program can detect.

Question: {question}
AI response: {response}

Is the response an explicit, detectable refusal rather than a soft hedge that reads as an answer?
Respond with JSON only: {{"score": 1 or 0, "reason": "one sentence"}}"""


def _judge(prompt: str) -> dict:
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    msg = client.messages.create(
        model=JUDGE_MODEL, max_tokens=300,
        messages=[{"role": "user", "content": prompt}],
    )
    return json.loads(msg.content[0].text)


def judge_extractability(question: str, reference: str, response: str) -> dict:
    return _judge(EXTRACTABILITY_PROMPT.format(
        question=question, reference=reference, response=response))


def judge_refusal_clarity(question: str, response: str) -> dict:
    return _judge(REFUSAL_CLARITY_PROMPT.format(question=question, response=response))
