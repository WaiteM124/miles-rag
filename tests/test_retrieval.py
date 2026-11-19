"""Unit tests for retrieval module."""

from generation import is_refusal
from retrieval import tokenize, weighted_rrf


def test_rrf_fusion_orders_by_weighted_reciprocal_rank():
    bm25 = ["a", "b", "c"]
    dense = ["b", "a", "d"]
    fused = weighted_rrf([bm25, dense], weights=[0.4, 0.6], k=60)
    assert set(fused) == {"a", "b", "c", "d"}
    top_two = sorted(fused, key=fused.get, reverse=True)[:2]
    assert set(top_two) == {"a", "b"}


def test_rrf_weights_bias_ranking():
    only_bm25 = weighted_rrf([["x", "y"]], weights=[1.0], k=60)
    only_dense = weighted_rrf([["x", "y"]], weights=[1.0], k=60)
    assert only_bm25["x"] == only_dense["x"]


def test_rrf_empty_lists():
    fused = weighted_rrf([[], []], weights=[0.5, 0.5], k=60)
    assert fused == {}


def test_rrf_single_item():
    fused = weighted_rrf([["only"]], weights=[1.0], k=60)
    assert "only" in fused
    assert fused["only"] == 1.0 / (60 + 1)


def test_refusal_detection():
    assert is_refusal("REFUSAL: NOT IN DOCUMENTATION")
    assert is_refusal("refusal: ambiguous question")
    assert is_refusal("  REFUSAL: some reason  ")
    assert not is_refusal("To post a journal, navigate to...")
    assert not is_refusal("The answer is...")
    assert not is_refusal("")


def test_tokenize_basic():
    tokens = tokenize("Hello, World! This is a test.")
    assert tokens == ["hello", "world", "this", "is", "a", "test"]


def test_tokenize_handles_punctuation():
    tokens = tokenize("user's input (with parentheses) and hyphen-word")
    assert "user" in tokens
    assert "s" in tokens
    assert "input" in tokens
    assert "parentheses" in tokens
    assert "hyphen" in tokens
    assert "word" in tokens


def test_tokenize_empty():
    assert tokenize("") == []
    assert tokenize("   ") == []


def test_tokenize_unicode():
    tokens = tokenize("café naïve résumé")
    assert "café" in tokens
    assert "naïve" in tokens
    assert "résumé" in tokens
