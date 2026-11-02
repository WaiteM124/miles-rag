from app.retrieval import weighted_rrf
from app.generation import is_refusal


def test_rrf_fusion_orders_by_weighted_reciprocal_rank():
    bm25 = ["a", "b", "c"]
    dense = ["b", "a", "d"]
    fused = weighted_rrf([bm25, dense], weights=[0.4, 0.6], k=60)
    # "a" and "b" appear in both lists, so they outrank single-list members
    assert set(fused) == {"a", "b", "c", "d"}
    top_two = sorted(fused, key=fused.get, reverse=True)[:2]
    assert set(top_two) == {"a", "b"}


def test_rrf_weights_bias_ranking():
    only_bm25 = weighted_rrf([["x", "y"], []][:1], weights=[1.0], k=60)
    only_dense = weighted_rrf([[], ["x", "y"]][1:], weights=[1.0], k=60)
    assert only_bm25["x"] == only_dense["x"]  # rank dominates over source list


def test_refusal_detection():
    assert is_refusal("REFUSAL: NOT IN DOCUMENTATION")
    assert is_refusal("refusal: ambiguous question")
    assert not is_refusal("To post a journal, navigate to...")
