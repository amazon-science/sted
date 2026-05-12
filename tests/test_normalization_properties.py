"""Unit tests for STED normalization boundary behavior.

Verifies the paper's Proposition 3.1 claims against the implementation:
  - Disjoint same-size objects score 0 (not 0.5, which would be the case
    under a |C_1|+|C_2| denominator with cost<=1 per match).
  - Identical objects score 1.
  - Partial overlap produces sensible intermediate scores.

These tests guard against regression if the Hungarian-padding normalization
is accidentally changed back to a sum-denominator formulation.
"""
import pytest

from sted.semantic_json_tree_consistency import SemanticJsonTreeConsistencyEvaluator


@pytest.fixture(scope="module")
def evaluator():
    return SemanticJsonTreeConsistencyEvaluator(
        model_id="all-MiniLM-L6-v2", embedding_dim=512
    )


def sted(evaluator, a, b):
    return evaluator.calculate_similarity_method["sted"](a, b, variation_type="combined")


def test_disjoint_single_key_is_zero(evaluator):
    """{"a":1} vs {"z":9}: totally disjoint 1-key objects should score 0."""
    assert sted(evaluator, {"a": 1}, {"z": 9}) < 0.01


def test_disjoint_multi_key_is_zero(evaluator):
    """Larger disjoint objects should also score 0."""
    a = {"a": 1, "b": 2, "c": 3}
    b = {"x": 7, "y": 8, "z": 9}
    assert sted(evaluator, a, b) < 0.01


def test_identical_objects_score_one(evaluator):
    assert sted(evaluator, {"a": 1}, {"a": 1}) == pytest.approx(1.0)


def test_partial_overlap_two_keys(evaluator):
    """One shared key out of two: expected ~0.5 under max normalization."""
    a = {"a": 1, "b": 2}
    b = {"a": 1, "z": 9}
    score = sted(evaluator, a, b)
    assert 0.4 < score < 0.6


def test_empty_vs_empty(evaluator):
    assert sted(evaluator, {}, {}) == pytest.approx(1.0)


def test_synonym_keys_same_value(evaluator):
    """email vs email_address with same value: embedding semantic match
    plus content match should give a clearly-positive score (>0.5)."""
    a = {"email": "foo@bar.com"}
    b = {"email_address": "foo@bar.com"}
    assert sted(evaluator, a, b) > 0.5
