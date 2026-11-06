"""Behavioral regression checks for JSON distinctions lost by path flattening."""
import pytest

from sted.semantic_json_tree_consistency import SemanticJsonTreeConsistencyEvaluator as EV
from sted._paths import EmptyContainer


@pytest.fixture
def evaluator():
    # These comparisons never need an encoder; isolate correspondence and types.
    ev = EV.__new__(EV)
    ev.align_calls_by_name = False
    ev.exact_name_fields = frozenset()
    return ev


@pytest.mark.parametrize("a,b", [
    ({"a.b": 1}, {"a": {"b": 1}}),
    ({"a[0]": 1}, {"a": [1]}),
    ({"0": 1}, [1]),
    ({"": {"x": 1}}, {"x": 1}),
    ({"(root)": None}, None),
    ({}, []),
    ({"x": []}, {"x": {}}),
    ({"x": []}, {}),
    ({"x": []}, {"x": None}),
    ({"x": True}, {"x": 1}),
    ({"x": False}, {"x": 0}),
])
def test_distinct_json_does_not_get_full_credit(evaluator, a, b):
    assert evaluator.calculate_path_f1_similarity(a, b) == 0
    assert evaluator.calculate_path_f1_similarity(b, a) == 0


@pytest.mark.parametrize("a", [{}, [], {"": []}, {"x": None}, {"x": True},
                               {"a.b": [], "a": {"b": {}}}])
def test_self_agreement(evaluator, a):
    assert evaluator.calculate_path_f1_similarity(a, a) == 1


def test_colliding_serializations_retain_both_fields():
    paths = EV._leaf_paths({"a.b": None, "a": {"b": True}})
    assert len(paths) == 2
    assert paths[(("key", "a.b"),)] is None
    assert paths[(("key", "a"), ("key", "b"))] is True


def test_added_empty_field_receives_missing_field_penalty(evaluator):
    assert evaluator.calculate_path_f1_similarity(
        {"keep": True}, {"keep": True, "extra": []}) == pytest.approx(2 / 3)


def test_object_order_invariance(evaluator):
    assert evaluator.calculate_path_f1_similarity(
        {"a": [], "b": True}, {"b": True, "a": []}) == 1


def test_alignment_holes_are_distinct_from_empty_containers(evaluator):
    p = evaluator._leaf_paths_skipping_holes([evaluator._UNMATCHED, []])
    assert p == {(("index", 1),): EmptyContainer.ARRAY}


def test_exact_callee_paths_survive_delimiter_keys(evaluator):
    evaluator.exact_name_fields = frozenset({"name"})
    obj = {"a.b": {"name": "f", "arguments": {}},
           "a": {"b": {"name": "person"}}}
    assert evaluator._callee_name_paths(obj) == {(("key", "a.b"), ("key", "name"))}
