"""Numerical regression test for the v0.2.0 refactor.

Before any code reorganization that touches semantic_json_tree_consistency.py,
we lock in these "golden" STED scores. The refactor MUST produce
bit-identical (or floating-point-identical to 6 decimals) values for
these inputs. If any of these tests fail, the refactor introduced a
behavioral change.

These values were captured with v0.1.1 of the library.
"""
from __future__ import annotations

import pytest

from sted import STED


GOLDEN = {
    "identity_dict": 1.000000,
    "empty_dicts": 1.000000,
    "tool_call_match": 1.000000,
    "array_reorder": 1.000000,
    "type_coercion_bool_str": 0.950000,
    "tool_call_arg_drift": 0.931212,
    "single_string_change": 0.917750,
    "nested_dict": 0.750000,
    "semantic_key_rename": 0.732952,
    "w0_content_only": 0.000000,
    "w1_struct_only": 0.000000,
}


@pytest.fixture(scope="module")
def evaluator():
    return STED()


def _check(actual: float, expected: float, name: str) -> None:
    """Assert match to 4 decimals (CPU-determinism + embedder rounding)."""
    assert abs(actual - expected) < 1e-4, (
        f"refactor regression on {name}: expected {expected:.6f}, got {actual:.6f} "
        f"(diff {abs(actual - expected):.2e})"
    )


def test_identity_dict(evaluator):
    s = evaluator.calculate_tree_edit_distance_fast(
        {"a": 1, "b": 2}, {"a": 1, "b": 2}
    )
    _check(s, GOLDEN["identity_dict"], "identity_dict")


def test_empty_dicts(evaluator):
    s = evaluator.calculate_tree_edit_distance_fast({}, {})
    _check(s, GOLDEN["empty_dicts"], "empty_dicts")


def test_tool_call_match(evaluator):
    s = evaluator.calculate_tree_edit_distance_fast(
        {"tool": "search", "args": {"q": "python"}},
        {"tool": "search", "args": {"q": "python"}},
    )
    _check(s, GOLDEN["tool_call_match"], "tool_call_match")


def test_array_reorder_invariant(evaluator):
    """Default order-invariant mode: reordered arrays should score 1.0."""
    s = evaluator.calculate_tree_edit_distance_fast(
        {"tags": ["a", "b", "c"]}, {"tags": ["c", "a", "b"]}
    )
    _check(s, GOLDEN["array_reorder"], "array_reorder")


def test_type_coercion_bool_str(evaluator):
    s = evaluator.calculate_tree_edit_distance_fast(
        {"active": True}, {"active": "true"}
    )
    _check(s, GOLDEN["type_coercion_bool_str"], "type_coercion_bool_str")


def test_tool_call_arg_drift(evaluator):
    s = evaluator.calculate_tree_edit_distance_fast(
        {"tool": "search", "args": {"q": "python"}},
        {"tool": "search", "args": {"q": "java"}},
    )
    _check(s, GOLDEN["tool_call_arg_drift"], "tool_call_arg_drift")


def test_single_string_change(evaluator):
    s = evaluator.calculate_tree_edit_distance_fast(
        {"name": "John Smith"}, {"name": "Jane Smith"}
    )
    _check(s, GOLDEN["single_string_change"], "single_string_change")


def test_nested_dict_value_change(evaluator):
    s = evaluator.calculate_tree_edit_distance_fast(
        {"user": {"name": "John", "age": 30}},
        {"user": {"name": "John", "age": 31}},
    )
    _check(s, GOLDEN["nested_dict"], "nested_dict")


def test_semantic_key_rename(evaluator):
    s = evaluator.calculate_tree_edit_distance_fast(
        {"email": "x@y.com"}, {"email_address": "x@y.com"}
    )
    _check(s, GOLDEN["semantic_key_rename"], "semantic_key_rename")


def test_w0_content_only(evaluator):
    """w=0 (content-only) projection: identical content, different keys -> 0."""
    s = evaluator.calculate_tree_edit_distance_fast(
        {"a": 1}, {"a": 2}, variation_type="content"
    )
    _check(s, GOLDEN["w0_content_only"], "w0_content_only")


def test_w1_struct_only(evaluator):
    """w=1 (struct-only) projection: same structure, different keys -> 0."""
    s = evaluator.calculate_tree_edit_distance_fast(
        {"a": 1}, {"b": 1}, variation_type="structural"
    )
    _check(s, GOLDEN["w1_struct_only"], "w1_struct_only")
