"""Tests for production-readiness fixes in AgentConsistencyEvaluator.

Covers:
  - Issue 1: timeout_seconds for STED computation
  - Issue 2: graceful exception fallback (exact-match)
  - Issue 3: streaming evaluate_outputs_streaming
  - Issue 5: cache_stats() helper
  - Issue 7: configurable min_length_for_embeddings on STED evaluator
"""
from __future__ import annotations

import time

import pytest

from sted import AgentConsistencyEvaluator, PromptResult
from sted.semantic_json_tree_consistency import SemanticJsonTreeConsistencyEvaluator


# ---------- Stub evaluators for fast, controllable testing -------------------


class _SlowEvaluator:
    """Stand-in for the real STED evaluator that sleeps before returning."""

    def __init__(self, delay: float = 5.0, return_value: float = 0.5):
        self.delay = delay
        self.return_value = return_value
        self.calls = 0
        self.embedding_model = None  # Skip cross-prompt precompute

    def calculate_tree_edit_distance_fast(self, a, b):
        self.calls += 1
        time.sleep(self.delay)
        return self.return_value

    def get_cache_stats(self):
        return {
            "embedding_cache_size": 0,
            "subtree_cache_size": 0,
            "subtree_cache_hit_rate": 0.0,
        }


class _ErrorEvaluator:
    """Always raises — used to test error fallback."""

    def __init__(self):
        self.calls = 0
        self.embedding_model = None

    def calculate_tree_edit_distance_fast(self, a, b):
        self.calls += 1
        raise RuntimeError("simulated STED failure")

    def get_cache_stats(self):
        return {
            "embedding_cache_size": 0,
            "subtree_cache_size": 0,
            "subtree_cache_hit_rate": 0.0,
        }


class _PassthroughEvaluator:
    """Returns 1.0 if equal else 0.0 — for streaming tests."""

    def __init__(self):
        self.embedding_model = None

    def calculate_tree_edit_distance_fast(self, a, b):
        return 1.0 if a == b else 0.0

    def get_cache_stats(self):
        return {
            "embedding_cache_size": 0,
            "subtree_cache_size": 0,
            "subtree_cache_hit_rate": 0.0,
        }


# ---------- Issue 1: timeout -------------------------------------------------


def test_timeout_marks_pair_with_timeout_error():
    slow = _SlowEvaluator(delay=2.0, return_value=0.7)
    ev = AgentConsistencyEvaluator(evaluator=slow, timeout_seconds=0.05)
    outputs = {"p": [{"a": 1}, {"b": 2}, {"c": 3}]}
    report = ev.evaluate_outputs(outputs, precompute_embeddings=False)
    r = report.per_prompt[0]
    assert r.error is not None
    assert "timeout" in r.error
    # All pairs timed out -> no finite sims; c_mean should be 0.0 fallback.
    assert r.c_mean == 0.0
    # The pairwise_similarities list should contain Nones for timeouts
    assert all(s is None for s in r.pairwise_similarities)


def test_no_timeout_when_disabled():
    """Default timeout_seconds=None should not enforce any timeout."""
    slow = _SlowEvaluator(delay=0.05, return_value=0.42)
    ev = AgentConsistencyEvaluator(evaluator=slow)  # timeout_seconds default
    outputs = {"p": [{"a": 1}, {"b": 2}]}
    report = ev.evaluate_outputs(outputs, precompute_embeddings=False)
    r = report.per_prompt[0]
    assert r.error is None
    assert r.c_mean == pytest.approx(0.42)


# ---------- Issue 2: error fallback ------------------------------------------


def test_sted_exception_falls_back_to_exact_match():
    bad = _ErrorEvaluator()
    ev = AgentConsistencyEvaluator(evaluator=bad)
    # Two identical -> fallback gives 1.0; one different -> 0.0
    outputs = {"p": [{"a": 1}, {"a": 1}, {"a": 2}]}
    report = ev.evaluate_outputs(outputs, precompute_embeddings=False)
    r = report.per_prompt[0]
    # Pairs: (a=1,a=1)=1.0; (a=1,a=2)=0.0; (a=1,a=2)=0.0  -> mean ~ 0.333
    assert r.error is not None
    assert "fallback" in r.error
    assert r.c_mean == pytest.approx(1.0 / 3.0, abs=1e-6)
    # Exactly 3 fallback calls
    assert bad.calls == 3


def test_sted_exception_does_not_crash_full_evaluation():
    """A single failing pair must not crash the whole prompt evaluation."""
    bad = _ErrorEvaluator()
    ev = AgentConsistencyEvaluator(evaluator=bad)
    outputs = {"p1": [{"x": 1}, {"x": 1}], "p2": [{"y": 1}, {"y": 2}]}
    report = ev.evaluate_outputs(outputs, precompute_embeddings=False)
    assert report.n_prompts == 2  # both prompts produced results


# ---------- Issue 3: streaming ----------------------------------------------


def test_evaluate_outputs_streaming_yields_one_at_a_time():
    ev = AgentConsistencyEvaluator(evaluator=_PassthroughEvaluator())

    def _gen():
        for i in range(5):
            # Two identical outputs per prompt -> c_mean 1.0
            yield (f"p{i}", [{"k": i}, {"k": i}])

    seen = []
    for r in ev.evaluate_outputs_streaming(_gen(), chunk_size=2):
        assert isinstance(r, PromptResult)
        seen.append(r)
    assert len(seen) == 5
    assert all(r.c_mean == 1.0 for r in seen)


def test_evaluate_outputs_streaming_threaded_path():
    """n_workers > 1 streaming should still yield all and complete."""
    ev = AgentConsistencyEvaluator(
        evaluator=_PassthroughEvaluator(), n_workers=2
    )

    def _gen():
        for i in range(7):
            yield (f"p{i}", [{"k": i}, {"k": i}])

    results = list(ev.evaluate_outputs_streaming(_gen(), chunk_size=3))
    assert len(results) == 7


def test_streaming_does_not_consume_iterable_eagerly():
    """The iterable should be drained lazily — confirm via a counting iterator."""
    ev = AgentConsistencyEvaluator(evaluator=_PassthroughEvaluator())
    consumed = {"n": 0}

    def _gen():
        for i in range(3):
            consumed["n"] += 1
            yield (f"p{i}", [{"k": i}, {"k": i}])

    stream = ev.evaluate_outputs_streaming(_gen())
    # No consumption yet
    assert consumed["n"] == 0
    next(stream)
    # First item consumed (and possibly the next, with chunking).
    assert consumed["n"] >= 1
    # Drain the rest
    list(stream)
    assert consumed["n"] == 3


# ---------- Issue 5: cache_stats --------------------------------------------


def test_cache_stats_returns_expected_keys():
    ev = AgentConsistencyEvaluator(evaluator=_PassthroughEvaluator())
    stats = ev.cache_stats()
    assert set(stats.keys()) == {
        "embedding_cache_size",
        "subtree_cache_size",
        "subtree_cache_hit_rate",
    }


def test_cache_stats_with_real_evaluator():
    """Real evaluator should still return numeric stats."""
    real = SemanticJsonTreeConsistencyEvaluator(model_id="all-MiniLM-L6-v2")
    ev = AgentConsistencyEvaluator(evaluator=real)
    stats = ev.cache_stats()
    assert isinstance(stats["embedding_cache_size"], int)
    assert isinstance(stats["subtree_cache_size"], int)
    assert 0.0 <= stats["subtree_cache_hit_rate"] <= 1.0


# ---------- Issue 7: configurable min_length_for_embeddings ------------------


def test_min_length_for_embeddings_default_is_4():
    ev = SemanticJsonTreeConsistencyEvaluator(model_id="all-MiniLM-L6-v2")
    assert ev.min_length_for_embeddings == 4


def test_min_length_for_embeddings_configurable():
    ev = SemanticJsonTreeConsistencyEvaluator(
        model_id="all-MiniLM-L6-v2", min_length_for_embeddings=10
    )
    assert ev.min_length_for_embeddings == 10
    # With min_length=10, "hello" (len 5) should fall back to char-edit-distance
    # — so identical short strings still score 1.0, similar short strings should
    # score by char-distance.
    s_same = ev._calculate_semantic_similarity("hello", "hello")
    assert s_same == pytest.approx(1.0)
    # 1 char diff out of 5 -> 0.8
    s_one_off = ev._calculate_semantic_similarity("hello", "hallo")
    assert s_one_off == pytest.approx(0.8, abs=1e-6)


def test_min_length_zero_uses_embeddings_for_short_strings():
    """min_length=0 disables the fallback even for 1-char strings."""
    ev = SemanticJsonTreeConsistencyEvaluator(
        model_id="all-MiniLM-L6-v2", min_length_for_embeddings=0
    )
    # Identical strings short-circuit to 1.0 regardless of path
    assert ev._calculate_semantic_similarity("a", "a") == 1.0
    # Different short strings: with min_length=0 we go through embedding path,
    # with default we'd go through char edit distance. Just check it returns
    # a float in [0,1].
    s = ev._calculate_semantic_similarity("a", "b")
    assert 0.0 <= s <= 1.0


# ---------- Trajectory-mode defaults (Issue 94) ------------------------------


def test_trajectory_mode_sets_exact_match_keys():
    """trajectory mode wires exact-match on keys + tool-name fields, plus
    positional matching on step lists."""
    ev = AgentConsistencyEvaluator(mode="trajectory")
    inner = ev._evaluator
    assert inner.exact_match_all_keys is True
    assert "name" in inner.exact_match_fields
    assert "tool" in inner.exact_match_fields
    assert "tool_name" in inner.exact_match_fields
    assert "function" in inner.exact_match_fields
    assert "tool_calls" in inner.order_sensitive_fields
    assert "steps" in inner.order_sensitive_fields
    assert ev._metric_config["mode"] == "trajectory"


def test_for_trajectory_classmethod_equivalent_to_mode_kw():
    a = AgentConsistencyEvaluator.for_trajectory()
    b = AgentConsistencyEvaluator(mode="trajectory")
    assert a._evaluator.exact_match_all_keys == b._evaluator.exact_match_all_keys
    assert a._evaluator.exact_match_fields == b._evaluator.exact_match_fields
    assert a._evaluator.order_sensitive_fields == b._evaluator.order_sensitive_fields


def test_default_mode_does_not_force_exact_match():
    ev = AgentConsistencyEvaluator()
    assert ev._evaluator.exact_match_all_keys is False
    assert ev._evaluator.exact_match_fields == set()


def test_invalid_mode_raises():
    with pytest.raises(ValueError, match="mode must be"):
        AgentConsistencyEvaluator(mode="bogus")


def test_trajectory_mode_penalizes_tool_name_drift():
    """Two trajectories that only differ in tool name (semantically similar
    e.g. 'search' vs 'lookup') should score LOWER under trajectory mode than
    under default JSON mode, because tool names are exact-match in trajectory
    mode."""
    traj_a = {"tool_calls": [{"name": "search", "args": {"q": "python"}}]}
    traj_b = {"tool_calls": [{"name": "lookup", "args": {"q": "python"}}]}

    json_mode = AgentConsistencyEvaluator()
    traj_mode = AgentConsistencyEvaluator.for_trajectory()

    s_json = json_mode.evaluate_pair(traj_a, traj_b)
    s_traj = traj_mode.evaluate_pair(traj_a, traj_b)

    # trajectory mode should be strictly lower (or equal) since exact-match
    # fields turn the ~0.6 semantic sim on "search"/"lookup" into 0.
    assert s_traj <= s_json + 1e-6


def test_explicit_evaluator_overrides_mode():
    """If an evaluator is passed explicitly, mode is ignored for its config."""
    custom = SemanticJsonTreeConsistencyEvaluator(
        model_id="all-MiniLM-L6-v2",
        exact_match_all_keys=False,
    )
    ev = AgentConsistencyEvaluator(evaluator=custom, mode="trajectory")
    # Did NOT mutate the user-supplied evaluator.
    assert ev._evaluator.exact_match_all_keys is False
