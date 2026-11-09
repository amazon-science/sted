#!/usr/bin/env python3
"""Tests for the key-identity blend (``sted._key_identity``) and its intervals.

Same three obligations as ``tests/test_array_order.py``, for the second interpretive axis:

1. **μ=0 reproduces the published metric exactly**, and **μ=1 reproduces
   ``exact_match_all_keys``** — the blend has to be an extension between two behaviours the
   library already had, not a third one.
2. **The bracketing lemma.** ``s_1 <= s_μ <= s_0``, which holds because the exact-match term
   never exceeds the embedding term (identical keys short-circuit to 1.0; different keys give
   0 against a kernel truncated at 0).
3. **A consistent renaming is no longer certified.** A high ``s_label`` with a much lower
   ``s_identifier`` is the point: the metric reports that its verdict depends on whether
   anything downstream declared those keys.

Plus one hazard specific to μ: unlike λ it is a single evaluator-wide constant, so it is
deliberately *not* in the subtree cache key — which makes cache invalidation on assignment
load-bearing.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from sted._key_identity import field_similarity, resolve_mu               # noqa: E402
from sted.semantic_json_tree_consistency import (                         # noqa: E402
    SemanticJsonTreeConsistencyEvaluator)

MODEL = 'all-MiniLM-L6-v2'

_EVALUATORS = {}


def _ev(**kwargs):
    """One evaluator per distinct configuration — each construction loads the model."""
    kwargs.setdefault("profile", "legacy")
    key = tuple(sorted(kwargs.items()))
    if key not in _EVALUATORS:
        _EVALUATORS[key] = SemanticJsonTreeConsistencyEvaluator(model_id=MODEL, **kwargs)
    return _EVALUATORS[key]


# ------------------------------------------------------------------ the blend in isolation

def test_field_similarity_endpoints_and_monotonicity():
    assert field_similarity(0.8, 0.0, 0.0) == 0.8
    assert field_similarity(0.8, 0.0, 1.0) == 0.0
    assert field_similarity(0.8, 0.0, 0.5) == pytest.approx(0.4)
    vals = [field_similarity(0.8, 0.0, i / 10) for i in range(11)]
    assert vals == sorted(vals, reverse=True)
    # Identical keys: both readings agree, so μ cannot move the score at all.
    assert all(field_similarity(1.0, 1.0, i / 10) == 1.0 for i in range(11))


def test_resolve_mu_forms():
    assert resolve_mu(None) == 0.0
    assert resolve_mu(None, default=0.4) == pytest.approx(0.4)
    assert resolve_mu(0.3) == pytest.approx(0.3)
    assert resolve_mu(5.0) == 1.0                      # clamped
    assert resolve_mu(-2.0) == 0.0                     # clamped
    # 'auto' exists for λ but must not silently do something here: whether keys are declared
    # elsewhere is not derivable from the pair.
    with pytest.raises(ValueError):
        resolve_mu('auto')
    # bool is an int subclass; True must not be accepted as μ=1 by accident.
    with pytest.raises(ValueError):
        resolve_mu(True)


def test_invalid_mu_is_rejected_at_assignment_not_at_scoring():
    ev = _ev()
    with pytest.raises(ValueError):
        ev.key_identity_mu = 'auto'
    assert ev.key_identity_mu is None                  # and the old value survives


# ------------------------------------------------------------------- end-to-end behaviour

RENAME = ({'name': 'f', 'arguments': {'n': 20, 'rate': 0.5}},
          {'name': 'f', 'arguments': {'n_value': 20, 'rate_value': 0.5}})

PAIRS = [
    RENAME,
    ({'name': 'f', 'arguments': {'n': 20}}, {'name': 'f', 'arguments': {'n': 20}}),
    ({'name': 'f', 'arguments': {'n': 20}}, {'name': 'f', 'arguments': {'n': 99}}),
    ({'a': {'b': [1, 2], 'c': 'hello world'}},
     {'a': {'b_val': [1, 2], 'c_val': 'hello world'}}),
    ({'name': 'f', 'arguments': {'xs': [1, 2, 3]}},
     {'name': 'g', 'arguments': {'ys': [3, 2, 1], 'extra': True}}),
]


@pytest.mark.parametrize('cost_combination', ['additive', 'product'])
@pytest.mark.parametrize('variation_type', ['combined', 'structural', 'content'])
def test_mu_zero_reproduces_the_published_metric(cost_combination, variation_type):
    """Exact equality, not approximate: μ=0 must be the same computation."""
    published = _ev(cost_combination=cost_combination)
    at_zero = _ev(cost_combination=cost_combination, key_identity_mu=0.0)
    for a, b in PAIRS:
        assert (published.calculate_tree_edit_distance_fast(a, b, variation_type)
                == at_zero.calculate_tree_edit_distance_fast(a, b, variation_type))


@pytest.mark.parametrize('variation_type', ['combined', 'structural', 'content'])
def test_mu_one_reproduces_exact_match_all_keys(variation_type):
    """μ=1 is not a new behaviour — it is the flag the library already had."""
    at_one = _ev(key_identity_mu=1.0)
    flag = _ev(exact_match_all_keys=True)
    for a, b in PAIRS:
        assert (at_one.calculate_tree_edit_distance_fast(a, b, variation_type)
                == pytest.approx(
                    flag.calculate_tree_edit_distance_fast(a, b, variation_type), abs=1e-12))


@pytest.mark.parametrize('cost_combination', ['additive', 'product'])
@pytest.mark.parametrize('variation_type', ['combined', 'structural', 'content'])
def test_similarity_is_monotone_in_mu(cost_combination, variation_type):
    mus = (0.0, 0.25, 0.5, 0.75, 1.0)
    evs = {m: _ev(cost_combination=cost_combination, key_identity_mu=m) for m in mus}
    for a, b in PAIRS:
        scores = [evs[m].calculate_tree_edit_distance_fast(a, b, variation_type) for m in mus]
        for earlier, later in zip(scores, scores[1:]):
            assert later <= earlier + 1e-9, (scores, a, b)


def test_interval_brackets_and_flags_a_consistent_rename():
    ev = _ev()
    s_ident, s_label = ev.compute_key_identity_interval(*RENAME)
    # Today's STED treats k -> k_value as near-synonymous, so the label end stays high...
    assert s_label > 0.75
    # ...and the interval is wide, so the score is reported as reading-dependent rather than
    # certifying a call the callee would reject.
    assert s_ident < s_label

    identical = ({'name': 'f', 'arguments': {'n': 20}}, {'name': 'f', 'arguments': {'n': 20}})
    assert ev.compute_key_identity_interval(*identical) == (1.0, 1.0)

    # Keys already agree, only a value differs: nothing rests on the label reading, so the
    # interval collapses onto the published score.
    values_only = ({'name': 'f', 'arguments': {'n': 20}},
                   {'name': 'f', 'arguments': {'n': 99}})
    s_ident, s_label = ev.compute_key_identity_interval(*values_only)
    assert s_ident == s_label == ev.calculate_tree_edit_distance_fast(*values_only)


def test_interval_leaves_the_evaluator_untouched():
    ev = _ev(key_identity_mu=0.5)
    ev.compute_key_identity_interval(*RENAME)
    assert ev.key_identity_mu == 0.5


def test_assumption_interval_is_the_corner_to_corner_bracket():
    """Both axes are monotone and touch disjoint terms, so (1,1) and (0,0) are the extremes."""
    a = {'name': 'f', 'arguments': {'xs': [1, 2, 3]}}
    b = {'name': 'f', 'arguments': {'xs_value': [3, 1, 2]}}
    ev = _ev()
    s_min, s_max = ev.compute_assumption_interval(a, b)
    assert s_min <= s_max
    # The published score is the (λ=0, μ=0) corner, i.e. the upper end.
    assert s_max == pytest.approx(ev.calculate_tree_edit_distance_fast(a, b), abs=1e-12)
    for lam, mu in ((0.0, 1.0), (1.0, 0.0)):
        mixed = _ev(array_order_lambda=lam, key_identity_mu=mu)
        s = mixed.calculate_tree_edit_distance_fast(a, b)
        assert s_min - 1e-9 <= s <= s_max + 1e-9, (lam, mu, s, s_min, s_max)
    # A permuted *and* renamed array is exactly the case where both assumptions are load-
    # bearing, so the interval must be wide rather than a point.
    assert s_max - s_min > 0.1


def test_changing_mu_invalidates_the_subtree_cache():
    """μ is not in the cache key, so the setter has to clear the cache.

    Without that, scoring at μ=0 and then at μ=1 on the same trees returns the μ=0 costs: the
    key is (tree1_hash, tree2_hash, variation_type) and no part of it can see μ.
    """
    a, b = RENAME
    cached = SemanticJsonTreeConsistencyEvaluator(
        model_id=MODEL, profile="legacy", subtree_cache_size=10000)
    cached.calculate_tree_edit_distance_fast(a, b)          # populate under the label reading
    cached.key_identity_mu = 1.0
    after = cached.calculate_tree_edit_distance_fast(a, b)
    fresh = _ev(key_identity_mu=1.0).calculate_tree_edit_distance_fast(a, b)
    assert after == pytest.approx(fresh, abs=1e-12)

    # Re-assigning the same value must not clear anything (it would make the cache useless
    # for callers that set μ once per pair), so only the *change* invalidates.
    cached.clear_subtree_cache()
    cached.calculate_tree_edit_distance_fast(a, b)
    before_hits = cached.get_cache_stats()['subtree_cache_hits']
    cached.key_identity_mu = 1.0
    cached.calculate_tree_edit_distance_fast(a, b)
    assert cached.get_cache_stats()['subtree_cache_hits'] > before_hits
