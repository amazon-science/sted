#!/usr/bin/env python3
"""Tests for the array-order blend (``sted._order``) and the similarity interval.

Three things need pinning here, in decreasing order of how badly a regression would hurt:

1. **λ=0 reproduces the published metric exactly.** Every number in the papers was computed
   with the blend absent. If turning the machinery on at λ=0 moved a score by even 1e-12, the
   whole feature would be a silent re-definition of STED rather than an extension of it.
2. **The bracketing lemma.** ``s_1 <= s_λ <= s_0`` is what makes the interval meaningful;
   it follows from the monotone matchings being a subset of all matchings, and the DP is
   checked against brute-force enumeration so that "subset" claim is actually true of the
   implementation.
3. **A permutation is no longer certified.** ``s_0 == 1.0`` with ``s_1 < 1.0`` is the point
   of the exercise: the metric reports that its verdict rests on the order assumption
   instead of returning an unfalsifiable 1.0.
"""

import itertools
import random
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from sted._order import (blend, monotone_alignment, monotone_alignment_cost,  # noqa: E402
                         order_lambda_arity_coupling, resolve_lambda)
from sted.json_tree_node import JsonNode                                      # noqa: E402
from sted.semantic_json_tree_consistency import (                             # noqa: E402
    SemanticJsonTreeConsistencyEvaluator)

MODEL = 'all-MiniLM-L6-v2'


# --------------------------------------------------------------------- the DP in isolation

def _brute_force_monotone(pair_costs, del_costs, ins_costs):
    """Cheapest order-preserving matching of exactly min(n1, n2) pairs, by enumeration."""
    n1, n2 = len(del_costs), len(ins_costs)
    if n1 == 0 or n2 == 0:
        return float(sum(del_costs) + sum(ins_costs))
    k = min(n1, n2)
    best = float('inf')
    for rows in itertools.combinations(range(n1), k):
        for cols in itertools.combinations(range(n2), k):
            cost = sum(pair_costs[i, j] for i, j in zip(rows, cols))
            cost += sum(c for i, c in enumerate(del_costs) if i not in rows)
            cost += sum(c for j, c in enumerate(ins_costs) if j not in cols)
            best = min(best, cost)
    return best


def _brute_force_free(pair_costs, del_costs, ins_costs):
    """Same, but over ALL matchings — the assignment's feasible set."""
    n1, n2 = len(del_costs), len(ins_costs)
    if n1 == 0 or n2 == 0:
        return float(sum(del_costs) + sum(ins_costs))
    k = min(n1, n2)
    best = float('inf')
    for rows in itertools.permutations(range(n1), k):
        for cols in itertools.permutations(range(n2), k):
            cost = sum(pair_costs[i, j] for i, j in zip(rows, cols))
            cost += sum(c for i, c in enumerate(del_costs) if i not in rows)
            cost += sum(c for j, c in enumerate(ins_costs) if j not in cols)
            best = min(best, cost)
    return best


def test_monotone_alignment_matches_brute_force_and_is_bracketed():
    """The DP is exact, and its optimum is never below the order-free optimum.

    Randomised over every shape up to 4x4 including the empty cases. The second assertion is
    the bracketing lemma at the level of a single array pair: because both sides sum the same
    per-pair costs and the same indel charges, ``free <= mono`` must hold elementwise, and
    that is what lets the evaluator claim ``s_1 <= s_0``.
    """
    rng = random.Random(0)
    for _ in range(400):
        n1, n2 = rng.randint(0, 4), rng.randint(0, 4)
        pair = np.array([[round(rng.random(), 3) for _ in range(n2)]
                         for _ in range(n1)], dtype=np.float64).reshape(n1, n2)
        dels = [round(rng.random(), 3) for _ in range(n1)]
        inss = [round(rng.random(), 3) for _ in range(n2)]

        total, pairs, skip1, skip2 = monotone_alignment(pair, dels, inss)
        assert total == pytest.approx(_brute_force_monotone(pair, dels, inss), abs=1e-9)
        assert _brute_force_free(pair, dels, inss) <= total + 1e-9

        if n1 and n2:
            assert len(pairs) == min(n1, n2)          # pair count pinned: only skips differ
            assert pairs == sorted(pairs)             # monotone on both sides
            assert len(skip1) == n1 - len(pairs)
            assert len(skip2) == n2 - len(pairs)
            recon = (sum(pair[i, j] for i, j in pairs)
                     + sum(dels[i] for i in skip1) + sum(inss[j] for j in skip2))
            assert recon == pytest.approx(total, abs=1e-9)  # returned path explains the total


def test_monotone_alignment_penalises_a_swap_the_assignment_forgives():
    swapped = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert monotone_alignment_cost(swapped, [1.0, 1.0], [1.0, 1.0]) == 2.0
    aligned = np.array([[0.0, 1.0], [1.0, 0.0]])
    assert monotone_alignment_cost(aligned, [1.0, 1.0], [1.0, 1.0]) == 0.0


def test_monotone_alignment_degenerate_shapes():
    assert monotone_alignment(np.zeros((0, 0)), [], []) == (0.0, [], [], [])
    assert monotone_alignment(np.zeros((0, 2)), [], [1.0, 2.0]) == (3.0, [], [], [0, 1])
    assert monotone_alignment(np.zeros((2, 0)), [1.0, 2.0], []) == (3.0, [], [0, 1], [])


def test_blend_endpoints_and_monotonicity():
    assert blend(0.2, 0.8, 0.0) == 0.2
    assert blend(0.2, 0.8, 1.0) == 0.8
    assert blend(0.2, 0.8, 0.5) == pytest.approx(0.5)
    lams = [i / 10 for i in range(11)]
    vals = [blend(0.2, 0.8, x) for x in lams]
    assert vals == sorted(vals)


# ------------------------------------------------------------------- λ resolution

def _tree(obj):
    return JsonNode.from_dict(obj, sort_arrays=False)


def test_arity_coupling_fires_only_on_equal_length_sibling_arrays():
    # x and y are the same length -> read elementwise -> ordered
    coupled = _tree({'x': [1, 2, 3], 'y': [4, 5, 6]}).children
    assert order_lambda_arity_coupling(coupled[0], coupled[0]) == 1.0
    # different lengths -> no positional correspondence to break
    uncoupled = _tree({'x': [1, 2, 3], 'y': [4, 5]}).children
    assert order_lambda_arity_coupling(uncoupled[0], uncoupled[0]) == 0.0
    # length 1 is excluded: a singleton has only one order
    single = _tree({'x': [1], 'y': [2]}).children
    assert order_lambda_arity_coupling(single[0], single[0]) == 0.0


def test_arity_coupling_degrades_to_default_without_parent_links():
    orphan = JsonNode('xs', node_type='array')
    for v in (1, 2):
        orphan.children.append(JsonNode('xs[0]', v))   # bypass add_child -> no parent set
    assert order_lambda_arity_coupling(orphan, orphan, default=0.3) == 0.3


def test_resolve_lambda_forms():
    n = _tree({'x': [1, 2], 'y': [3, 4]}).children[0]
    assert resolve_lambda(None, n, n, default=0.7) == 0.7
    assert resolve_lambda(0.4, n, n) == pytest.approx(0.4)
    assert resolve_lambda(5.0, n, n) == 1.0            # clamped
    assert resolve_lambda(-1.0, n, n) == 0.0           # clamped
    assert resolve_lambda('auto', n, n) == 1.0
    assert resolve_lambda(lambda a, b: 0.25, n, n) == pytest.approx(0.25)
    with pytest.raises(ValueError):
        resolve_lambda('sometimes', n, n)


def test_requesting_a_blend_disables_array_sorting():
    """Sorting would destroy the order before the alignment could read it, so it is off.

    Without this the feature would silently do nothing: ``from_dict`` would hand the DP two
    already-canonicalised sequences and ``c_ord`` would equal ``c_free`` on every pair.
    """
    off = SemanticJsonTreeConsistencyEvaluator(model_id=MODEL, profile="legacy")
    assert off.sort_arrays is True
    on = SemanticJsonTreeConsistencyEvaluator(
        model_id=MODEL, profile="legacy", array_order_lambda=0.0)
    assert on.sort_arrays is False
    on.array_order_lambda = None                       # and it comes back when disabled
    assert on.sort_arrays is True


# ------------------------------------------------------------------- end-to-end behaviour

PAIRS = [
    ({'name': 'f', 'arguments': {'xs': [1, 2, 3]}},
     {'name': 'f', 'arguments': {'xs': [3, 1, 2]}}),
    ({'name': 'f', 'arguments': {'xs': [1, 2, 3]}},
     {'name': 'f', 'arguments': {'xs': [1, 2, 3]}}),
    ({'name': 'f', 'arguments': {'xs': [1, 2, 3], 'ys': [4, 5, 6]}},
     {'name': 'f', 'arguments': {'xs': [3, 1, 2], 'ys': [4, 5, 6]}}),
    ({'name': 'g', 'arguments': {'pts': [[1, 2], [3, 4]]}},
     {'name': 'g', 'arguments': {'pts': [[2, 1], [4, 3]]}}),
    ({'name': 'f', 'arguments': {'xs': [1, 2, 3, 4, 5]}},
     {'name': 'f', 'arguments': {'xs': [1, 2, 3]}}),
    ({'name': 'f', 'arguments': {'xs': []}},
     {'name': 'f', 'arguments': {'xs': [1]}}),
    ({'name': 'h', 'arguments': {'a': 1, 'b': 'text here'}},
     {'name': 'h2', 'arguments': {'a': 2, 'b': 'other stuff'}}),
    ({'calls': [{'name': 'read', 'arguments': {}}, {'name': 'write', 'arguments': {}}]},
     {'calls': [{'name': 'write', 'arguments': {}}, {'name': 'read', 'arguments': {}}]}),
]


@pytest.mark.parametrize('cost_combination', ['additive', 'product'])
@pytest.mark.parametrize('variation_type', ['combined', 'structural', 'content'])
def test_lambda_zero_reproduces_the_published_metric(cost_combination, variation_type):
    """Exact equality, not approximate: λ=0 must be the same computation, not a close one."""
    published = SemanticJsonTreeConsistencyEvaluator(
        model_id=MODEL, profile="legacy", cost_combination=cost_combination)
    at_zero = SemanticJsonTreeConsistencyEvaluator(
        model_id=MODEL, profile="legacy", cost_combination=cost_combination,
        array_order_lambda=0.0)
    for a, b in PAIRS:
        assert (published.calculate_tree_edit_distance_fast(a, b, variation_type)
                == at_zero.calculate_tree_edit_distance_fast(a, b, variation_type))


@pytest.mark.parametrize('cost_combination', ['additive', 'product'])
@pytest.mark.parametrize('variation_type', ['combined', 'structural', 'content'])
def test_similarity_is_monotone_in_lambda(cost_combination, variation_type):
    evs = {lam: SemanticJsonTreeConsistencyEvaluator(
        model_id=MODEL, profile="legacy", cost_combination=cost_combination,
        array_order_lambda=lam)
        for lam in (0.0, 0.25, 0.5, 0.75, 1.0)}
    for a, b in PAIRS:
        scores = [evs[lam].calculate_tree_edit_distance_fast(a, b, variation_type)
                  for lam in (0.0, 0.25, 0.5, 0.75, 1.0)]
        for earlier, later in zip(scores, scores[1:]):
            assert later <= earlier + 1e-9, (scores, a, b)


def test_interval_brackets_and_flags_permutations():
    ev = SemanticJsonTreeConsistencyEvaluator(model_id=MODEL, profile="legacy")
    rotation = ({'name': 'f', 'arguments': {'xs': [1, 2, 3]}},
                {'name': 'f', 'arguments': {'xs': [2, 3, 1]}})
    s_ord, s_free = ev.compute_similarity_interval(*rotation)
    # The defect: today's STED cannot distinguish a rotation from the original at all.
    assert s_free == 1.0
    # The repair: the interval is wide, so the 1.0 is reported as assumption-dependent.
    assert s_ord < 1.0

    identical = ({'name': 'f', 'arguments': {'xs': [1, 2, 3]}},
                 {'name': 'f', 'arguments': {'xs': [1, 2, 3]}})
    assert ev.compute_similarity_interval(*identical) == (1.0, 1.0)

    # No array, no ambiguity: the interval collapses and s_free is the published score.
    scalars = ({'name': 'f', 'arguments': {'a': 1}}, {'name': 'f', 'arguments': {'a': 2}})
    s_ord, s_free = ev.compute_similarity_interval(*scalars)
    assert s_ord == s_free == ev.calculate_tree_edit_distance_fast(*scalars)


def test_interval_leaves_the_evaluator_untouched():
    """The λ override is temporary — a caller's configuration must survive the call."""
    ev = SemanticJsonTreeConsistencyEvaluator(
        model_id=MODEL, profile="legacy", array_order_lambda='auto')
    ev.compute_similarity_interval({'a': [1, 2]}, {'a': [2, 1]})
    assert ev.array_order_lambda == 'auto'
    assert ev.sort_arrays is False


CACHE_COLLISION = (
    # calls[0].xs and calls[1].xs strip to the same label pattern and hold the same values, so
    # they hash identically -- but calls[0].xs has an equal-length sibling (ys) and calls[1].xs
    # does not, so 'auto' gives them different λ. This is the shape that makes λ load-bearing
    # in the cache key rather than merely tidy.
    {'calls': [{'xs': [1, 2, 3], 'ys': [4, 5, 6]}, {'xs': [1, 2, 3], 'ys': [4, 5]}]},
    {'calls': [{'xs': [2, 3, 1], 'ys': [4, 5, 6]}, {'xs': [2, 3, 1], 'ys': [4, 5]}]},
)


def test_the_cache_collision_shape_is_real():
    """Guards the premise of the next test: equal hashes, different λ."""
    ev = SemanticJsonTreeConsistencyEvaluator(
        model_id=MODEL, profile="legacy", array_order_lambda='auto')
    tree = JsonNode.from_dict({'root': CACHE_COLLISION[0]}, sort_arrays=False)
    calls = tree.children[0].children[0]                     # root -> calls
    xs0 = [c for c in calls.children[0].children if c.label.endswith('xs')][0]
    xs1 = [c for c in calls.children[1].children if c.label.endswith('xs')][0]
    assert ev._compute_tree_hash(xs0) == ev._compute_tree_hash(xs1)
    assert order_lambda_arity_coupling(xs0, xs0) == 1.0
    assert order_lambda_arity_coupling(xs1, xs1) == 0.0


def test_subtree_cache_does_not_confuse_two_lambdas():
    """λ depends on a node's *siblings*, which the subtree hash cannot see.

    So it has to be part of the cache key; otherwise the second array pair's lookup returns
    the first pair's cost. Enabling the cache must not change any score.
    """
    a, b = CACHE_COLLISION
    for lam in ('auto', 1.0):
        uncached = SemanticJsonTreeConsistencyEvaluator(
            model_id=MODEL, profile="legacy", array_order_lambda=lam,
            subtree_cache_size=0)
        cached = SemanticJsonTreeConsistencyEvaluator(
            model_id=MODEL, profile="legacy", array_order_lambda=lam,
            subtree_cache_size=10000)
        assert (uncached.calculate_tree_edit_distance_fast(a, b)
                == cached.calculate_tree_edit_distance_fast(a, b))


def test_greedy_matching_is_bypassed_under_a_blend():
    """Greedy would break the lemma, so λ>0 must fall back to the exact path.

    A greedy assignment total can exceed the monotone optimum, which would make the
    "order-free" cost the *larger* of the two and invert the interval.
    """
    a = {'name': 'f', 'arguments': {'xs': list(range(12))}}
    b = {'name': 'f', 'arguments': {'xs': list(range(11, -1, -1))}}
    ev = SemanticJsonTreeConsistencyEvaluator(
        model_id=MODEL, profile="legacy", array_order_lambda=1.0)
    ev.use_greedy_matching = True
    s_ord = ev.calculate_tree_edit_distance_fast(a, b)
    ev.array_order_lambda = 0.0
    s_free = ev.calculate_tree_edit_distance_fast(a, b)
    assert s_ord <= s_free + 1e-9
