#!/usr/bin/env python3
"""Tests for the child-aggregation axis (``sted._aggregation``).

Three obligations, as with the order and key-identity axes:

1. **p = 1 reproduces the published metric exactly.** Bit-identical, not close — the call
   sites keep their original expression rather than routing through a p = 1 special case, and
   these tests are what pins that.
2. **The bracketing lemma.** The score is non-decreasing in ``p``, which requires each node's
   correspondence to be the argmax of *its own* aggregate; ``objective_matrix`` is what makes
   that true, so a test that only checked ``aggregate`` would miss the load-bearing part.
3. **The defect is narrowed by exactly the amount claimed, and no more.** A single
   meaning-destroying leaf still gets cheaper as the document grows — but on an ``n^(1/|p|)``
   law rather than the mean's ``1/n``. Both the ratio and the fitted exponent are pinned, so
   the axis cannot later be described as removing the dilution.
4. **``p = -inf`` closes it.** The bottleneck end is the only member of the family whose score
   does not mention ``n``, which is what lets ``score >= theta`` certify every matched pair.
   That claim is worth exactly as much as its solver is exact, so the bottleneck assignment and
   the min-max monotone DP are both checked against brute force, and the size-freedom itself is
   measured on the same ``_flip_first`` ladder that exhibits the dilution.

Plus the same cache hazard as μ: ``p`` is evaluator-wide and so deliberately absent from the
subtree cache key.
"""

import itertools
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from sted._aggregation import (                                             # noqa: E402
    aggregate, assign, bottleneck_assignment, combine_mode, is_active, is_bottleneck,
    objective_matrix, resolve_p)
from sted._order import monotone_alignment                                  # noqa: E402
from sted.semantic_json_tree_consistency import (                           # noqa: E402
    SemanticJsonTreeConsistencyEvaluator)

MODEL = 'all-MiniLM-L6-v2'
CONJ = -8.0
NEG_INF = float('-inf')

_EVALUATORS = {}


def _ev(**kwargs):
    """One evaluator per distinct configuration — each construction loads the model."""
    kwargs.setdefault("profile", "legacy")
    key = tuple(sorted(kwargs.items()))
    if key not in _EVALUATORS:
        _EVALUATORS[key] = SemanticJsonTreeConsistencyEvaluator(model_id=MODEL, **kwargs)
    return _EVALUATORS[key]


# --------------------------------------------------------------------- the math in isolation

def test_resolve_p_forms():
    assert resolve_p(None) == 1.0
    assert resolve_p(None, default=-2.0) == -2.0
    assert resolve_p(1.0) == 1.0
    assert resolve_p(0) == 0.0
    assert resolve_p(-8.0) == -8.0
    assert resolve_p('mean') == 1.0
    assert resolve_p('geometric') == 0.0
    # p > 1 is rejected rather than clamped: it would put the published score *inside* the
    # interval instead of at its permissive end.
    with pytest.raises(ValueError):
        resolve_p(1.5)
    with pytest.raises(ValueError):
        resolve_p(True)                       # bool is an int subclass
    with pytest.raises(ValueError):
        resolve_p('auto')
    with pytest.raises(ValueError):
        resolve_p(float('nan'))
    # -inf is a different algorithm, not a different exponent, but it is a valid setting.
    assert resolve_p(NEG_INF) == NEG_INF
    assert resolve_p('min') == NEG_INF
    assert resolve_p('bottleneck') == NEG_INF


def test_is_active_only_off_at_one():
    assert not is_active(1.0)
    assert is_active(0.0)
    assert is_active(-8.0)
    assert is_active(NEG_INF)


def test_is_bottleneck_only_at_minus_inf():
    assert is_bottleneck(NEG_INF)
    for p in (1.0, 0.0, -8.0, -1e30):
        assert not is_bottleneck(p)
    assert combine_mode(NEG_INF) == 'max'
    for p in (1.0, 0.0, -8.0):
        assert combine_mode(p) == 'sum'


def test_aggregate_reduces_to_the_mean_at_p_one():
    costs = [0.0, 0.25, 0.5, 1.0]
    assert aggregate(costs, 1.0) == pytest.approx(sum(costs) / len(costs))
    assert aggregate([], 1.0) == 0.0


def test_aggregate_is_monotone_in_p_and_hits_the_known_means():
    costs = [0.0, 0.2, 0.6]
    sims = [1.0, 0.8, 0.4]
    assert aggregate(costs, 0.0) == pytest.approx(1.0 - (1.0 * 0.8 * 0.4) ** (1 / 3))
    assert aggregate(costs, -1.0) == pytest.approx(
        1.0 - 3.0 / (1 / 1.0 + 1 / 0.8 + 1 / 0.4))
    ps = [1.0, 0.5, 0.0, -1.0, -2.0, -8.0, -32.0]
    scores = [1.0 - aggregate(costs, p) for p in ps]
    for earlier, later in zip(scores, scores[1:]):
        assert later <= earlier + 1e-12, scores
    # ...and it approaches the minimum similarity from above, without reaching it.
    assert scores[-1] > min(sims) - 1e-9
    assert scores[-1] == pytest.approx(min(sims), abs=0.02)


def test_a_constant_vector_is_a_fixed_point_for_every_p():
    """The power mean of a constant vector is that constant, so p cannot move a uniform node."""
    for c in (0.0, 0.3, 1.0):
        for p in (1.0, 0.0, -1.0, -8.0):
            assert aggregate([c] * 5, p) == pytest.approx(c, abs=1e-12)


def test_a_zero_similarity_saturates_every_non_positive_p():
    """A fully wrong child makes the node fully wrong under any conjunctive reading."""
    for p in (0.0, -0.5, -8.0):
        assert aggregate([0.1, 1.0, 0.1], p) == 1.0
    # ...but not under the mean, which is the defect.
    assert aggregate([0.1, 1.0, 0.1], 1.0) == pytest.approx(0.4)


def test_objective_matrix_is_the_identity_at_p_one_and_increasing_elsewhere():
    m = np.array([[0.0, 0.5], [1.0, 0.25]])
    assert objective_matrix(m, 1.0) is m
    for p in (0.5, 0.0, -1.0, -8.0):
        obj = objective_matrix(np.array([[0.0, 0.25, 0.5, 0.75, 1.0]]), p)[0]
        assert list(obj) == sorted(obj), (p, obj)
    # An infeasible entry must stay the most expensive option, not become the cheapest.
    with_inf = objective_matrix(np.array([[0.0, np.inf]]), -2.0)[0]
    assert with_inf[1] > with_inf[0]
    assert math.isfinite(with_inf[1])
    for p in (0.5, 0.0):
        row = objective_matrix(np.array([[0.0, np.inf]]), p)[0]
        assert row[1] > row[0] and math.isfinite(row[1])
    # At the bottleneck end there is nothing to reparameterise: max is already monotone in cost,
    # so the change is the solver and the DP accumulator, not the matrix.
    assert objective_matrix(m, NEG_INF) is m


# ------------------------------------------------------------------- the bottleneck end (p=-inf)

def test_aggregate_at_minus_inf_is_the_worst_child_and_ignores_n():
    """``1 - min_i s_i``: the size-free reading, stated as an identity rather than a limit."""
    assert aggregate([0.0, 0.2, 0.6], NEG_INF) == pytest.approx(0.6)
    # The whole point: padding a node with perfect children cannot dilute the worst one.
    for n in (1, 2, 10, 1000):
        assert aggregate([0.6] + [0.0] * n, NEG_INF) == pytest.approx(0.6)
    assert aggregate([], NEG_INF) == 0.0
    # And it is the bottom of the bracket, below every finite p on the same vector.
    costs = [0.0, 0.2, 0.6]
    for p in (1.0, 0.0, -8.0, -256.0):
        assert aggregate(costs, NEG_INF) >= aggregate(costs, p) - 1e-12


def _brute_bottleneck(m):
    """min over assignments of the largest matched cost, by enumeration."""
    rows, cols = m.shape
    if rows <= cols:
        best = min(max(m[i, c] for i, c in enumerate(perm))
                   for perm in itertools.permutations(range(cols), rows))
    else:
        best = min(max(m[r, j] for j, r in enumerate(perm))
                   for perm in itertools.permutations(range(rows), cols))
    return best


@pytest.mark.parametrize('shape', [(1, 1), (2, 2), (3, 3), (2, 4), (4, 2), (3, 5)])
def test_bottleneck_assignment_matches_brute_force(shape):
    """The certificate is only worth as much as the solver is exact."""
    rng = np.random.default_rng(11)
    for _ in range(30):
        m = np.round(rng.random(shape), 2)
        r, c = bottleneck_assignment(m)
        assert len(r) == min(shape) and sorted(set(r)) == sorted(r)
        assert sorted(set(c)) == sorted(c)
        assert max(m[i, j] for i, j in zip(r, c)) == pytest.approx(_brute_bottleneck(m))
    # It must also be the *cheapest* min-bottleneck assignment, so the result is deterministic
    # rather than an artefact of the matching algorithm's internal order: both assignments here
    # achieve bottleneck 0.5, and only one of them keeps the total at 0.5.
    m = np.array([[0.5, 0.5], [0.0, 0.4]])
    r, c = bottleneck_assignment(m)
    assert sorted(zip(r, c)) == [(0, 1), (1, 0)]
    assert sum(m[i, j] for i, j in zip(r, c)) == pytest.approx(0.5)


def test_bottleneck_assignment_handles_forbidden_cells_and_empty_matrices():
    r, c = bottleneck_assignment(np.zeros((0, 3)))
    assert len(r) == 0 and len(c) == 0
    # An inf cell is a forbidden match; a feasible assignment avoiding it must be found.
    m = np.array([[np.inf, 0.7], [0.3, np.inf]])
    r, c = bottleneck_assignment(m)
    assert sorted(zip(r, c)) == [(0, 1), (1, 0)]
    # When every cell is forbidden there is nothing to optimise and we must still return a
    # complete assignment rather than raise.
    r, c = bottleneck_assignment(np.full((2, 2), np.inf))
    assert len(r) == 2
    # assign() dispatches to the right solver; at p=1 it is linear_sum_assignment verbatim.
    m = np.array([[0.9, 0.1], [0.5, 0.5]])
    assert sorted(zip(*assign(m, NEG_INF))) == [(0, 1), (1, 0)]
    assert sorted(zip(*assign(m, 1.0))) == [(0, 1), (1, 0)]


def _brute_monotone(pair_costs, del_costs, ins_costs, combine):
    """Cheapest order-preserving matching of exactly min(n1, n2) pairs, by enumeration."""
    n1, n2 = len(del_costs), len(ins_costs)
    agg = max if combine == 'max' else sum
    best = float('inf')
    if n1 <= n2:
        for cols in itertools.combinations(range(n2), n1):
            charges = ([pair_costs[i, j] for i, j in enumerate(cols)]
                       + [ins_costs[j] for j in range(n2) if j not in cols])
            best = min(best, agg(charges) if charges else 0.0)
    else:
        for rows in itertools.combinations(range(n1), n2):
            charges = ([pair_costs[i, j] for j, i in enumerate(rows)]
                       + [del_costs[i] for i in range(n1) if i not in rows])
            best = min(best, agg(charges) if charges else 0.0)
    return best


@pytest.mark.parametrize('combine', ['sum', 'max'])
@pytest.mark.parametrize('shape', [(1, 1), (2, 2), (3, 3), (2, 4), (4, 2), (3, 5)])
def test_monotone_alignment_is_optimal_for_both_accumulators(combine, shape):
    """``combine='max'`` is the order-preserving bottleneck, and the DP still finds its optimum."""
    rng = np.random.default_rng(23)
    n1, n2 = shape
    for _ in range(20):
        pair = np.round(rng.random(shape), 2)
        dels = list(np.round(rng.random(n1), 2))
        inss = list(np.round(rng.random(n2), 2))
        total, pairs, s1, s2 = monotone_alignment(pair, dels, inss, combine=combine)
        assert len(pairs) == min(shape)
        assert [i for i, _ in pairs] == sorted(i for i, _ in pairs)
        assert [j for _, j in pairs] == sorted(j for _, j in pairs)
        assert len(s1) == n1 - min(shape) and len(s2) == n2 - min(shape)
        assert total == pytest.approx(_brute_monotone(pair, dels, inss, combine))
    # The two accumulators agree only by coincidence; max is never above sum.
    pair = np.array([[0.1, 0.9], [0.9, 0.1]])
    assert (monotone_alignment(pair, [1.0, 1.0], [1.0, 1.0], combine='max')[0]
            <= monotone_alignment(pair, [1.0, 1.0], [1.0, 1.0])[0])


def test_monotone_alignment_rejects_an_unknown_accumulator():
    with pytest.raises(ValueError):
        monotone_alignment(np.zeros((1, 1)), [0.0], [0.0], combine='product')


# --------------------------------------------------------------------- end-to-end behaviour

PAIRS = [
    ({'name': 'f', 'arguments': {'n': 20, 'rate': 0.5}},
     {'name': 'f', 'arguments': {'n': 20, 'rate': 0.9}}),
    ({'name': 'f', 'arguments': {'n': 20}}, {'name': 'f', 'arguments': {'n': 20}}),
    ({'name': 'f', 'arguments': {'xs': [1, 2, 3, 4]}},
     {'name': 'f', 'arguments': {'xs': [1, 2, 3, 99]}}),
    ({'a': {'b': [1, 2], 'c': 'hello world'}},
     {'a': {'b': [1, 2], 'c': 'goodbye world'}}),
    ({'name': 'f', 'arguments': {'xs': [1, 2, 3]}},
     {'name': 'g', 'arguments': {'ys': [3, 2, 1], 'extra': True}}),
]


@pytest.mark.parametrize('cost_combination', ['additive', 'product'])
@pytest.mark.parametrize('variation_type', ['combined', 'structural', 'content'])
def test_p_one_reproduces_the_published_metric(cost_combination, variation_type):
    """Exact equality, not approximate: p=1 must be the same computation."""
    published = _ev(cost_combination=cost_combination)
    at_one = _ev(cost_combination=cost_combination, aggregation_p=1.0)
    for a, b in PAIRS:
        assert (published.calculate_tree_edit_distance_fast(a, b, variation_type)
                == at_one.calculate_tree_edit_distance_fast(a, b, variation_type))


@pytest.mark.parametrize('cost_combination', ['additive', 'product'])
@pytest.mark.parametrize('variation_type', ['combined', 'structural', 'content'])
def test_similarity_is_monotone_in_p(cost_combination, variation_type):
    ps = (1.0, 0.5, 0.0, -1.0, -4.0, -8.0, NEG_INF)
    evs = {p: _ev(cost_combination=cost_combination, aggregation_p=p) for p in ps}
    for a, b in PAIRS:
        scores = [evs[p].calculate_tree_edit_distance_fast(a, b, variation_type) for p in ps]
        for earlier, later in zip(scores, scores[1:]):
            assert later <= earlier + 1e-9, (scores, a, b)


def _flip_first(n):
    """A call whose first array element has its sign flipped: one destroyed leaf out of n."""
    xs = list(range(1, n + 1))
    return ({'name': 'calculate_mean', 'arguments': {'numbers': xs}},
            {'name': 'calculate_mean', 'arguments': {'numbers': [-xs[0]] + xs[1:]}})


def test_a_destroyed_leaf_gets_cheaper_in_a_bigger_array_and_p_slows_that_down():
    """The defect, and how much of it a finite p actually removes.

    Under the arithmetic mean the *same* edit costs 1/n of the level, so the penalty is divided
    by 10 when the array grows tenfold — the mechanism behind the +0.639 correlation between
    document size and score on the sign-flip operator and behind Instrument D's ceiling false
    accepts. A negative p does **not** remove that: with one destroyed child of similarity s
    among n, M_p ~ s * n^(1/|p|), so the score still climbs, polynomially more slowly. What is
    tested here is the exponent, because that is the whole of what the axis buys.
    """
    sizes = (2, 20)
    mean_scores = [_ev().calculate_tree_edit_distance_fast(*_flip_first(n)) for n in sizes]
    conj_scores = [_ev(aggregation_p=CONJ).calculate_tree_edit_distance_fast(*_flip_first(n))
                   for n in sizes]

    # Measured: 0.8750, 0.9500, 0.9875 for n = 2, 5, 20 — exactly 1 - 0.25/n, so the deficit
    # falls by the full factor of 10 across a tenfold size increase.
    assert mean_scores == sorted(mean_scores)
    mean_ratio = (1.0 - mean_scores[0]) / (1.0 - mean_scores[1])
    assert mean_ratio == pytest.approx(10.0, rel=0.02), mean_scores
    # At p = -8 the deficit falls by only ~1.84 over the same range (0.4063 -> 0.2210).
    conj_ratio = (1.0 - conj_scores[0]) / (1.0 - conj_scores[1])
    assert 1.0 < conj_ratio < mean_ratio / 3.0, (conj_scores, conj_ratio, mean_ratio)
    # And it is a bracket, pointwise.
    for m, c in zip(mean_scores, conj_scores):
        assert c <= m + 1e-9


def test_the_dilution_exponent_is_one_over_abs_p():
    """``M_p ~ s * n^(1/|p|)``: the residual size dependence has a closed form.

    Fitted on the arithmetic mean's own scale, the slope of log score against log n comes out
    at 1/|p| to four decimals for p <= -16 (0.0625 and 0.0312 measured). This is the honest
    statement of what the axis achieves: the dilution exponent drops from 1 to 1/|p|, and
    ``sup_n`` of the score is still 1 for every finite p, so no finite p makes a threshold
    sound. Only p = -inf, which is a bottleneck assignment and not implemented here, is
    size-free.
    """
    sizes = (2, 5, 10, 20, 50)
    logs = [math.log(n) for n in sizes]
    for p in (-16.0, -32.0):
        ev = _ev(aggregation_p=p)
        scores = [ev.calculate_tree_edit_distance_fast(*_flip_first(n)) for n in sizes]
        assert max(scores) < 0.999, (p, scores)          # fit is only valid away from the cap
        ys = [math.log(s) for s in scores]
        mx, my = sum(logs) / len(logs), sum(ys) / len(ys)
        slope = (sum((x - mx) * (y - my) for x, y in zip(logs, ys))
                 / sum((x - mx) ** 2 for x in logs))
        assert slope == pytest.approx(1.0 / abs(p), abs=0.01), (p, slope, scores)


def test_the_bottleneck_end_is_size_free():
    """The whole justification for ``p = -inf``: the same edit costs the same at every ``n``.

    Under the mean the score is 1 - 0.25/n (0.8750 at n=2, 0.9875 at n=20); under any finite p it
    still climbs as n^(1/|p|). Here it must be *flat*, because ``min_i s_i`` does not mention n.
    Flat is asserted to 1e-6, not approximately: any residual n-dependence would mean some level
    is still averaging, and the threshold-soundness theorem would be false.
    """
    sizes = (2, 5, 10, 20, 50)
    ev = _ev(aggregation_p=NEG_INF)
    scores = [ev.calculate_tree_edit_distance_fast(*_flip_first(n)) for n in sizes]
    assert max(scores) - min(scores) < 1e-6, scores
    assert max(scores) < 0.9, scores            # and it is the worst child, not a diluted one

    # Which is what makes a threshold mean something: it is below every finite p's score, and
    # every finite p's score rises with n while this one does not.
    for p in (-8.0, -32.0):
        finite = _ev(aggregation_p=p)
        for n, s in zip(sizes, scores):
            assert s <= finite.calculate_tree_edit_distance_fast(*_flip_first(n)) + 1e-9


def test_padding_a_call_with_correct_arguments_cannot_hide_a_wrong_one():
    """The dilution proposition, run directly against the two ends.

    Adding *correct* material to a call with one wrong argument raises the mean's score without
    bound — that is the defect in its purest form, with the destroyed child held fixed. At the
    bottleneck end the score cannot move at all.
    """
    def pair(n_extra):
        extra = {f'opt{i}': f'value{i}' for i in range(n_extra)}
        return ({'name': 'f', 'arguments': dict({'mode': 'ascending'}, **extra)},
                {'name': 'f', 'arguments': dict({'mode': 'descending'}, **extra)})

    means = [_ev().calculate_tree_edit_distance_fast(*pair(k)) for k in (0, 4, 16)]
    mins = [_ev(aggregation_p=NEG_INF).calculate_tree_edit_distance_fast(*pair(k))
            for k in (0, 4, 16)]
    assert means[0] < means[1] < means[2], means
    assert means[2] > 0.95, means               # 16 correct arguments hide the wrong one
    assert max(mins) - min(mins) < 1e-6, mins
    assert max(mins) <= means[0] + 1e-9, (mins, means)


def test_p_one_is_still_bit_identical_after_the_bottleneck_path_exists():
    """The bottleneck end added a solver and a DP accumulator; neither may touch p = 1.

    Duplicates ``test_p_one_reproduces_the_published_metric`` deliberately: that test predates
    ``assign``/``combine_mode``, and this one names the regression it is guarding against.
    """
    published, at_one = _ev(), _ev(aggregation_p=1.0)
    for a, b in PAIRS + [_flip_first(20)]:
        for vt in ('combined', 'structural', 'content'):
            assert (published.calculate_tree_edit_distance_fast(a, b, vt)
                    == at_one.calculate_tree_edit_distance_fast(a, b, vt))


def test_interval_brackets_and_leaves_the_evaluator_untouched():
    a, b = _flip_first(20)
    ev = _ev(aggregation_p=-2.0)
    s_conj, s_agg = ev.compute_aggregation_interval(a, b)
    assert s_conj <= s_agg
    assert s_agg == pytest.approx(_ev().calculate_tree_edit_distance_fast(a, b), abs=1e-12)
    # The default lower end is now the bottleneck, i.e. the exact minimum, not a finite p.
    assert s_conj == pytest.approx(
        _ev(aggregation_p=NEG_INF).calculate_tree_edit_distance_fast(a, b), abs=1e-12)
    assert s_conj <= ev.compute_aggregation_interval(a, b, p=-2.0)[0] + 1e-9
    # One destroyed leaf in a 20-element array is the widest case there is.
    assert s_agg - s_conj > 0.1
    assert ev.aggregation_p == -2.0

    identical = ({'name': 'f', 'arguments': {'n': 20}}, {'name': 'f', 'arguments': {'n': 20}})
    assert ev.compute_aggregation_interval(*identical) == (1.0, 1.0)


def test_assumption_interval_covers_all_three_axes():
    """The corner ``(λ, μ, p) = (1, 1, p_conj)`` must bound every mixed reading."""
    a = {'name': 'f', 'arguments': {'xs': [1, 2, 3, 4]}}
    b = {'name': 'f', 'arguments': {'xs_value': [4, 1, 2, 3]}}
    ev = _ev()
    s_min, s_max = ev.compute_assumption_interval(a, b)
    assert s_min <= s_max
    # The published score is the fully permissive corner, i.e. the upper end.
    assert s_max == pytest.approx(ev.calculate_tree_edit_distance_fast(a, b), abs=1e-12)
    for lam, mu, p in ((1.0, 0.0, 1.0), (0.0, 1.0, 1.0), (0.0, 0.0, CONJ),
                       (1.0, 1.0, 1.0), (0.0, 1.0, CONJ)):
        mixed = _ev(array_order_lambda=lam, key_identity_mu=mu, aggregation_p=p)
        s = mixed.calculate_tree_edit_distance_fast(a, b)
        assert s_min - 1e-9 <= s <= s_max + 1e-9, (lam, mu, p, s, s_min, s_max)


def test_greedy_matching_is_bypassed_under_an_active_p():
    """Greedy returns a total, from which no power mean can be recovered."""
    a, b = _flip_first(12)                    # > 5 children, so greedy would otherwise fire
    ev = SemanticJsonTreeConsistencyEvaluator(
        model_id=MODEL, profile="legacy", aggregation_p=CONJ)
    ev.use_greedy_matching = False
    exact = ev.calculate_tree_edit_distance_fast(a, b)
    ev.use_greedy_matching = True
    assert ev.calculate_tree_edit_distance_fast(a, b) == pytest.approx(exact, abs=1e-12)


def test_changing_p_invalidates_the_subtree_cache():
    """p is not in the cache key, so the setter has to clear the cache."""
    a, b = _flip_first(6)
    cached = SemanticJsonTreeConsistencyEvaluator(
        model_id=MODEL, profile="legacy", subtree_cache_size=10000)
    cached.calculate_tree_edit_distance_fast(a, b)           # populate under the mean
    cached.aggregation_p = CONJ
    after = cached.calculate_tree_edit_distance_fast(a, b)
    fresh = _ev(aggregation_p=CONJ).calculate_tree_edit_distance_fast(a, b)
    assert after == pytest.approx(fresh, abs=1e-12)

    # Re-assigning the same value must not clear anything, so only the *change* invalidates.
    cached.clear_subtree_cache()
    cached.calculate_tree_edit_distance_fast(a, b)
    before_hits = cached.get_cache_stats()['subtree_cache_hits']
    cached.aggregation_p = CONJ
    cached.calculate_tree_edit_distance_fast(a, b)
    assert cached.get_cache_stats()['subtree_cache_hits'] > before_hits


def test_invalid_p_is_rejected_at_assignment_not_at_scoring():
    ev = _ev()
    with pytest.raises(ValueError):
        ev.aggregation_p = 2.0
    assert ev.aggregation_p is None
