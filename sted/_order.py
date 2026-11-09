"""Order-sensitivity for array nodes: the two extremal costs and the family between them.

Why this module exists
----------------------
STED treats every array as a *bag*. Two mechanisms enforce it, and both are unconditional:
``JsonNode.from_dict`` sorts array elements into a canonical order at construction time, and
every cost function strips array indices out of paths (``re.sub(r'\\[\\d+\\]', '[*]', ...)``).
So a permutation of an array is not merely cheap — it is the *identical tree*, and the
similarity is exactly 1.0. No threshold can separate a permuted call from its original.

Measured consequence (results/metric_verification_v2/adversarial.json,
results/metric_verification_v2/order_signal.json): on the 208-call BFCL executable suite,
39 of 207 gold calls admit a semantically different variant STED scores >= 0.999, all of
them list permutations; and 119 of the 153 (call, array-argument) pairs are order-sensitive
under execution, i.e. STED's bag assumption is wrong on 78% of the arrays it sees.

Why the obvious fix is wrong
----------------------------
"Declare which fields are ordered" is the existing ``order_sensitive_fields`` API, and it
cannot be made correct. Two witnesses from the same suite:

  * ``polygon_area(vertices)`` is order-*insensitive* at 3 vertices (every permutation of
    three points is the same triangle) and order-*sensitive* at 4 (the shoelace formula
    reads consecutive pairs).
  * ``maxPoints(points)`` pools its points, but each element ``points[i] = [x, y]`` is a
    positional pair — and transposing it is inert for some values and not for others.

Order-sensitivity is therefore a property of (callee, argument, *value*), not of a field
name or a path, so no static annotation — no schema keyword, no name set, no learned prior
on the parameter name — is a correct predictor of it. Any metric that commits to one answer
is silently wrong on some inputs; the only sound move is to stop committing.

What this module does instead
-----------------------------
For each array pair it computes both extremes of the same edit problem:

  ``c_free``  minimum over ALL matchings of min(n1, n2) element pairs (the Hungarian
              assignment STED already computes) — the bag reading.
  ``c_ord``   minimum over the MONOTONE such matchings — the sequence reading.

and exposes the one-parameter family

  ``c(lambda) = (1 - lambda) * c_free + lambda * c_ord``,  lambda in [0, 1].

**Lemma (bracketing).** Let ``F`` be the set of matchings that pair exactly
``min(n1, n2)`` elements, leaving ``|n1 - n2|`` unmatched, and ``F_mono ⊂ F`` those that
preserve order. Both costs sum the same per-pair costs and the same insert/delete charges
and both divide by ``max(n1, n2)``, so ``c_free = min_F <= min_{F_mono} = c_ord``: the
order-free cost is a minimum over a superset. Hence ``c(lambda)`` is non-decreasing in
lambda and ``s_1 <= s_lambda <= s_0`` for the induced similarities. ∎

That is the point. ``s_0`` is what STED reports today and ``s_1`` is what it would report
if every array were a sequence; the *width* ``s_0 - s_1`` is exactly the part of the answer
that rests on the array-order assumption. A permuted call has ``s_0 = 1`` and ``s_1 < 1``,
so instead of an unfalsifiable 1.0 the metric can report that its verdict is
assumption-dependent — a guarantee that does not require guessing lambda correctly.

``order_lambda_arity_coupling`` is offered as a *default point estimate* for callers who
need a single number, not as a solution to the prediction problem: equal-length sibling
arrays in one call are in elementwise correspondence (``cosine_similarity(vecA, vecB)``,
``linear_regression(x, y)``, a matrix's rows), so permuting one side alone breaks the
pairing. It reaches 0.889 accuracy against the execution oracle (13 order-sensitive arrays
missed, 4 order-free arrays over-penalised) versus 0.222 for the current all-bags
assumption and 0.778 for all-sequences. The residual errors are why the interval is the
headline and the point estimate is not.
"""
from __future__ import annotations

from typing import List, Sequence, Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - only for type checkers
    from .json_tree_node import JsonNode


_MATCH, _SKIP1, _SKIP2 = 0, 1, 2


def monotone_alignment(
    pair_costs: np.ndarray,
    del_costs: Sequence[float],
    ins_costs: Sequence[float],
    combine: str = 'sum',
) -> Tuple[float, List[Tuple[int, int]], List[int], List[int]]:
    """Cheapest order-preserving matching of exactly ``min(n1, n2)`` pairs.

    ``pair_costs[i, j]`` is the cost of matching element *i* of the first array to element
    *j* of the second. Exactly ``|n1 - n2|`` elements of the longer array are skipped and
    charged their delete (resp. insert) cost; the rest are matched in increasing index
    order on both sides.

    O(n1 * n2) dynamic program with backpointers: ``dp[i][j]`` is the best cost for the
    first *i* and *j* elements. Because the pair count is pinned to ``min(n1, n2)``, only
    the longer side may skip — that is what keeps the feasible set a *subset* of the
    assignment's and makes the bracketing lemma in the module docstring hold with the same
    denominator. Reachability is pruned on ``|i - j| <= |n1 - n2|`` for the same reason:
    any state outside that band has already used more skips than the pair count allows.

    ``combine`` selects what the DP accumulates. ``'sum'`` (the default, and the only
    behaviour before the aggregation axis existed) minimises the total, which is what a
    level normalising by an arithmetic mean needs. ``'max'`` minimises the *largest* charged
    cost along the alignment — the order-preserving analogue of the bottleneck assignment,
    needed when the level aggregates its children with ``p = -inf`` (see
    ``sted._aggregation``). Both are optimal for their own objective by the same exchange
    argument; ``max`` is valid in the DP because it is associative, monotone in each argument
    and has 0 as its identity on non-negative costs, which is all the recursion uses.

    Returns ``(total, matched_pairs, skipped_from_1, skipped_from_2)`` with the total
    *un-normalised*, so callers divide by ``max(n1, n2)`` exactly as the assignment path
    does. Under ``combine='max'`` the total is already the level's cost and callers must not
    divide it; every caller that passes ``'max'`` recharges the returned path through
    ``_aggregation.aggregate`` instead of using the total, so this asymmetry is not relied on.
    The path is returned because the callers that need it charge several cost
    channels along one shared correspondence; recovering the channels from separate
    per-channel alignments would charge totals from a correspondence that was never chosen.
    """
    if combine not in ('sum', 'max'):
        raise ValueError(f"combine must be 'sum' or 'max', got {combine!r}")
    n1, n2 = len(del_costs), len(ins_costs)
    if n1 == 0 and n2 == 0:
        return 0.0, [], [], []
    if combine == 'max':
        def acc(prev, cost):
            return prev if prev > cost else cost

        def charge_all(costs):
            return float(max(costs)) if len(costs) else 0.0
    else:
        def acc(prev, cost):
            return prev + cost

        def charge_all(costs):
            return float(sum(costs))
    if n1 == 0:
        return charge_all(ins_costs), [], [], list(range(n2))
    if n2 == 0:
        return charge_all(del_costs), [], list(range(n1)), []

    inf = float('inf')
    slack = abs(n1 - n2)
    dp = np.full((n1 + 1, n2 + 1), inf, dtype=np.float64)
    bp = np.full((n1 + 1, n2 + 1), -1, dtype=np.int8)
    dp[0, 0] = 0.0
    for j in range(1, n2 + 1):
        if j - 0 > slack or n2 <= n1:
            break
        dp[0, j] = acc(dp[0, j - 1], float(ins_costs[j - 1]))
        bp[0, j] = _SKIP2
    for i in range(1, n1 + 1):
        if i <= slack and n1 > n2:
            dp[i, 0] = acc(dp[i - 1, 0], float(del_costs[i - 1]))
            bp[i, 0] = _SKIP1
        for j in range(1, n2 + 1):
            if abs(i - j) > slack:
                continue
            best, arg = inf, -1
            if dp[i - 1, j - 1] < inf:                 # match i-1 with j-1
                best, arg = acc(dp[i - 1, j - 1], float(pair_costs[i - 1, j - 1])), _MATCH
            if n1 > n2 and dp[i - 1, j] < inf:         # skip element i-1 (longer side)
                cand = acc(dp[i - 1, j], float(del_costs[i - 1]))
                if cand < best:
                    best, arg = cand, _SKIP1
            if n2 > n1 and dp[i, j - 1] < inf:         # skip element j-1 (longer side)
                cand = acc(dp[i, j - 1], float(ins_costs[j - 1]))
                if cand < best:
                    best, arg = cand, _SKIP2
            dp[i, j], bp[i, j] = best, arg

    total = float(dp[n1, n2])
    if total == inf:                                    # unreachable: charge everything
        return (charge_all(list(del_costs) + list(ins_costs)), [],
                list(range(n1)), list(range(n2)))

    pairs: List[Tuple[int, int]] = []
    skip1: List[int] = []
    skip2: List[int] = []
    i, j = n1, n2
    while i > 0 or j > 0:
        step = bp[i, j]
        if step == _MATCH:
            pairs.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif step == _SKIP1:
            skip1.append(i - 1)
            i -= 1
        elif step == _SKIP2:
            skip2.append(j - 1)
            j -= 1
        else:                                           # pragma: no cover - defensive
            break
    pairs.reverse()
    skip1.reverse()
    skip2.reverse()
    return total, pairs, skip1, skip2


def monotone_alignment_cost(
    pair_costs: np.ndarray,
    del_costs: Sequence[float],
    ins_costs: Sequence[float],
) -> float:
    """``monotone_alignment`` when only the total is needed."""
    return monotone_alignment(pair_costs, del_costs, ins_costs)[0]


def blend(c_free: float, c_ord: float, lam: float) -> float:
    """``(1 - lam) * c_free + lam * c_ord``, clamped to [0, 1].

    The clamp is defensive only: both inputs are already normalised into [0, 1] by their
    callers and a convex combination of two such values cannot leave the interval.
    """
    if lam <= 0.0:
        return c_free
    if lam >= 1.0:
        return min(max(c_ord, 0.0), 1.0)
    return min(max((1.0 - lam) * c_free + lam * c_ord, 0.0), 1.0)


# ------------------------------------------------------------------ lambda estimation

def _sibling_array_lengths(node: "JsonNode") -> List[int]:
    """Lengths of the array siblings of ``node`` under its recorded parent, if any."""
    parent = getattr(node, 'parent', None)
    if parent is None:
        return []
    out = []
    for sib in parent.children:
        if sib is node:
            continue
        if sib.node_type == 'array':
            out.append(len(sib.children))
    return out


def order_lambda_arity_coupling(
    node1: "JsonNode", node2: "JsonNode", default: float = 0.0
) -> float:
    """Point estimate of order-sensitivity from equal-length sibling arrays.

    Returns 1.0 when some *other* array under the same parent has the same length as this
    one (and that length exceeds 1), else ``default``. The signal is that two equal-length
    arrays in one object are read elementwise — vector against vector, x against y, a
    matrix's rows — so permuting one side alone destroys a correspondence the callee
    depends on.

    Requires ``JsonNode.parent`` to be populated, which ``from_dict`` only does when asked;
    with no parent link the function degrades to ``default`` rather than guessing.

    This is a 0.889-accuracy heuristic (see the module docstring), not a decision
    procedure. Callers that need a sound answer should read the interval instead.
    """
    for node in (node1, node2):
        n = len(node.children)
        if n <= 1:
            continue
        if any(length == n for length in _sibling_array_lengths(node)):
            return 1.0
    return default


def resolve_lambda(
    spec, node1: "JsonNode", node2: "JsonNode", default: float = 0.0
) -> float:
    """Turn an ``array_order_lambda`` setting into a number for one array pair.

    Accepted forms: ``None`` (-> ``default``), a float in [0, 1], the string ``'auto'``
    (-> ``order_lambda_arity_coupling``), or any callable ``(node1, node2) -> float``.
    """
    if spec is None:
        return default
    if isinstance(spec, (int, float)) and not isinstance(spec, bool):
        return min(max(float(spec), 0.0), 1.0)
    if spec == 'auto':
        return order_lambda_arity_coupling(node1, node2, default=default)
    if callable(spec):
        return min(max(float(spec(node1, node2)), 0.0), 1.0)
    raise ValueError(
        f"array_order_lambda must be None, a float in [0, 1], 'auto', or a callable; "
        f"got {spec!r}")


# ------------------------------------------------------------------------- diagnostics

def array_pairs_with_order_ambiguity(
    tree1: "JsonNode", tree2: "JsonNode"
) -> List[Tuple[str, int, int]]:
    """Array node pairs whose element multisets could differ only by order.

    A cheap structural screen for "is the order assumption load-bearing here": paired array
    nodes of equal length greater than one. Used for reporting, not for scoring.
    """
    out: List[Tuple[str, int, int]] = []

    def walk(a: "JsonNode", b: "JsonNode") -> None:
        if a.node_type == 'array' and b.node_type == 'array':
            n1, n2 = len(a.children), len(b.children)
            if n1 > 1 and n1 == n2:
                out.append((a.label, n1, n2))
        for c1, c2 in zip(a.children, b.children):
            walk(c1, c2)

    walk(tree1, tree2)
    return out
