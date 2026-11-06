"""How a node's children combine: the aggregation axis, and the dilution it repairs.

Why this module exists
----------------------
``_order`` and ``_key_identity`` each bracket an *interpretive* ambiguity — something the
document does not determine. This axis is different: it starts from a defect that is provable
from the code, not a reading that is arguable.

Every level of the recursion aggregates its matched children by an **arithmetic mean**
(``total_cost / len(row_indices)`` in ``_matching``). So:

**Proposition (dilution).** Let a node have ``n`` matched child pairs, of which exactly one
has cost ``c`` and the rest cost 0. The node's cost is ``c / n`` and its similarity
``1 - c/n``, which tends to 1 as ``n`` grows. Hence for any fixed local edit, however
meaning-destroying, ``sup`` over documents of the reported similarity is exactly 1. ∎

The consequence is not hypothetical. On the 208-call BFCL executable suite the correlation
between a pair's leaf count and its STED score is **+0.639** on the sign-flip operator (N6),
**+0.640** on numeric jitter (N1) and **+0.548** on the argument-value swap (N2), while the
correlation between leaf count and the *execution* verdict is slightly negative (−0.153,
+0.018, −0.203). The score is therefore driven by how much of the document the edit touched
and the oracle by whether the callee is invariant to it, and on this suite the two run in
opposite directions: within-operator AUC is 0.3225 on N6, i.e. below chance. Splitting N6 by
document size shows the confound directly — scalar-argument calls (<= 4 leaves) mean 0.8577
with a 15.6% oracle-equivalence rate, array calls (>= 10 leaves) mean 0.9725 with a 0.0%
rate. Flipping the sign of one element of a 20-number array scores 0.9917; flipping the sign
of ``math_gcd``'s first argument, which the callee ignores, scores 0.875.

This is also the general form of Instrument D's ceiling false accepts, and it is not specific
to STED: SLOT averages over key paths (0.3259 on N6) and every string baseline normalises by
length.

What this module does instead
-----------------------------
It replaces the mean over children with the **power mean of the children's similarities**,

    s_p(node) = ( (1/n) * sum_i s_i^p )^(1/p),      p in (-inf, 1]
    gamma_p(node) = 1 - s_p(node)

with ``p = 1`` the arithmetic mean (exactly today's behaviour, since
``(1/n) sum (1 - c_i) = 1 - (1/n) sum c_i``), ``p -> 0`` the geometric mean, and ``p -> -inf``
the minimum. The axis it parameterises is what a JSON object *is* with respect to its
children: at ``p = 1`` an **aggregate** — an average of independent evidence, where a good
field can offset a bad one — and as ``p`` falls a **conjunction** — a set of requirements, all
of which must hold. A tool callee is a conjunction: every argument has to be right. A record
being scored for overall fidelity is an aggregate. Which one applies is a fact about what the
score will be used for, so, as with ``mu``, there is no ``'auto'``.

**Lemma (bracketing).** ``s_p`` is non-decreasing in ``p``, so ``s_p <= s_1`` for all
``p <= 1``: the published score is the permissive end of the interval, and the width
``s_1 - s_p`` is the part of the verdict that rests on reading the object as an aggregate.

How much of the dilution a finite ``p`` actually removes
-------------------------------------------------------
Less than all of it, and the residue has a closed form. With one destroyed child of
similarity ``s`` among ``n``, ``M_p = ((s^p + n - 1)/n)^(1/p)``, and for ``p < 0`` with
``s^|p| << n`` this is ``s * n^(1/|p|) + O(...)``. So the score still climbs with document
size, just polynomially more slowly. Measured on ``calculate_mean`` with the first array
element's sign flipped (``s ~ 0.5``), score against ``n = 1, 2, 5, 10, 20, 50, 100``:

    p =   1   0.7500 0.8750 0.9500 0.9750 0.9875 0.9950 0.9975   log-log slope 0.0300
    p =   0   0.7071 0.8409 0.9330 0.9659 0.9828 0.9931 0.9965                 0.0391
    p =  -2   0.6325 0.7559 0.8771 0.9325 0.9645 0.9853 0.9926                 0.0646
    p =  -8   0.5450 0.5937 0.6639 0.7206 0.7790 0.8535 0.9024                 0.1077
    p = -16   0.5221 0.5453 0.5774 0.6029 0.6296 0.6667 0.6961                 0.0625
    p = -32   0.5109 0.5221 0.5373 0.5491 0.5611 0.5774 0.5900                 0.0312
    p = -inf  0.5000 0.5000 0.5000 0.5000 0.5000 0.5000 0.5000                 0.0000

For ``p <= -16`` the fitted slope is ``1/|p|`` to four decimals (0.0625, 0.0312); at ``p = 1``
and ``p = -8`` the fit is contaminated by saturation near 1. The right way to read the table is
by *deficit*: at ``p = 1`` a tenfold larger array divides the penalty by 10.0
(0.1250 -> 0.0125), at ``p = -8`` by 1.84 (0.4063 -> 0.2210), at ``p = -32`` by 1.08, and at
``p = -inf`` by exactly 1.00 — the last row is flat to 1e-6, which is the whole content of the
bottleneck section below.

Two consequences, stated plainly because they bound what this axis can claim:

* **The dilution proposition survives every finite ``p``.** ``sup_n M_p = 1``, so no finite
  ``p`` makes a threshold sound; it only raises the ``n`` at which a fixed edit becomes
  invisible, from ``O(1/eps)`` to ``O(eps^-|p|)``.
* **Only ``p = -inf`` is size-free**, because ``min_i s_i`` does not mention ``n``. That is the
  reading under which ``score >= theta`` certifies that *every* matched pair at every level
  scores at least ``theta`` — the only member of the family that can carry a threshold. It is
  also the one that stops being a linear assignment problem; see ``bottleneck_assignment``.

*Proof.* By the power-mean inequality (Hardy–Littlewood–Pólya, Thm. 16), for a fixed vector
of similarities ``M_p <= M_q`` whenever ``p <= q``. Induct on depth. At a leaf ``s_p`` does
not depend on ``p``. At an interior node let ``S^{(p)}`` be the matrix of child similarities
computed at ``p``; by the inductive hypothesis ``S^{(p)} <= S^{(q)}`` entrywise. ``M_q`` is
non-decreasing in each argument, so for the correspondence ``A_p`` this node selects at ``p``,

    s_p = M_p(S^{(p)}|A_p) <= M_q(S^{(p)}|A_p) <= M_q(S^{(q)}|A_p) <= max_A M_q(S^{(q)}|A) = s_q

where the last step needs the node's correspondence to be the **argmax of its own
aggregate**. ∎

That last step is why ``objective_matrix`` exists. If the correspondence were held at the
``p = 1`` optimum and only the aggregation changed, the final inequality would compare a
maximum against an arbitrary feasible point in the wrong direction and the lemma would fail
— monotonicity is not a free consequence of using a power mean, it is a constraint on which
correspondence is chosen. Fortunately maximising ``M_p`` over assignments is still a *linear*
assignment problem, because ``M_p`` is a strictly monotone function of ``sum_i s_i^p``:

    p > 0   maximise sum s^p        -> minimise  -(s^p)
    p = 0   maximise sum log s      -> minimise  -log s
    p < 0   maximise M_p            -> minimise  sum s^p   (t^(1/p) is decreasing)

so ``scipy.optimize.linear_sum_assignment`` on the reparameterised matrix returns the exact
optimum of the aggregate actually being reported. At ``p = 1`` the reparameterisation is the
identity, so both the assignment and the score are bit-identical to the published metric.

The bottleneck end, and what it buys
------------------------------------
At ``p = -inf`` the node's cost is ``max_i c_i`` over the charged pairs and the correspondence
must minimise that maximum. This is the **bottleneck assignment problem**: not an LAP, because
the objective is not a sum, and not solvable by reparameterising the matrix. It is solvable
exactly, by binary search on the sorted distinct costs with a bipartite-matching feasibility
test at each candidate threshold (``bottleneck_assignment``), which is what makes the following
statable at all.

**Theorem (threshold soundness).** Let ``s_min(a, b)`` be the score computed with
``p = -inf`` at every level. Then ``s_min(a, b) = min`` over all charged pairs of the recursive
correspondence of their similarities, and therefore

    s_min(a, b) >= theta   ==>   every matched pair at every level, and in particular every
                                 matched leaf pair, has similarity >= theta,

with no dependence on any node's arity.

*Proof.* Induction on depth: the max of maxima over a nested family of correspondences is the
max over their union, so ``1 - s_min`` is the largest charge anywhere in the correspondence,
including the flat indel charges for unmatched children. ``n`` appears nowhere. ∎

This is exactly the guarantee the dilution proposition denies for every finite ``p``, and it
holds for *any* correspondence, optimal or not — optimality only affects how many equivalent
pairs are falsely rejected, never the soundness of the certificate. The price is that the score
is a minimum: it reports the worst part of the document and nothing about the rest, so it is a
certificate, not a graded similarity. The two readings answer different questions and the
interval between them is the honest object.

Scope and limits
----------------
* ``p = -inf`` **is** offered, and it is the only member of the family that closes the
  dilution rather than narrowing it (see ``bottleneck_assignment`` and the soundness theorem
  above). It is not a power mean and needs its own solver, so it is a separate code path
  rather than a limiting value of the ``p`` arithmetic. Finite large-negative ``p`` remains
  useful as a cheap approximation from above.
* ``p != 1`` implies the *per-pair* channel combination. Aggregating the structural and
  content channels separately and combining the two totals is only equivalent to combining
  per pair when the aggregation is linear, which is exactly what ``p != 1`` gives up. This is
  the same reasoning that gives ``cost_combination='product'`` its own joint channel in
  ``_matching.get_channel_costs``, and it means ``p`` and ``product`` are the conjunctive
  reading applied at two different levels: ``product`` conjoins the two channels of one node,
  ``p`` conjoins the children of one node.
* The order-sensitive (``order_sensitive_fields``) positional paths are not covered, the same
  scope as ``array_order_lambda``; they keep the mean. ``_aggregation_p_active`` is checked at
  every site that is covered, so this is a gap in coverage, not a silent inconsistency.
"""

import math
from typing import Any, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching

# Below this the reparameterised objective would overflow float64 for plausible p. It bounds
# the *assignment* only: the reported aggregate uses the true similarities, and a child with
# similarity exactly 0 makes the aggregate exactly 0 for every p <= 0 regardless.
_SIM_FLOOR = 1e-9
_OBJ_CAP = 1e30


def resolve_p(spec: Any, default: float = 1.0) -> float:
    """Validate an aggregation exponent. ``None`` means the published arithmetic mean.

    ``p > 1`` is rejected rather than clamped. It is well defined mathematically and would
    make the score *more* dilution-prone than the published metric, which no deployment has a
    reason to want, and it would put the published value inside the interval instead of at its
    permissive end — the property every claim in this module is stated against.
    """
    if spec is None:
        return float(default)
    if isinstance(spec, str):
        if spec == 'mean':
            return 1.0
        if spec == 'geometric':
            return 0.0
        if spec in ('min', 'bottleneck'):
            return -math.inf
        raise ValueError(
            f"aggregation_p must be None or a number <= 1 "
            f"(or 'mean'/'geometric'/'min'), got {spec!r}")
    if isinstance(spec, bool) or not isinstance(spec, (int, float)):
        raise ValueError(f"aggregation_p must be None or a number <= 1, got {spec!r}")
    p = float(spec)
    if math.isnan(p):
        raise ValueError('aggregation_p must not be NaN')
    if p > 1.0:
        raise ValueError(f'aggregation_p must be <= 1 (1.0 is the published mean), got {p}')
    if p == math.inf:                                   # pragma: no cover - p > 1 caught above
        raise ValueError('aggregation_p=+inf is not a valid aggregation')
    return p


def is_bottleneck(p: float) -> bool:
    """True for the ``p = -inf`` end, which is a different algorithm, not a different exponent."""
    return p == -math.inf


def is_active(p: float) -> bool:
    """True when ``p`` changes anything. Call sites keep their published expression when False."""
    return p != 1.0


def aggregate(costs: Sequence[float], p: float) -> float:
    """The node's cost under the power mean of its matched children's similarities.

    ``costs`` is the vector of per-pair costs actually charged at this level, including the
    flat insert/delete charges for unmatched children — the same terms the published mean
    sums. Its length is the level's denominator, so nothing about the normalisation changes
    except the shape of the average.
    """
    n = len(costs)
    if n == 0:
        return 0.0
    sims = [1.0 - c for c in costs]
    sims = [1.0 if s > 1.0 else (0.0 if s < 0.0 else s) for s in sims]
    if is_bottleneck(p):
        # The worst child, and nothing else. No n anywhere: this is the size-free end.
        return 1.0 - min(sims)
    if p == 1.0:
        mean = math.fsum(sims) / n
    elif p == 0.0:
        if any(s <= 0.0 for s in sims):
            mean = 0.0
        else:
            mean = math.exp(math.fsum(math.log(s) for s in sims) / n)
    elif p < 0.0:
        if any(s <= 0.0 for s in sims):
            mean = 0.0
        else:
            mean = (math.fsum(s ** p for s in sims) / n) ** (1.0 / p)
    else:
        mean = (math.fsum(s ** p for s in sims) / n) ** (1.0 / p)
    cost = 1.0 - mean
    return 1.0 if cost > 1.0 else (0.0 if cost < 0.0 else cost)


def objective_matrix(cost_matrix: np.ndarray, p: float) -> np.ndarray:
    """Reparameterise a cost matrix so that its minimum-sum assignment maximises ``M_p``.

    Returns ``cost_matrix`` itself at ``p = 1``, so the published assignment is untouched, and
    also at ``p = -inf``, where the objective is ``max`` of the raw costs and ``max`` is already
    monotone in cost — there is nothing to reparameterise, only a different solver
    (``bottleneck_assignment``) and a different DP accumulator (``combine_mode``).
    See the module docstring for the three finite cases; the transform is strictly increasing in
    cost in every one of them, which is what makes the reparameterisation faithful.
    """
    if p == 1.0 or is_bottleneck(p):
        return cost_matrix
    sims = np.clip(1.0 - np.asarray(cost_matrix, dtype=np.float64), _SIM_FLOOR, 1.0)
    with np.errstate(over='ignore', divide='ignore'):
        if p == 0.0:
            obj = -np.log(sims)
        elif p < 0.0:
            obj = np.power(sims, p)
        else:
            obj = -np.power(sims, p)
    # inf entries mark forbidden matches in the padded region of the caller's matrix; they
    # must stay large-but-finite so linear_sum_assignment does not report infeasibility.
    return np.nan_to_num(obj, nan=_OBJ_CAP, posinf=_OBJ_CAP, neginf=-_OBJ_CAP)


def combine_mode(p: float) -> str:
    """What an order-preserving DP over the same children must accumulate at this ``p``.

    ``'max'`` only at the bottleneck end; ``'sum'`` everywhere else, because for finite ``p``
    the aggregate is a strictly monotone function of the *sum* of the reparameterised costs.
    """
    return 'max' if is_bottleneck(p) else 'sum'


def bottleneck_assignment(cost_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Assignment minimising the **largest** matched cost, with ties broken by total cost.

    Same signature and contract as ``scipy.optimize.linear_sum_assignment``: returns
    ``(row_indices, col_indices)`` selecting ``min(n_rows, n_cols)`` cells, one per row and per
    column. The objective is different — ``min over A of max over (i,j) in A of C[i,j]`` — and
    that objective is *not* a sum, so no reparameterisation of ``C`` turns it into an LAP; this
    is why ``p = -inf`` needs its own solver rather than an exponent.

    Algorithm: the optimal bottleneck value is one of the entries of ``C``, and feasibility is
    monotone in the threshold (allowing more edges can only help), so binary search the sorted
    distinct finite entries and test each candidate ``t`` by asking whether the bipartite graph
    of edges ``C <= t`` admits a matching of size ``min(n_rows, n_cols)``. Hopcroft–Karp answers
    that in O(E sqrt(V)), giving O(E sqrt(V) log |C|) overall — irrelevant at the sizes here (a
    node's children), and exact rather than approximate, which is the point.

    Among the assignments achieving the optimal bottleneck there are usually many, and they are
    not equally good: they differ in the costs the bottleneck does not constrain. Rather than let
    the matching algorithm's internal order decide, the final step runs a normal LAP on the true
    costs with the infeasible edges priced at ``_OBJ_CAP``. Since a feasible assignment exists
    within the surviving edges and every true cost is <= 1, that LAP can never prefer a capped
    edge, so it returns a minimum-total assignment among the minimum-bottleneck ones. This makes
    the result deterministic and keeps the unconstrained pairs sensible; the reported score
    depends only on the bottleneck, so it is unaffected either way.
    """
    m = np.asarray(cost_matrix, dtype=np.float64)
    if m.size == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    need = min(m.shape)
    finite = m[np.isfinite(m)]

    def feasible(t: float) -> bool:
        adj = csr_matrix(m <= t)
        if adj.nnz == 0:
            return need == 0
        matched = maximum_bipartite_matching(adj, perm_type='column')
        return int(np.count_nonzero(matched >= 0)) >= need

    if finite.size == 0 or not feasible(float(finite.max())):
        # Either every cell is forbidden, or no complete assignment avoids the forbidden ones.
        # There is no bottleneck left to optimise, but the contract is a complete assignment, so
        # price the forbidden cells at the cap rather than letting the LAP refuse the problem.
        return linear_sum_assignment(np.where(np.isfinite(m), m, _OBJ_CAP))
    values = np.unique(finite)
    lo, hi = 0, len(values) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if feasible(float(values[mid])):
            hi = mid
        else:
            lo = mid + 1
    t_star = float(values[lo])
    restricted = np.where(m <= t_star, m, _OBJ_CAP)
    return linear_sum_assignment(restricted)


def assign(cost_matrix: np.ndarray, p: float) -> Tuple[np.ndarray, np.ndarray]:
    """The correspondence that is the argmax of *this node's own* aggregate.

    The bracketing lemma's last step needs exactly this and nothing weaker; see the module
    docstring. ``p = 1`` is ``linear_sum_assignment(cost_matrix)`` verbatim, so the published
    assignment — including its tie-breaking — is untouched.
    """
    if is_bottleneck(p):
        return bottleneck_assignment(cost_matrix)
    return linear_sum_assignment(objective_matrix(cost_matrix, p))
