"""Hungarian / matching algorithms for STED.

Extracted from ``semantic_json_tree_consistency.py`` during the v0.2.0 refactor.
The original methods on ``SemanticJsonTreeConsistencyEvaluator`` are preserved
as thin wrappers that delegate to these module-level helpers.

All helpers take the evaluator instance as their first argument so they can
read/write the evaluator's caches and configuration (``_subtree_cache``,
``_compute_tree_hash``, ``update_cost``/``insert_cost``/``delete_cost``,
``structural_update_cost``/``content_update_cost``, ``structural_weight``,
``use_greedy_matching``, ``early_pruning_threshold``, ``type_change_cost``,
``_is_order_sensitive_field``).
"""
from __future__ import annotations

from typing import List, Optional, Tuple, TYPE_CHECKING

import numpy as np
from scipy.optimize import linear_sum_assignment

from . import _aggregation, _key_identity, _order

if TYPE_CHECKING:  # pragma: no cover - only for type checkers
    from .json_tree_node import JsonNode


def _aggregation_p(evaluator) -> float:
    """The child-aggregation exponent, or 1.0 (the published arithmetic mean) when unset.

    Unlike λ this is a single evaluator-wide constant, so it is resolved once per call site
    rather than per node pair, and it is *not* part of the subtree cache key — the
    ``aggregation_p`` setter invalidates that cache instead. See ``sted._aggregation``.
    """
    return _aggregation.resolve_p(getattr(evaluator, 'aggregation_p', None))


def _obj_skips(costs, p):
    """``costs`` reparameterised for the alignment DP, whose skip charges are summed with the
    pair costs and so must live in the same objective as ``_assign``'s matrix."""
    return [float(x) for x in _aggregation.objective_matrix(np.asarray(costs, dtype=float), p)]


def _assign(matrix, p):
    """Assignment indices for ``matrix`` under the ``p``-aggregate's own objective.

    At ``p = 1`` this is ``linear_sum_assignment(matrix)`` verbatim; for finite ``p != 1`` the
    matrix is reparameterised first, and at ``p = -inf`` a bottleneck solver replaces the LAP
    entirely. That is what makes the reported aggregate the optimum of the objective it is
    computed from rather than the value of a different objective's argmin. The costs *charged*
    are always read back from ``matrix``, never from the reparameterisation.
    """
    return _aggregation.assign(matrix, p)


def _align(pair_costs, del_costs, ins_costs, p):
    """``_order.monotone_alignment`` with the accumulator this ``p`` requires.

    The order-preserving path has to optimise the same objective as the assignment path, or the
    two ends of the λ blend would come from different aggregations and the interval would not
    bracket anything.
    """
    return _order.monotone_alignment(
        _aggregation.objective_matrix(pair_costs, p),
        _obj_skips(del_costs, p), _obj_skips(ins_costs, p),
        combine=_aggregation.combine_mode(p))


def _array_order_lambda(evaluator, tree1: "JsonNode", tree2: "JsonNode") -> float:
    """Order-sensitivity weight for this array pair, or 0.0 when the feature is off.

    Returns 0.0 -- i.e. the pure Hungarian bag reading STED has always used -- unless the
    evaluator carries an ``array_order_lambda`` setting and both nodes are arrays. Keeping
    the gate here means every existing call site is bit-identical by default.
    """
    spec = getattr(evaluator, 'array_order_lambda', None)
    if spec is None:
        return 0.0
    if tree1.node_type != 'array' or tree2.node_type != 'array':
        return 0.0
    return _order.resolve_lambda(
        spec, tree1, tree2,
        default=getattr(evaluator, 'array_order_lambda_default', 0.0))


def greedy_matching(
    evaluator,
    children1: List["JsonNode"],
    children2: List["JsonNode"],
    variation_type: str,
) -> Tuple[List[Tuple[int, int]], float]:
    """Greedy matching approximation - O(B^2) instead of O(B^3) Hungarian.

    Was ``SemanticJsonTreeConsistencyEvaluator._greedy_matching``.
    """
    n1, n2 = len(children1), len(children2)
    matched_pairs: List[Tuple[int, int]] = []
    total_cost = 0.0
    used_j = set()

    # For each node in children1, find best match in children2
    for i in range(n1):
        best_j = -1
        best_cost = float('inf')

        for j in range(n2):
            if j in used_j:
                continue

            # Calculate cost for this pair
            cost = calculate_optimal_matching_cost_fast(
                evaluator, children1[i], children2[j], variation_type
            )

            if cost < best_cost:
                best_cost = cost
                best_j = j

        if best_j >= 0:
            matched_pairs.append((i, best_j))
            used_j.add(best_j)
            total_cost += best_cost
        else:
            # No match found, count as deletion
            total_cost += evaluator.delete_cost(children1[i])

    # Add insertion costs for unmatched children2
    for j in range(n2):
        if j not in used_j:
            total_cost += evaluator.insert_cost(children2[j])

    return matched_pairs, total_cost


def greedy_matching_streaming(
    evaluator,
    children1: List["JsonNode"],
    children2: List["JsonNode"],
    variation_type: str,
) -> float:
    """Space-optimized greedy matching - O(B) space instead of O(B^2).

    Was ``SemanticJsonTreeConsistencyEvaluator._greedy_matching_streaming``.
    """
    n1, n2 = len(children1), len(children2)
    total_cost = 0.0
    used_j = set()  # O(min(n1, n2)) space

    # For each node in children1, find best match in children2
    for i in range(n1):
        best_j = -1
        best_cost = float('inf')

        # Stream through children2, computing costs on-demand
        for j in range(n2):
            if j in used_j:
                continue

            # Calculate cost for this pair (on-the-fly, not stored)
            cost = calculate_optimal_matching_cost_fast(
                evaluator, children1[i], children2[j], variation_type
            )

            if cost < best_cost:
                best_cost = cost
                best_j = j

        if best_j >= 0:
            used_j.add(best_j)
            total_cost += best_cost
        else:
            # No match found, count as deletion
            total_cost += evaluator.delete_cost(children1[i])

    # Add insertion costs for unmatched children2
    for j in range(n2):
        if j not in used_j:
            total_cost += evaluator.insert_cost(children2[j])

    return total_cost


def early_prune_check(
    evaluator, node1: "JsonNode", node2: "JsonNode"
) -> Optional[float]:
    """Check if we can skip detailed comparison based on early pruning criteria.

    Was ``SemanticJsonTreeConsistencyEvaluator._early_prune_check``.
    """
    # Type mismatch with incompatible types
    type_cost = evaluator.type_change_cost.get((node1.node_type, node2.node_type), 1.0)
    if type_cost >= evaluator.early_pruning_threshold:
        # Skip detailed comparison, return high cost
        return type_cost

    # Structural mismatch: one side has children and the other does not, so every child of the
    # non-empty side is unmatched at a cost of 1.0 and there is nothing to match it against. Under
    # Eq. (3) the level is normalised by max(|C_1|, |C_2|) = the number of those children, so the
    # cost is exactly 1.0 for any subtree size. This is a genuine shortcut: it returns the value the
    # recursion would return, not an estimate of it. It previously returned 0.9, which is below the
    # only value the true cost can take and disagreed by 0.1 with the interior recursion's answer
    # for the same configuration (see D3a).
    if bool(node1.children) != bool(node2.children):
        return 1.0

    # No size-ratio shortcut. This used to return `min(1.0, 0.3 + 0.1 * ratio)` whenever the child
    # counts differed by more than 3x, which is neither the true cost nor a bound on it. With
    # n1 < n2 children at least n2 - n1 are unmatched at 1.0 each and matched pairs cost >= 0, so
    # the true cost lies in [1 - min/max, 1.0]: at ratio 4 the formula returns 0.70 against a floor
    # of 0.75, while from ratio 7 up it clamps to 1.0 and overshoots (1-vs-10 keys: 1.0 returned,
    # 0.9 exact). The error therefore ran in both directions and was not a controlled approximation.
    # Skipping the comparison also cannot be justified on the grounds that the answer is nearly
    # saturated: the min(n1, n2) elements that *do* pair up are exactly what decides how much of the
    # level matched. Dropping the branch costs one Hungarian solve on lopsided nodes and makes the
    # projections agree with combined mode, which never reached this code at all.

    return None  # No pruning, proceed with full comparison


def calculate_optimal_matching_cost_fast(
    evaluator,
    tree1: "JsonNode",
    tree2: "JsonNode",
    variation_type: str = "combined",
) -> float:
    """Optimized version of ``_calculate_optimal_matching_cost``.

    Was ``SemanticJsonTreeConsistencyEvaluator._calculate_optimal_matching_cost_fast``.
    """
    # === Optimization 1: Memoization with LRU cache ===
    # lam is part of the key because it is a property of the node pair's *context* (its
    # equal-length siblings), not of the subtree the hash covers, and the hash strips array
    # indices. Two array pairs with identical content but different order-sensitivity would
    # otherwise share an entry. It is (0.0,) whenever the blend is off, so keys are
    # unchanged in shape for every existing caller.
    lam = _array_order_lambda(evaluator, tree1, tree2)
    cache_key = (
        evaluator._compute_tree_hash(tree1),
        evaluator._compute_tree_hash(tree2),
        variation_type,
        lam,
    )
    cached_result = evaluator._subtree_cache.get(cache_key)
    if cached_result is not None:
        return cached_result

    # === Optimization 5: Early pruning ===
    pruned_cost = early_prune_check(evaluator, tree1, tree2)
    if pruned_cost is not None:
        evaluator._subtree_cache.set(cache_key, pruned_cost)
        return pruned_cost

    # Base case: both are leaf nodes
    if not tree1.children and not tree2.children:
        if variation_type == "structural":
            cost = evaluator.structural_update_cost(tree1, tree2)
        elif variation_type == "content":
            cost = evaluator.content_update_cost(tree1, tree2)
        else:  # combined
            cost = evaluator.update_cost(tree1, tree2)
        evaluator._subtree_cache.set(cache_key, cost)
        return cost

    # If one is leaf and other is not.
    # Normalize to [0, 1] per Eq. (3): divide total edit cost by max(|C_1|, |C_2|).
    # Without this normalization, cost scales with the non-leaf subtree size and the
    # outer `similarity = 1 - distance` can fall below 0 when this branch fires at
    # the root.
    if not tree1.children and tree2.children:
        raw_cost = evaluator.delete_cost(tree1) + sum(
            evaluator.insert_cost(c) for c in tree2.children
        )
        denom = max(1, len(tree2.children))
        cost = min(raw_cost / denom, 1.0)
        evaluator._subtree_cache.set(cache_key, cost)
        return cost
    if tree1.children and not tree2.children:
        raw_cost = sum(evaluator.delete_cost(c) for c in tree1.children) + evaluator.insert_cost(tree2)
        denom = max(1, len(tree1.children))
        cost = min(raw_cost / denom, 1.0)
        evaluator._subtree_cache.set(cache_key, cost)
        return cost

    # Both have children
    children1 = tree1.children
    children2 = tree2.children

    # Check for order-sensitive arrays
    is_order_sensitive = (
        tree1.node_type == "array" and tree2.node_type == "array" and
        (evaluator._is_order_sensitive_field(tree1.label) or evaluator._is_order_sensitive_field(tree2.label))
    )

    if is_order_sensitive:
        cost = calculate_sequential_matching_cost(evaluator, tree1, tree2, variation_type)
        evaluator._subtree_cache.set(cache_key, cost)
        return cost

    n1, n2 = len(children1), len(children2)

    if n1 == 0 and n2 == 0:
        if variation_type == "structural":
            cost = evaluator.structural_update_cost(tree1, tree2)
        elif variation_type == "content":
            cost = evaluator.content_update_cost(tree1, tree2)
        else:
            cost = evaluator.update_cost(tree1, tree2)
        evaluator._subtree_cache.set(cache_key, cost)
        return cost

    # === Optimization 3: Greedy matching with streaming (O(B) space) ===
    # Skipped for array pairs under an order blend: the blend needs the full pair-cost
    # matrix for the alignment DP, so the O(B) space saving is not available anyway, and
    # greedy's approximate assignment would break the bracketing lemma (a greedy total can
    # exceed the monotone optimum, making the "order-free" side the larger of the two).
    # Skipped under an aggregation exponent for the same reason: greedy returns a total, and a
    # power mean cannot be recovered from one. It is also an approximate assignment, which
    # would break the bracketing lemma the same way it breaks the order one.
    if (evaluator.use_greedy_matching and max(n1, n2) > 5
            and _array_order_lambda(evaluator, tree1, tree2) <= 0.0
            and not _aggregation.is_active(_aggregation_p(evaluator))):
        total_cost = greedy_matching_streaming(evaluator, children1, children2, variation_type)
        normalized_cost = min(total_cost / max(n1, n2, 1), 1.0)
        evaluator._subtree_cache.set(cache_key, normalized_cost)
        return normalized_cost

    # === Optimization 2: Single-pass combined calculation ===
    if variation_type == "combined":
        cost = single_pass_combined_matching(
            evaluator, children1, children2, n1, n2,
            lam=_array_order_lambda(evaluator, tree1, tree2),
            p=_aggregation_p(evaluator))
        evaluator._subtree_cache.set(cache_key, cost)
        return cost

    # For structural or content only, use NumPy array for memory efficiency
    max_size = max(n1, n2)
    cost_matrix = np.full((max_size, max_size), np.inf, dtype=np.float32)

    for i in range(n1):
        for j in range(n2):
            cost_matrix[i, j] = calculate_optimal_matching_cost_fast(
                evaluator, children1[i], children2[j], variation_type
            )

    # Add deletion/insertion costs (vectorized for efficiency)
    for i in range(n1):
        del_cost = evaluator.delete_cost(children1[i])
        cost_matrix[i, n2:max_size] = del_cost
    for j in range(n2):
        ins_cost = evaluator.insert_cost(children2[j])
        cost_matrix[n1:max_size, j] = ins_cost

    # Hungarian algorithm
    p = _aggregation_p(evaluator)
    row_indices, col_indices = _assign(cost_matrix, p)

    if _aggregation.is_active(p):
        normalized_cost = _aggregation.aggregate(
            [float(cost_matrix[i, j]) for i, j in zip(row_indices, col_indices)], p)
    else:
        total_cost = float(cost_matrix[row_indices, col_indices].sum())
        normalized_cost = min(total_cost / len(row_indices), 1.0)

    # Order-sensitivity blend for array pairs. lam == 0.0 (the default, and every non-array
    # node) leaves ``normalized_cost`` untouched, so this is inert unless asked for.
    lam = _array_order_lambda(evaluator, tree1, tree2)
    if lam > 0.0:
        del_costs = [evaluator.delete_cost(c) for c in children1]
        ins_costs = [evaluator.insert_cost(c) for c in children2]
        if _aggregation.is_active(p):
            # Align on the p-objective the assignment used, then aggregate the true costs
            # along that alignment -- the same discipline as the free side above.
            _, pairs, skip1, skip2 = _align(
                cost_matrix[:n1, :n2], del_costs, ins_costs, p)
            charged = [float(cost_matrix[i, j]) for i, j in pairs]
            charged += [del_costs[i] for i in skip1] + [ins_costs[j] for j in skip2]
            c_ord = _aggregation.aggregate(charged, p)
        else:
            c_ord = min(_order.monotone_alignment_cost(
                cost_matrix[:n1, :n2], del_costs, ins_costs) / max(n1, n2, 1), 1.0)
        normalized_cost = _order.blend(normalized_cost, c_ord, lam)

    evaluator._subtree_cache.set(cache_key, normalized_cost)
    return normalized_cost


def single_pass_combined_matching(
    evaluator,
    children1: List["JsonNode"],
    children2: List["JsonNode"],
    n1: int,
    n2: int,
    lam: float = 0.0,
    p: float = 1.0,
) -> float:
    """Single-pass combined calculation that computes structural and content costs together.

    Was ``SemanticJsonTreeConsistencyEvaluator._single_pass_combined_matching``.

    ``lam`` is the order-sensitivity weight for this node when it is an array pair; the
    caller resolves it (``_array_order_lambda``) because the child lists alone do not say
    which nodes they came from. At the default 0.0 the function is unchanged.

    ``p`` is the child-aggregation exponent (``sted._aggregation``). At the default 1.0 the
    function is unchanged. Any other value routes through the ``joint_costs`` branch even
    under ``'additive'``: a nonlinear aggregate of ``w * sum(s) + (1-w) * sum(c)`` is not the
    aggregate of the per-pair combination, and it is the per-pair combination that Eq. (2)
    describes. The two coincide at ``p = 1``, which is why the additive branch below can keep
    charging the channel totals.
    """
    max_size = max(n1, n2)

    # Build combined cost matrix in single pass using NumPy (float32 saves memory)
    structural_costs = np.zeros((max_size, max_size), dtype=np.float32)
    content_costs = np.zeros((max_size, max_size), dtype=np.float32)

    joint_costs = np.zeros((max_size, max_size), dtype=np.float32)

    for i in range(n1):
        for j in range(n2):
            # Get all three costs in single recursive call
            s_cost, c_cost, j_cost = get_channel_costs(
                evaluator, children1[i], children2[j]
            )
            structural_costs[i, j] = s_cost
            content_costs[i, j] = c_cost
            joint_costs[i, j] = j_cost

    # Add deletion costs (vectorized)
    for i in range(n1):
        del_cost = evaluator.delete_cost(children1[i])
        structural_costs[i, n2:max_size] = del_cost
        content_costs[i, n2:max_size] = del_cost
        joint_costs[i, n2:max_size] = evaluator.combine_channels(del_cost, del_cost)

    # Add insertion costs (vectorized)
    for j in range(n2):
        ins_cost = evaluator.insert_cost(children2[j])
        structural_costs[n1:max_size, j] = ins_cost
        content_costs[n1:max_size, j] = ins_cost
        joint_costs[n1:max_size, j] = evaluator.combine_channels(ins_cost, ins_cost)

    # Match on the objective the metric actually declares.
    #
    # This used to run linear_sum_assignment on ``structural_costs`` alone and then pay whatever
    # content cost the structurally-chosen correspondence happened to incur, so the returned value
    # was not the minimum of Eq. (1) over correspondences — it was Eq. (1) evaluated at the
    # argmin of a different function. The two disagree whenever structure does not separate the
    # candidates: array elements under one key have identical index-stripped paths, which makes the
    # structural matrix constant and hands the assignment to ``linear_sum_assignment``'s tie-break
    # order. A permuted pair of tool calls with a perfect content correspondence available scored
    # 0.157 where the objective's minimum is 0.0 (probe P15).
    if evaluator.cost_combination == 'additive' and not _aggregation.is_active(p):
        combined_costs = combine_channel_matrix(evaluator, structural_costs, content_costs)
        row_indices, col_indices = linear_sum_assignment(combined_costs)

        # Calculate final costs using the matched pairs (vectorized)
        structural_total = float(structural_costs[row_indices, col_indices].sum())
        content_total = float(content_costs[row_indices, col_indices].sum())

        # Paper Eq.(1): w * structural + (1-w) * content
        total_cost = (
            evaluator.structural_weight * structural_total
            + (1 - evaluator.structural_weight) * content_total
        )
        if lam > 0.0:
            # Align on the same objective the assignment minimised, then charge the two
            # channels along that one correspondence -- the same discipline the assignment
            # branch follows, and for the same reason.
            _, pairs, skip1, skip2 = _order.monotone_alignment(
                combined_costs[:n1, :n2],
                [evaluator.delete_cost(c) for c in children1],
                [evaluator.insert_cost(c) for c in children2],
            )
            s_ord = sum(float(structural_costs[i, j]) for i, j in pairs)
            c_ord = sum(float(content_costs[i, j]) for i, j in pairs)
            s_ord += sum(evaluator.delete_cost(children1[i]) for i in skip1)
            c_ord += sum(evaluator.delete_cost(children1[i]) for i in skip1)
            s_ord += sum(evaluator.insert_cost(children2[j]) for j in skip2)
            c_ord += sum(evaluator.insert_cost(children2[j]) for j in skip2)
            ord_total = (evaluator.structural_weight * s_ord
                         + (1 - evaluator.structural_weight) * c_ord)
            denom = max(len(row_indices), 1)
            return _order.blend(min(total_cost / denom, 1.0),
                                min(ord_total / denom, 1.0), lam)
    else:
        # ``joint_costs`` is already the per-pair combination, conjoined at the leaves and
        # aggregated by the same normalise-then-sum recursion, so it is both the objective to
        # minimise and the total to charge. Conjoining ``structural_total`` with ``content_total``
        # instead would be the error this whole change exists to remove: a t-norm of two sums is
        # not the sum of the t-norms, and the defect lives at individual leaves.
        row_indices, col_indices = _assign(joint_costs, p)
        j_del = [evaluator.combine_channels(evaluator.delete_cost(c), evaluator.delete_cost(c))
                 for c in children1]
        j_ins = [evaluator.combine_channels(evaluator.insert_cost(c), evaluator.insert_cost(c))
                 for c in children2]
        if _aggregation.is_active(p):
            free_cost = _aggregation.aggregate(
                [float(joint_costs[i, j]) for i, j in zip(row_indices, col_indices)], p)
            if lam > 0.0:
                _, pairs, skip1, skip2 = _align(
                    joint_costs[:n1, :n2], j_del, j_ins, p)
                charged = [float(joint_costs[i, j]) for i, j in pairs]
                charged += [j_del[i] for i in skip1] + [j_ins[j] for j in skip2]
                return _order.blend(free_cost, _aggregation.aggregate(charged, p), lam)
            return free_cost
        total_cost = float(joint_costs[row_indices, col_indices].sum())
        if lam > 0.0:
            ord_total = _order.monotone_alignment_cost(
                joint_costs[:n1, :n2], j_del, j_ins)
            denom = max(len(row_indices), 1)
            return _order.blend(min(total_cost / denom, 1.0),
                                min(ord_total / denom, 1.0), lam)
    normalized_cost = min(total_cost / len(row_indices), 1.0)

    return normalized_cost


def combine_channel_matrix(evaluator, s_costs, c_costs):
    """Elementwise channel combination, matching ``evaluator.combine_channels``.

    Under ``'additive'`` this is Eq. (2) and callers keep their original arithmetic, so published
    numbers are reproduced bit-for-bit. Under ``'product'`` it is the t-norm
    ``1 - (1-s)^w * (1-c)^(1-w)``, which is 1.0 whenever either channel is fully wrong.

    Applying it per *pair* and then summing is the whole point, and is not interchangeable with
    combining the two channel totals: a t-norm of two sums is not the sum of the t-norms. The defect
    being repaired is that an individual leaf receives structural credit its content never earned
    (results/rebuttal/hallucination_vs_omission.json), so the conjunction has to bind at the leaf.

    Unmatched rows/columns are unaffected either way: delete/insert write the same value into both
    channels, and the conjunction of (1.0, 1.0) is 1.0 exactly as the weighted sum is.
    """
    w = evaluator.structural_weight
    if evaluator.cost_combination == 'additive':
        return w * s_costs + (1 - w) * c_costs
    if evaluator.cost_combination == 'product':
        s_sim = np.clip(1.0 - s_costs, 0.0, None)
        c_sim = np.clip(1.0 - c_costs, 0.0, None)
        return 1.0 - np.power(s_sim, w) * np.power(c_sim, 1.0 - w)
    raise ValueError(
        f"cost_combination must be 'additive' or 'product', "
        f"got {evaluator.cost_combination!r}")


def get_structural_and_content_costs(
    evaluator, tree1: "JsonNode", tree2: "JsonNode"
) -> Tuple[float, float]:
    """Get both structural and content costs in a single traversal.

    Was ``SemanticJsonTreeConsistencyEvaluator._get_structural_and_content_costs``. Kept as a
    two-value wrapper because callers outside this module (the evaluator's own thin wrapper and the
    probe harness) read the two channels; ``get_channel_costs`` carries the third.
    """
    s_cost, c_cost, _ = get_channel_costs(evaluator, tree1, tree2)
    return s_cost, c_cost


def get_channel_costs(
    evaluator, tree1: "JsonNode", tree2: "JsonNode"
) -> Tuple[float, float, float]:
    """Return ``(structural, content, joint)`` for a subtree pair in one traversal.

    The first two are the separately-normalised channels the additive Eq. (1) weights at the root.
    The third is the *leaf-bound* combination: at a leaf it is ``combine_channels`` of that leaf's
    two costs, and at an interior node it is the same normalise-then-sum aggregation applied to the
    children's joint costs.

    It has to be carried as its own channel rather than reconstructed at the root, because
    reconstruction is only valid when the combination is linear. Under ``'additive'`` it is, so this
    third value is redundant there and callers ignore it; under ``'product'`` it is not, and
    conjoining the two channel *totals* would give a leaf structural credit its content never
    earned -- the hallucination-over-omission defect in
    ``results/rebuttal/hallucination_vs_omission.json``.
    """
    # Check memoization cache for both (single hash computation, reused).
    #
    # The tags below are deliberately *not* "structural" and "content". Those are the
    # ``variation_type`` values ``calculate_optimal_matching_cost_fast`` uses in its own cache key
    # (see its ``cache_key``), and the two quantities are different: this function returns the
    # structural and content *channels* of the combined recursion, while the fast path caches the
    # full structural or content *projection* of the same subtree pair, computed by a different
    # algorithm. Sharing the key space meant an evaluator reused across ``variation_type`` values
    # could serve one path's number to the other. Measured before the rename: a combined-mode run
    # wrote 147 entries, 146 of them into the projections' namespace. No wrong answer was ever
    # observed — two of the three reuse directions cannot leak because this function only consults
    # its cache when *both* channel keys are present and a projection run writes only one, and the
    # third scored zero cross-path hits — so this is hazard removal, not a behaviour change.
    tree1_hash = evaluator._compute_tree_hash(tree1)
    tree2_hash = evaluator._compute_tree_hash(tree2)
    # lam joins the key for the reason given at ``calculate_optimal_matching_cost_fast``: it is a
    # property of the pair's *context* (its equal-length sibling arrays), which the subtree hash
    # cannot see, so two structurally identical array pairs in differently-shaped parents would
    # otherwise share an entry. It is 0.0 whenever the blend is off.
    lam_key = _array_order_lambda(evaluator, tree1, tree2)
    s_key = (tree1_hash, tree2_hash, "combined_channel_structural", lam_key)
    c_key = (tree1_hash, tree2_hash, "combined_channel_content", lam_key)
    # The joint channel is mode-dependent where the other two are not, so it gets its own namespace
    # tagged with the mode. Without the tag an evaluator reused across modes would serve an additive
    # joint cost to the product path.
    j_key = (tree1_hash, tree2_hash, "combined_channel_joint", evaluator.cost_combination,
             lam_key)

    s_cached = evaluator._subtree_cache.get(s_key)
    c_cached = evaluator._subtree_cache.get(c_key)
    j_cached = evaluator._subtree_cache.get(j_key)

    if s_cached is not None and c_cached is not None and j_cached is not None:
        # All three already cached (hits counted in .get())
        return s_cached, c_cached, j_cached

    # Base case: both are leaf nodes
    if not tree1.children and not tree2.children:
        s_cost = evaluator.structural_update_cost(tree1, tree2)
        c_cost = evaluator.content_update_cost(tree1, tree2)
        j_cost = evaluator.combine_channels(s_cost, c_cost)
        evaluator._subtree_cache.set(s_key, s_cost)
        evaluator._subtree_cache.set(c_key, c_cost)
        evaluator._subtree_cache.set(j_key, j_cost)
        return s_cost, c_cost, j_cost

    # If one is leaf and other is not.
    # Normalize to [0, 1] per Eq. (3), exactly as the ``variation_type``-specific path does at
    # ``calculate_optimal_matching_cost_fast``. This branch used to return the raw sum, which is
    # ``1 + |children|`` and so grows without bound: a leaf against a five-key object returned
    # 6.0. That value went into ``_subtree_cache`` (breaking the [0, 1] invariant the cache is
    # assumed to hold) and into the parent's cost matrix alongside normalized siblings, where it
    # dominated the assignment and then saturated the parent's own mean. One leaf-vs-subtree
    # mismatch among two siblings pushed the parent to 1.0 and erased the fact that the other
    # sibling matched exactly.
    #
    # The normalized value here is still exactly 1.0, because ``insert_cost`` is a flat 1.0 per
    # child regardless of that child's size; that residual saturation is a separate defect of the
    # insert/delete costs, not of this normalization.
    if not tree1.children and tree2.children:
        raw_cost = evaluator.delete_cost(tree1) + sum(
            evaluator.insert_cost(c) for c in tree2.children
        )
        cost = min(raw_cost / max(1, len(tree2.children)), 1.0)
        j_cost = evaluator.combine_channels(cost, cost)
        evaluator._subtree_cache.set(s_key, cost)
        evaluator._subtree_cache.set(c_key, cost)
        evaluator._subtree_cache.set(j_key, j_cost)
        return cost, cost, j_cost
    if tree1.children and not tree2.children:
        raw_cost = sum(evaluator.delete_cost(c) for c in tree1.children) + evaluator.insert_cost(tree2)
        cost = min(raw_cost / max(1, len(tree1.children)), 1.0)
        j_cost = evaluator.combine_channels(cost, cost)
        evaluator._subtree_cache.set(s_key, cost)
        evaluator._subtree_cache.set(c_key, cost)
        evaluator._subtree_cache.set(j_key, j_cost)
        return cost, cost, j_cost

    # Both have children - compute costs recursively
    children1 = tree1.children
    children2 = tree2.children
    n1, n2 = len(children1), len(children2)

    if n1 == 0 and n2 == 0:
        s_cost = evaluator.structural_update_cost(tree1, tree2)
        c_cost = evaluator.content_update_cost(tree1, tree2)
        j_cost = evaluator.combine_channels(s_cost, c_cost)
        evaluator._subtree_cache.set(s_key, s_cost)
        evaluator._subtree_cache.set(c_key, c_cost)
        evaluator._subtree_cache.set(j_key, j_cost)
        return s_cost, c_cost, j_cost

    # Order-sensitive arrays match by position, not by assignment.
    #
    # ``calculate_optimal_matching_cost_fast`` has the same dispatch, but in
    # ``variation_type="combined"`` it runs only once, at the root, and every interior node is
    # reached through this function instead — so before this branch existed
    # ``order_sensitive_fields`` had no effect at all in combined mode. The array was left unsorted
    # at tree-construction time and its elements were then matched with the Hungarian algorithm,
    # which is order-invariant by construction: swapping two elements of *different shape* cost
    # exactly 0.0, the same as with order sensitivity switched off entirely.
    if (
        tree1.node_type == "array" and tree2.node_type == "array" and
        (evaluator._is_order_sensitive_field(tree1.label) or evaluator._is_order_sensitive_field(tree2.label))
    ):
        s_cost, c_cost, j_cost = sequential_structural_and_content_costs(
            evaluator, tree1, tree2)
        evaluator._subtree_cache.set(s_key, s_cost)
        evaluator._subtree_cache.set(c_key, c_cost)
        evaluator._subtree_cache.set(j_key, j_cost)
        return s_cost, c_cost, j_cost

    # Use NumPy arrays for memory efficiency (float32 saves 50% vs float64)
    max_size = max(n1, n2)
    s_matrix = np.full((max_size, max_size), np.inf, dtype=np.float32)
    c_matrix = np.full((max_size, max_size), np.inf, dtype=np.float32)
    j_matrix = np.full((max_size, max_size), np.inf, dtype=np.float32)

    for i in range(n1):
        for j in range(n2):
            s, c, jc = get_channel_costs(evaluator, children1[i], children2[j])
            s_matrix[i, j] = s
            c_matrix[i, j] = c
            j_matrix[i, j] = jc

    # Vectorized deletion/insertion costs
    for i in range(n1):
        del_cost = evaluator.delete_cost(children1[i])
        s_matrix[i, n2:max_size] = del_cost
        c_matrix[i, n2:max_size] = del_cost
        j_matrix[i, n2:max_size] = evaluator.combine_channels(del_cost, del_cost)

    for j in range(n2):
        ins_cost = evaluator.insert_cost(children2[j])
        s_matrix[n1:max_size, j] = ins_cost
        c_matrix[n1:max_size, j] = ins_cost
        j_matrix[n1:max_size, j] = evaluator.combine_channels(ins_cost, ins_cost)

    # Match on the combined objective, not on structure alone (see
    # single_pass_combined_matching for why). The two channels are still returned separately so
    # the caller can weight them; only the choice of correspondence changes here. Under
    # ``'product'`` the objective is the leaf-bound joint channel, which is the cost actually
    # charged, so assignment and payment stay consistent.
    # ``p != 1`` routes the objective to the joint channel for the reason given at
    # ``single_pass_combined_matching``: under a nonlinear aggregate only the per-pair
    # combination is meaningful, so that is what the correspondence must minimise. The two
    # objectives coincide at p = 1 under 'additive', so nothing moves by default.
    p = _aggregation_p(evaluator)
    if evaluator.cost_combination == 'additive' and not _aggregation.is_active(p):
        objective = combine_channel_matrix(evaluator, s_matrix, c_matrix)
    else:
        objective = j_matrix
    row_indices, col_indices = _assign(objective, p)

    if _aggregation.is_active(p):
        # Each channel aggregated along the one correspondence the objective selected. The
        # structural and content values are report-only under an active p (the parent reads
        # the joint channel), but they are still aggregated the same way so that a caller
        # inspecting them sees the same reading the score was computed under.
        s_normalized = _aggregation.aggregate(
            [float(s_matrix[i, j]) for i, j in zip(row_indices, col_indices)], p)
        c_normalized = _aggregation.aggregate(
            [float(c_matrix[i, j]) for i, j in zip(row_indices, col_indices)], p)
        j_normalized = _aggregation.aggregate(
            [float(j_matrix[i, j]) for i, j in zip(row_indices, col_indices)], p)
    else:
        # Vectorized sum
        s_total = float(s_matrix[row_indices, col_indices].sum())
        c_total = float(c_matrix[row_indices, col_indices].sum())
        j_total = float(j_matrix[row_indices, col_indices].sum())

        s_normalized = min(s_total / len(row_indices), 1.0)
        c_normalized = min(c_total / len(row_indices), 1.0)
        j_normalized = min(j_total / len(row_indices), 1.0)

    # Order-sensitivity blend, applied per channel along ONE shared monotone alignment of
    # the same objective the assignment used. Inert at lam == 0.0, which is the default and
    # every non-array node, so the recursion is bit-identical unless the blend is enabled.
    lam = lam_key
    if lam > 0.0:
        del_costs = [evaluator.delete_cost(c) for c in children1]
        ins_costs = [evaluator.insert_cost(c) for c in children2]
        j_del = [evaluator.combine_channels(d, d) for d in del_costs]
        j_ins = [evaluator.combine_channels(i, i) for i in ins_costs]
        if _aggregation.is_active(p):
            _, pairs, skip1, skip2 = _align(objective[:n1, :n2], j_del, j_ins, p)
            s_charged = [float(s_matrix[i, j]) for i, j in pairs]
            c_charged = [float(c_matrix[i, j]) for i, j in pairs]
            j_charged = [float(j_matrix[i, j]) for i, j in pairs]
            for i in skip1:
                s_charged.append(del_costs[i])
                c_charged.append(del_costs[i])
                j_charged.append(j_del[i])
            for j in skip2:
                s_charged.append(ins_costs[j])
                c_charged.append(ins_costs[j])
                j_charged.append(j_ins[j])
            s_normalized = _order.blend(
                s_normalized, _aggregation.aggregate(s_charged, p), lam)
            c_normalized = _order.blend(
                c_normalized, _aggregation.aggregate(c_charged, p), lam)
            j_normalized = _order.blend(
                j_normalized, _aggregation.aggregate(j_charged, p), lam)
        else:
            _, pairs, skip1, skip2 = _order.monotone_alignment(
                objective[:n1, :n2], del_costs, ins_costs)
            s_ord = sum(float(s_matrix[i, j]) for i, j in pairs)
            c_ord = sum(float(c_matrix[i, j]) for i, j in pairs)
            j_ord = sum(float(j_matrix[i, j]) for i, j in pairs)
            for i in skip1:
                s_ord += del_costs[i]
                c_ord += del_costs[i]
                j_ord += j_del[i]
            for j in skip2:
                s_ord += ins_costs[j]
                c_ord += ins_costs[j]
                j_ord += j_ins[j]
            denom = max(len(row_indices), 1)
            s_normalized = _order.blend(s_normalized, min(s_ord / denom, 1.0), lam)
            c_normalized = _order.blend(c_normalized, min(c_ord / denom, 1.0), lam)
            j_normalized = _order.blend(j_normalized, min(j_ord / denom, 1.0), lam)

    # Identifier keys apply to internal nodes too. The recursion above charges a key only at
    # leaves, through their path indicator, so renaming an object-valued key (BFCL's outer tool
    # name, a renamed argument object) scored 0.707 under the product profiles although mu = 1
    # declares a field name an identity. Under an active mu the node's own key/path term gates
    # its subtree: mu = 1 and a different key voids the subtree (cost 1), a fractional mu grades
    # it, and equal keys leave every value untouched. The legacy profile (mu unset) is bit-identical.
    mu_spec = getattr(evaluator, "key_identity_mu", None)
    if mu_spec is not None:
        mu = _key_identity.resolve_mu(mu_spec)
        if mu > 0.0:
            s_node = evaluator.structural_update_cost(tree1, tree2)
            if s_node > 0.0:
                gate = min(1.0, mu * s_node)
                s_normalized = max(s_normalized, gate)
                j_normalized = max(j_normalized, gate)

    evaluator._subtree_cache.set(s_key, s_normalized)
    evaluator._subtree_cache.set(c_key, c_normalized)
    evaluator._subtree_cache.set(j_key, j_normalized)

    return s_normalized, c_normalized, j_normalized


def sequential_structural_and_content_costs(
    evaluator, tree1: "JsonNode", tree2: "JsonNode"
) -> Tuple[float, float, float]:
    """Positional (order-sensitive) matching that returns structural and content costs together.

    The combined-path counterpart of ``calculate_sequential_matching_cost``: same pairing rule
    (element *i* against element *i*), same insert/delete charge for the length difference and same
    normalisation, but it recurses through
    ``get_structural_and_content_costs`` so the two cost channels stay separated for
    ``single_pass_combined_matching`` to weight.
    """
    children1 = tree1.children
    children2 = tree2.children
    n1, n2 = len(children1), len(children2)

    if n1 == 0 and n2 == 0:
        return 0.0, 0.0, 0.0

    s_total = 0.0
    c_total = 0.0
    j_total = 0.0
    matched_count = min(n1, n2)

    # Compare elements by position
    for i in range(matched_count):
        s, c, jc = get_channel_costs(evaluator, children1[i], children2[i])
        s_total += s
        c_total += c
        j_total += jc

    # Extra elements on either side are deletions / insertions
    for i in range(matched_count, n1):
        del_cost = evaluator.delete_cost(children1[i])
        s_total += del_cost
        c_total += del_cost
        j_total += evaluator.combine_channels(del_cost, del_cost)
    for j in range(matched_count, n2):
        ins_cost = evaluator.insert_cost(children2[j])
        s_total += ins_cost
        c_total += ins_cost
        j_total += evaluator.combine_channels(ins_cost, ins_cost)

    # No separate length penalty: the loops above already charge a full insert/delete cost of 1.0
    # for every unmatched element, which is the maximum a single element can cost. Adding
    # `0.1 * length_diff` on top charged the length difference twice and pushed the total past the
    # "delete everything, insert everything" bound, leaving only the final clamp to keep it in
    # range. See ``calculate_sequential_matching_cost`` for the measured numbers.
    denom = max(n1, n2)
    return (min(s_total / denom, 1.0), min(c_total / denom, 1.0),
            min(j_total / denom, 1.0))


def calculate_sequential_matching_cost(
    evaluator,
    tree1: "JsonNode",
    tree2: "JsonNode",
    variation_type: str = "combined",
) -> float:
    """Calculate matching cost using sequential (positional) matching for order-sensitive arrays.

    Was ``SemanticJsonTreeConsistencyEvaluator._calculate_sequential_matching_cost``.

    Implementation note: this dispatches inner-element costs through
    ``evaluator._calculate_optimal_matching_cost`` (the legacy non-fast
    version) rather than ``calculate_optimal_matching_cost_fast``. This was
    flagged during the v0.2.0 refactor as a possible perf bug, but
    benchmarking shows the sequential path remains ~2× faster than the
    fast-Hungarian path on realistic agent-trace inputs because sequential
    matching is O(n) inner-element calls instead of O(n²). Switching to the
    fast inner path would speed sequential further but is not on the
    critical path. Leaving as-is preserves bit-identical regression.
    """
    children1 = tree1.children
    children2 = tree2.children
    n1, n2 = len(children1), len(children2)

    if n1 == 0 and n2 == 0:
        return 0.0

    total_cost = 0.0
    matched_count = min(n1, n2)

    # Compare elements by position
    for i in range(matched_count):
        if variation_type == "structural":
            total_cost += evaluator._calculate_optimal_matching_cost(
                children1[i], children2[i], "structural"
            )
        elif variation_type == "content":
            total_cost += evaluator._calculate_optimal_matching_cost(
                children1[i], children2[i], "content"
            )
        else:  # combined
            total_cost += evaluator._calculate_optimal_matching_cost(
                children1[i], children2[i], "combined"
            )

    # Add deletion costs for extra elements in tree1
    for i in range(matched_count, n1):
        total_cost += evaluator.delete_cost(children1[i])

    # Add insertion costs for extra elements in tree2
    for j in range(matched_count, n2):
        total_cost += evaluator.insert_cost(children2[j])

    # No separate length penalty. The two loops above already charge a full insert/delete cost of
    # 1.0 for each of the ``|n1 - n2|`` unmatched elements, which is the most a single element can
    # cost, so a further ``0.1 * length_diff`` charged the same fact twice. Under Eq. (3)'s
    # per-level normalisation the honest cost of k unmatched elements out of n is exactly k/n, the
    # fraction of the level that failed to match; the penalty pushed 1-vs-2 to 0.55 instead of 0.50
    # and 1-vs-3 to 0.7333 instead of 0.6667, i.e. the excess grew with the length difference and
    # was only kept inside [0, 1] by the final clamp. It also made the two array paths disagree on
    # identical input: order-*insensitive* Hungarian matching of the same 1-vs-2 arrays returns
    # 0.50, so declaring a field order-sensitive silently changed the price of a length difference.

    # Normalize by total elements
    num_elements = max(n1, n2)
    if num_elements == 0:
        return 0.0

    normalized_cost = total_cost / num_elements
    return min(normalized_cost, 1.0)
