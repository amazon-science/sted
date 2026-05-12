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

if TYPE_CHECKING:  # pragma: no cover - only for type checkers
    from .json_tree_node import JsonNode


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

    # Structural mismatch: one has children, other doesn't
    if bool(node1.children) != bool(node2.children):
        # Different structure levels
        return 0.9

    # Large size difference (> 3x)
    if node1.children and node2.children:
        n1, n2 = len(node1.children), len(node2.children)
        if n1 > 0 and n2 > 0:
            ratio = max(n1, n2) / min(n1, n2)
            if ratio > 3:
                # Very different sizes, use approximate cost
                return min(1.0, 0.3 + 0.1 * ratio)

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
    cache_key = (
        evaluator._compute_tree_hash(tree1),
        evaluator._compute_tree_hash(tree2),
        variation_type,
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
    if evaluator.use_greedy_matching and max(n1, n2) > 5:
        total_cost = greedy_matching_streaming(evaluator, children1, children2, variation_type)
        normalized_cost = min(total_cost / max(n1, n2, 1), 1.0)
        evaluator._subtree_cache.set(cache_key, normalized_cost)
        return normalized_cost

    # === Optimization 2: Single-pass combined calculation ===
    if variation_type == "combined":
        cost = single_pass_combined_matching(evaluator, children1, children2, n1, n2)
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
    row_indices, col_indices = linear_sum_assignment(cost_matrix)

    total_cost = float(cost_matrix[row_indices, col_indices].sum())
    normalized_cost = min(total_cost / len(row_indices), 1.0)

    evaluator._subtree_cache.set(cache_key, normalized_cost)
    return normalized_cost


def single_pass_combined_matching(
    evaluator,
    children1: List["JsonNode"],
    children2: List["JsonNode"],
    n1: int,
    n2: int,
) -> float:
    """Single-pass combined calculation that computes structural and content costs together.

    Was ``SemanticJsonTreeConsistencyEvaluator._single_pass_combined_matching``.
    """
    max_size = max(n1, n2)

    # Build combined cost matrix in single pass using NumPy (float32 saves memory)
    structural_costs = np.zeros((max_size, max_size), dtype=np.float32)
    content_costs = np.zeros((max_size, max_size), dtype=np.float32)

    for i in range(n1):
        for j in range(n2):
            # Get both costs in single recursive call
            s_cost, c_cost = get_structural_and_content_costs(
                evaluator, children1[i], children2[j]
            )
            structural_costs[i, j] = s_cost
            content_costs[i, j] = c_cost

    # Add deletion costs (vectorized)
    for i in range(n1):
        del_cost = evaluator.delete_cost(children1[i])
        structural_costs[i, n2:max_size] = del_cost
        content_costs[i, n2:max_size] = del_cost

    # Add insertion costs (vectorized)
    for j in range(n2):
        ins_cost = evaluator.insert_cost(children2[j])
        structural_costs[n1:max_size, j] = ins_cost
        content_costs[n1:max_size, j] = ins_cost

    # Use structural costs for matching (as per original algorithm)
    row_indices, col_indices = linear_sum_assignment(structural_costs)

    # Calculate final costs using the matched pairs (vectorized)
    structural_total = float(structural_costs[row_indices, col_indices].sum())
    content_total = float(content_costs[row_indices, col_indices].sum())

    # Paper Eq.(1): w * structural + (1-w) * content
    total_cost = (
        evaluator.structural_weight * structural_total
        + (1 - evaluator.structural_weight) * content_total
    )
    normalized_cost = min(total_cost / len(row_indices), 1.0)

    return normalized_cost


def get_structural_and_content_costs(
    evaluator, tree1: "JsonNode", tree2: "JsonNode"
) -> Tuple[float, float]:
    """Get both structural and content costs in a single traversal.

    Was ``SemanticJsonTreeConsistencyEvaluator._get_structural_and_content_costs``.
    """
    # Check memoization cache for both (single hash computation, reused)
    tree1_hash = evaluator._compute_tree_hash(tree1)
    tree2_hash = evaluator._compute_tree_hash(tree2)
    s_key = (tree1_hash, tree2_hash, "structural")
    c_key = (tree1_hash, tree2_hash, "content")

    s_cached = evaluator._subtree_cache.get(s_key)
    c_cached = evaluator._subtree_cache.get(c_key)

    if s_cached is not None and c_cached is not None:
        # Both already cached (hits counted in .get())
        return s_cached, c_cached

    # Base case: both are leaf nodes
    if not tree1.children and not tree2.children:
        s_cost = evaluator.structural_update_cost(tree1, tree2)
        c_cost = evaluator.content_update_cost(tree1, tree2)
        evaluator._subtree_cache.set(s_key, s_cost)
        evaluator._subtree_cache.set(c_key, c_cost)
        return s_cost, c_cost

    # If one is leaf and other is not
    if not tree1.children and tree2.children:
        cost = evaluator.delete_cost(tree1) + sum(
            evaluator.insert_cost(c) for c in tree2.children
        )
        evaluator._subtree_cache.set(s_key, cost)
        evaluator._subtree_cache.set(c_key, cost)
        return cost, cost
    if tree1.children and not tree2.children:
        cost = sum(evaluator.delete_cost(c) for c in tree1.children) + evaluator.insert_cost(tree2)
        evaluator._subtree_cache.set(s_key, cost)
        evaluator._subtree_cache.set(c_key, cost)
        return cost, cost

    # Both have children - compute costs recursively
    children1 = tree1.children
    children2 = tree2.children
    n1, n2 = len(children1), len(children2)

    if n1 == 0 and n2 == 0:
        s_cost = evaluator.structural_update_cost(tree1, tree2)
        c_cost = evaluator.content_update_cost(tree1, tree2)
        evaluator._subtree_cache.set(s_key, s_cost)
        evaluator._subtree_cache.set(c_key, c_cost)
        return s_cost, c_cost

    # Use NumPy arrays for memory efficiency (float32 saves 50% vs float64)
    max_size = max(n1, n2)
    s_matrix = np.full((max_size, max_size), np.inf, dtype=np.float32)
    c_matrix = np.full((max_size, max_size), np.inf, dtype=np.float32)

    for i in range(n1):
        for j in range(n2):
            s, c = get_structural_and_content_costs(evaluator, children1[i], children2[j])
            s_matrix[i, j] = s
            c_matrix[i, j] = c

    # Vectorized deletion/insertion costs
    for i in range(n1):
        del_cost = evaluator.delete_cost(children1[i])
        s_matrix[i, n2:max_size] = del_cost
        c_matrix[i, n2:max_size] = del_cost

    for j in range(n2):
        ins_cost = evaluator.insert_cost(children2[j])
        s_matrix[n1:max_size, j] = ins_cost
        c_matrix[n1:max_size, j] = ins_cost

    # Use structural for matching
    row_indices, col_indices = linear_sum_assignment(s_matrix)

    # Vectorized sum
    s_total = float(s_matrix[row_indices, col_indices].sum())
    c_total = float(c_matrix[row_indices, col_indices].sum())

    s_normalized = min(s_total / len(row_indices), 1.0)
    c_normalized = min(c_total / len(row_indices), 1.0)

    evaluator._subtree_cache.set(s_key, s_normalized)
    evaluator._subtree_cache.set(c_key, c_normalized)

    return s_normalized, c_normalized


def calculate_sequential_matching_cost(
    evaluator,
    tree1: "JsonNode",
    tree2: "JsonNode",
    variation_type: str = "combined",
) -> float:
    """Calculate matching cost using sequential (positional) matching for order-sensitive arrays.

    Was ``SemanticJsonTreeConsistencyEvaluator._calculate_sequential_matching_cost``.

    Note: This implementation calls ``evaluator._calculate_optimal_matching_cost`` (the
    legacy non-fast version) for inner-element costs, matching the original behavior.
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

    # Add length penalty
    length_diff = abs(n1 - n2)
    max_len = max(n1, n2)
    length_penalty = length_diff * 0.1 if max_len > 0 else 0

    total_cost += length_penalty

    # Normalize by total elements
    num_elements = max(n1, n2)
    if num_elements == 0:
        return 0.0

    normalized_cost = total_cost / num_elements
    return min(normalized_cost, 1.0)
