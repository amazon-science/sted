"""
Semantic JSON Structural Consistency Evaluation using Tree Edit Distance

This module enhances the JSON tree consistency evaluation with semantic similarity capabilities.
It combines tree edit distance algorithms with embedding-based semantic similarity to provide
more accurate structural consistency evaluation for JSON outputs.
"""

import os
import asyncio
import json
import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Set
from functools import lru_cache
from collections import OrderedDict
import warnings
import re
from bert_score import score as bert_score

from scipy.optimize import linear_sum_assignment
import zss

from langchain_text_splitters import RecursiveCharacterTextSplitter

from sentence_transformers import SentenceTransformer

from deepdiff import DeepDiff


from .json_tree_node import JsonNode
from .similarity_cache import StringSimilarityCache
from .utils import get_embeddings, create_bedrock_client, count_json_elements
from botocore.config import Config
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from tqdm.asyncio import tqdm as atqdm

from transformers import logging
logging.set_verbosity_error()


class LRUCache:
    """Memory-bounded LRU cache for subtree comparison results."""

    def __init__(self, maxsize: int = 10000):
        self.maxsize = maxsize
        self._cache: OrderedDict = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, key: Tuple) -> Optional[float]:
        """Get value from cache, moving to end (most recently used)."""
        if key in self._cache:
            self._cache.move_to_end(key)
            self.hits += 1
            return self._cache[key]
        self.misses += 1
        return None

    def set(self, key: Tuple, value: float) -> None:
        """Set value in cache, evicting LRU if at capacity."""
        if key in self._cache:
            self._cache.move_to_end(key)
        else:
            if len(self._cache) >= self.maxsize:
                # Evict least recently used (first item)
                self._cache.popitem(last=False)
            self._cache[key] = value

    def clear(self) -> None:
        """Clear the cache."""
        self._cache.clear()
        self.hits = 0
        self.misses = 0

    def __len__(self) -> int:
        return len(self._cache)

    def __contains__(self, key: Tuple) -> bool:
        return key in self._cache

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


def _get_default_type_change_costs() -> Dict[Tuple[str, str], float]:
    """Define default costs for type changes."""
    costs = {}
    types = ["object", "array", "string", "number", "boolean", "null"]

    # Default cost is 1.0
    for t1 in types:
        for t2 in types:
            costs[(t1, t2)] = 1.0

    # Same type has zero cost
    for t in types:
        costs[(t, t)] = 0.0

    # Lower costs for some type conversions
    costs[("string", "number")] = costs[("number", "string")] = 0.1
    costs[("boolean", "string")] = costs[("string", "boolean")] = 0.1
    costs[("number", "boolean")] = costs[("boolean", "number")] = 0.1
    costs[("null", "string")] = costs[("string", "null")] = 0.1
    costs[("null", "number")] = costs[("number", "null")] = 0.1

    # Higher costs for structure changes
    costs[("object", "array")] = costs[("array", "object")] = 0.5
    costs[("object", "string")] = costs[("string", "object")] = 0.5
    costs[("object", "number")] = costs[("number", "object")] = 0.5
    costs[("array", "string")] = costs[("string", "array")] = 0.5
    costs[("array", "number")] = costs[("number", "array")] = 1

    return costs


from transformers import logging
logging.set_verbosity_error()

class SemanticJsonTreeConsistencyEvaluator:
    """Evaluator for JSON structural consistency using Tree Edit Distance with semantic similarity."""

    def __init__(self,
                 path_weight_decay: float = 1.0,
                 type_change_cost: Dict[Tuple[str, str], float] = None,
                 required_fields: Set[str] = None,
                 order_sensitive_fields: Set[str] = None,
                 exact_match_fields: Set[str] = None,
                 exact_match_all_keys: bool = False,
                 model_id: str = 'all-MiniLM-L6-v2',
                 chunk_size: int = 300,
                 chunk_overlap: int = 50,
                 region_name: str = "us-west-2",
                 structural_weight: float = 0.5,  # Paper: w in Eq.(1)
                 sort_keys: bool = True,
                 sort_arrays: bool = True,
                 embedding_dim: int = 512,
                 # Space optimization options
                 subtree_cache_size: int = 10000,
                 use_fp16_embeddings: bool = False,
                 max_embedding_cache_size: int = 100000
        ):
        """
        Initialize the evaluator with semantic capabilities.

        Args:
            path_weight_decay: Weight decay factor for deeper paths (0-1)
            type_change_cost: Custom costs for type changes
            required_fields: Set of required field paths
            order_sensitive_fields: Set of field names where array order matters
                                    (e.g., {"trace", "steps", "calls"} for agent traces).
                                    For these fields, sequential comparison is used instead
                                    of Hungarian algorithm optimal matching.
            exact_match_fields: Set of field names that require exact string matching
                               (e.g., {"name"} for function/tool names).
                               For these fields, semantic similarity is bypassed and
                               only exact value equality is considered (1.0 if equal, 0.0 otherwise).
            exact_match_all_keys: If True, use exact string matching for all field names (keys)
                                 instead of semantic similarity. Useful for tool calling where
                                 parameter names must match exactly.
            model_id: Name of the sentence transformer model or Bedrock model ID
            chunk_size: Size of chunks for text splitting
            chunk_overlap: Overlap between chunks
            structural_weight: Structural vs content weight w (0-1), default 0.5. Paper Eq.(1): γ_upd = w·γ_struct + (1-w)·γ_content
            embedding_dim: Embedding dimension for Bedrock models (256, 384, 512, or 1024). Default: 512
            subtree_cache_size: Max entries in subtree LRU cache (default 10000). Set to 0 for unbounded.
            use_fp16_embeddings: Store embeddings as float16 to save ~50% memory (slight precision loss)
            max_embedding_cache_size: Max embeddings to cache (default 100000). Set to 0 for unbounded.
        """
        self.path_weight_decay = path_weight_decay
        self.type_change_cost = type_change_cost or _get_default_type_change_costs()
        self.required_fields = required_fields or set()
        self.order_sensitive_fields = order_sensitive_fields or set()
        self.exact_match_fields = exact_match_fields or set()
        self.exact_match_all_keys = exact_match_all_keys

        # Initialize embedding model if available
        self.embedding_model = None
        self.bedrock_client = None
        self.model_id = model_id
        self.embedding_dim = embedding_dim

        boto_config = Config(
            retries={
                'max_attempts': 10,
                'mode': 'adaptive'
            },
            max_pool_connections=50  # Increase connection pool size
        )

        if self.model_id in ["amazon.titan-embed-text-v1", "amazon.titan-embed-text-v2:0",
                             "cohere.embed-multilingual-v3", "cohere.embed-v4:0", "us.cohere.embed-v4:0"]:
            self.bedrock_client = create_bedrock_client(region_name=region_name, config=boto_config)
        else:
            self.embedding_model = SentenceTransformer(self.model_id)
            # Warm up the model
            self.embedding_model.encode(["test"], show_progress_bar=False)

        # Cache for embeddings
        self._cache = StringSimilarityCache()

        # Pre-computed embedding dictionary: string -> np.ndarray
        self._embedding_dict: Dict[str, np.ndarray] = {}
        self._embedding_dict_populated = False

        # Text splitting configuration
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.calculate_similarity_method = {
            "ted": self.calculate_tree_edit_distance,
            "sted": self.calculate_tree_edit_distance_opt,
            "sted_fast": self.calculate_tree_edit_distance_fast,
            "bertscore": self.calculate_bertscore,
            "deepdiff": self.calculate_similarity_with_deepdiff,
            "deepdiff_opt": self.calculate_similarity_with_deepdiff_opt,
        }

        self.structural_weight = structural_weight  # Paper Eq.(1): w

        self.sort_keys = sort_keys
        self.sort_arrays = sort_arrays

        self.batch_size_bertscore = 2000

        # === Optimization settings ===
        # Space optimization config
        self.use_fp16_embeddings = use_fp16_embeddings
        self.max_embedding_cache_size = max_embedding_cache_size
        self.subtree_cache_size = subtree_cache_size

        # Memoization cache for subtree comparisons: (tree1_hash, tree2_hash, variation_type) -> cost
        # Use LRU cache with bounded size for space efficiency
        if subtree_cache_size > 0:
            self._subtree_cache = LRUCache(maxsize=subtree_cache_size)
        else:
            # Unbounded cache (original behavior)
            self._subtree_cache = LRUCache(maxsize=float('inf'))

        # Greedy approximation flag (optional, trades accuracy for speed)
        self.use_greedy_matching = False

        # Early pruning threshold (skip detailed comparison if type mismatch cost exceeds this)
        self.early_pruning_threshold = 0.8

        # Vectorized leaf comparison: pre-collected leaf pairs for batch processing
        self._pending_leaf_pairs: List[Tuple[str, str]] = []

        # Hash cache for nodes (lazy computation)
        self._node_hash_cache: Dict[int, str] = {}

    # === Optimization Helper Methods ===

    def _compute_tree_hash(self, node: 'JsonNode') -> str:
        """
        Compute a hash for a tree node for memoization with lazy caching.

        The hash captures the structural and content signature of the subtree.
        Uses node id() for caching to avoid recomputation.
        """
        import hashlib

        node_id = id(node)
        if node_id in self._node_hash_cache:
            return self._node_hash_cache[node_id]

        def _hash_node(n: 'JsonNode') -> str:
            n_id = id(n)
            if n_id in self._node_hash_cache:
                return self._node_hash_cache[n_id]

            # Include node type, label pattern (without indices), and value
            label_pattern = re.sub(r'\[\d+\]', '[*]', n.label)
            parts = [n.node_type, label_pattern]

            if n.value is not None and not n.children:
                # For leaf nodes, include value hash
                parts.append(str(n.value)[:100])  # Truncate long values

            # Include children hashes
            if n.children:
                child_hashes = sorted([_hash_node(c) for c in n.children])
                parts.extend(child_hashes)

            result = hashlib.md5('|'.join(parts).encode()).hexdigest()[:16]
            self._node_hash_cache[n_id] = result
            return result

        return _hash_node(node)

    def clear_subtree_cache(self):
        """Clear the subtree comparison cache and node hash cache."""
        self._subtree_cache.clear()
        self._node_hash_cache.clear()

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics for performance monitoring."""
        return {
            'subtree_cache_size': len(self._subtree_cache),
            'subtree_cache_hits': self._subtree_cache.hits,
            'subtree_cache_misses': self._subtree_cache.misses,
            'subtree_cache_hit_rate': self._subtree_cache.hit_rate,
            'subtree_cache_max_size': self.subtree_cache_size,
            'embedding_cache_size': len(self._embedding_dict),
            'node_hash_cache_size': len(self._node_hash_cache),
            'using_fp16': self.use_fp16_embeddings
        }

    def set_greedy_matching(self, enabled: bool):
        """Enable or disable greedy matching approximation."""
        self.use_greedy_matching = enabled

    def _collect_leaf_pairs(self, tree1: 'JsonNode', tree2: 'JsonNode') -> List[Tuple[str, str]]:
        """
        Collect all potential leaf value pairs for vectorized comparison.

        This enables batch similarity computation before tree traversal.
        """
        pairs = []

        def collect_leaves(node: 'JsonNode') -> List[str]:
            if not node.children:
                if node.value is not None and isinstance(node.value, str):
                    return [str(node.value)]
                return []
            leaves = []
            for child in node.children:
                leaves.extend(collect_leaves(child))
            return leaves

        leaves1 = collect_leaves(tree1)
        leaves2 = collect_leaves(tree2)

        # Create all pairs for batch processing
        for l1 in leaves1:
            for l2 in leaves2:
                if l1 != l2:  # Skip identical pairs
                    pairs.append((l1, l2))

        return pairs

    def _precompute_leaf_similarities(self, tree1: 'JsonNode', tree2: 'JsonNode'):
        """
        Pre-compute all leaf string similarities before tree comparison.

        This enables vectorized embedding computation for better performance.
        """
        pairs = self._collect_leaf_pairs(tree1, tree2)

        # Filter pairs not already in cache
        uncached_pairs = [(s1, s2) for s1, s2 in pairs if self._cache.get(s1, s2) is None]

        if not uncached_pairs:
            return

        # Batch compute similarities
        for s1, s2 in uncached_pairs:
            sim = self._calculate_semantic_similarity(s1, s2)
            self._cache.set(s1, s2, sim)

    def _greedy_matching(self, children1: List['JsonNode'], children2: List['JsonNode'],
                         variation_type: str) -> Tuple[List[Tuple[int, int]], float]:
        """
        Greedy matching approximation - O(B²) instead of O(B³) Hungarian.

        Matches each node in children1 to its best available partner in children2.
        Not optimal but much faster for large branching factors.

        Returns:
            Tuple of (matched_pairs, total_cost)
        """
        n1, n2 = len(children1), len(children2)
        matched_pairs = []
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
                cost = self._calculate_optimal_matching_cost_fast(
                    children1[i], children2[j], variation_type
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
                total_cost += self.delete_cost(children1[i])

        # Add insertion costs for unmatched children2
        for j in range(n2):
            if j not in used_j:
                total_cost += self.insert_cost(children2[j])

        return matched_pairs, total_cost

    def _greedy_matching_streaming(self, children1: List['JsonNode'], children2: List['JsonNode'],
                                    variation_type: str) -> float:
        """
        Space-optimized greedy matching - O(B) space instead of O(B²).

        This version doesn't build a full cost matrix. Instead, it computes costs
        on-the-fly and only tracks which indices are used (O(B) space for the set).

        For very large branching factors (B > 100), this significantly reduces memory.

        Returns:
            total_cost (float)
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
                cost = self._calculate_optimal_matching_cost_fast(
                    children1[i], children2[j], variation_type
                )

                if cost < best_cost:
                    best_cost = cost
                    best_j = j

            if best_j >= 0:
                used_j.add(best_j)
                total_cost += best_cost
            else:
                # No match found, count as deletion
                total_cost += self.delete_cost(children1[i])

        # Add insertion costs for unmatched children2
        for j in range(n2):
            if j not in used_j:
                total_cost += self.insert_cost(children2[j])

        return total_cost

    def _early_prune_check(self, node1: 'JsonNode', node2: 'JsonNode') -> Optional[float]:
        """
        Check if we can skip detailed comparison based on early pruning criteria.

        Returns:
            Cost if pruning applies, None otherwise
        """
        # Type mismatch with incompatible types
        type_cost = self.type_change_cost.get((node1.node_type, node2.node_type), 1.0)
        if type_cost >= self.early_pruning_threshold:
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

    def _calculate_optimal_matching_cost_fast(
        self, tree1: 'JsonNode', tree2: 'JsonNode', variation_type: str = "combined"
    ) -> float:
        """
        Optimized version of _calculate_optimal_matching_cost with:
        1. Memoization (LRU cache with bounded size)
        2. Single-pass combined calculation
        3. Optional greedy matching (O(B) space with streaming)
        4. Early pruning
        5. Lazy hash computation
        6. NumPy arrays for cost matrices (memory efficient)
        """
        # === Optimization 1: Memoization with LRU cache ===
        cache_key = (self._compute_tree_hash(tree1), self._compute_tree_hash(tree2), variation_type)
        cached_result = self._subtree_cache.get(cache_key)
        if cached_result is not None:
            return cached_result

        # === Optimization 5: Early pruning ===
        pruned_cost = self._early_prune_check(tree1, tree2)
        if pruned_cost is not None:
            self._subtree_cache.set(cache_key, pruned_cost)
            return pruned_cost

        # Base case: both are leaf nodes
        if not tree1.children and not tree2.children:
            if variation_type == "structural":
                cost = self.structural_update_cost(tree1, tree2)
            elif variation_type == "content":
                cost = self.content_update_cost(tree1, tree2)
            else:  # combined
                cost = self.update_cost(tree1, tree2)
            self._subtree_cache.set(cache_key, cost)
            return cost

        # If one is leaf and other is not
        if not tree1.children and tree2.children:
            cost = self.delete_cost(tree1) + sum(self.insert_cost(c) for c in tree2.children)
            self._subtree_cache.set(cache_key, cost)
            return cost
        if tree1.children and not tree2.children:
            cost = sum(self.delete_cost(c) for c in tree1.children) + self.insert_cost(tree2)
            self._subtree_cache.set(cache_key, cost)
            return cost

        # Both have children
        children1 = tree1.children
        children2 = tree2.children

        # Check for order-sensitive arrays
        is_order_sensitive = (
            tree1.node_type == "array" and tree2.node_type == "array" and
            (self._is_order_sensitive_field(tree1.label) or self._is_order_sensitive_field(tree2.label))
        )

        if is_order_sensitive:
            cost = self._calculate_sequential_matching_cost(tree1, tree2, variation_type)
            self._subtree_cache.set(cache_key, cost)
            return cost

        n1, n2 = len(children1), len(children2)

        if n1 == 0 and n2 == 0:
            if variation_type == "structural":
                cost = self.structural_update_cost(tree1, tree2)
            elif variation_type == "content":
                cost = self.content_update_cost(tree1, tree2)
            else:
                cost = self.update_cost(tree1, tree2)
            self._subtree_cache.set(cache_key, cost)
            return cost

        # === Optimization 3: Greedy matching with streaming (O(B) space) ===
        if self.use_greedy_matching and max(n1, n2) > 5:
            total_cost = self._greedy_matching_streaming(children1, children2, variation_type)
            normalized_cost = min(total_cost / max(n1, n2, 1), 1.0)
            self._subtree_cache.set(cache_key, normalized_cost)
            return normalized_cost

        # === Optimization 2: Single-pass combined calculation ===
        if variation_type == "combined":
            cost = self._single_pass_combined_matching(children1, children2, n1, n2)
            self._subtree_cache.set(cache_key, cost)
            return cost

        # For structural or content only, use NumPy array for memory efficiency
        max_size = max(n1, n2)
        cost_matrix = np.full((max_size, max_size), np.inf, dtype=np.float32)

        for i in range(n1):
            for j in range(n2):
                cost_matrix[i, j] = self._calculate_optimal_matching_cost_fast(
                    children1[i], children2[j], variation_type
                )

        # Add deletion/insertion costs (vectorized for efficiency)
        for i in range(n1):
            del_cost = self.delete_cost(children1[i])
            cost_matrix[i, n2:max_size] = del_cost
        for j in range(n2):
            ins_cost = self.insert_cost(children2[j])
            cost_matrix[n1:max_size, j] = ins_cost

        # Hungarian algorithm
        row_indices, col_indices = linear_sum_assignment(cost_matrix)

        total_cost = float(cost_matrix[row_indices, col_indices].sum())
        normalized_cost = min(total_cost / len(row_indices), 1.0)

        self._subtree_cache.set(cache_key, normalized_cost)
        return normalized_cost

    def _single_pass_combined_matching(
        self, children1: List['JsonNode'], children2: List['JsonNode'], n1: int, n2: int
    ) -> float:
        """
        Single-pass combined calculation that computes structural and content costs together.

        Uses NumPy arrays for memory efficiency (float32 instead of Python float).

        Instead of:
        1. Build structural cost matrix
        2. Run Hungarian
        3. Re-compute combined costs for matched pairs

        This does:
        1. Build combined cost matrix (structural + content in one pass)
        2. Run Hungarian once
        """
        max_size = max(n1, n2)

        # Build combined cost matrix in single pass using NumPy (float32 saves memory)
        structural_costs = np.zeros((max_size, max_size), dtype=np.float32)
        content_costs = np.zeros((max_size, max_size), dtype=np.float32)

        for i in range(n1):
            for j in range(n2):
                # Get both costs in single recursive call
                s_cost, c_cost = self._get_structural_and_content_costs(
                    children1[i], children2[j]
                )
                structural_costs[i, j] = s_cost
                content_costs[i, j] = c_cost

        # Add deletion costs (vectorized)
        for i in range(n1):
            del_cost = self.delete_cost(children1[i])
            structural_costs[i, n2:max_size] = del_cost
            content_costs[i, n2:max_size] = del_cost

        # Add insertion costs (vectorized)
        for j in range(n2):
            ins_cost = self.insert_cost(children2[j])
            structural_costs[n1:max_size, j] = ins_cost
            content_costs[n1:max_size, j] = ins_cost

        # Use structural costs for matching (as per original algorithm)
        row_indices, col_indices = linear_sum_assignment(structural_costs)

        # Calculate final costs using the matched pairs (vectorized)
        structural_total = float(structural_costs[row_indices, col_indices].sum())
        content_total = float(content_costs[row_indices, col_indices].sum())

        # Paper Eq.(1): w * structural + (1-w) * content
        total_cost = self.structural_weight * structural_total + (1 - self.structural_weight) * content_total
        normalized_cost = min(total_cost / len(row_indices), 1.0)

        return normalized_cost

    def _get_structural_and_content_costs(
        self, tree1: 'JsonNode', tree2: 'JsonNode'
    ) -> Tuple[float, float]:
        """
        Get both structural and content costs in a single traversal.

        Uses NumPy arrays for memory efficiency and LRU cache for bounded memory.

        Returns:
            Tuple of (structural_cost, content_cost)
        """
        # Check memoization cache for both (single hash computation, reused)
        tree1_hash = self._compute_tree_hash(tree1)
        tree2_hash = self._compute_tree_hash(tree2)
        s_key = (tree1_hash, tree2_hash, "structural")
        c_key = (tree1_hash, tree2_hash, "content")

        s_cached = self._subtree_cache.get(s_key)
        c_cached = self._subtree_cache.get(c_key)

        if s_cached is not None and c_cached is not None:
            # Both already cached (hits counted in .get())
            return s_cached, c_cached

        # Base case: both are leaf nodes
        if not tree1.children and not tree2.children:
            s_cost = self.structural_update_cost(tree1, tree2)
            c_cost = self.content_update_cost(tree1, tree2)
            self._subtree_cache.set(s_key, s_cost)
            self._subtree_cache.set(c_key, c_cost)
            return s_cost, c_cost

        # If one is leaf and other is not
        if not tree1.children and tree2.children:
            cost = self.delete_cost(tree1) + sum(self.insert_cost(c) for c in tree2.children)
            self._subtree_cache.set(s_key, cost)
            self._subtree_cache.set(c_key, cost)
            return cost, cost
        if tree1.children and not tree2.children:
            cost = sum(self.delete_cost(c) for c in tree1.children) + self.insert_cost(tree2)
            self._subtree_cache.set(s_key, cost)
            self._subtree_cache.set(c_key, cost)
            return cost, cost

        # Both have children - compute costs recursively
        children1 = tree1.children
        children2 = tree2.children
        n1, n2 = len(children1), len(children2)

        if n1 == 0 and n2 == 0:
            s_cost = self.structural_update_cost(tree1, tree2)
            c_cost = self.content_update_cost(tree1, tree2)
            self._subtree_cache.set(s_key, s_cost)
            self._subtree_cache.set(c_key, c_cost)
            return s_cost, c_cost

        # Use NumPy arrays for memory efficiency (float32 saves 50% vs float64)
        max_size = max(n1, n2)
        s_matrix = np.full((max_size, max_size), np.inf, dtype=np.float32)
        c_matrix = np.full((max_size, max_size), np.inf, dtype=np.float32)

        for i in range(n1):
            for j in range(n2):
                s, c = self._get_structural_and_content_costs(children1[i], children2[j])
                s_matrix[i, j] = s
                c_matrix[i, j] = c

        # Vectorized deletion/insertion costs
        for i in range(n1):
            del_cost = self.delete_cost(children1[i])
            s_matrix[i, n2:max_size] = del_cost
            c_matrix[i, n2:max_size] = del_cost

        for j in range(n2):
            ins_cost = self.insert_cost(children2[j])
            s_matrix[n1:max_size, j] = ins_cost
            c_matrix[n1:max_size, j] = ins_cost

        # Use structural for matching
        row_indices, col_indices = linear_sum_assignment(s_matrix)

        # Vectorized sum
        s_total = float(s_matrix[row_indices, col_indices].sum())
        c_total = float(c_matrix[row_indices, col_indices].sum())

        s_normalized = min(s_total / len(row_indices), 1.0)
        c_normalized = min(c_total / len(row_indices), 1.0)

        self._subtree_cache.set(s_key, s_normalized)
        self._subtree_cache.set(c_key, c_normalized)

        return s_normalized, c_normalized

    @lru_cache(maxsize=2000)
    def _get_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get embedding for a text with caching.

        First checks the pre-computed embedding dictionary, then falls back
        to individual computation if not found.
        """
        # Check pre-computed dictionary first
        if text in self._embedding_dict:
            return self._embedding_dict[text]

        # Fall back to individual computation
        if self.bedrock_client:
            try:
                emb = get_embeddings(text, self.model_id, self.bedrock_client, output_embedding_length=self.embedding_dim)
                self._embedding_dict[text] = emb  # Cache for future use
                return emb
            except Exception as e:
                warnings.warn(f"Failed to get Bedrock embedding for '{text}': {e}")
                return None
        elif self.embedding_model:
            try:
                # Preprocess key names for better semantic understanding
                processed_text = self._preprocess_key_name(text)
                embedding = self.embedding_model.encode(processed_text, show_progress_bar=False)
                self._embedding_dict[text] = embedding  # Cache for future use
                return embedding
            except Exception as e:
                warnings.warn(f"Failed to get embedding for '{text}': {e}")
                return None

        return None

    def _preprocess_key_name(self, key: str) -> str:
        """Preprocess key names for better semantic understanding."""
        if not key:
            return ""

        # Handle acronyms (e.g., "HTTPSConnection" -> "HTTPS Connection")
        processed = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1 \2', key)
        processed = re.sub(r'([a-z\d])([A-Z])', r'\1 \2', processed)

        # Replace separators
        processed = re.sub(r'[_\-\.]+', ' ', processed)

        # Clean up and lowercase
        return ' '.join(processed.lower().split())

    def collect_strings_from_json(self, json_obj: Any, strings: Set[str], min_length: int = 4) -> None:
        """
        Recursively collect all string values and keys from a JSON object.

        Also collects preprocessed versions of keys (used by _calculate_structural_similarity)
        to ensure they're in the embedding cache.

        Args:
            json_obj: The JSON object to extract strings from
            strings: Set to add strings to
            min_length: Minimum string length for embedding (shorter strings use edit distance)
        """
        if isinstance(json_obj, dict):
            for key, value in json_obj.items():
                # Add key if long enough
                if len(key) >= min_length:
                    strings.add(key)
                    # Also add preprocessed version (used by structural similarity)
                    preprocessed = self._preprocess_key_name(key)
                    if len(preprocessed) >= min_length:
                        strings.add(preprocessed)
                self.collect_strings_from_json(value, strings, min_length)
        elif isinstance(json_obj, list):
            for item in json_obj:
                self.collect_strings_from_json(item, strings, min_length)
        elif isinstance(json_obj, str):
            if len(json_obj) >= min_length and len(json_obj) < self.chunk_size:
                strings.add(json_obj)

    def precompute_embeddings(self, json_objects: List[Any], batch_size: int = 64,
                               show_progress: bool = True, max_workers: int = 10,
                               use_batch_inference: bool = False,
                               use_async: bool = False,
                               max_concurrent: int = 50,
                               s3_bucket: str = None,
                               s3_prefix: str = "bedrock-batch/embeddings",
                               role_arn: str = None,
                               auto_save_cache: bool = False,
                               cache_dir: str = None) -> int:
        """
        Pre-compute embeddings for all unique strings in the given JSON objects.

        This should be called BEFORE running similarity calculations to batch all
        embedding computations and significantly speed up the process.

        Args:
            json_objects: List of JSON objects to extract strings from
            batch_size: Batch size for SentenceTransformer encoding
            show_progress: Whether to show progress bar
            max_workers: Max parallel workers for Bedrock API calls (ThreadPoolExecutor)
            use_batch_inference: If True, use Bedrock Batch Inference (S3-based async)
                                 instead of parallel single API calls. Recommended for
                                 large datasets (>10,000 strings).
            use_async: If True, use asyncio with aioboto3 for Bedrock API calls.
                       This is more efficient than ThreadPoolExecutor for I/O-bound
                       operations. Requires aioboto3 package.
            max_concurrent: Maximum concurrent API calls when use_async=True (default 50)
            s3_bucket: S3 bucket for batch inference input/output (required if use_batch_inference=True)
            s3_prefix: S3 prefix for batch inference files
            role_arn: IAM role ARN for Bedrock Batch Inference
            auto_save_cache: If True, automatically save embeddings to a cache file with
                             model_id, dimension, and timestamp in the filename
            cache_dir: Directory to save the auto-generated cache file (default: current directory)

        Returns:
            Number of unique strings embedded
        """
        # Collect all unique strings
        all_strings: Set[str] = set()
        for json_obj in json_objects:
            if json_obj:  # Skip None/empty
                self.collect_strings_from_json(json_obj, all_strings)

        # Filter out strings already in the embedding dict
        new_strings = [s for s in all_strings if s not in self._embedding_dict]

        if not new_strings:
            self._embedding_dict_populated = True
            return 0

        if show_progress:
            print(f"Pre-computing embeddings for {len(new_strings)} unique strings...")

        if self.embedding_model:
            # Use SentenceTransformer batch encoding
            self._batch_encode_sentence_transformer(new_strings, batch_size, show_progress)
        elif self.bedrock_client:
            if use_batch_inference:
                # Use true Bedrock Batch Inference (S3-based async)
                self._batch_encode_bedrock_batch_inference(
                    new_strings, show_progress,
                    s3_bucket=s3_bucket, s3_prefix=s3_prefix, role_arn=role_arn
                )
            elif use_async:
                # Use async API calls with aioboto3
                if show_progress:
                    print(f"Using async Bedrock API calls (max {max_concurrent} concurrent)...")
                self._batch_encode_bedrock_async(
                    new_strings, show_progress, max_concurrent=max_concurrent
                )
            else:
                # Auto-select best method based on string count
                self._batch_encode_bedrock_auto(
                    new_strings, show_progress, max_workers=max_workers,
                    s3_bucket=s3_bucket, s3_prefix=s3_prefix, role_arn=role_arn
                )

        self._embedding_dict_populated = True

        # Auto-save cache if requested
        if auto_save_cache and len(new_strings) > 0:
            import os
            from datetime import datetime

            # Generate descriptive filename
            model_short = self.model_id.replace("amazon.", "").replace(":", "_").replace(".", "_")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            cache_filename = f"embeddings_{model_short}_dim{self.embedding_dim}_{timestamp}.npz"

            if cache_dir:
                os.makedirs(cache_dir, exist_ok=True)
                cache_path = os.path.join(cache_dir, cache_filename)
            else:
                cache_path = cache_filename

            self.save_embedding_dict(cache_path)
            if show_progress:
                print(f"Auto-saved embedding cache to: {cache_path}")

        return len(new_strings)

    def _store_embedding(self, text: str, embedding: np.ndarray) -> None:
        """Store embedding with optional fp16 conversion and cache size limit."""
        # Convert to fp16 if enabled (saves ~50% memory)
        if self.use_fp16_embeddings:
            embedding = embedding.astype(np.float16)

        # Check cache size limit
        if self.max_embedding_cache_size > 0 and len(self._embedding_dict) >= self.max_embedding_cache_size:
            # Simple eviction: remove oldest (first) entry
            if self._embedding_dict:
                oldest_key = next(iter(self._embedding_dict))
                del self._embedding_dict[oldest_key]

        self._embedding_dict[text] = embedding

    def _batch_encode_sentence_transformer(self, strings: List[str], batch_size: int = 64,
                                            show_progress: bool = True) -> None:
        """Batch encode strings using SentenceTransformer."""
        # Preprocess all strings
        processed_strings = [self._preprocess_key_name(s) for s in strings]

        # Encode in batches
        iterator = range(0, len(processed_strings), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Encoding batches", total=len(processed_strings) // batch_size + 1)

        for i in iterator:
            batch = processed_strings[i:i + batch_size]
            original_batch = strings[i:i + batch_size]

            try:
                embeddings = self.embedding_model.encode(batch, show_progress_bar=False, batch_size=batch_size)
                for orig_str, emb in zip(original_batch, embeddings):
                    self._store_embedding(orig_str, emb)
            except Exception as e:
                warnings.warn(f"Batch encoding failed: {e}")

    def _batch_encode_bedrock(self, strings: List[str], show_progress: bool = True,
                               max_workers: int = 10) -> None:
        """Batch encode strings using Bedrock API with parallel calls."""
        def encode_single(text: str) -> Tuple[str, Optional[np.ndarray]]:
            try:
                emb = get_embeddings(text, self.model_id, self.bedrock_client, output_embedding_length=self.embedding_dim)
                return (text, emb)
            except Exception as e:
                warnings.warn(f"Bedrock embedding failed for '{text[:50]}...': {e}")
                return (text, None)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(encode_single, s): s for s in strings}

            iterator = as_completed(futures)
            if show_progress:
                iterator = tqdm(iterator, total=len(strings), desc="Bedrock embeddings")

            for future in iterator:
                text, emb = future.result()
                if emb is not None:
                    self._store_embedding(text, emb)

    def _batch_encode_bedrock_async(self, strings: List[str], show_progress: bool = True,
                                     max_concurrent: int = 50, region_name: str = None) -> None:
        """
        Batch encode strings using Bedrock API with async calls.

        This method uses aioboto3 for true async I/O, which is more efficient than
        ThreadPoolExecutor for I/O-bound operations like API calls.

        Args:
            strings: List of strings to embed
            show_progress: Whether to show progress bar
            max_concurrent: Maximum concurrent API calls (default 50)
            region_name: AWS region name (default: use client's region)
        """
        try:
            import aioboto3
        except ImportError:
            warnings.warn("aioboto3 not installed. Falling back to ThreadPoolExecutor. "
                         "Install with: pip install aioboto3")
            self._batch_encode_bedrock(strings, show_progress, max_workers=max_concurrent)
            return

        if region_name is None:
            if hasattr(self.bedrock_client, '_client_config'):
                region_name = self.bedrock_client._client_config.region_name
            else:
                region_name = "us-west-2"

        async def encode_single_async(
            text: str,
            client,
            semaphore: asyncio.Semaphore
        ) -> Tuple[str, Optional[np.ndarray]]:
            """Encode a single string asynchronously."""
            async with semaphore:
                try:
                    # Build request body based on model
                    if self.model_id == "amazon.titan-embed-text-v1":
                        request_body = {"inputText": text}
                    elif self.model_id == "amazon.titan-embed-text-v2:0":
                        request_body = {
                            "inputText": text,
                            "dimensions": self.embedding_dim,
                            "normalize": True,
                            "embeddingTypes": ["float"]
                        }
                    elif self.model_id == "cohere.embed-multilingual-v3":
                        # Cohere Embed Multilingual V3 returns fixed 1024-dim embeddings
                        request_body = {
                            "texts": [text],
                            "input_type": "clustering",
                            "truncate": "END"
                        }
                    elif self.model_id in ["cohere.embed-v4:0", "us.cohere.embed-v4:0"]:
                        request_body = {
                            "texts": [text],
                            "input_type": "search_document",
                            "embedding_types": ["float"]
                        }
                    else:
                        request_body = {"inputText": text}

                    # Call Bedrock API
                    response = await client.invoke_model(
                        modelId=self.model_id,
                        body=json.dumps(request_body),
                        contentType="application/json",
                        accept="application/json"
                    )

                    # Read response body
                    response_body = await response['body'].read()
                    result = json.loads(response_body)

                    # Extract embedding based on model
                    if self.model_id == "cohere.embed-multilingual-v3":
                        embedding = np.array(result["embeddings"][0])
                    elif self.model_id in ["cohere.embed-v4:0", "us.cohere.embed-v4:0"]:
                        # Cohere Embed V4 returns embeddings as dict with 'float' key
                        embedding = np.array(result["embeddings"]["float"][0])
                    elif "embeddingsByType" in result:
                        embedding = np.array(result["embeddingsByType"]["float"])
                    else:
                        embedding = np.array(result.get("embedding", []))

                    return (text, embedding)

                except Exception as e:
                    warnings.warn(f"Async Bedrock embedding failed for '{text[:50]}...': {e}")
                    return (text, None)

        async def encode_all_async():
            """Encode all strings concurrently."""
            session = aioboto3.Session()
            semaphore = asyncio.Semaphore(max_concurrent)

            async with session.client('bedrock-runtime', region_name=region_name) as client:
                tasks = [encode_single_async(s, client, semaphore) for s in strings]

                if show_progress:
                    results = []
                    for coro in atqdm(asyncio.as_completed(tasks), total=len(tasks),
                                      desc="Async Bedrock embeddings"):
                        result = await coro
                        results.append(result)
                else:
                    results = await asyncio.gather(*tasks)

                return results

        # Run the async function
        try:
            # Check if we're already in an event loop
            loop = asyncio.get_running_loop()
            # If we are, use nest_asyncio or run in executor
            import nest_asyncio
            nest_asyncio.apply()
            results = asyncio.run(encode_all_async())
        except RuntimeError:
            # No running event loop, safe to use asyncio.run
            results = asyncio.run(encode_all_async())

        # Store results in embedding dict
        for text, emb in results:
            if emb is not None:
                self._embedding_dict[text] = emb

    def _batch_encode_bedrock_auto(
        self, strings: List[str], show_progress: bool = True,
        max_workers: int = 10, s3_bucket: str = None,
        s3_prefix: str = "bedrock-batch/embeddings", role_arn: str = None,
        batch_threshold: int = 5000
    ) -> None:
        """
        Auto-select the best embedding method based on string count.

        This method automatically chooses between parallel API calls and
        batch inference based on the number of strings to embed:

        - < 100 strings: Parallel API (batch inference not supported)
        - 100 - batch_threshold: Parallel API (faster, no job overhead)
        - > batch_threshold: Batch inference (no rate limits, better throughput)

        Args:
            strings: List of strings to embed
            show_progress: Whether to show progress
            max_workers: Max parallel workers for API calls
            s3_bucket: S3 bucket for batch inference
            s3_prefix: S3 prefix for batch inference files
            role_arn: IAM role ARN for batch inference
            batch_threshold: String count threshold for batch inference (default 5000)
        """
        num_strings = len(strings)

        # Check if batch inference is configured
        batch_configured = (s3_bucket is not None or
                           os.environ.get('BEDROCK_BATCH_S3_BUCKET') is not None)

        if num_strings < 100:
            # Batch inference not supported below 100
            if show_progress:
                print(f"Using parallel API calls ({num_strings} strings < 100 minimum for batch)")
            self._batch_encode_bedrock(strings, show_progress, max_workers)

        elif num_strings <= batch_threshold or not batch_configured:
            # Parallel API is faster for medium-sized datasets
            if show_progress:
                if not batch_configured:
                    print(f"Using parallel API calls ({num_strings} strings, batch not configured)")
                else:
                    print(f"Using parallel API calls ({num_strings} strings <= {batch_threshold} threshold)")
            self._batch_encode_bedrock(strings, show_progress, max_workers)

        else:
            # Large dataset - use batch inference with async parallel chunks
            if show_progress:
                print(f"Using batch inference ({num_strings} strings > {batch_threshold} threshold)")
            self._batch_encode_bedrock_batch_inference_async(
                strings, show_progress,
                s3_bucket=s3_bucket, s3_prefix=s3_prefix, role_arn=role_arn
            )

    def _batch_encode_bedrock_batch_inference_async(
        self, strings: List[str], show_progress: bool = True,
        s3_bucket: str = None, s3_prefix: str = "bedrock-batch/embeddings",
        role_arn: str = None, chunk_size: int = 50000
    ) -> None:
        """
        Batch encode using async parallel batch inference jobs.

        For very large datasets, this method chunks the strings and runs
        multiple batch inference jobs concurrently using asyncio.

        Args:
            strings: List of strings to embed
            show_progress: Whether to show progress
            s3_bucket: S3 bucket for batch inference
            s3_prefix: S3 prefix for files
            role_arn: IAM role ARN for batch inference
            chunk_size: Strings per batch job (default 50000)
        """
        import asyncio
        from .bedrock_utils import (
            prepare_batch_embedding_input,
            upload_to_s3,
            create_batch_inference_job,
            get_batch_job_status,
            parse_batch_embedding_output
        )
        import uuid
        import os as _os

        # Get configuration
        if s3_bucket is None:
            s3_bucket = _os.environ.get('BEDROCK_BATCH_S3_BUCKET')
        if role_arn is None:
            role_arn = _os.environ.get('BEDROCK_BATCH_ROLE_ARN')

        region = "us-west-2"
        if hasattr(self.bedrock_client, '_client_config'):
            region = self.bedrock_client._client_config.region_name

        # Chunk the strings
        chunks = [strings[i:i + chunk_size] for i in range(0, len(strings), chunk_size)]
        num_chunks = len(chunks)

        if show_progress:
            print(f"Splitting {len(strings)} strings into {num_chunks} batch job(s)...")

        async def submit_and_wait_job(chunk_strings: List[str], chunk_idx: int) -> dict:
            """Submit a batch job and wait for completion."""
            job_id = uuid.uuid4().hex[:8]

            # Prepare input
            input_path = prepare_batch_embedding_input(chunk_strings, self.model_id, embedding_dim=self.embedding_dim)

            try:
                # Upload to S3
                input_s3_key = f"{s3_prefix}/input/{job_id}/input.jsonl"
                input_s3_uri = upload_to_s3(input_path, s3_bucket, input_s3_key, region)

                output_s3_prefix = f"{s3_prefix}/output/{job_id}/"
                output_s3_uri = f"s3://{s3_bucket}/{output_s3_prefix}"

                # Create batch job
                job_response = create_batch_inference_job(
                    input_s3_uri=input_s3_uri,
                    output_s3_uri=output_s3_uri,
                    model_id=self.model_id,
                    role_arn=role_arn,
                    job_name=f"embed-chunk{chunk_idx}-{job_id}",
                    region=region
                )
                job_arn = job_response['jobArn']

                if show_progress:
                    print(f"  Chunk {chunk_idx + 1}/{num_chunks}: Job submitted ({len(chunk_strings)} strings)")

                # Poll for completion (async sleep)
                while True:
                    status = get_batch_job_status(job_arn, region)
                    job_status = status.get('status', 'Unknown')

                    if job_status == 'Completed':
                        # Parse results
                        embeddings = parse_batch_embedding_output(output_s3_uri, region)

                        # Map back to original strings
                        result = {}
                        for idx, text in enumerate(chunk_strings):
                            record_id = str(idx)
                            if record_id in embeddings:
                                result[text] = embeddings[record_id]

                        if show_progress:
                            print(f"  Chunk {chunk_idx + 1}/{num_chunks}: Completed ({len(result)} embeddings)")
                        return result

                    if job_status in ['Failed', 'Stopped', 'Expired']:
                        error_msg = status.get('message', 'Unknown error')
                        warnings.warn(f"Chunk {chunk_idx + 1} failed: {error_msg}")
                        return {}

                    await asyncio.sleep(15)  # Non-blocking sleep

            finally:
                # Cleanup local temp file
                if _os.path.exists(input_path):
                    _os.remove(input_path)

        async def run_all_jobs():
            """Run all batch jobs concurrently."""
            tasks = [submit_and_wait_job(chunk, idx) for idx, chunk in enumerate(chunks)]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            all_embeddings = {}
            for result in results:
                if isinstance(result, dict):
                    all_embeddings.update(result)
                elif isinstance(result, Exception):
                    warnings.warn(f"Batch job failed: {result}")

            return all_embeddings

        # Run async jobs
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        if loop.is_running():
            # If already in async context, create task
            import nest_asyncio
            nest_asyncio.apply()
            embeddings = loop.run_until_complete(run_all_jobs())
        else:
            embeddings = loop.run_until_complete(run_all_jobs())

        # Update embedding dict
        self._embedding_dict.update(embeddings)

        if show_progress:
            print(f"Batch inference completed: {len(embeddings)} total embeddings")

    def _batch_encode_bedrock_batch_inference(
        self, strings: List[str], show_progress: bool = True,
        s3_bucket: str = None, s3_prefix: str = "bedrock-batch/embeddings",
        role_arn: str = None, min_records: int = 100
    ) -> None:
        """
        Batch encode strings using true Bedrock Batch Inference (S3-based async).

        This method uses Bedrock's batch inference API which is more efficient
        for large datasets (>100 strings) compared to parallel single API calls.

        IMPORTANT: Bedrock Batch Inference requires a minimum of 100 records.
        For smaller datasets, this method will automatically fall back to
        parallel API calls.

        Args:
            strings: List of strings to embed
            show_progress: Whether to show progress
            s3_bucket: S3 bucket for input/output (required for batch inference)
            s3_prefix: S3 prefix for files
            role_arn: IAM role ARN for Bedrock Batch Inference
            min_records: Minimum records for batch inference (default 100)
        """
        # Check minimum records requirement
        if len(strings) < min_records:
            if show_progress:
                print(f"Only {len(strings)} strings (< {min_records} minimum). "
                      f"Using parallel API calls instead of batch inference.")
            self._batch_encode_bedrock(strings, show_progress)
            return

        from .bedrock_utils import batch_compute_embeddings_chunked

        if show_progress:
            print(f"Using Bedrock Batch Inference for {len(strings)} strings...")

        try:
            # Get region from bedrock client if possible
            region = "us-west-2"
            if hasattr(self.bedrock_client, '_client_config'):
                region = self.bedrock_client._client_config.region_name

            embeddings = batch_compute_embeddings_chunked(
                strings=strings,
                model_id=self.model_id,
                s3_bucket=s3_bucket,
                s3_prefix=s3_prefix,
                role_arn=role_arn,
                region=region,
                show_progress=show_progress,
                embedding_dim=self.embedding_dim
            )

            # Update embedding dict
            self._embedding_dict.update(embeddings)

            if show_progress:
                print(f"Batch inference completed: {len(embeddings)} embeddings computed")

        except Exception as e:
            warnings.warn(f"Batch inference failed: {e}. Falling back to parallel API calls.")
            # Fallback to parallel single API calls
            self._batch_encode_bedrock(strings, show_progress)

    def clear_embedding_cache(self) -> None:
        """Clear the pre-computed embedding dictionary."""
        self._embedding_dict.clear()
        self._embedding_dict_populated = False

    def save_embedding_dict(self, filepath: str) -> int:
        """
        Save the pre-computed embedding dictionary to disk for reuse.

        Args:
            filepath: Path to save the embedding dictionary (.npz format recommended)

        Returns:
            Number of embeddings saved
        """
        if not self._embedding_dict:
            warnings.warn("No embeddings to save. Run precompute_embeddings() first.")
            return 0

        # Separate strings and embeddings for numpy savez
        strings = list(self._embedding_dict.keys())
        embeddings = np.array([self._embedding_dict[s] for s in strings])

        # Save metadata along with embeddings (include dimension for compatibility check)
        np.savez_compressed(
            filepath,
            strings=np.array(strings, dtype=object),
            embeddings=embeddings,
            model_id=np.array([self.model_id], dtype=object),
            embedding_dim=np.array([self.embedding_dim])
        )

        print(f"Saved {len(strings)} embeddings (model={self.model_id}, dim={self.embedding_dim}) to {filepath}")
        return len(strings)

    def load_embedding_dict(self, filepath: str, strict_model_check: bool = True,
                             strict_dimension_check: bool = True) -> int:
        """
        Load a pre-computed embedding dictionary from disk.

        Args:
            filepath: Path to the saved embedding dictionary (.npz file)
            strict_model_check: If True, warn if model_id doesn't match
            strict_dimension_check: If True, raise error if embedding dimension doesn't match

        Returns:
            Number of embeddings loaded
        """
        import os
        if not os.path.exists(filepath):
            warnings.warn(f"Embedding file not found: {filepath}")
            return 0

        data = np.load(filepath, allow_pickle=True)

        # Check model compatibility
        if strict_model_check and 'model_id' in data:
            saved_model_id = str(data['model_id'][0])
            if saved_model_id != self.model_id:
                warnings.warn(
                    f"Model ID mismatch: saved with '{saved_model_id}', "
                    f"current is '{self.model_id}'. Embeddings may not be compatible."
                )

        # Check dimension compatibility
        if 'embedding_dim' in data:
            saved_dim = int(data['embedding_dim'][0])
            if saved_dim != self.embedding_dim:
                msg = (f"Embedding dimension mismatch: saved with dim={saved_dim}, "
                       f"current is dim={self.embedding_dim}. "
                       f"Cannot mix different dimension embeddings.")
                if strict_dimension_check:
                    raise ValueError(msg)
                else:
                    warnings.warn(msg)

        strings = data['strings']
        embeddings = data['embeddings']

        # Verify actual embedding dimensions match
        if len(embeddings) > 0:
            actual_dim = embeddings[0].shape[0]
            if actual_dim != self.embedding_dim:
                msg = (f"Actual embedding dimension ({actual_dim}) doesn't match "
                       f"current setting ({self.embedding_dim}). Cannot load cache.")
                raise ValueError(msg)

        # Populate the dictionary
        for s, emb in zip(strings, embeddings):
            self._embedding_dict[str(s)] = emb

        self._embedding_dict_populated = True
        saved_dim_str = f", dim={data['embedding_dim'][0]}" if 'embedding_dim' in data else ""
        print(f"Loaded {len(strings)} embeddings{saved_dim_str} from {filepath}")
        return len(strings)

    def _calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate semantic similarity between two texts using embeddings.

        For very short strings (< 4 characters), uses character-level edit distance
        instead of embeddings, as embeddings are unreliable for short strings.
        """
        if text1 == text2:
            return 1.0

        # For very short strings, use character-level edit distance instead of embeddings
        # Embeddings are unreliable for strings shorter than 4 characters
        MIN_LENGTH_FOR_EMBEDDINGS = 4
        if len(text1) < MIN_LENGTH_FOR_EMBEDDINGS or len(text2) < MIN_LENGTH_FOR_EMBEDDINGS:
            # Use normalized Levenshtein distance
            max_len = max(len(text1), len(text2))
            if max_len == 0:
                return 1.0

            # Calculate edit distance
            # Simple implementation: count character differences
            edit_distance = sum(c1 != c2 for c1, c2 in zip(text1, text2))
            edit_distance += abs(len(text1) - len(text2))  # Add length difference

            # Normalize to [0, 1] and invert (0 = different, 1 = same)
            similarity = 1.0 - (edit_distance / max_len)
            return max(0.0, similarity)

        emb1 = self._get_embedding(text1)
        emb2 = self._get_embedding(text2)

        if emb1 is None or emb2 is None:
            return 0.0

        # Ensure embeddings are 1D arrays
        emb1 = emb1.flatten()
        emb2 = emb2.flatten()

        # Cosine similarity
        dot_product = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        similarity = dot_product / (norm1 * norm2)
        similarity = np.clip(similarity, -1.0, 1.0)

        # Scale from [-1, 1] to [0, 1]
        similarity = (similarity + 1) / 2.0

        return float(similarity)

    def calculate_path_weight(self, path: str) -> float:
        """
        Calculate weight for a path based on its depth.
        Args:
            path: The path to calculate weight for
            
        Returns:
            A weight factor between 0 and 1
        """
        # Count the number of path segments
        if path == "root":
            depth = 0
        else:
            # Count dots and array indices
            depth = path.count('.') + path.count('[')

        # Apply exponential decay based on depth
        weight = self.path_weight_decay ** depth

        # Increase weight for required fields
        if path in self.required_fields:
            weight *= 1.5

        return weight

    def insert_cost(self, node: JsonNode) -> float:
        """
        Calculate the cost of inserting a node.

        Args:
            node: The node to insert

        Returns:
            The cost of insertion, bounded in [0, 1]
        """
        # Base cost is 1.0
        cost = 1.0

        # Apply path-based weighting (depth decay ensures cost <= 1.0)
        cost *= self.calculate_path_weight(node.path)

        # Ensure boundedness for formal properties
        return min(1.0, cost)

    def delete_cost(self, node: JsonNode) -> float:
        """
        Calculate the cost of deleting a node.

        Args:
            node: The node to delete

        Returns:
            The cost of deletion, bounded in [0, 1]
        """
        # Base cost is 1.0
        cost = 1.0

        # Apply path-based weighting (depth decay ensures cost <= 1.0)
        cost *= self.calculate_path_weight(node.path)

        # Ensure boundedness for formal properties
        return min(1.0, cost)

    def _compare_strings(self, str1: str, str2: str, method="cosine") -> float:
        """Compare two strings with optional semantic similarity and chunking for long text."""
        # Quick equality check

        if str1 == str2:
            return 1.0

        # Check cache first
        cached_sim = self._cache.get(str1, str2)
        if cached_sim is not None:
            return cached_sim

        if len(str1) < self.chunk_size and len(str2) < self.chunk_size:
            if method == "bertscore":
                P, R, F1 = bert_score([str1], [str2], lang="en")
                sim = float(F1.item())
            else:
                sim = self._calculate_semantic_similarity(str1, str2)
            self._cache.set(str1, str2, sim)
            return sim
        else:
            chunks1 = self._split_natural_text(str1)
            chunks2 = self._split_natural_text(str2)

            return 1 - self._compare_arrays_unordered(chunks1, chunks2, "str1_chunks", "str2_chunks")

    def _split_natural_text(self, text: str) -> List[str]:
        """Split natural language text into sentences or paragraphs using LangChain if available and enabled."""
        # For short text, don't split at all
        if len(text) < self.chunk_size:
            return [text]

        # Create a text splitter that tries to create semantically meaningful chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size - self.chunk_overlap,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", ";", ":", " "]
        )

        try:
            # Split the text
            return text_splitter.split_text(text)
        except Exception as e:
            warnings.warn(f"Failed to split text: {e}\n{text}")
            return [text]

    def _compare_numbers(self, num1: float, num2: float) -> float:
        """Compare two numbers with tolerance."""
        return 0 if num1 != num2 else 1

    def update_cost(self, node1: JsonNode, node2: JsonNode) -> float:
        """
        Calculate the cost of updating a node.

        Paper Eq. (2): γ_upd = α · γ_struct + (1-α) · γ_content

        Args:
            node1: The source node
            node2: The target node

        Returns:
            The cost of update (0 = identical, 1 = completely different)
        """
        structural_cost = self.structural_update_cost(node1, node2)
        content_cost = self.content_update_cost(node1, node2)

        # Paper Eq.(1): γ_upd = w · γ_struct + (1-w) · γ_content
        return self.structural_weight * structural_cost + (1 - self.structural_weight) * content_cost

    def _calculate_structural_similarity(self, node1: JsonNode, node2: JsonNode) -> float:
        """Calculate structural similarity: paths and field names only"""
        path1 = re.sub(r'\[\d+\]', '[*]', node1.path)
        path2 = re.sub(r'\[\d+\]', '[*]', node2.path)
        path_match = path1 == path2

        key1 = node1.label.split('.')[-1] if '.' in node1.label else node1.label
        key2 = node2.label.split('.')[-1] if '.' in node2.label else node2.label
        key1 = re.sub(r'\[\d+\]', '', key1)
        key2 = re.sub(r'\[\d+\]', '', key2)

        # Use exact match for field names if enabled (e.g., for tool calling)
        if self.exact_match_all_keys:
            # Exact string match for field names
            field_similarity = 1.0 if key1 == key2 else 0.0
        else:
            # Semantic similarity for field names (default behavior)
            # replace all special characters such as -, _, % with space
            key1_normalized = re.sub(r'[^a-zA-Z0-9]', ' ', key1)
            key2_normalized = re.sub(r'[^a-zA-Z0-9]', ' ', key2)
            field_similarity = self._calculate_semantic_similarity(key1_normalized, key2_normalized)

        structural_similarity = (field_similarity + (1 if path_match else 0)) / 2

        return structural_similarity

    def _calculate_content_similarity(self, node1: JsonNode, node2: JsonNode) -> float:
        """Calculate content similarity: value comparison only.

        Note: Structural similarity (field names, paths) should be evaluated separately.
        This function focuses purely on value similarity for leaf nodes.
        """
        if not node1.children and not node2.children:
            # Check if this field requires exact matching (bypass semantic similarity)
            if self.exact_match_fields:
                # Extract field name from path (e.g., "root.name" -> "name")
                field_name = node1.path.split('.')[-1] if node1.path else node1.label
                # Also check label directly (handles array elements like "name[0]")
                label_name = re.sub(r'\[\d+\]', '', node1.label.split('.')[-1])

                if field_name in self.exact_match_fields or label_name in self.exact_match_fields:
                    # Use exact string matching instead of semantic similarity
                    return 1.0 if str(node1.value) == str(node2.value) else 0.0

            # Same type comparisons
            if node1.node_type == node2.node_type:
                if node1.node_type == "string":
                    return self._compare_strings(str(node1.value), str(node2.value))
                elif node1.node_type == "number":
                    return self._compare_numbers(float(node1.value), float(node2.value))
                elif node1.node_type == "array":
                    return 1 - self._compare_arrays_unordered(list(node1.value), list(node2.value), node1.label, node2.label, variation_type="content")
                elif node1.node_type == "object":
                    return self._calculate_content_similarity(node1.value, node2.value)
                else:
                    # boolean, null - exact match
                    return 1.0 if node1.value == node2.value else 0.0
            else:
                # Cross-type comparison: use semantic similarity with type penalty
                val1_str = str(node1.value)
                val2_str = str(node2.value)

                semantic_sim = self._calculate_semantic_similarity(val1_str, val2_str)
                type_cost = self.type_change_cost.get((node1.node_type, node2.node_type), 1.0)
                return max(0.0, semantic_sim - type_cost)

        return 0.0  # Non-leaf nodes have no direct content to compare

    def structural_update_cost(self, node1: JsonNode, node2: JsonNode) -> float:
        """Calculate structural update cost only, bounded in [0, 1]"""
        # Early exit only for leaf nodes with same structure
        if (not node1.children and not node2.children and
            node1.label == node2.label and
            node1.node_type == node2.node_type):
            return 0.0  # Structurally identical leaf nodes

        structural_similarity = self._calculate_structural_similarity(node1, node2)
        cost = 1 - structural_similarity

        # Apply path weighting
        path_weight1 = self.calculate_path_weight(node1.path)
        path_weight2 = self.calculate_path_weight(node2.path)
        avg_path_weight = (path_weight1 + path_weight2) / 2.0

        # Ensure boundedness for formal properties
        return min(1.0, cost * avg_path_weight)

    def content_update_cost(self, node1: JsonNode, node2: JsonNode) -> float:
        """Calculate content update cost only, bounded in [0, 1]"""
        if (node1.node_type == node2.node_type and
            node1.value == node2.value):
            return 0.0  # Content identical

        content_similarity = self._calculate_content_similarity(node1, node2)
        cost = 1 - content_similarity

        # Apply path weighting
        path_weight1 = self.calculate_path_weight(node1.path)
        path_weight2 = self.calculate_path_weight(node2.path)
        avg_path_weight = (path_weight1 + path_weight2) / 2.0

        # Ensure boundedness for formal properties
        return min(1.0, cost * avg_path_weight)

    def _compare_arrays_unordered(self, arr1: List[Any], arr2: List[Any], arr1_label: str, arr2_label: str, variation_type="content") -> float:
        """
        Compare arrays using optimal matching (order-insensitive by default).

        If the field is in order_sensitive_fields, delegates to _compare_arrays_ordered
        for sequential comparison instead.
        """
        # Check if this field should use order-sensitive comparison
        if self._is_order_sensitive_field(arr1_label) or self._is_order_sensitive_field(arr2_label):
            return self._compare_arrays_ordered(arr1, arr2, arr1_label, arr2_label, variation_type)

        if len(arr1) == 0 and len(arr2) == 0:
            return 0  # Both empty arrays are identical
        if len(arr1) == 0 or len(arr2) == 0:
            return 1  # One empty, one not

        # Create similarity matrix
        cost_matrix = np.ones((len(arr1), len(arr2)))

        for i, item1 in enumerate(arr1):
            for j, item2 in enumerate(arr2):
                # Handle different types appropriately
                if not isinstance(item1, type(item2)):
                    cost_matrix[i, j] = 1.0
                elif isinstance(item1, str):
                    cost_matrix[i, j] = 1-self._compare_strings(str(item1), str(item2))
                elif isinstance(item1, (int, float)):
                    cost_matrix[i, j] = 1-self._compare_numbers(float(item1), float(item2))
                elif isinstance(item1, dict):
                    # Recursive comparison for nested objects
                    tree1 = JsonNode.from_dict(item1, f"{arr1_label}", sort_arrays=self.sort_arrays, sort_keys=self.sort_keys,
                                               order_sensitive_fields=self.order_sensitive_fields)
                    tree2 = JsonNode.from_dict(item2, f"{arr2_label}", sort_arrays=self.sort_arrays, sort_keys=self.sort_keys,
                                               order_sensitive_fields=self.order_sensitive_fields)
                    cost_matrix[i, j] = self._calculate_optimal_matching_cost(tree1, tree2, variation_type=variation_type)
                elif isinstance(item1, list):
                    # Recursive array comparison
                    cost_matrix[i, j] = self._compare_arrays_unordered(item1, item2, arr1_label, arr2_label, variation_type=variation_type)
                elif item1 and item1 == item2:
                    cost_matrix[i, j] = 0.0
                else:
                    cost_matrix[i, j] = 1.0

        # Use Hungarian algorithm for optimal matching
        len1, len2 = len(arr1), len(arr2)
        max_len = max(len1, len2)
        
        if cost_matrix.shape[0] != cost_matrix.shape[1]:
            # Pad matrix for Hungarian algorithm
            padded_matrix = np.ones((max_len, max_len))
            padded_matrix[:len1, :len2] = cost_matrix
            cost_matrix = padded_matrix

        row_indices, col_indices = linear_sum_assignment(cost_matrix)

        # Calculate total cost including unmatched elements
        total_cost = 0
        matched_elements = 0

        for i, j in zip(row_indices, col_indices):
            if i < len1 and j < len2:
                total_cost += cost_matrix[i, j]
                matched_elements += 1
            elif i < len1:
                total_cost += 1.0  # Full deletion cost
            elif j < len2:
                total_cost += 1.0  # Full insertion cost

        # Also account for any unmatched elements not covered by the assignment
        # (This handles cases where one array is larger)
        unmatched_from_arr1 = len1 - matched_elements
        unmatched_from_arr2 = len2 - matched_elements
        total_cost += max(max(0, unmatched_from_arr1), max(0, unmatched_from_arr2))

        # Normalize by the maximum array length
        normalized_cost = total_cost / max_len if max_len > 0 else 0

        return min(normalized_cost, 1)

    def _compare_arrays_ordered(self, arr1: List[Any], arr2: List[Any], arr1_label: str, arr2_label: str, variation_type="content") -> float:
        """
        Compare arrays with order sensitivity using sequential alignment.

        This method compares elements in order (index 0 vs 0, 1 vs 1, etc.) and handles
        length differences as insertions/deletions. Use this for fields where order matters
        (e.g., agent call traces, execution steps).

        Args:
            arr1: First array
            arr2: Second array
            arr1_label: Label for first array (for nested comparison)
            arr2_label: Label for second array (for nested comparison)
            variation_type: Type of variation ("structural", "content", or "combined")

        Returns:
            Cost value between 0 (identical) and 1 (completely different)
        """
        if len(arr1) == 0 and len(arr2) == 0:
            return 0  # Both empty arrays are identical
        if len(arr1) == 0 or len(arr2) == 0:
            return 1  # One empty, one not

        len1, len2 = len(arr1), len(arr2)
        max_len = max(len1, len2)
        min_len = min(len1, len2)

        # Compare elements sequentially
        total_cost = 0.0

        for i in range(min_len):
            item1 = arr1[i]
            item2 = arr2[i]

            # Calculate cost for this position
            if not isinstance(item1, type(item2)):
                cost = 1.0
            elif isinstance(item1, str):
                cost = 1 - self._compare_strings(str(item1), str(item2))
            elif isinstance(item1, (int, float)):
                cost = 1 - self._compare_numbers(float(item1), float(item2))
            elif isinstance(item1, dict):
                # Recursive comparison for nested objects
                tree1 = JsonNode.from_dict(item1, f"{arr1_label}[{i}]", sort_arrays=self.sort_arrays, sort_keys=self.sort_keys,
                                           order_sensitive_fields=self.order_sensitive_fields)
                tree2 = JsonNode.from_dict(item2, f"{arr2_label}[{i}]", sort_arrays=self.sort_arrays, sort_keys=self.sort_keys,
                                           order_sensitive_fields=self.order_sensitive_fields)
                cost = self._calculate_optimal_matching_cost(tree1, tree2, variation_type=variation_type)
            elif isinstance(item1, list):
                # Check if nested list should also be order-sensitive
                nested_label1 = f"{arr1_label}[{i}]"
                nested_label2 = f"{arr2_label}[{i}]"
                if self._is_order_sensitive_field(nested_label1) or self._is_order_sensitive_field(nested_label2):
                    cost = self._compare_arrays_ordered(item1, item2, nested_label1, nested_label2, variation_type=variation_type)
                else:
                    cost = self._compare_arrays_unordered(item1, item2, nested_label1, nested_label2, variation_type=variation_type)
            elif item1 == item2:
                cost = 0.0
            else:
                cost = 1.0

            total_cost += cost

        # Add insertion/deletion costs for length differences
        # Each extra element costs 1.0 (full insertion/deletion cost)
        length_diff = abs(len1 - len2)
        total_cost += length_diff

        # Normalize by the maximum array length
        normalized_cost = total_cost / max_len if max_len > 0 else 0

        return min(normalized_cost, 1)

    def _is_order_sensitive_field(self, field_label: str) -> bool:
        """
        Check if a field should use order-sensitive comparison.

        Args:
            field_label: The field label/path to check

        Returns:
            True if the field is in order_sensitive_fields, False otherwise
        """
        if not self.order_sensitive_fields:
            return False

        # Extract field name from label (handle nested paths like "root.trace[0]")
        # Check if any part of the path matches order_sensitive_fields
        field_parts = field_label.replace('[', '.').replace(']', '').split('.')

        for part in field_parts:
            # Remove array indices if present
            clean_part = part.strip()
            if clean_part and not clean_part.isdigit():
                if clean_part in self.order_sensitive_fields:
                    return True

        return False

    def calculate_field_level_similarity(self, json1: Dict[str, Any], json2: Dict[str, Any], exact_match_fields: Set[str] = None) -> Dict[str, Dict[str, Any]]:
        """Calculate content similarity for each matching field pair."""
        json1 = {"root": json1} if isinstance(json1, dict) else json1
        json2 = {"root": json2} if isinstance(json2, dict) else json2

        tree1 = JsonNode.from_dict(json1, sort_arrays=self.sort_arrays, sort_keys=self.sort_keys,
                                   order_sensitive_fields=self.order_sensitive_fields)
        tree2 = JsonNode.from_dict(json2, sort_arrays=self.sort_arrays, sort_keys=self.sort_keys,
                                   order_sensitive_fields=self.order_sensitive_fields)

        field_similarities = {}
        self.exact_match_fields = exact_match_fields or set()
        self._collect_field_similarities(tree1, tree2, field_similarities)
        return field_similarities

    def _collect_field_similarities(self, tree1: JsonNode, tree2: JsonNode, similarities: Dict[str, float]):
        """Recursively collect field-level similarities."""
        if not tree1.children and not tree2.children:
            structural_sim = self._calculate_structural_similarity(tree1, tree2)
            if structural_sim > 0.3:
                # Check if field requires exact match
                field_name = tree1.path.split('.')[-1]
                if hasattr(self, 'exact_match_fields') and field_name in self.exact_match_fields:
                    content_sim = 1.0 if tree1.value == tree2.value else 0.0
                else:
                    content_sim = self._calculate_content_similarity(tree1, tree2)

                # For dict values, include both structural and content similarity
                if tree1.node_type == "object" and tree2.node_type == "object":
                    combined_sim = (structural_sim + content_sim) / 2
                    similarities[tree1.path] = {"similarity": combined_sim, "matched_with": tree2.path, "structural": structural_sim, "content": content_sim}
                else:
                    similarities[tree1.path] = {"similarity": content_sim, "matched_with": tree2.path}

        if tree1.children and tree2.children:
            parent_structural_sim = self._calculate_structural_similarity(tree1, tree2)
            if parent_structural_sim > 0.5:
                n1, n2 = len(tree1.children), len(tree2.children)
                cost_matrix = [[1.0] * n2 for _ in range(n1)]

                for i, child1 in enumerate(tree1.children):
                    for j, child2 in enumerate(tree2.children):
                        structural_sim = self._calculate_structural_similarity(child1, child2)
                        cost_matrix[i][j] = 1.0 - structural_sim

                from scipy.optimize import linear_sum_assignment
                row_indices, col_indices = linear_sum_assignment(cost_matrix)

                for i, j in zip(row_indices, col_indices):
                    if i < n1 and j < n2 and cost_matrix[i][j] < 0.7:
                        self._collect_field_similarities(tree1.children[i], tree2.children[j], similarities)

    def _calculate_sequential_matching_cost(
        self, tree1: JsonNode, tree2: JsonNode, variation_type: str = "combined"
    ) -> float:
        """
        Calculate matching cost using sequential (positional) matching for order-sensitive arrays.
        Elements are compared by position (index 0 with index 0, etc.).

        Args:
            tree1: First tree (array node)
            tree2: Second tree (array node)
            variation_type: "structural", "content", or "combined"

        Returns:
            Normalized cost between 0 and 1
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
                total_cost += self._calculate_optimal_matching_cost(
                    children1[i], children2[i], "structural"
                )
            elif variation_type == "content":
                total_cost += self._calculate_optimal_matching_cost(
                    children1[i], children2[i], "content"
                )
            else:  # combined
                total_cost += self._calculate_optimal_matching_cost(
                    children1[i], children2[i], "combined"
                )

        # Add deletion costs for extra elements in tree1
        for i in range(matched_count, n1):
            total_cost += self.delete_cost(children1[i])

        # Add insertion costs for extra elements in tree2
        for j in range(matched_count, n2):
            total_cost += self.insert_cost(children2[j])

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

    def _calculate_optimal_matching_cost(
        self, tree1: JsonNode, tree2: JsonNode, variation_type: str = "combined"
    ) -> float:
        """
        Calculate optimal matching cost between two trees using Hungarian algorithm.
        For order-sensitive array fields, uses sequential (positional) matching instead.

        Args:
            tree1: First tree
            tree2: Second tree
            variation_type: "structural", "content", or "combined"

        For "combined" type:
            - First finds optimal structural matching using Hungarian algorithm
            - Then calculates content costs based on that fixed structural matching
            - Returns weighted average: 0.5 * structural_cost + 0.5 * content_cost
        """
        # Base case: both are leaf nodes
        if not tree1.children and not tree2.children:
            if variation_type == "structural":
                return self.structural_update_cost(tree1, tree2)
            elif variation_type == "content":
                return self.content_update_cost(tree1, tree2)
            else:  # combined
                return self.update_cost(tree1, tree2)

        # If one is leaf and other is not, use insert/delete costs
        if not tree1.children and tree2.children:
            return self.delete_cost(tree1) + sum(
                self.insert_cost(child) for child in tree2.children
            )
        if tree1.children and not tree2.children:
            return sum(
                self.delete_cost(child) for child in tree1.children
            ) + self.insert_cost(tree2)

        # Both have children
        children1 = tree1.children
        children2 = tree2.children

        # Check if this is an order-sensitive array field
        is_order_sensitive = (
            tree1.node_type == "array" and
            tree2.node_type == "array" and
            (self._is_order_sensitive_field(tree1.label) or self._is_order_sensitive_field(tree2.label))
        )

        # For order-sensitive arrays, use sequential matching
        if is_order_sensitive:
            return self._calculate_sequential_matching_cost(tree1, tree2, variation_type)

        if not children1 and not children2:
            if variation_type == "structural":
                return self.structural_update_cost(tree1, tree2)
            elif variation_type == "content":
                return self.content_update_cost(tree1, tree2)
            else:
                return self.update_cost(tree1, tree2)

        # Create cost matrix for Hungarian algorithm
        n1, n2 = len(children1), len(children2)
        max_size = max(n1, n2)

        # For "combined" type, use structural matching first
        if variation_type == "combined":
            # Build structural cost matrix
            structural_cost_matrix = [[float('inf')] * max_size for _ in range(max_size)]

            for i in range(n1):
                for j in range(n2):
                    structural_cost_matrix[i][j] = self._calculate_optimal_matching_cost(
                        children1[i], children2[j], "structural"
                    )

            # Add deletion costs for unmatched nodes in tree1
            for i in range(n1):
                for j in range(n2, max_size):
                    structural_cost_matrix[i][j] = self.delete_cost(children1[i])

            # Add insertion costs for unmatched nodes in tree2
            for i in range(n1, max_size):
                for j in range(n2):
                    structural_cost_matrix[i][j] = self.insert_cost(children2[j])

            # Solve assignment problem using Hungarian algorithm on structural costs
            from scipy.optimize import linear_sum_assignment
            row_indices, col_indices = linear_sum_assignment(structural_cost_matrix)

            # Calculate structural cost
            structural_total_cost = sum(structural_cost_matrix[i][j] for i, j in zip(row_indices, col_indices))

            # Now calculate content cost based on the structural matching
            # IMPORTANT: Use content_update_cost directly to avoid re-running Hungarian
            content_total_cost = 0.0
            for i, j in zip(row_indices, col_indices):
                if i < n1 and j < n2:
                    # Matched pair - calculate content cost directly
                    # For leaves: use content_update_cost
                    # For non-leaves: recursively use combined matching (structure-guided)
                    if not children1[i].children and not children2[j].children:
                        content_total_cost += self.content_update_cost(children1[i], children2[j])
                    else:
                        # For non-leaf nodes, recursively calculate combined cost
                        # (which also uses structural matching to guide content)
                        content_total_cost += self._calculate_optimal_matching_cost(
                            children1[i], children2[j], "combined"
                        )
                elif i < n1:
                    # Deletion
                    content_total_cost += self.delete_cost(children1[i])
                elif j < n2:
                    # Insertion
                    content_total_cost += self.insert_cost(children2[j])

            # Tree-level weighted average, Eq. (2): γ_upd = w · γ_struct + (1-w) · γ_content
            total_cost = self.structural_weight * structural_total_cost + (1 - self.structural_weight) * content_total_cost

            normalized_cost = total_cost / len(row_indices)
            normalized_cost = min(normalized_cost, 1.0)

            return normalized_cost

        # For "structural" or "content" types, use original algorithm
        cost_matrix = [[float('inf')] * max_size for _ in range(max_size)]

        # Fill cost matrix
        for i in range(n1):
            for j in range(n2):
                # Cost of matching child i with child j
                cost_matrix[i][j] = self._calculate_optimal_matching_cost(
                    children1[i], children2[j], variation_type
                )

        # Add deletion costs for unmatched nodes in tree1
        for i in range(n1):
            for j in range(n2, max_size):
                cost_matrix[i][j] = self.delete_cost(children1[i])

        # Add insertion costs for unmatched nodes in tree2
        for i in range(n1, max_size):
            for j in range(n2):
                cost_matrix[i][j] = self.insert_cost(children2[j])

        # Solve assignment problem using Hungarian algorithm
        from scipy.optimize import linear_sum_assignment
        row_indices, col_indices = linear_sum_assignment(cost_matrix)

        total_cost = sum(cost_matrix[i][j] for i, j in zip(row_indices, col_indices))

        normalized_cost = total_cost / len(row_indices)
        normalized_cost = min(normalized_cost, 1.0)

        return normalized_cost

    def calculate_tree_edit_distance_opt(self, json1: Dict[str, Any], json2: Dict[str, Any], variation_type: str = "combined") -> float:
        return self.calculate_tree_edit_distance(json1, json2, original_zss=False, variation_type=variation_type)

    def calculate_tree_edit_distance_fast(self, json1: Dict[str, Any], json2: Dict[str, Any],
                                          variation_type: str = "combined",
                                          use_greedy: bool = None) -> float:
        """
        Optimized tree edit distance calculation with all performance improvements.

        Optimizations included:
        1. Memoization for subtree comparisons
        2. Single-pass combined calculation
        3. Optional greedy matching (O(B²) instead of O(B³))
        4. Vectorized leaf comparison
        5. Early pruning for mismatched structures

        Args:
            json1: First JSON object
            json2: Second JSON object
            variation_type: "structural", "content", or "combined"
            use_greedy: Override greedy matching setting (None uses instance setting)

        Returns:
            similarity_score (0-1, higher is more similar)
        """
        # Temporarily override greedy setting if specified
        original_greedy = self.use_greedy_matching
        if use_greedy is not None:
            self.use_greedy_matching = use_greedy

        try:
            json1 = {"root": json1} if isinstance(json1, dict) else json1
            json2 = {"root": json2} if isinstance(json2, dict) else json2

            # CRITICAL: Clear node hash cache to prevent id() reuse bugs.
            # Python may reuse memory addresses for new objects after old ones are garbage collected.
            self._node_hash_cache.clear()

            # Convert JSONs to trees
            tree1 = JsonNode.from_dict(json1, sort_arrays=self.sort_arrays, sort_keys=self.sort_keys,
                                       order_sensitive_fields=self.order_sensitive_fields)
            tree2 = JsonNode.from_dict(json2, sort_arrays=self.sort_arrays, sort_keys=self.sort_keys,
                                       order_sensitive_fields=self.order_sensitive_fields)

            # === Optimization 4: Vectorized leaf comparison ===
            # Pre-compute all leaf similarities before tree traversal
            self._precompute_leaf_similarities(tree1, tree2)

            # Use optimized matching cost calculation
            distance = self._calculate_optimal_matching_cost_fast(tree1, tree2, variation_type)

            similarity = 1.0 - distance
            return similarity

        finally:
            # Restore original greedy setting
            self.use_greedy_matching = original_greedy

    def calculate_tree_edit_distance(self, json1: Dict[str, Any], json2: Dict[str, Any], original_zss=True, variation_type: str = "combined") -> float:
        """
        Calculate tree edit distance between two JSON objects.

        Args:
            json1: First JSON object
            json2: Second JSON object
            original_zss: Whether to use original ZSS algorithm
            variation_type: "structural", "content", or "combined"

        Returns:
            similarity_score
        """
        json1 = {"root": json1} if isinstance(json1, dict) else json1
        json2 = {"root": json2} if isinstance(json2, dict) else json2

        # CRITICAL: Clear node hash cache to prevent id() reuse bugs.
        # Python may reuse memory addresses for new objects after old ones are garbage collected.
        # The _node_hash_cache uses id(node) as key, so stale entries could return wrong hashes
        # for new nodes that happen to reuse the same memory address.
        self._node_hash_cache.clear()

        # Convert JSONs to trees (pass order_sensitive_fields to preserve order for those arrays)
        tree1 = JsonNode.from_dict(json1, sort_arrays=self.sort_arrays, sort_keys=self.sort_keys,
                                   order_sensitive_fields=self.order_sensitive_fields)
        tree2 = JsonNode.from_dict(json2, sort_arrays=self.sort_arrays, sort_keys=self.sort_keys,
                                   order_sensitive_fields=self.order_sensitive_fields)

        if original_zss:
            try:
                # For ZSS, use appropriate cost function based on variation_type
                """
                if variation_type == "structural":
                    update_cost_func = self.structural_update_cost
                elif variation_type == "content":
                    update_cost_func = self.content_update_cost
                else:
                    update_cost_func = self.update_cost

                distance = zss.distance(
                    tree1, tree2,
                    get_children=lambda x: x.get_children(),
                    insert_cost=self.insert_cost,
                    remove_cost=self.delete_cost,
                    update_cost=update_cost_func
                )
                """
                distance = zss.simple_distance(tree1, tree2, get_children=lambda x: x.get_children())
                max_node = max(tree1.count_nodes(), tree2.count_nodes())
                return 1 - distance/max_node

            except TypeError as e:
                raise TypeError(
                    f"Failed to calculate tree distance. Ensure zss is properly installed "
                    f"and trees have compatible structure: {str(e)}"
                ) from e
        else:
            # Use optimized version with all improvements
            distance = self._calculate_optimal_matching_cost_fast(tree1, tree2, variation_type)

        similarity = 1.0 - distance
        return similarity

    def calculate_bertscore(self, json1: Dict[str, Any], json2: Dict[str, Any], **kwargs) -> float:
        # Preprocess JSONs to make them order-invariant
        tree1 = JsonNode.from_dict(json1, sort_arrays=self.sort_arrays, sort_keys=self.sort_keys)
        tree2 = JsonNode.from_dict(json2, sort_arrays=self.sort_arrays, sort_keys=self.sort_keys)
        processed_json1 = tree1.reconstruct_json()
        processed_json2 = tree2.reconstruct_json()

        P, R, F1 = bert_score([str(processed_json1)], [str(processed_json2)], lang="en")
        return float(F1.item())

    def calculate_similarity_with_deepdiff(self, json1: [Dict[str, Any], List], json2: [Dict[str, Any], List], **kwargs) -> float:
        diff = DeepDiff(json1, json2, ignore_order=True, cache_size=5000, get_deep_distance=True)
        return 1- diff['deep_distance']

    def calculate_similarity_with_deepdiff_opt(self, json1: [Dict[str, Any], List], json2: [Dict[str, Any], List], variation_type: str = "combined", structural_weight=0.5, **kwargs) -> float:
        """
        Calculate similarity using DeepDiff with enhanced value comparison.
        Uses semantic similarity for strings and proper comparison for numbers.
        
        Args:
            json1: First JSON object
            json2: Second JSON object
            variation_type: "structural", "content", or "combined"
            
        Returns:
            Similarity score between 0 and 1 (1 = identical, 0 = completely different)
        """
        try:
            # Use DeepDiff to find structural differences
            diff = DeepDiff(json1, json2, ignore_order=True, cache_size=5000)

            # If no differences found, return perfect similarity
            if not diff:
                return 1.0

            # Calculate similarity based on variation type
            if variation_type == "structural":
                return self._calculate_deepdiff_structural_only(diff, json1, json2)
            elif variation_type == "content":
                return self._calculate_deepdiff_content_only(diff, json1, json2)
            else:  # combined
                structural_sim = self._calculate_deepdiff_structural_only(diff, json1, json2)
                content_sim = self._calculate_deepdiff_content_only(diff, json1, json2)
                return structural_sim * structural_weight + (1 - structural_weight) * content_sim

        except Exception as e:
            warnings.warn(f"Error in DeepDiff calculation: {str(e)}")
            return 0.0

    def _calculate_deepdiff_structural_only(self, diff: dict, json1, json2) -> float:
        """Calculate structural similarity only (schema organization changes)"""
        structural_changes = 0

        # Count structural changes only (schema organization)
        if 'dictionary_item_added' in diff:
            structural_changes += len(diff['dictionary_item_added'])

        if 'dictionary_item_removed' in diff:
            structural_changes += len(diff['dictionary_item_removed'])

        if 'iterable_item_added' in diff:
            structural_changes += len(diff['iterable_item_added'])

        if 'iterable_item_removed' in diff:
            structural_changes += len(diff['iterable_item_removed'])

        # Estimate total structural elements
        total_elements = count_json_elements(json1) + count_json_elements(json2)
        if total_elements == 0:
            return 1.0

        # Calculate structural similarity
        structural_similarity = max(0.0, 1.0 - (structural_changes * 2) / total_elements)
        return structural_similarity

    def _calculate_deepdiff_content_only(self, diff: dict, json1, json2) -> float:
        """Calculate content similarity only (reuse original value processing logic)"""
        # First check structural similarity
        structural_sim = self._calculate_deepdiff_structural_only(diff, json1, json2)

        if structural_sim < 0.5:  # Same threshold as other methods
            return 0.0

        # Reuse original value processing logic
        total_similarity_score = 0.0
        total_comparisons = 0

        # Handle value changes with semantic comparison (original logic)
        if 'values_changed' in diff:
            for path, change in diff['values_changed'].items():
                old_value = change['old_value']
                new_value = change['new_value']

                # Use appropriate comparison method based on value types
                if isinstance(old_value, str) and isinstance(new_value, str):
                    value_similarity = self._compare_strings(old_value, new_value)
                elif isinstance(old_value, (int, float)) and isinstance(new_value, (int, float)):
                    value_similarity = self._compare_numbers(float(old_value), float(new_value))
                elif isinstance(old_value, bool) and isinstance(new_value, bool):
                    value_similarity = 1.0 if old_value == new_value else 0.0
                elif old_value is None or new_value is None:
                    value_similarity = 1.0 if old_value == new_value else 0.0
                elif isinstance(old_value, (list, dict)) and isinstance(new_value, (list, dict)):
                    value_similarity = self.calculate_similarity_with_deepdiff(old_value, new_value)
                else:
                    try:
                        value_similarity = self._compare_numbers(float(old_value), float(new_value))
                    except (ValueError, TypeError):
                        value_similarity = self._compare_strings(str(old_value), str(new_value))

                total_similarity_score += value_similarity
                total_comparisons += 1

        # Handle type changes as content changes (moved from structural)
        if 'type_changes' in diff:
            for path, change in diff['type_changes'].items():
                old_type = change['old_type'].__name__ if hasattr(change['old_type'], '__name__') else str(change['old_type'])
                new_type = change['new_type'].__name__ if hasattr(change['new_type'], '__name__') else str(change['new_type'])

                # Map Python types to our type system
                type_mapping = {
                    'str': 'string',
                    'int': 'number',
                    'float': 'number',
                    'bool': 'boolean',
                    'NoneType': 'null',
                    'list': 'array',
                    'dict': 'object'
                }

                old_type_mapped = type_mapping.get(old_type, old_type)
                new_type_mapped = type_mapping.get(new_type, new_type)

                # Use type_change_cost to calculate type similarity
                type_cost = self.type_change_cost.get((old_type_mapped, new_type_mapped), 1.0)
                type_similarity = 1.0 - type_cost

                total_similarity_score += type_similarity
                total_comparisons += 1

        # Return content similarity
        if total_comparisons == 0:
            return 1.0  # No value or type changes

        return total_similarity_score / total_comparisons


if __name__ == "__main__":
    # Example usage
    evaluator = SemanticJsonTreeConsistencyEvaluator(
        model_id='amazon.titan-embed-text-v2:0',
    )

    json1 = {'name': 'John', 'age': 30, 'city': 'New York'}
    json2 = {'name': 'John', 'age': 30, 'location': 'NYC'}

    similarity = evaluator.calculate_tree_edit_distance_opt(json1, json2, variation_type="combined")
    print(f"Similarity: {similarity:.4f}")
