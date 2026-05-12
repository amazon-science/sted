"""
Semantic JSON Structural Consistency Evaluation using Tree Edit Distance

This module enhances the JSON tree consistency evaluation with semantic similarity capabilities.
It combines tree edit distance algorithms with embedding-based semantic similarity to provide
more accurate structural consistency evaluation for JSON outputs.
"""

# === MODULE MAP (after v0.2.0 refactor) ===
# - LRUCache:                       sted._lru_cache
# - Type-change costs:              sted._costs
# - Bedrock embedding backends:     sted.embeddings.bedrock
# - SentenceTransformer backend:    sted.embeddings.sentence_transformers
# - BERTScore / DeepDiff baselines: sted._baselines
# - Hungarian / matching algorithms: sted._matching
# - Core class:                     this file (lines below)

import os
import asyncio
import json
import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Set
from functools import lru_cache
from collections import OrderedDict
import warnings
import re

# bert_score is optional (only used by the bertscore_pair / bertscore_json
# helpers below). Lazy-imported so users without the bertscore extra can
# still use the STED metric.
try:
    from bert_score import score as bert_score
except ImportError:
    bert_score = None

from scipy.optimize import linear_sum_assignment
import zss

from langchain_text_splitters import RecursiveCharacterTextSplitter

from sentence_transformers import SentenceTransformer

from deepdiff import DeepDiff


from .json_tree_node import JsonNode
from .similarity_cache import StringSimilarityCache
from .utils import get_embeddings, create_bedrock_client, count_json_elements
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from tqdm.asyncio import tqdm as atqdm

# botocore is optional (only needed for Bedrock embedding backends).
try:
    from botocore.config import Config as _BotoConfig
except ImportError:
    _BotoConfig = None

from transformers import logging
logging.set_verbosity_error()


# LRUCache definition moved to ./_lru_cache.py during the v0.2.0 refactor.
# Re-imported here so existing
#     from sted.semantic_json_tree_consistency import LRUCache
# imports keep working.
from ._lru_cache import LRUCache  # noqa: E402, F401


# Default type-change costs moved to ./_costs.py during the v0.2.0 refactor.
# Re-imported here so existing imports keep working.
from ._costs import _get_default_type_change_costs  # noqa: E402, F401


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
                 subtree_cache_size: int = 0,
                 use_fp16_embeddings: bool = False,
                 max_embedding_cache_size: int = 100000,
                 min_length_for_embeddings: int = 4
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
            subtree_cache_size: Size of the subtree LRU cache. Default 0 (disabled).
                The cache is only useful when you re-compare the same subtree pair
                multiple times — most production workloads don't trigger this. Set
                to >0 (e.g., 10000) only for benchmarks comparing many similar trees.
            use_fp16_embeddings: Store embeddings as float16 to save ~50% memory (slight precision loss)
            max_embedding_cache_size: Max embeddings to cache (default 100000). Set to 0 for unbounded.
            min_length_for_embeddings: Strings shorter than this fall back to character-level
                edit distance instead of embeddings (which are unreliable for very short
                strings). Default 4.
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

        # boto_config is only used by Bedrock embedding paths below; build it
        # lazily so users without the bedrock extra can still instantiate the
        # evaluator with a non-Bedrock model (e.g. local sentence-transformers).
        if _BotoConfig is not None:
            boto_config = _BotoConfig(
                retries={
                    'max_attempts': 10,
                    'mode': 'adaptive'
                },
                max_pool_connections=50  # Increase connection pool size
            )
        else:
            boto_config = None

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
        # Default is disabled (size=0) since most production workloads do not
        # benefit from this cache; set subtree_cache_size>0 (e.g. 10000) to enable.
        if subtree_cache_size > 0:
            self._subtree_cache = LRUCache(maxsize=subtree_cache_size)
        elif subtree_cache_size == 0:
            # Disabled: get() always returns None, set() is a no-op.
            self._subtree_cache = LRUCache(maxsize=0)
        else:
            # Negative => unbounded (legacy/back-compat).
            self._subtree_cache = LRUCache(maxsize=float('inf'))

        # Min string length for embedding-based similarity (else fall back to char edit distance)
        self.min_length_for_embeddings = int(min_length_for_embeddings)

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
        """Thin wrapper. Implementation lives in :mod:`sted._matching`."""
        from ._matching import greedy_matching
        return greedy_matching(self, children1, children2, variation_type)

    def _greedy_matching_streaming(self, children1: List['JsonNode'], children2: List['JsonNode'],
                                    variation_type: str) -> float:
        """Thin wrapper. Implementation lives in :mod:`sted._matching`."""
        from ._matching import greedy_matching_streaming
        return greedy_matching_streaming(self, children1, children2, variation_type)

    def _early_prune_check(self, node1: 'JsonNode', node2: 'JsonNode') -> Optional[float]:
        """Thin wrapper. Implementation lives in :mod:`sted._matching`."""
        from ._matching import early_prune_check
        return early_prune_check(self, node1, node2)

    def _calculate_optimal_matching_cost_fast(
        self, tree1: 'JsonNode', tree2: 'JsonNode', variation_type: str = "combined"
    ) -> float:
        """Thin wrapper. Implementation lives in :mod:`sted._matching`."""
        from ._matching import calculate_optimal_matching_cost_fast
        return calculate_optimal_matching_cost_fast(self, tree1, tree2, variation_type)

    def _single_pass_combined_matching(
        self, children1: List['JsonNode'], children2: List['JsonNode'], n1: int, n2: int
    ) -> float:
        """Thin wrapper. Implementation lives in :mod:`sted._matching`."""
        from ._matching import single_pass_combined_matching
        return single_pass_combined_matching(self, children1, children2, n1, n2)

    def _get_structural_and_content_costs(
        self, tree1: 'JsonNode', tree2: 'JsonNode'
    ) -> Tuple[float, float]:
        """Thin wrapper. Implementation lives in :mod:`sted._matching`."""
        from ._matching import get_structural_and_content_costs
        return get_structural_and_content_costs(self, tree1, tree2)

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
        """Batch encode strings using SentenceTransformer.

        Body moved to ``sted.embeddings.sentence_transformers`` during the
        v0.2.0 refactor.
        """
        from .embeddings.sentence_transformers import batch_encode_sentence_transformer
        return batch_encode_sentence_transformer(self, strings, batch_size, show_progress)

    def _batch_encode_bedrock(self, strings: List[str], show_progress: bool = True,
                               max_workers: int = 10) -> None:
        """Batch encode strings using Bedrock API with parallel calls.

        Body moved to ``sted.embeddings.bedrock`` during the v0.2.0 refactor.
        """
        from .embeddings.bedrock import batch_encode_bedrock
        return batch_encode_bedrock(self, strings, show_progress, max_workers)

    def _batch_encode_bedrock_async(self, strings: List[str], show_progress: bool = True,
                                     max_concurrent: int = 50, region_name: str = None) -> None:
        """Batch encode strings using Bedrock API with async calls.

        Body moved to ``sted.embeddings.bedrock`` during the v0.2.0 refactor.
        """
        from .embeddings.bedrock import batch_encode_bedrock_async
        return batch_encode_bedrock_async(self, strings, show_progress, max_concurrent, region_name)

    def _batch_encode_bedrock_auto(
        self, strings: List[str], show_progress: bool = True,
        max_workers: int = 10, s3_bucket: str = None,
        s3_prefix: str = "bedrock-batch/embeddings", role_arn: str = None,
        batch_threshold: int = 5000
    ) -> None:
        """Auto-select the best embedding method based on string count.

        Body moved to ``sted.embeddings.bedrock`` during the v0.2.0 refactor.
        """
        from .embeddings.bedrock import batch_encode_bedrock_auto
        return batch_encode_bedrock_auto(
            self, strings, show_progress, max_workers,
            s3_bucket, s3_prefix, role_arn, batch_threshold,
        )

    def _batch_encode_bedrock_batch_inference_async(
        self, strings: List[str], show_progress: bool = True,
        s3_bucket: str = None, s3_prefix: str = "bedrock-batch/embeddings",
        role_arn: str = None, chunk_size: int = 50000
    ) -> None:
        """Batch encode using async parallel batch inference jobs.

        Body moved to ``sted.embeddings.bedrock`` during the v0.2.0 refactor.
        """
        from .embeddings.bedrock import batch_encode_bedrock_batch_inference_async
        return batch_encode_bedrock_batch_inference_async(
            self, strings, show_progress,
            s3_bucket, s3_prefix, role_arn, chunk_size,
        )

    def _batch_encode_bedrock_batch_inference(
        self, strings: List[str], show_progress: bool = True,
        s3_bucket: str = None, s3_prefix: str = "bedrock-batch/embeddings",
        role_arn: str = None, min_records: int = 100
    ) -> None:
        """Batch encode strings using true Bedrock Batch Inference (S3-based async).

        Body moved to ``sted.embeddings.bedrock`` during the v0.2.0 refactor.
        """
        from .embeddings.bedrock import batch_encode_bedrock_batch_inference
        return batch_encode_bedrock_batch_inference(
            self, strings, show_progress,
            s3_bucket, s3_prefix, role_arn, min_records,
        )

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

        For very short strings (< self.min_length_for_embeddings characters), uses
        character-level edit distance instead of embeddings, as embeddings are
        unreliable for short strings.
        """
        if text1 == text2:
            return 1.0

        # For very short strings, use character-level edit distance instead of embeddings
        # Embeddings are unreliable for very short strings.
        if (len(text1) < self.min_length_for_embeddings
                or len(text2) < self.min_length_for_embeddings):
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
        """Thin wrapper. Implementation lives in :mod:`sted._matching`."""
        from ._matching import calculate_sequential_matching_cost
        return calculate_sequential_matching_cost(self, tree1, tree2, variation_type)

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
        import warnings
        warnings.warn(
            "calculate_tree_edit_distance{,_opt} is deprecated; use calculate_tree_edit_distance_fast instead. "
            "These will be removed in v0.3.0.",
            DeprecationWarning,
            stacklevel=2,
        )
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

            # === Optimization: batch-encode all unique strings in this pair
            # before running the recursive comparison. Without this, every
            # leaf/key comparison triggers a separate single-string encode
            # call (~50ms each on MiniLM), making large/deep JSON pairs O(n)
            # encode calls. With this, encoding is O(1) batch call.
            # Only does anything when we have a SentenceTransformer (Bedrock
            # batch path is handled by user-driven precompute_embeddings).
            if self.embedding_model is not None and not getattr(self, '_skip_auto_precompute', False):
                try:
                    pair_strings: Set[str] = set()
                    self.collect_strings_from_json(json1, pair_strings)
                    self.collect_strings_from_json(json2, pair_strings)
                    new_strings = [s for s in pair_strings if s not in self._embedding_dict]
                    # Heuristic: skip the batch path for tiny pairs (overhead
                    # exceeds savings) — single-encode caching covers them.
                    if len(new_strings) >= 4:
                        self._batch_encode_sentence_transformer(
                            new_strings, batch_size=64, show_progress=False
                        )
                except Exception:
                    pass  # never block real computation on a pre-warm failure

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

            # Enforce Proposition 1 boundedness: similarity in [0, 1].
            similarity = max(0.0, min(1.0, 1.0 - distance))
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
        import warnings
        warnings.warn(
            "calculate_tree_edit_distance{,_opt} is deprecated; use calculate_tree_edit_distance_fast instead. "
            "These will be removed in v0.3.0.",
            DeprecationWarning,
            stacklevel=2,
        )
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
                # Correct normalization: ZSS unit-cost edit distance is bounded by |T1|+|T2|
                # (delete all of T1, insert all of T2), so divide by the sum to guarantee
                # similarity in [0, 1] per Proposition 1 / Eq. (3). Using max(|T1|,|T2|)
                # could produce values outside [0, 1] because update operations are counted
                # separately from insert/delete, allowing distance > max(|T1|,|T2|).
                total_nodes = tree1.count_nodes() + tree2.count_nodes()
                if total_nodes == 0:
                    return 1.0
                return max(0.0, 1.0 - distance / total_nodes)

            except TypeError as e:
                raise TypeError(
                    f"Failed to calculate tree distance. Ensure zss is properly installed "
                    f"and trees have compatible structure: {str(e)}"
                ) from e
        else:
            # Use optimized version with all improvements
            distance = self._calculate_optimal_matching_cost_fast(tree1, tree2, variation_type)

        # Enforce Proposition 1 boundedness: similarity in [0, 1].
        similarity = max(0.0, min(1.0, 1.0 - distance))
        return similarity

    def calculate_bertscore(self, json1: Dict[str, Any], json2: Dict[str, Any], **kwargs) -> float:
        """BERTScore baseline.

        Body moved to ``sted._baselines`` during the v0.2.0 refactor.
        """
        from ._baselines import calculate_bertscore
        return calculate_bertscore(self, json1, json2, **kwargs)

    def calculate_similarity_with_deepdiff(self, json1: [Dict[str, Any], List], json2: [Dict[str, Any], List], **kwargs) -> float:
        """DeepDiff-based similarity baseline.

        Body moved to ``sted._baselines`` during the v0.2.0 refactor.
        """
        from ._baselines import calculate_similarity_with_deepdiff
        return calculate_similarity_with_deepdiff(self, json1, json2, **kwargs)

    def calculate_similarity_with_deepdiff_opt(self, json1: [Dict[str, Any], List], json2: [Dict[str, Any], List], variation_type: str = "combined", structural_weight=0.5, **kwargs) -> float:
        """DeepDiff-based similarity with structural/content split.

        Body moved to ``sted._baselines`` during the v0.2.0 refactor.
        """
        from ._baselines import calculate_similarity_with_deepdiff_opt
        return calculate_similarity_with_deepdiff_opt(
            self, json1, json2, variation_type, structural_weight, **kwargs
        )

    def _calculate_deepdiff_structural_only(self, diff: dict, json1, json2) -> float:
        """Calculate structural similarity only (schema organization changes).

        Body moved to ``sted._baselines`` during the v0.2.0 refactor.
        """
        from ._baselines import _calculate_deepdiff_structural_only
        return _calculate_deepdiff_structural_only(self, diff, json1, json2)

    def _calculate_deepdiff_content_only(self, diff: dict, json1, json2) -> float:
        """Calculate content similarity only (reuse original value processing logic).

        Body moved to ``sted._baselines`` during the v0.2.0 refactor.
        """
        from ._baselines import _calculate_deepdiff_content_only
        return _calculate_deepdiff_content_only(self, diff, json1, json2)


if __name__ == "__main__":
    # Example usage
    evaluator = SemanticJsonTreeConsistencyEvaluator(
        model_id='amazon.titan-embed-text-v2:0',
    )

    json1 = {'name': 'John', 'age': 30, 'city': 'New York'}
    json2 = {'name': 'John', 'age': 30, 'location': 'NYC'}

    similarity = evaluator.calculate_tree_edit_distance_opt(json1, json2, variation_type="combined")
    print(f"Similarity: {similarity:.4f}")
