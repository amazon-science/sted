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
# - Interpretation axes:            sted._order (λ), sted._key_identity (μ),
#                                   sted._aggregation (p), sted._leaf (τ)
# - Core class:                     this file (lines below)

import os
import asyncio
import json
import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Set
from functools import lru_cache
from collections import OrderedDict
import threading
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
from .rules import numbers_within
from ._paths import EmptyContainer, Path as JsonPath, terminal_paths
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
from . import _aggregation  # noqa: E402
from . import _key_identity  # noqa: E402
from . import _leaf  # noqa: E402
from . import _order  # noqa: E402


from transformers import logging
logging.set_verbosity_error()


_PROFILE_UNSET = object()

_ST_MODELS: Dict[str, Any] = {}
_ST_LOCK = threading.Lock()


def _load_sentence_transformer(model_id: str):
    """One SentenceTransformer per model id per process, warmed up once. Evaluators that
    differ only in profile or field rules share it instead of each loading the weights."""
    with _ST_LOCK:
        model = _ST_MODELS.get(model_id)
        if model is None:
            model = SentenceTransformer(model_id)
            model.encode(["test"], show_progress_bar=False)
            _ST_MODELS[model_id] = model
        return model

STED_PROFILES = {
    # The certificate structure with a *typed* leaf: prose is read as a form
    # (embedding), literals -- numbers, coordinates, dates, amounts, identifiers,
    # code -- as tokens (exact), and a JSON number equals a numeric string of the
    # same value. Measured against the certificate profile on the 1,752 headline
    # oracle pairs and the 704-pair annotation pool (experiment sted_variant_sweep):
    # false accepts at 0.95 stay at 0, false rejects fall from 314 to ~114 (200 of
    # them were number-vs-numeric-string pairs scoring exactly 0.949), and
    # agreement with the human on structural pool tasks rises from 69% to 77%.
    "typed": {
        "cost_combination": "product",
        "array_order_lambda": 1.0,
        "key_identity_mu": 1.0,
        "aggregation_p": float("-inf"),
        "leaf_literalism_tau": None,
        "leaf_literal_exact": True,
        "numeric_string_equality": True,
        "aggregation_mode": "tree",
        "exact_name_fields": ("name",),
        "align_calls_by_name": False,
        "literal_near_miss": False,
    },
    # Ranking corner. Same typed leaf kernel, exact callee names, but the document
    # score is the harmonic mean of path-anchored soft precision (over the
    # candidate's leaf paths; a path absent from the reference scores 0) and recall
    # (over the reference's leaf paths; a path absent from the candidate scores 0).
    # Every deviation is charged in full and none is fatal, which is how the
    # annotators judge: on the structural stratum of the annotation pool this
    # decides 203 of 215 tasks and agrees with the rater on 79.8% of them, against
    # 78.0% on the 59 the bottleneck decides. It is NOT a certificate -- a high
    # score bounds nothing locally -- so it must not be thresholded.
    "ranking": {
        "cost_combination": "product",
        "array_order_lambda": 1.0,
        "key_identity_mu": 1.0,
        "aggregation_p": 1.0,
        "leaf_literalism_tau": None,
        "leaf_literal_exact": True,
        "numeric_string_equality": True,
        "aggregation_mode": "path_f1",
        "exact_name_fields": ("name",),
        "align_calls_by_name": False,
        "literal_near_miss": True,
    },
    # Ranking corner of the axis family (the paper's STED/rank): same-length sibling
    # arrays read as sequences, otherwise bags; identifier keys; product channels;
    # bottleneck aggregation; the legacy length-switched leaf kernel. Best
    # within-operator discrimination on the execution oracle; NOT a certificate.
    "rank": {
        "cost_combination": "product",
        "array_order_lambda": "auto",
        "key_identity_mu": 1.0,
        "aggregation_p": float("-inf"),
        "leaf_literalism_tau": None,
        "leaf_literal_exact": False,
        "numeric_string_equality": False,
        "aggregation_mode": "tree",
        "exact_name_fields": (),
        "align_calls_by_name": False,
        "literal_near_miss": False,
    },
    # Earlier threshold profile: bottleneck with literal character leaves.
    "certificate": {
        "cost_combination": "product",
        "array_order_lambda": 1.0,
        "key_identity_mu": 1.0,
        "aggregation_p": float("-inf"),
        "leaf_literalism_tau": 1.0,
        "leaf_literal_exact": False,
        "numeric_string_equality": False,
        "aggregation_mode": "tree",
        "exact_name_fields": (),
        "align_calls_by_name": False,
        "literal_near_miss": False,
    },
    "legacy": {
        "cost_combination": "additive",
        "array_order_lambda": None,
        "key_identity_mu": None,
        "aggregation_p": None,
        "leaf_literalism_tau": None,
        "leaf_literal_exact": False,
        "numeric_string_equality": False,
        "aggregation_mode": "tree",
        "exact_name_fields": (),
        "align_calls_by_name": False,
        "literal_near_miss": False,
    },
}


# The paper calls the path-anchored profile STED/path-F1; "ranking" is kept as an alias.
STED_PROFILES["path_f1"] = STED_PROFILES["ranking"]

# The mean readout of STED/typed: the same trees, matching, leaf kernel and callee
# identity, with children aggregated by the arithmetic mean (p=1) instead of the
# bottleneck. It grades how far two documents are apart and is what a consistency
# monitor averages; it is not a certificate (a local edit is diluted by its siblings),
# so decisions of "the same" read STED/typed itself.
STED_PROFILES["typed_mean"] = {**STED_PROFILES["typed"], "aggregation_p": 1.0}


# Literal classes whose unequal members a ranker may grade by edit distance.
_NEAR_MISS_CLASSES = frozenset({"identifier", "code", "word"})
_NEAR_MISS_MAX_LEN = 128


try:  # unit-cost Levenshtein in C when available; the pure-Python loop below is the reference
    from rapidfuzz.distance.Levenshtein import distance as _rf_levenshtein
except Exception:  # noqa: BLE001
    _rf_levenshtein = None


def _levenshtein(a: str, b: str) -> int:
    """Levenshtein edit distance (insertions, deletions, substitutions).

    Used by the short-string fallback in ``_calculate_semantic_similarity``. Before v0.2.1 that
    fallback counted per-position character mismatches plus the length difference, which is
    Hamming distance, not Levenshtein: a single insertion shifts every later character and the
    score collapses. ``("abc", "bc")`` scored 0.0 where normalised Levenshtein gives 0.667.
    Short strings are exactly where identifiers, enums and abbreviations live, so the fallback
    has to handle shifts.
    """
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    if _rf_levenshtein is not None:            # C implementation, same integer
        return int(_rf_levenshtein(a, b))
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1,          # deletion
                           cur[j - 1] + 1,       # insertion
                           prev[j - 1] + (ca != cb)))  # substitution
        prev = cur
    return prev[-1]


class _Whole:
    """An unordered array reported as one field in calculate_field_differences."""
    __slots__ = ("value",)

    def __init__(self, value):
        self.value = value

    def __eq__(self, other):
        return isinstance(other, _Whole) and json.dumps(self.value, sort_keys=True, default=str) == \
            json.dumps(other.value, sort_keys=True, default=str)


def format_json_path(path) -> str:
    """Typed path -> readable JSONPath-like string: ``arguments.days``, ``[0].name``,
    ``["a.b"]`` for keys that are not plain identifiers, ``$`` for the root."""
    parts = []
    for kind, step in path:
        if kind == "index":
            parts.append(f"[{step}]")
        elif isinstance(step, str) and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_-]*", step):
            parts.append(("." if parts else "") + step)
        else:
            parts.append(f"[{json.dumps(step)}]")
    return "".join(parts) or "$"


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
                 profile: str = "typed",
                 cost_combination=None,
                 sort_keys: bool = True,
                 sort_arrays: bool = True,
                 array_order_lambda=_PROFILE_UNSET,
                 array_order_lambda_default: float = 0.0,
                 key_identity_mu=_PROFILE_UNSET,
                 aggregation_p=_PROFILE_UNSET,
                 leaf_literalism_tau=_PROFILE_UNSET,
                 leaf_literal_exact=_PROFILE_UNSET,
                 numeric_string_equality=_PROFILE_UNSET,
                 literal_near_miss=_PROFILE_UNSET,
                 aggregation_mode=_PROFILE_UNSET,
                 exact_name_fields=_PROFILE_UNSET,
                 align_calls_by_name=_PROFILE_UNSET,
                 field_rules=None,
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
            profile: Named interpretation profile. ``"certificate"`` (default) uses positional
                arrays, identifier keys, product channel composition, bottleneck child
                aggregation, and literal string leaves. ``"legacy"`` reproduces the published
                mean-pooled configuration. Explicit axis arguments override the profile.
            cost_combination: How the two channels combine. If omitted, selected by ``profile``.
                ``'additive'`` is Eq.(2); ``'product'`` conjoins the channels so a fully wrong
                value costs 1.0 regardless of key similarity.
            array_order_lambda: How much of an array's cost comes from the *sequence* reading
                rather than the *bag* reading. If omitted, selected by ``profile``. ``None``
                explicitly disables the machinery and reproduces every published number.
                Otherwise: a float in [0, 1]
                (0 = bag, i.e. the Hungarian assignment STED has always used; 1 = sequence,
                i.e. a monotone alignment), the string ``'auto'`` for the arity-coupling
                estimate, or a callable ``(node1, node2) -> float``. Because the two costs
                bracket each other (``c_free <= c_ord``, see ``sted._order``), λ only ever
                interpolates between two computable extremes — use
                ``compute_similarity_interval`` to get both instead of picking one.
            array_order_lambda_default: λ for array pairs where ``'auto'`` (or a callable
                returning None-like) has no signal. Default 0.0, i.e. fall back to the
                bag reading.
            key_identity_mu: How much of a field name's cost comes from the *identifier*
                reading (a key is a name some consumer declared, so ``n`` != ``n_value``)
                rather than the *label* reading (a key describes its value, so the two are
                near-synonyms). If omitted, selected by ``profile``. ``None`` disables it and
                reproduces every published number; 1.0 is equivalent to
                ``exact_match_all_keys``. The two
                readings bracket each other (see ``sted._key_identity``), so prefer
                ``compute_key_identity_interval`` over picking one. There is no ``'auto'``:
                whether keys are declared elsewhere is a fact about the deployment, not
                about the pair.
            aggregation_p: How a node's matched children combine into the node's score --
                the exponent of the power mean over child *similarities*. If omitted, selected
                by ``profile``. ``None`` and 1.0 are the arithmetic mean, i.e. every published
                number; as p falls
                towards -inf the mean approaches the minimum, so one bad child can no longer
                be averaged away by many good ones. This is the axis that fixes cost
                dilution (``sted._aggregation``): today a single meaning-destroying leaf in
                an n-child node costs only 1/n of the score, so similarity tends to 1 as the
                document grows. Must be <= 1. ``-inf`` (or ``'min'``) is the exact minimum,
                solved by a bottleneck assignment instead of an LAP; it is the only setting
                whose score does not depend on any node's arity, and therefore the only one on
                which ``score >= theta`` certifies every matched pair. Use
                ``compute_aggregation_interval`` for both ends.
            leaf_literalism_tau: How much of a leaf *value*'s similarity comes from the
                *token* reading (the value is interpreted literally, so ``'lambda x: ...'``
                != ``'LAMBDA X: ...'``) rather than the *form* reading (surface details like
                case are semantically inert, so ``'male'`` == ``'MALE'``). If omitted, selected
                by ``profile``. ``None`` reproduces every published number, and is **not** an
                endpoint: published STED
                is the token reading below ``min_length_for_embeddings`` characters and the
                form reading at or above it. Unlike μ the two ends are **not** ordered, so
                there is no bracketing lemma; see ``sted._leaf``. What ``tau = 1`` buys is
                that a leaf threshold becomes checkable -- ``k >= theta`` bounds the
                Levenshtein distance by ``(1 - theta) * max(|u|, |v|)`` -- which is what makes
                ``aggregation_p = -inf``'s certificate say anything about the values. Numbers
                are unaffected: their kernel is already exact.
            leaf_literal_exact: When True, a pair of string leaves that are both *literals*
                -- numbers, coordinate/number lists, dates and times, amounts, identifiers,
                code (``sted._leaf.literal_class``) -- is scored by an exact kernel instead of
                the string kernel: numeric equality where the strings parse as numbers,
                verbatim equality otherwise. Reason: for these values one edit is not "a
                small change"; ``"40.3772,-105.5217"`` against ``"40.3772,105.5217"`` is a
                different hemisphere and the string kernel prices it at 1/17, and two
                different lambda polynomials score 0.96 under the uncased encoder. Set by
                ``profile`` (on in ``"typed"``, off in ``"certificate"``/``"legacy"``).
            aggregation_mode: ``"tree"`` (the level-wise tree edit score, every profile
                until now) or ``"path_f1"`` (the ranking corner): leaves are keyed by their
                full path -- positional array indices, exact keys -- and the score is the
                harmonic mean of soft precision over the candidate's paths and soft recall
                over the reference's paths, each leaf scored with the profile's leaf kernel
                and an unmatched path scoring 0. No level means, no bottleneck, no partial
                credit for a wrong subtree whose argument keys happen to overlap.
            exact_name_fields: Leaf keys whose string value is an identity, not a text --
                a callee name -- and is therefore compared exactly whatever tau says.
            align_calls_by_name: In path_f1 mode, pair the elements of a list of named
                objects (tool calls) by exact name before scoring, so a reordering of
                the same calls scores 1. Off everywhere, including the ranking profile:
                measured on the annotation pool it changed 24 of rater A's verdicts, 4 for
                and 20 against -- the annotator reads the order of calls as part of the
                meaning (a candidate in the reference's order with one wrong value beat a
                perfect reordering). Kept as an option so that result is reproducible.
            field_rules: Optional ``sted.rules.FieldRules`` (ignore / unordered / tolerance
                by path pattern). Ignored fields are removed from both documents before any
                comparison; arrays matching an ``unordered`` pattern are compared as
                multisets (lambda = 0) while the others keep the profile's lambda; numbers
                matching a ``tolerance`` pattern are equal within that tolerance.
            literal_near_miss: When True (ranking profile), two unequal literal strings are
                scored by a graded near-miss kernel -- 0.95 for a case-only difference,
                otherwise 0.9 x (1 - edit distance / length) -- instead of 0. A ranker needs
                to order wrong literals; a certificate does not, so typed/certificate keep 0.
            numeric_string_equality: When True, a JSON number against a string that denotes
                a number is compared by value (1 if equal up to float noise, else 0) instead
                of "string kernel minus a type penalty". The penalty of 0.05 put every
                ``20`` vs ``"20"`` pair at exactly 0.949 -- 200 sandbox-equivalent oracle
                pairs rejected at the 0.95 threshold by a constant. Set by ``profile``.
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
        if profile not in STED_PROFILES:
            raise ValueError(
                f"profile must be one of {sorted(STED_PROFILES)}, got {profile!r}")
        profile_values = STED_PROFILES[profile]
        if cost_combination is None:
            cost_combination = profile_values["cost_combination"]
        if array_order_lambda is _PROFILE_UNSET:
            array_order_lambda = profile_values["array_order_lambda"]
        if key_identity_mu is _PROFILE_UNSET:
            key_identity_mu = profile_values["key_identity_mu"]
        if aggregation_p is _PROFILE_UNSET:
            aggregation_p = profile_values["aggregation_p"]
        if leaf_literalism_tau is _PROFILE_UNSET:
            leaf_literalism_tau = profile_values["leaf_literalism_tau"]
        if leaf_literal_exact is _PROFILE_UNSET:
            leaf_literal_exact = profile_values["leaf_literal_exact"]
        if numeric_string_equality is _PROFILE_UNSET:
            numeric_string_equality = profile_values["numeric_string_equality"]
        if literal_near_miss is _PROFILE_UNSET:
            literal_near_miss = profile_values.get("literal_near_miss", False)
        if aggregation_mode is _PROFILE_UNSET:
            aggregation_mode = profile_values["aggregation_mode"]
        if exact_name_fields is _PROFILE_UNSET:
            exact_name_fields = profile_values["exact_name_fields"]
        if align_calls_by_name is _PROFILE_UNSET:
            align_calls_by_name = profile_values["align_calls_by_name"]
        if aggregation_mode not in ("tree", "path_f1"):
            raise ValueError(f"aggregation_mode must be 'tree' or 'path_f1', got {aggregation_mode!r}")

        self.profile = profile
        self.path_weight_decay = path_weight_decay
        self.type_change_cost = type_change_cost or _get_default_type_change_costs()
        self.required_fields = required_fields or set()
        self.order_sensitive_fields = order_sensitive_fields or set()
        self.exact_match_fields = exact_match_fields or set()
        self.exact_match_all_keys = exact_match_all_keys
        # Via the property, which validates and invalidates the subtree cache on change. The
        # setter tolerates running before that cache exists.
        self.key_identity_mu = key_identity_mu
        self.aggregation_p = aggregation_p
        self.leaf_literalism_tau = leaf_literalism_tau
        self.leaf_literal_exact = bool(leaf_literal_exact)
        self.numeric_string_equality = bool(numeric_string_equality)
        self.literal_near_miss = bool(literal_near_miss)
        self.aggregation_mode = aggregation_mode
        self.exact_name_fields = frozenset(exact_name_fields or ())
        self.align_calls_by_name = bool(align_calls_by_name)

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
            self.embedding_model = _load_sentence_transformer(self.model_id)

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
        # See combine_channels(). The certificate profile uses the product t-norm; the
        # published additive Eq.(2) remains available through profile='legacy'.
        self.cost_combination = cost_combination

        self.sort_keys = sort_keys
        # Assign array_order_lambda BEFORE sort_arrays: the sort_arrays property below reads it.
        from .rules import FieldRules
        self.field_rules = field_rules if field_rules is not None else FieldRules()
        if self.field_rules.unordered and array_order_lambda is not None:
            base_lambda, rules_ = array_order_lambda, self.field_rules

            def _lambda_with_rules(node1, node2, _base=base_lambda, _rules=rules_):
                if _rules.is_unordered(node1.path) or _rules.is_unordered(node2.path):
                    return 0.0
                return _order.resolve_lambda(_base, node1, node2, self.array_order_lambda_default)
            array_order_lambda = _lambda_with_rules
        self.array_order_lambda = array_order_lambda
        self.array_order_lambda_default = float(array_order_lambda_default)
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

    # === Array order ===

    @property
    def sort_arrays(self) -> bool:
        """Whether ``from_dict`` canonicalises array element order.

        A property rather than a plain attribute because the order blend is unable to see
        any order that ``from_dict`` has already destroyed: with ``sort_arrays=True`` a
        permuted array becomes the identical tree, so the monotone alignment would be handed
        two already-aligned sequences and ``c_ord`` would equal ``c_free`` on every pair.
        Requesting a blend therefore implies unsorted arrays, and this property enforces it
        rather than leaving the caller to remember.

        This does not change the bag reading. Every cost function strips array indices out of
        paths, so the assignment's cost matrix under a permutation is the same matrix with its
        rows permuted and the optimal total is unchanged — λ=0 on unsorted arrays reproduces
        the published score. ``tests/test_array_order.py`` pins that.
        """
        if getattr(self, 'array_order_lambda', None) is not None:
            return False
        return self._sort_arrays

    @sort_arrays.setter
    def sort_arrays(self, value: bool) -> None:
        self._sort_arrays = bool(value)

    def compute_similarity_interval(self, json1: Dict[str, Any], json2: Dict[str, Any],
                                    variation_type: str = "combined") -> Tuple[float, float]:
        """``(s_ordered, s_free)`` — the similarity under both array-order readings.

        ``s_free`` (λ=0) treats every array as a bag and is what STED reports today;
        ``s_ordered`` (λ=1) treats every array as a sequence. By the bracketing lemma in
        ``sted._order`` the true similarity under *any* per-array mixture of the two readings
        lies in ``[s_ordered, s_free]``, so the width is exactly the part of the answer that
        rests on an assumption this metric cannot verify without executing the call.

        A pure permutation of distinct elements gives ``s_free == 1.0`` and
        ``s_ordered < 1.0``: instead of an unfalsifiable perfect score, the metric reports
        that its verdict is order-dependent.
        """
        original = self.array_order_lambda
        try:
            self.array_order_lambda = 1.0
            s_ord = self.calculate_tree_edit_distance_fast(json1, json2, variation_type)
            self.array_order_lambda = 0.0
            s_free = self.calculate_tree_edit_distance_fast(json1, json2, variation_type)
        finally:
            self.array_order_lambda = original
        # Clamp only the lemma's direction, and only against float noise: the two costs are
        # minima over a set and a subset of it, so s_ord <= s_free exactly in exact
        # arithmetic. If they ever cross by more than rounding, something is wrong upstream.
        if s_ord > s_free:
            if s_ord - s_free > 1e-9:
                raise AssertionError(
                    f"bracketing violated: s_ordered={s_ord!r} > s_free={s_free!r}")
            s_ord = s_free
        return s_ord, s_free

    # === Key identity ===

    @property
    def key_identity_mu(self):
        """See ``sted._key_identity``. ``None`` means published behaviour (label reading)."""
        return self._key_identity_mu

    @key_identity_mu.setter
    def key_identity_mu(self, value) -> None:
        # Validate eagerly so a typo surfaces at configuration time rather than as a silently
        # wrong score thousands of node comparisons later.
        _key_identity.resolve_mu(value)
        previous = getattr(self, '_key_identity_mu', 'unset')
        self._key_identity_mu = value
        # Unlike λ, μ is a single evaluator-wide constant, so it is not part of the subtree
        # cache key — which is exactly why *changing* it has to invalidate the cache: the
        # stored costs were computed under the old reading and the key cannot tell them apart.
        if previous != 'unset' and previous != value:
            if getattr(self, '_subtree_cache', None) is not None:
                self.clear_subtree_cache()

    def compute_key_identity_interval(self, json1: Dict[str, Any], json2: Dict[str, Any],
                                      variation_type: str = "combined") -> Tuple[float, float]:
        """``(s_identifier, s_label)`` — the similarity under both readings of a JSON key.

        ``s_label`` (μ=0) scores field names with the embedding kernel and is what STED
        reports today; ``s_identifier`` (μ=1) requires them to match exactly. By the
        bracketing lemma in ``sted._key_identity`` every intermediate reading lies between,
        so the width is the part of the answer that rests on keys being descriptions rather
        than declared names — a question about the deployment, not about the two documents.

        A consistent renaming of every key gives a high ``s_label`` and a much lower
        ``s_identifier``: instead of certifying a call that will not execute, the metric
        reports that its verdict depends on whether anything downstream declared those keys.
        """
        original = self.key_identity_mu
        try:
            self.key_identity_mu = 1.0
            s_ident = self.calculate_tree_edit_distance_fast(json1, json2, variation_type)
            self.key_identity_mu = 0.0
            s_label = self.calculate_tree_edit_distance_fast(json1, json2, variation_type)
        finally:
            self.key_identity_mu = original
        if s_ident > s_label:
            if s_ident - s_label > 1e-9:
                raise AssertionError(
                    f"bracketing violated: s_identifier={s_ident!r} > s_label={s_label!r}")
            s_ident = s_label
        return s_ident, s_label

    # === Child aggregation ===

    @property
    def aggregation_p(self):
        """See ``sted._aggregation``. ``None`` means published behaviour (arithmetic mean)."""
        return self._aggregation_p

    @aggregation_p.setter
    def aggregation_p(self, value) -> None:
        _aggregation.resolve_p(value)
        previous = getattr(self, '_aggregation_p', 'unset')
        self._aggregation_p = value
        # Same hazard and same remedy as μ: p is evaluator-wide and therefore absent from the
        # subtree cache key, so a change has to invalidate the entries computed under the old
        # exponent. Under 'additive' it also changes which correspondence is chosen, so the
        # stale entries would be wrong in two independent ways.
        if previous != 'unset' and previous != value:
            if getattr(self, '_subtree_cache', None) is not None:
                self.clear_subtree_cache()

    def compute_aggregation_interval(self, json1: Dict[str, Any], json2: Dict[str, Any],
                                     variation_type: str = "combined",
                                     p: float = float('-inf')) -> Tuple[float, float]:
        """``(s_conjunctive, s_aggregate)`` — the two readings of what an object *is*.

        ``s_aggregate`` (p=1) averages a node's children, which is what STED reports today
        and which lets many correct fields average away one destroyed field: a single
        meaning-changing leaf among ``n`` children costs exactly ``1/n`` of the level.
        ``s_conjunctive`` uses a strongly negative ``p``, approaching the minimum, where the
        worst child dominates. By the power-mean lemma in ``sted._aggregation`` the score is
        non-decreasing in ``p``, so these bracket every intermediate reading.

        ``p`` defaults to the exact minimum (``-inf``, i.e. the bottleneck assignment), which
        is the only reading whose score is independent of any node's arity and therefore the
        only one a threshold can be applied to soundly — see the soundness theorem in
        ``sted._aggregation``. Passing a finite ``p`` instead approaches that end from above and
        makes the reported width a *lower* bound on the true one, which is cheaper (a plain LAP
        rather than a binary search over matchings) but no longer size-free.
        """
        original = self.aggregation_p
        try:
            self.aggregation_p = _aggregation.resolve_p(p)
            s_conj = self.calculate_tree_edit_distance_fast(json1, json2, variation_type)
            self.aggregation_p = 1.0
            s_agg = self.calculate_tree_edit_distance_fast(json1, json2, variation_type)
        finally:
            self.aggregation_p = original
        if s_conj > s_agg:
            if s_conj - s_agg > 1e-9:
                raise AssertionError(
                    f"bracketing violated: s_conjunctive={s_conj!r} > s_aggregate={s_agg!r}")
            s_conj = s_agg
        return s_conj, s_agg

    # === Leaf literalism ===

    @property
    def leaf_literalism_tau(self):
        """See ``sted._leaf``. ``None`` means published behaviour (the length switch)."""
        return self._leaf_literalism_tau

    @leaf_literalism_tau.setter
    def leaf_literalism_tau(self, value) -> None:
        _leaf.resolve_tau(value)
        previous = getattr(self, '_leaf_literalism_tau', 'unset')
        self._leaf_literalism_tau = value
        if previous != 'unset' and previous != value:
            # Two caches, not one. τ is evaluator-wide and so absent from the subtree cache key,
            # the same hazard μ and p have. It is *also* absent from ``_cache``'s key, which is
            # only ``(s1, s2)`` — and unlike μ and p, τ changes exactly the number that cache
            # stores. Leaving it would silently score the new reading with the old kernel.
            if getattr(self, '_subtree_cache', None) is not None:
                self.clear_subtree_cache()
            if getattr(self, '_cache', None) is not None:
                self._cache.clear()

    def compute_leaf_literalism_interval(self, json1: Dict[str, Any], json2: Dict[str, Any],
                                         variation_type: str = "combined") -> Tuple[float, float]:
        """``(s_token, s_form)`` — the similarity under both readings of a leaf *value*.

        ``s_form`` (τ=0) scores every string leaf with the embedding cosine, so case,
        whitespace and other surface differences are nearly free; ``s_token`` (τ=1) scores them
        with length-normalised edit distance, so they are not. Published STED is at neither end:
        it is τ=1 below ``min_length_for_embeddings`` characters and τ=0 at or above it.

        **This pair is not ordered**, unlike the other three axes' intervals. The two kernels
        cross — on Instrument J's 360 string records the character kernel exceeds the embedding
        kernel on 28 of them — so there is no bracketing lemma and this method deliberately does
        not assert one. Take ``min``/``max`` of the returned pair for the interval, and read the
        *signed* difference as which reading is more permissive for this particular document.

        The reason to want τ=1 despite it being the worse classifier (AUC 0.7936 against 0.9823)
        is that a threshold on it is checkable: see the edit-radius certificate in ``sted._leaf``.
        """
        original = self.leaf_literalism_tau
        try:
            self.leaf_literalism_tau = 1.0
            s_token = self.calculate_tree_edit_distance_fast(json1, json2, variation_type)
            self.leaf_literalism_tau = 0.0
            s_form = self.calculate_tree_edit_distance_fast(json1, json2, variation_type)
        finally:
            self.leaf_literalism_tau = original
        return s_token, s_form

    def compute_assumption_interval(self, json1: Dict[str, Any], json2: Dict[str, Any],
                                    variation_type: str = "combined",
                                    p: float = float('-inf')) -> Tuple[float, float]:
        """``(s_min, s_max)`` over all three interpretation axes at once.

        A similarity metric on JSON must decide three things the documents themselves do not
        settle: whether an array is a sequence or a bag (λ, ``sted._order``), whether a key
        is an identifier or a label (μ, ``sted._key_identity``), and whether a node's
        children are a conjunction of requirements or an aggregate of evidence
        (p, ``sted._aggregation``).

        Each is monotone on its own, and the score is non-decreasing in ``p`` and
        non-increasing in λ and μ, so the extremes over *all* combinations of readings are the
        two corners ``(λ, μ, p) = (1, 1, p_conj)`` and ``(0, 0, 1)``. λ and μ additionally act
        on disjoint terms of Eq. (1) — λ on array child matching, μ on the field-name half of
        the structural channel — and ``p`` acts on the aggregation of those terms rather than
        on the terms, so no axis can cancel another.

        The width is what STED cannot know from the pair alone. Reporting it is the honest
        alternative to picking a convention and presenting the result as a fact.

        τ (``sted._leaf``) is a fourth such decision and is deliberately **not** included. The
        corner argument above needs each axis to be monotone in a known direction; τ is not, so
        adding it would turn a proved interval into an unproved one. Use
        ``compute_leaf_literalism_interval`` separately and take ``min``/``max`` across the two.
        """
        lam0, mu0, p0 = self.array_order_lambda, self.key_identity_mu, self.aggregation_p
        try:
            self.array_order_lambda, self.key_identity_mu = 1.0, 1.0
            self.aggregation_p = _aggregation.resolve_p(p)
            s_min = self.calculate_tree_edit_distance_fast(json1, json2, variation_type)
            self.array_order_lambda, self.key_identity_mu = 0.0, 0.0
            self.aggregation_p = 1.0
            s_max = self.calculate_tree_edit_distance_fast(json1, json2, variation_type)
        finally:
            self.array_order_lambda, self.key_identity_mu = lam0, mu0
            self.aggregation_p = p0
        if s_min > s_max:
            if s_min - s_max > 1e-9:
                raise AssertionError(f"bracketing violated: {s_min!r} > {s_max!r}")
            s_min = s_max
        return s_min, s_max

    # === Optimization Helper Methods ===

    def _compute_tree_hash(self, node: 'JsonNode') -> str:
        """
        Compute a hash for a tree node for memoization with lazy caching.

        The hash captures the structural and content signature of the subtree.
        Uses node id() for caching to avoid recomputation.

        Two properties this hash must have, because it is the key of
        ``_subtree_cache`` and a collision silently returns the cost of a
        *different* pair:

        1. It is sensitive to child order. Child hashes used to be ``sorted()``
           before being folded in, which made the hash order-invariant. Arrays
           whose field is in ``order_sensitive_fields`` are deliberately left
           unsorted by ``JsonNode.from_dict``, so a permutation of such an
           array used to hash the same as the original and hit its cache entry:
           with ``subtree_cache_size > 0`` a fully permuted tool-call sequence
           scored 0.0 distance where the cold computation gives 0.1187. Every
           other array is canonically sorted at tree-construction time, so
           preserving order here costs nothing and is what makes the key honest.
        2. It reads the whole leaf value. Values used to be truncated to 100
           characters, so two long strings sharing a prefix collided — and long
           strings are exactly the ones the chunking path exists for.

        Parts are length-prefixed before joining so that no arrangement of
        values and child hashes can be re-parsed as another.
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
                # For leaf nodes, include the full value: a prefix is not a signature.
                parts.append(str(n.value))

            # Include children hashes in order (see property 1 above)
            if n.children:
                parts.extend(_hash_node(c) for c in n.children)

            payload = '|'.join(f'{len(p)}:{p}' for p in parts)
            result = hashlib.md5(payload.encode()).hexdigest()[:16]
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

        # Batch compute similarities. Must use the τ-aware kernel, not the published one: this
        # writes into the same cache ``_compare_strings`` reads, so a mismatch here would let a
        # precomputed entry silently override the configured reading.
        for s1, s2 in uncached_pairs:
            sim = self._leaf_value_similarity(s1, s2)
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
            max_len = max(len(text1), len(text2))
            if max_len == 0:
                return 1.0

            edit_distance = _levenshtein(text1, text2)

            # Normalize to [0, 1] and invert (0 = different, 1 = same)
            similarity = 1.0 - (edit_distance / max_len)
            return max(0.0, similarity)

        return self._embedding_cosine(text1, text2)

    def _embedding_cosine(self, text1: str, text2: str) -> float:
        """The embedding branch of ``_calculate_semantic_similarity``, on its own.

        Extracted so ``sted._leaf``'s τ axis can ask for the *form* reading of a leaf value
        without the length switch deciding for it. Callers must handle ``text1 == text2``
        themselves; this function does not shortcut it.
        """
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

        # Truncate at zero rather than affinely rescaling [-1, 1] onto [0, 1].
        #
        # `(cos + 1) / 2` maps the *theoretical* cosine range onto [0, 1], but sentence-embedding
        # cosines for natural text are effectively non-negative: unrelated pairs sit near 0.0-0.1,
        # so the rescaling compressed the whole usable range into [0.47, 1.0] and priced "completely
        # unrelated" at ~0.5 similar. That is a different scale from every other leaf kernel in the
        # same tree — the short-string branch of this very function returns
        # `max(0, 1 - d/L)`, and numbers, booleans and `exact_match_fields` are 0/1 — so the same
        # amount of disagreement was priced 2.1x lower for a long string than for a number (probe
        # P16: 0.235 vs 0.5). Eq. (1) then mixes those incommensurable units.
        #
        # Truncation is the standard convention for embedding cosines and introduces no parameter.
        similarity = max(0.0, float(similarity))

        return similarity

    def _leaf_value_similarity(self, text1: str, text2: str) -> float:
        """The leaf **value** kernel, at whatever point on the τ axis is configured.

        ``τ = None`` (the default) delegates to ``_calculate_semantic_similarity`` unchanged, so
        every published number is reproduced bit-for-bit. Any other τ replaces that function's
        length switch with the explicit blend in ``sted._leaf``.

        Field names deliberately do **not** come through here — they are μ's axis, and a key is a
        name rather than a datum. See ``_calculate_structural_similarity``.
        """
        tau = _leaf.resolve_tau(self.leaf_literalism_tau)
        if tau is None:
            return self._calculate_semantic_similarity(text1, text2)
        if text1 == text2:
            return 1.0
        char_sim = _leaf.char_kernel(text1, text2)
        if tau >= 1.0:
            # Skip the cosine entirely: at the token end it carries no weight, so consulting it
            # could only introduce a difference. The batch pre-warm still encodes these strings,
            # because field names are μ's axis and go on using embeddings.
            return char_sim
        return _leaf.blend(self._embedding_cosine(text1, text2), char_sim, tau)

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
        # For the ranker a case-only difference ("MID" / "mid", "News" / "news") is a near
        # miss on any string: the short-string edit kernel would otherwise score "MID" vs
        # "mid" as three substitutions, below an unrelated word. Certificates keep charging it.
        if self.literal_near_miss and str1.strip().casefold() == str2.strip().casefold():
            return 0.95
        # Literal values (numbers, coordinates, dates, ids) are tokens a consumer parses;
        # one edit is a different value, so the string kernel's 1 - d/L is the wrong ruler.
        if self.leaf_literal_exact and _leaf.literal_class(str1) and _leaf.literal_class(str2):
            if _leaf.literal_equal(str1, str2):
                return 1.0
            if not self.literal_near_miss:
                return 0.0
            # A ranker still has to order two wrong literals: "b12345" against "B12345"
            # should beat "my-bot-id". For token-like literals (identifiers, codes, words)
            # a case-only difference scores 0.95 and anything else the character edit
            # ratio scaled below it, so no near-miss reaches equality. Numeric literals
            # (numbers, coordinate lists, amounts) use the relative-error kernel in
            # _leaf.numeric_near_miss; dates stay 0. The certificate profiles keep the exact kernel
            # for every class. On BFCL-500 the exact kernel tied or inverted 23 triples
            # whose accepted alternative differed from the reference only in case.
            u_num, v_num = _leaf.numeric_values(str1), _leaf.numeric_values(str2)
            if u_num is not None and v_num is not None:
                return _leaf.numeric_near_miss(u_num, v_num)
            if _leaf.literal_class(str1) in _NEAR_MISS_CLASSES and _leaf.literal_class(str2) in _NEAR_MISS_CLASSES:
                u, v = str1.strip(), str2.strip()
                if len(u) <= _NEAR_MISS_MAX_LEN and len(v) <= _NEAR_MISS_MAX_LEN:
                    # Token-sized literals only: the code class also catches whole source
                    # files, and the quadratic edit distance on two of those held a benchmark
                    # worker for half an hour. A long code literal that differs is a miss.
                    return 0.9 * (1.0 - _levenshtein(u, v) / max(len(u), len(v), 1))
            return 0.0

        # Check cache first
        cached_sim = self._cache.get(str1, str2)
        if cached_sim is not None:
            return cached_sim

        if len(str1) < self.chunk_size and len(str2) < self.chunk_size:
            if method == "bertscore":
                P, R, F1 = bert_score([str1], [str2], lang="en")
                sim = float(F1.item())
            else:
                sim = self._leaf_value_similarity(str1, str2)
            self._cache.set(str1, str2, sim)
            return sim
        else:
            sim = self._long_text_similarity(str1, str2)
            self._cache.set(str1, str2, sim)
            return sim

    def _long_text_similarity(self, str1: str, str2: str) -> float:
        """Similarity of two strings too long for one encoder pass.

        Both are split into chunks and each chunk is matched to its best counterpart
        on the other side; the score is the symmetric mean of those best matches
        (chunk-level BERTScore F, with the leaf kernel -- so tau still decides
        whether a chunk pair is read as a form or as a token). Order of chunks plays
        no part and the two texts may have any number of chunks.

        This replaces a call to ``_compare_arrays_unordered`` on the chunk lists,
        which inherited the profile's *array* semantics: under the certificate
        (positional arrays, bottleneck aggregation) any two texts with different
        chunk counts scored exactly 0, and equal-count texts scored their worst
        aligned chunk. Measured on the annotation pool: four long `thought` fields
        with whole-text cosine 0.71-0.86 scored 0.000; 423 of 704 tasks have a
        differing leaf over the chunk size. The oracle corpus has no string that
        long, so its numbers are unaffected.
        """
        chunks1 = self._split_natural_text(str1)
        chunks2 = self._split_natural_text(str2)
        if not chunks1 or not chunks2:
            return 0.0
        sims = [[self._leaf_value_similarity(a, b) for b in chunks2] for a in chunks1]
        best1 = sum(max(row) for row in sims) / len(chunks1)
        best2 = sum(max(sims[i][j] for i in range(len(chunks1))) for j in range(len(chunks2))) / len(chunks2)
        return max(0.0, min(1.0, 0.5 * (best1 + best2)))

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

    def _numeric_string_match(self, a: Any, b: Any) -> float:
        """1.0 when one side is a number and the other a string denoting the same
        number (and numeric_string_equality is on); 0.0 otherwise."""
        if not self.numeric_string_equality:
            return 0.0
        if isinstance(a, bool) or isinstance(b, bool):
            return 0.0
        if not ((isinstance(a, (int, float)) and isinstance(b, str)) or (isinstance(b, (int, float)) and isinstance(a, str))):
            return 0.0
        x, y = _leaf.numeric_value(a), _leaf.numeric_value(b)
        if x is None or y is None:
            return 0.0
        if _leaf.numbers_close(x, y):
            return 1.0
        return _leaf.numeric_near_miss([x], [y]) if self.literal_near_miss else 0.0

    def _numeric_tolerance(self, node1: JsonNode, node2: JsonNode):
        """The field rule's (abs, rel) tolerance for this pair of number nodes, or None."""
        if not self.field_rules.tolerance:
            return None
        return self.field_rules.tolerance_for(node1.path) or self.field_rules.tolerance_for(node2.path)

    def _compare_numbers(self, num1: float, num2: float) -> float:
        """Exact numeric kernel, up to float noise.

        Equal within a relative 1e-9 (absolute 1e-12 near zero) scores 1, anything
        else 0 -- the kernel is binary, so 20 vs 21 costs the same as 20 vs 21000.
        The tolerance only absorbs representation noise such as 0.1 + 0.2 vs 0.3;
        no pair in the annotation pool or the execution-oracle corpus differs by
        that little, so it changes no published number.
        """
        if _leaf.numbers_close(num1, num2):
            return 1
        if self.literal_near_miss:
            # ranking only: a wrong number that is nearly right ranks above one that is
            # not (annotator A: 9 of 11 numeric-only pairs), see _leaf.numeric_near_miss
            return _leaf.numeric_near_miss([num1], [num2])
        return 0

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
        return self.combine_channels(structural_cost, content_cost)

    def combine_channels(self, structural_cost: float, content_cost: float) -> float:
        """Combine the structural and content channels into one update cost.

        ``'additive'`` is paper Eq. (2), γ_upd = w · γ_struct + (1-w) · γ_content, and is the
        default so every published number stays reproducible.

        ``'product'`` is a weighted geometric mean of the *similarities*,

            γ_upd = 1 - (1 - γ_struct)^w · (1 - γ_content)^(1-w)

        which is a t-norm rather than a weighted sum. The reason for it is a measured defect, not
        elegance. ``delete_cost`` is a hard 1.0, while under the additive form a substitution costs
        1 - (something strictly positive) for *any* pair of keys, because two arbitrary key names
        always have nonzero embedding similarity. So substitution always undercuts the deletion it
        replaces, and STED prefers hallucinating a field to omitting one -- measured at 29.06% of
        695 argument insertions in docs/analysis/known_order_validation.md §3, with the arithmetic
        closed in results/rebuttal/hallucination_vs_omission.json (content channel 1.0, structural
        0.898-0.945, combined 0.922-0.972 < 1.0).

        A conjunction removes it by construction: if either channel is fully wrong the cost is
        exactly 1.0, so key-name similarity can only ever modulate credit that the content channel
        actually earned. It cannot manufacture credit on its own. The two forms agree exactly at
        w = 0 and w = 1, so the endpoints are unchanged.
        """
        w = self.structural_weight
        if self.cost_combination == 'additive':
            return w * structural_cost + (1 - w) * content_cost
        if self.cost_combination == 'product':
            s_sim = max(0.0, 1.0 - structural_cost)
            c_sim = max(0.0, 1.0 - content_cost)
            return 1.0 - (s_sim ** w) * (c_sim ** (1.0 - w))
        raise ValueError(
            f"cost_combination must be 'additive' or 'product', got {self.cost_combination!r}")

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
            # μ blends in the identifier reading of a key. μ=0 (the default) leaves the line
            # above untouched; μ=1 is exact_match_all_keys. Note the exact term compares the
            # *raw* keys, not the normalized ones: normalization exists to help the embedding
            # kernel, and applying it here would make `a_b` and `a-b` identical identifiers.
            mu = _key_identity.resolve_mu(self.key_identity_mu)
            if mu > 0.0:
                field_similarity = _key_identity.field_similarity(
                    field_similarity, 1.0 if key1 == key2 else 0.0, mu)

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
                    tol = self._numeric_tolerance(node1, node2)
                    if tol is not None:
                        return 1.0 if numbers_within(float(node1.value), float(node2.value), tol) else 0.0
                    return self._compare_numbers(float(node1.value), float(node2.value))
                elif node1.node_type == "array":
                    return 1 - self._compare_arrays_unordered(list(node1.value), list(node2.value), node1.label, node2.label, variation_type="content")
                elif node1.node_type == "object":
                    return self._calculate_content_similarity(node1.value, node2.value)
                else:
                    # boolean, null - exact match
                    return 1.0 if node1.value == node2.value else 0.0
            else:
                # A number against a numeric string: compare by value, not by type. The
                # consumer parses the string, so 20 and "20" are one call (all 200 such
                # oracle pairs execute identically) and 20 and "21" are two.
                if self.numeric_string_equality and {node1.node_type, node2.node_type} == {"number", "string"}:
                    a, b = _leaf.numeric_value(node1.value), _leaf.numeric_value(node2.value)
                    if a is not None and b is not None:
                        tol = self._numeric_tolerance(node1, node2)
                        if tol is not None:
                            return 1.0 if numbers_within(a, b, tol) else 0.0
                        return 1.0 if _leaf.numbers_close(a, b) else 0.0
                # Cross-type comparison: use semantic similarity with type penalty
                val1_str = str(node1.value)
                val2_str = str(node2.value)

                semantic_sim = self._leaf_value_similarity(val1_str, val2_str)
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

    def _is_callee_name_node(self, node: JsonNode) -> bool:
        """A string leaf keyed by one of exact_name_fields whose object also carries
        `arguments`: the name of the function being called. Same scoping as
        _callee_name_paths on the path-F1 side."""
        if not self.exact_name_fields or node.node_type != "string" or node.parent is None:
            return False
        key = node.path.rsplit(".", 1)[-1]
        if key not in self.exact_name_fields:
            return False
        return any(c.path.rsplit(".", 1)[-1] == "arguments" for c in node.parent.children)

    def content_update_cost(self, node1: JsonNode, node2: JsonNode) -> float:
        """Calculate content update cost only, bounded in [0, 1]"""
        if (node1.node_type == node2.node_type and
            node1.value == node2.value):
            return 0.0  # Content identical
        if self._is_callee_name_node(node1) and self._is_callee_name_node(node2):
            # A different callee is a different call: the adversary's best move against the
            # typed profile was a substituted function name scoring 0.966 through the
            # embedding kernel (9 of 207 source calls above 0.95).
            return min(1.0, (self.calculate_path_weight(node1.path) + self.calculate_path_weight(node2.path)) / 2.0)

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
                    cost_matrix[i, j] = 1.0 - self._numeric_string_match(item1, item2)
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

        position_costs: List[float] = []
        for i in range(min_len):
            item1 = arr1[i]
            item2 = arr2[i]

            # Calculate cost for this position
            if not isinstance(item1, type(item2)):
                cost = 1.0 - self._numeric_string_match(item1, item2)
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
            position_costs.append(cost)

        # Add insertion/deletion costs for length differences
        # Each extra element costs 1.0 (full insertion/deletion cost)
        length_diff = abs(len1 - len2)
        p = _aggregation.resolve_p(self.aggregation_p)
        if _aggregation.is_active(p):
            # The positional reading used to keep the arithmetic mean whatever the profile's
            # p, so under the certificate (p = -inf, lambda = 1) an array node diluted its
            # worst element exactly the way the soundness theorem says a mean must; a
            # rotated list could score 1 - 1/n while every leaf inside it was wrong. The
            # per-position costs now go through the same power mean as object children,
            # with the flat indel charges included -- the bottleneck end takes the worst
            # position or, if lengths differ, 1.
            return _aggregation.aggregate(position_costs + [1.0] * length_diff, p)
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
            # Under 'product' the channels are conjoined per matched pair and then summed, rather
            # than summed per channel and conjoined once at the end. Only the per-pair order is
            # meaningful: a t-norm of two sums is not a sum of t-norms, and it is the individual
            # leaf that must not receive structural credit its content never earned. The Hungarian
            # assignment above is untouched, so matching stays structure-guided either way.
            pair_total_cost = 0.0
            for i, j in zip(row_indices, col_indices):
                if i < n1 and j < n2:
                    # Matched pair - calculate content cost directly
                    # For leaves: use content_update_cost
                    # For non-leaves: recursively use combined matching (structure-guided)
                    if not children1[i].children and not children2[j].children:
                        c = self.content_update_cost(children1[i], children2[j])
                        content_total_cost += c
                        pair_total_cost += self.combine_channels(
                            structural_cost_matrix[i][j], c)
                    else:
                        # For non-leaf nodes, recursively calculate combined cost
                        # (which also uses structural matching to guide content). The recursion
                        # already returns a channel-combined value, so it is used as the pair cost
                        # directly rather than being conjoined a second time.
                        rec = self._calculate_optimal_matching_cost(
                            children1[i], children2[j], "combined")
                        content_total_cost += rec
                        pair_total_cost += rec
                elif i < n1:
                    # Deletion
                    content_total_cost += self.delete_cost(children1[i])
                    pair_total_cost += self.delete_cost(children1[i])
                elif j < n2:
                    # Insertion
                    content_total_cost += self.insert_cost(children2[j])
                    pair_total_cost += self.insert_cost(children2[j])

            if self.cost_combination == 'additive':
                # Tree-level weighted average, Eq. (2): γ_upd = w · γ_struct + (1-w) · γ_content
                total_cost = self.structural_weight * structural_total_cost + (1 - self.structural_weight) * content_total_cost
            else:
                total_cost = pair_total_cost

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

    # ------------------------------------------------------------------ path-F1 (ranking)
    @staticmethod
    def _leaf_paths(obj: Any, prefix: JsonPath = ()) -> Dict[JsonPath, Any]:
        """Primitive values and empty containers at unambiguous typed paths."""
        return terminal_paths(obj, prefix)

    @staticmethod
    def _normalise_path(p):
        """Typed paths need no serialization or delimiter normalization."""
        return p

    def _callee_name_paths(self, obj: Any, prefix: JsonPath = ()) -> set:
        """Paths of name-field leaves that identify a *callee*: a name field whose
        object also carries an `arguments` field (a tool call). A `name` inside a
        person record or a product is a value, not an identity, and keeps the
        string kernel. Without this scoping `{"name": "John"}` vs `{"name": "Jane"}`
        scored 0 under the ranking profile."""
        out: set = set()
        if isinstance(obj, dict):
            if "arguments" in obj:
                for f in self.exact_name_fields:
                    if f in obj:
                        out.add(prefix + (("key", f),))
            for k, v in obj.items():
                out |= self._callee_name_paths(v, prefix + (("key", k),))
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                out |= self._callee_name_paths(v, prefix + (("index", i),))
        return out

    def _path_leaf_similarity(self, path: JsonPath, a: Any, b: Any, callee_paths: Optional[set] = None) -> float:
        """The profile's leaf kernel applied to two values at the same path."""
        if isinstance(a, EmptyContainer) or isinstance(b, EmptyContainer):
            return float(a is b)
        if isinstance(a, bool) or isinstance(b, bool) or a is None or b is None:
            return float(type(a) is type(b) and a == b)
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            return float(self._compare_numbers(float(a), float(b)))
        if isinstance(a, (int, float)) or isinstance(b, (int, float)):
            return self._numeric_string_match(a, b)
        if isinstance(a, str) and isinstance(b, str):
            if a == b:
                return 1.0
            if callee_paths is not None and path in callee_paths:
                return 0.0
            return self._compare_strings(a, b)
        return 0.0

    _UNMATCHED = object()

    def _align_named_lists(self, ref: Any, cand: Any) -> Any:
        """Return `cand` re-indexed so that, in every list of objects carrying a
        name field, the element with the same name as the reference's element at
        position i sits at position i. This is the one place the ranking corner
        keeps STED's alignment: a candidate that issues the same calls in another
        order is the same set of calls, and positional paths would otherwise score
        every one of them wrong (15 of the 704 pool pairs are such reorderings).
        Matching is by exact name -- identity, not similarity -- preferring the
        same index when names repeat; unmatched reference positions become holes
        (no leaves, so they count as missing), unmatched candidate elements are
        appended after the reference's length (extra). Everything else recurses
        positionally."""
        if isinstance(ref, dict) and isinstance(cand, dict):
            return {k: (self._align_named_lists(ref[k], v) if k in ref else v) for k, v in cand.items()}
        if isinstance(ref, list) and isinstance(cand, list):
            named = lambda xs: all(isinstance(x, dict) and any(f in x for f in self.exact_name_fields) for x in xs)
            if self.exact_name_fields and ref and cand and named(ref) and named(cand):
                name = lambda d: tuple(d.get(f) for f in sorted(self.exact_name_fields))
                used, out = set(), [None] * len(ref)
                for i, r in enumerate(ref):
                    j = next((k for k, c in enumerate(cand) if k not in used and name(c) == name(r) and k == i), None)
                    if j is None:
                        j = next((k for k, c in enumerate(cand) if k not in used and name(c) == name(r)), None)
                    if j is not None:
                        used.add(j)
                        out[i] = self._align_named_lists(r, cand[j])
                # Positional fallback for what identity could not pair: a call with a
                # different name still sits opposite the reference call at that slot,
                # so its arguments are priced by the kernel rather than zeroed. (The
                # zeroing alternative -- the identity gate -- lost 16:3 against the
                # annotator on the structural stratum.)
                rest = [k for k in range(len(cand)) if k not in used]
                for i in range(len(ref)):
                    if out[i] is None:
                        out[i] = self._align_named_lists(ref[i], cand[rest.pop(0)]) if rest else self._UNMATCHED
                out.extend(cand[k] for k in rest)
                return out
            return [self._align_named_lists(ref[i], c) if i < len(ref) else c for i, c in enumerate(cand)]
        return cand

    def _leaf_paths_skipping_holes(self, obj: Any) -> Dict[JsonPath, Any]:
        """_leaf_paths, but an unmatched hole contributes no leaves."""
        return terminal_paths(obj, skip=self._UNMATCHED, skip_enabled=True)

    def calculate_field_differences(self, json1: Any, json2: Any) -> List[Dict[str, Any]]:
        """Per-field scores behind the tree score, for the fields that differ.

        Fields are terminal paths (primitives and empty containers, typed key/index
        steps). A field present on one side only is ``missing`` (in json2) or
        ``extra`` (in json2) and scores 0. For a field present on both sides with
        different values, the score is the tree score of ``json1`` against a copy of
        ``json1`` with only that field replaced by json2's value: siblings, path and
        callee identity stay in context, so the field is scored exactly as it is
        inside the full comparison. Under a bottleneck profile with positional
        arrays and exact keys (``typed``), the tree score equals the minimum over
        these scores (1.0 when the list is empty).

        Arrays matched by an ``unordered`` field rule are reported as one field (the
        whole array), since element positions do not correspond. Ignored fields never
        appear.

        Returns dicts ``{"path", "status", "a", "b", "score"}`` sorted by score.
        """
        from ._paths import EmptyContainer, terminal_paths

        json1, json2 = self.field_rules.strip(json1), self.field_rules.strip(json2)
        pa, pb = terminal_paths(json1), terminal_paths(json2)
        if self.field_rules.unordered:
            pa, pb = self._collapse_unordered(json1, pa), self._collapse_unordered(json2, pb)

        def show(v):
            if isinstance(v, EmptyContainer):
                return {} if v is EmptyContainer.OBJECT else []
            return v

        def show(v, _show=show):
            return v.value if isinstance(v, _Whole) else _show(v)

        def replaced(doc, path, value):
            doc = json.loads(json.dumps(doc))
            if not path:
                return value
            node = doc
            for _, step in path[:-1]:
                node = node[step]
            node[path[-1][1]] = value
            return doc

        out: List[Dict[str, Any]] = []
        for p in sorted(set(pa) | set(pb), key=repr):
            if p not in pb:
                out.append({"path": format_json_path(p), "status": "missing", "a": show(pa[p]), "b": None, "score": 0.0})
            elif p not in pa:
                out.append({"path": format_json_path(p), "status": "extra", "a": None, "b": show(pb[p]), "score": 0.0})
            else:
                x, y = pa[p], pb[p]
                if isinstance(x, _Whole) or isinstance(y, _Whole):
                    xv = x.value if isinstance(x, _Whole) else show(x)
                    yv = y.value if isinstance(y, _Whole) else show(y)
                    score = float(self.calculate_tree_edit_distance_fast(json1, replaced(json1, p, yv)))
                    if score >= 1.0:
                        continue
                    out.append({"path": format_json_path(p), "status": "changed", "a": xv, "b": yv, "score": score})
                    continue
                if isinstance(x, EmptyContainer) or isinstance(y, EmptyContainer):
                    if x is y:
                        continue
                    score = 0.0
                elif type(x) is type(y) and x == y:
                    continue
                else:
                    score = float(self.calculate_tree_edit_distance_fast(json1, replaced(json1, p, y)))
                    if score >= 1.0:
                        continue   # e.g. 3 vs "3": not a difference under this profile
                out.append({"path": format_json_path(p), "status": "changed", "a": show(x), "b": show(y), "score": score})
        out.sort(key=lambda d: d["score"])
        return out

    def _collapse_unordered(self, doc: Any, leaves: Dict[JsonPath, Any]) -> Dict[JsonPath, Any]:
        """Replace the leaves under each outermost unordered array by one _Whole entry."""
        out: Dict[JsonPath, Any] = {}
        for p, v in leaves.items():
            cut = next((i for i in range(len(p) + 1)
                        if self._is_unordered_array(doc, p[:i])), None)
            if cut is None:
                out[p] = v
            else:
                q = p[:cut]
                if q not in out:
                    node = doc
                    for _, step in q:
                        node = node[step]
                    out[q] = _Whole(node)
        return out

    def _is_unordered_array(self, doc: Any, path: JsonPath) -> bool:
        node = doc
        try:
            for _, step in path:
                node = node[step]
        except (KeyError, IndexError, TypeError):
            return False
        return isinstance(node, list) and self.field_rules.is_unordered(path)

    def calculate_readouts(self, json1: Any, json2: Any) -> Dict[str, float]:
        """One leaf table, two readouts.

        ``score``     -- path-anchored soft precision/recall harmonic mean: the ranking
                         readout, every deviation charged, none fatal.
        ``certified`` -- the worst leaf (unmatched path = 0): the threshold readout.
                         ``certified >= theta`` means every reference leaf has a
                         counterpart at the same path with leaf similarity >= theta and
                         the candidate has no path the reference lacks.
        The paper's theorem is why there are two numbers and not one: a mean-type
        aggregate cannot carry a sound local threshold, and the bottleneck that can
        ties too often to rank. Both are computed from the same leaf kernel and the
        same alignment, so a pair that passes ``certified`` also has every leaf that
        ``score`` averaged at or above the same bar.
        """
        lr = {self._normalise_path(k): v for k, v in self._leaf_paths(json1).items()}
        if self.align_calls_by_name:
            aligned = self._align_named_lists(json1, json2)
            lc = {self._normalise_path(k): v for k, v in self._leaf_paths_skipping_holes(aligned).items()}
        else:
            lc = {self._normalise_path(k): v for k, v in self._leaf_paths(json2).items()}
        if not lr and not lc:
            return {"score": 1.0, "certified": 1.0}
        if not lr or not lc:
            return {"score": 0.0, "certified": 0.0}
        callee = self._callee_name_paths(json1) | self._callee_name_paths(json2)
        sims = {k: (self._path_leaf_similarity(k, lr[k], lc[k], callee) if k in lc else 0.0) for k in lr}
        recall = sum(sims.values()) / len(lr)
        precision = sum(sims[k] if k in lr else 0.0 for k in lc) / len(lc)
        score = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        certified = min(min(sims.values()), 0.0 if set(lc) - set(lr) else 1.0)
        return {"score": score, "certified": certified}

    def calculate_path_f1_similarity(self, json1: Any, json2: Any) -> float:
        """Harmonic mean of path-anchored soft recall (json1 as reference) and soft
        precision (json2 as candidate), after aligning named lists by identity.
        Symmetric in the two arguments up to that alignment. Equal to
        ``calculate_readouts(json1, json2)["score"]``."""
        lr = {self._normalise_path(k): v for k, v in self._leaf_paths(json1).items()}
        if self.align_calls_by_name:
            aligned = self._align_named_lists(json1, json2)
            lc = {self._normalise_path(k): v for k, v in self._leaf_paths_skipping_holes(aligned).items()}
        else:
            lc = {self._normalise_path(k): v for k, v in self._leaf_paths(json2).items()}
        if not lr and not lc:
            return 1.0
        if not lr or not lc:
            return 0.0
        # No identity gate on a mismatched call's arguments. Zeroing them is the
        # principled reading (a different callee has different parameters even
        # when the keys coincide), and it was tried: on the structural annotation
        # stratum it changed 22 verdicts, 16 against the annotator and 3 for.
        # Callees in this corpus usually differ by a variant suffix and the
        # annotator credits the shared arguments, so only the name leaf is exact.
        callee = self._callee_name_paths(json1) | self._callee_name_paths(json2)
        sim = lambda k: self._path_leaf_similarity(k, lr[k], lc[k], callee)
        recall = sum(sim(k) if k in lc else 0.0 for k in lr) / len(lr)
        precision = sum(sim(k) if k in lr else 0.0 for k in lc) / len(lc)
        if precision + recall == 0.0:
            return 0.0
        return 2.0 * precision * recall / (precision + recall)

    def calculate_tree_edit_distance_opt(self, json1: Dict[str, Any], json2: Dict[str, Any], variation_type: str = "combined") -> float:
        json1, json2 = self.field_rules.strip(json1), self.field_rules.strip(json2)
        if self.aggregation_mode == "path_f1":
            return self.calculate_path_f1_similarity(json1, json2)
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
        json1, json2 = self.field_rules.strip(json1), self.field_rules.strip(json2)
        if self.aggregation_mode == "path_f1":
            return self.calculate_path_f1_similarity(json1, json2)
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
        if self.aggregation_mode == "path_f1":
            return self.calculate_path_f1_similarity(json1, json2)
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
        if self.aggregation_mode == "path_f1":
            return self.calculate_path_f1_similarity(json1, json2)
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
