"""BERTScore and DeepDiff baseline metrics for STED.

Extracted from semantic_json_tree_consistency.py during the v0.2.0 refactor.
The original methods on ``SemanticJsonTreeConsistencyEvaluator`` are thin
wrappers that delegate to these module-level helpers, so existing call sites
and import paths keep working.
"""
from __future__ import annotations

import warnings
from typing import Any, Dict, List

from deepdiff import DeepDiff

# bert_score is optional (only used by the bertscore_pair / bertscore_json
# helpers). Lazy-imported so users without the bertscore extra can still use
# the STED metric.
try:
    from bert_score import score as bert_score
except ImportError:
    bert_score = None

from .json_tree_node import JsonNode
from .utils import count_json_elements


def calculate_bertscore(evaluator, json1: Dict[str, Any], json2: Dict[str, Any], **kwargs) -> float:
    # Preprocess JSONs to make them order-invariant
    tree1 = JsonNode.from_dict(json1, sort_arrays=evaluator.sort_arrays, sort_keys=evaluator.sort_keys)
    tree2 = JsonNode.from_dict(json2, sort_arrays=evaluator.sort_arrays, sort_keys=evaluator.sort_keys)
    processed_json1 = tree1.reconstruct_json()
    processed_json2 = tree2.reconstruct_json()

    P, R, F1 = bert_score([str(processed_json1)], [str(processed_json2)], lang="en")
    return float(F1.item())


def calculate_similarity_with_deepdiff(
    evaluator,
    json1,
    json2,
    **kwargs,
) -> float:
    diff = DeepDiff(json1, json2, ignore_order=True, cache_size=5000, get_deep_distance=True)
    return 1 - diff['deep_distance']


def calculate_similarity_with_deepdiff_opt(
    evaluator,
    json1,
    json2,
    variation_type: str = "combined",
    structural_weight: float = 0.5,
    **kwargs,
) -> float:
    """
    Calculate similarity using DeepDiff with enhanced value comparison.
    Uses semantic similarity for strings and proper comparison for numbers.

    Args:
        evaluator: SemanticJsonTreeConsistencyEvaluator instance
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
            return _calculate_deepdiff_structural_only(evaluator, diff, json1, json2)
        elif variation_type == "content":
            return _calculate_deepdiff_content_only(evaluator, diff, json1, json2)
        else:  # combined
            structural_sim = _calculate_deepdiff_structural_only(evaluator, diff, json1, json2)
            content_sim = _calculate_deepdiff_content_only(evaluator, diff, json1, json2)
            return structural_sim * structural_weight + (1 - structural_weight) * content_sim

    except Exception as e:
        warnings.warn(f"Error in DeepDiff calculation: {str(e)}")
        return 0.0


def _calculate_deepdiff_structural_only(evaluator, diff: dict, json1, json2) -> float:
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


def _calculate_deepdiff_content_only(evaluator, diff: dict, json1, json2) -> float:
    """Calculate content similarity only (reuse original value processing logic)"""
    # First check structural similarity
    structural_sim = _calculate_deepdiff_structural_only(evaluator, diff, json1, json2)

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
                value_similarity = evaluator._compare_strings(old_value, new_value)
            elif isinstance(old_value, (int, float)) and isinstance(new_value, (int, float)):
                value_similarity = evaluator._compare_numbers(float(old_value), float(new_value))
            elif isinstance(old_value, bool) and isinstance(new_value, bool):
                value_similarity = 1.0 if old_value == new_value else 0.0
            elif old_value is None or new_value is None:
                value_similarity = 1.0 if old_value == new_value else 0.0
            elif isinstance(old_value, (list, dict)) and isinstance(new_value, (list, dict)):
                value_similarity = evaluator.calculate_similarity_with_deepdiff(old_value, new_value)
            else:
                try:
                    value_similarity = evaluator._compare_numbers(float(old_value), float(new_value))
                except (ValueError, TypeError):
                    value_similarity = evaluator._compare_strings(str(old_value), str(new_value))

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
            type_cost = evaluator.type_change_cost.get((old_type_mapped, new_type_mapped), 1.0)
            type_similarity = 1.0 - type_cost

            total_similarity_score += type_similarity
            total_comparisons += 1

    # Return content similarity
    if total_comparisons == 0:
        return 1.0  # No value or type changes

    return total_similarity_score / total_comparisons
