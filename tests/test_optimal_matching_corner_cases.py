#!/usr/bin/env python3
"""
Test corner cases for _calculate_optimal_matching_cost function.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sted.semantic_json_tree_consistency import SemanticJsonTreeConsistencyEvaluator


def test_corner_cases():
    """Test various corner cases for optimal matching cost calculation."""

    evaluator = SemanticJsonTreeConsistencyEvaluator()

    results = []

    # ===========================================
    # Case 1: Empty objects
    # ===========================================
    print("\n" + "="*60)
    print("Case 1: Empty objects")
    print("="*60)
    json1 = {}
    json2 = {}
    similarity = evaluator.calculate_tree_edit_distance(json1, json2, original_zss=False)
    print(f"  {{}} vs {{}}")
    print(f"  Similarity: {similarity:.4f}")
    print(f"  Expected: 1.0 (identical)")
    results.append(("Empty objects", similarity, 1.0))

    # ===========================================
    # Case 2: Empty vs non-empty
    # ===========================================
    print("\n" + "="*60)
    print("Case 2: Empty vs non-empty")
    print("="*60)
    json1 = {}
    json2 = {"a": 1}
    similarity = evaluator.calculate_tree_edit_distance(json1, json2, original_zss=False)
    print(f"  {{}} vs {json2}")
    print(f"  Similarity: {similarity:.4f}")
    print(f"  Expected: < 1.0 (one has content)")
    results.append(("Empty vs non-empty", similarity, "< 1.0"))

    # ===========================================
    # Case 3: Single identical field
    # ===========================================
    print("\n" + "="*60)
    print("Case 3: Single identical field")
    print("="*60)
    json1 = {"name": "John"}
    json2 = {"name": "John"}
    similarity = evaluator.calculate_tree_edit_distance(json1, json2, original_zss=False)
    print(f"  {json1} vs {json2}")
    print(f"  Similarity: {similarity:.4f}")
    print(f"  Expected: 1.0 (identical)")
    results.append(("Single identical field", similarity, 1.0))

    # ===========================================
    # Case 4: Single field, different values
    # ===========================================
    print("\n" + "="*60)
    print("Case 4: Single field, different values")
    print("="*60)
    json1 = {"name": "John"}
    json2 = {"name": "Jane"}
    similarity = evaluator.calculate_tree_edit_distance(json1, json2, original_zss=False)
    print(f"  {json1} vs {json2}")
    print(f"  Similarity: {similarity:.4f}")
    print(f"  Expected: > 0.5 (same key, similar values)")
    results.append(("Single field, different values", similarity, "> 0.5"))

    # ===========================================
    # Case 5: Single field, completely different values
    # ===========================================
    print("\n" + "="*60)
    print("Case 5: Single field, completely different values")
    print("="*60)
    json1 = {"name": "John"}
    json2 = {"name": 12345}
    similarity = evaluator.calculate_tree_edit_distance(json1, json2, original_zss=False)
    print(f"  {json1} vs {json2}")
    print(f"  Similarity: {similarity:.4f}")
    print(f"  Expected: > 0 (same key, type mismatch)")
    results.append(("Single field, type mismatch", similarity, "> 0"))

    # ===========================================
    # Case 6: Different keys, same values
    # ===========================================
    print("\n" + "="*60)
    print("Case 6: Different keys, same values")
    print("="*60)
    json1 = {"name": "John"}
    json2 = {"username": "John"}
    similarity = evaluator.calculate_tree_edit_distance(json1, json2, original_zss=False)
    print(f"  {json1} vs {json2}")
    print(f"  Similarity: {similarity:.4f}")
    print(f"  Expected: > 0.5 (semantically similar keys)")
    results.append(("Different keys, same values", similarity, "> 0.5"))

    # ===========================================
    # Case 7: Completely different structures
    # ===========================================
    print("\n" + "="*60)
    print("Case 7: Completely different structures")
    print("="*60)
    json1 = {"a": 1, "b": 2}
    json2 = {"x": "hello", "y": "world", "z": True}
    similarity = evaluator.calculate_tree_edit_distance(json1, json2, original_zss=False)
    print(f"  {json1} vs {json2}")
    print(f"  Similarity: {similarity:.4f}")
    print(f"  Expected: low (different keys and values)")
    results.append(("Completely different", similarity, "low"))

    # ===========================================
    # Case 8: Nested identical objects
    # ===========================================
    print("\n" + "="*60)
    print("Case 8: Nested identical objects")
    print("="*60)
    json1 = {"user": {"name": "John", "age": 30}}
    json2 = {"user": {"name": "John", "age": 30}}
    similarity = evaluator.calculate_tree_edit_distance(json1, json2, original_zss=False)
    print(f"  {json1}")
    print(f"  vs {json2}")
    print(f"  Similarity: {similarity:.4f}")
    print(f"  Expected: 1.0 (identical)")
    results.append(("Nested identical", similarity, 1.0))

    # ===========================================
    # Case 9: Nested with partial difference
    # ===========================================
    print("\n" + "="*60)
    print("Case 9: Nested with partial difference")
    print("="*60)
    json1 = {"user": {"name": "John", "age": 30}}
    json2 = {"user": {"name": "John", "age": 31}}
    similarity = evaluator.calculate_tree_edit_distance(json1, json2, original_zss=False)
    print(f"  {json1}")
    print(f"  vs {json2}")
    print(f"  Similarity: {similarity:.4f}")
    print(f"  Expected: > 0.9 (only age differs slightly)")
    results.append(("Nested partial diff", similarity, "> 0.9"))

    # ===========================================
    # Case 10: Empty arrays
    # ===========================================
    print("\n" + "="*60)
    print("Case 10: Empty arrays")
    print("="*60)
    json1 = {"items": []}
    json2 = {"items": []}
    similarity = evaluator.calculate_tree_edit_distance(json1, json2, original_zss=False)
    print(f"  {json1} vs {json2}")
    print(f"  Similarity: {similarity:.4f}")
    print(f"  Expected: 1.0 (identical)")
    results.append(("Empty arrays", similarity, 1.0))

    # ===========================================
    # Case 11: Empty array vs non-empty array
    # ===========================================
    print("\n" + "="*60)
    print("Case 11: Empty array vs non-empty array")
    print("="*60)
    json1 = {"items": []}
    json2 = {"items": [1, 2, 3]}
    similarity = evaluator.calculate_tree_edit_distance(json1, json2, original_zss=False)
    print(f"  {json1} vs {json2}")
    print(f"  Similarity: {similarity:.4f}")
    print(f"  Expected: < 1.0 (different array contents)")
    results.append(("Empty vs non-empty array", similarity, "< 1.0"))

    # ===========================================
    # Case 12: Arrays with same elements, different order
    # ===========================================
    print("\n" + "="*60)
    print("Case 12: Arrays with same elements, different order")
    print("="*60)
    json1 = {"items": [1, 2, 3]}
    json2 = {"items": [3, 2, 1]}
    similarity = evaluator.calculate_tree_edit_distance(json1, json2, original_zss=False)
    print(f"  {json1} vs {json2}")
    print(f"  Similarity: {similarity:.4f}")
    print(f"  Expected: 1.0 (order-invariant matching)")
    results.append(("Array reordered", similarity, 1.0))

    # ===========================================
    # Case 13: Null values
    # ===========================================
    print("\n" + "="*60)
    print("Case 13: Null values")
    print("="*60)
    json1 = {"value": None}
    json2 = {"value": None}
    similarity = evaluator.calculate_tree_edit_distance(json1, json2, original_zss=False)
    print(f"  {json1} vs {json2}")
    print(f"  Similarity: {similarity:.4f}")
    print(f"  Expected: 1.0 (identical)")
    results.append(("Null values", similarity, 1.0))

    # ===========================================
    # Case 14: Null vs non-null
    # ===========================================
    print("\n" + "="*60)
    print("Case 14: Null vs non-null")
    print("="*60)
    json1 = {"value": None}
    json2 = {"value": "something"}
    similarity = evaluator.calculate_tree_edit_distance(json1, json2, original_zss=False)
    print(f"  {json1} vs {json2}")
    print(f"  Similarity: {similarity:.4f}")
    print(f"  Expected: < 1.0 (type mismatch)")
    results.append(("Null vs non-null", similarity, "< 1.0"))

    # ===========================================
    # Case 15: Boolean values
    # ===========================================
    print("\n" + "="*60)
    print("Case 15: Boolean values")
    print("="*60)
    json1 = {"active": True}
    json2 = {"active": False}
    similarity = evaluator.calculate_tree_edit_distance(json1, json2, original_zss=False)
    print(f"  {json1} vs {json2}")
    print(f"  Similarity: {similarity:.4f}")
    print(f"  Expected: > 0.5 (same key, different boolean)")
    results.append(("Boolean diff", similarity, "> 0.5"))

    # ===========================================
    # Case 16: Deep nesting (5 levels)
    # ===========================================
    print("\n" + "="*60)
    print("Case 16: Deep nesting (5 levels)")
    print("="*60)
    json1 = {"a": {"b": {"c": {"d": {"e": 1}}}}}
    json2 = {"a": {"b": {"c": {"d": {"e": 1}}}}}
    similarity = evaluator.calculate_tree_edit_distance(json1, json2, original_zss=False)
    print(f"  {json1}")
    print(f"  vs {json2}")
    print(f"  Similarity: {similarity:.4f}")
    print(f"  Expected: 1.0 (identical)")
    results.append(("Deep nesting identical", similarity, 1.0))

    # ===========================================
    # Case 17: Deep nesting with leaf difference
    # ===========================================
    print("\n" + "="*60)
    print("Case 17: Deep nesting with leaf difference")
    print("="*60)
    json1 = {"a": {"b": {"c": {"d": {"e": 1}}}}}
    json2 = {"a": {"b": {"c": {"d": {"e": 2}}}}}
    similarity = evaluator.calculate_tree_edit_distance(json1, json2, original_zss=False)
    print(f"  {json1}")
    print(f"  vs {json2}")
    print(f"  Similarity: {similarity:.4f}")
    print(f"  Expected: > 0.8 (only leaf differs)")
    results.append(("Deep nesting leaf diff", similarity, "> 0.8"))

    # ===========================================
    # Case 18: Large flat object
    # ===========================================
    print("\n" + "="*60)
    print("Case 18: Large flat object (20 fields)")
    print("="*60)
    json1 = {f"field_{i}": i for i in range(20)}
    json2 = {f"field_{i}": i for i in range(20)}
    similarity = evaluator.calculate_tree_edit_distance(json1, json2, original_zss=False)
    print(f"  20 identical fields")
    print(f"  Similarity: {similarity:.4f}")
    print(f"  Expected: 1.0 (identical)")
    results.append(("Large flat identical", similarity, 1.0))

    # ===========================================
    # Case 19: Large flat object with 1 difference
    # ===========================================
    print("\n" + "="*60)
    print("Case 19: Large flat object with 1 difference")
    print("="*60)
    json1 = {f"field_{i}": i for i in range(20)}
    json2 = {f"field_{i}": i for i in range(20)}
    json2["field_10"] = 999  # Change one field
    similarity = evaluator.calculate_tree_edit_distance(json1, json2, original_zss=False)
    print(f"  20 fields, 1 different value")
    print(f"  Similarity: {similarity:.4f}")
    print(f"  Expected: > 0.9 (only 1/20 differs)")
    results.append(("Large flat 1 diff", similarity, "> 0.9"))

    # ===========================================
    # Case 20: Mixed types in array
    # ===========================================
    print("\n" + "="*60)
    print("Case 20: Mixed types in array")
    print("="*60)
    json1 = {"data": [1, "two", True, None, {"nested": "obj"}]}
    json2 = {"data": [1, "two", True, None, {"nested": "obj"}]}
    similarity = evaluator.calculate_tree_edit_distance(json1, json2, original_zss=False)
    print(f"  {json1}")
    print(f"  vs {json2}")
    print(f"  Similarity: {similarity:.4f}")
    print(f"  Expected: 1.0 (identical)")
    results.append(("Mixed array identical", similarity, 1.0))

    # ===========================================
    # Summary
    # ===========================================
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    passed = 0
    failed = 0

    for name, actual, expected in results:
        if isinstance(expected, float):
            # Exact match (with tolerance)
            if abs(actual - expected) < 0.01:
                status = "✓ PASS"
                passed += 1
            else:
                status = "✗ FAIL"
                failed += 1
        elif expected == "< 1.0":
            if actual < 1.0:
                status = "✓ PASS"
                passed += 1
            else:
                status = "✗ FAIL"
                failed += 1
        elif expected == "> 0.5":
            if actual > 0.5:
                status = "✓ PASS"
                passed += 1
            else:
                status = "✗ FAIL"
                failed += 1
        elif expected == "> 0.8":
            if actual > 0.8:
                status = "✓ PASS"
                passed += 1
            else:
                status = "✗ FAIL"
                failed += 1
        elif expected == "> 0.9":
            if actual > 0.9:
                status = "✓ PASS"
                passed += 1
            else:
                status = "✗ FAIL"
                failed += 1
        elif expected == "> 0":
            if actual > 0:
                status = "✓ PASS"
                passed += 1
            else:
                status = "✗ FAIL"
                failed += 1
        elif expected == "low":
            if actual < 0.5:
                status = "✓ PASS"
                passed += 1
            else:
                status = "✗ FAIL"
                failed += 1
        else:
            status = "? UNKNOWN"

        print(f"  {status}: {name} (actual={actual:.4f}, expected={expected})")

    print(f"\nTotal: {passed} passed, {failed} failed")

    return failed == 0


if __name__ == "__main__":
    success = test_corner_cases()
    sys.exit(0 if success else 1)
