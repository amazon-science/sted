#!/usr/bin/env python3
"""
Tests for order-sensitive array comparison functionality.

This tests the order_sensitive_fields parameter which controls whether
arrays are compared sequentially (order matters) or using optimal matching
(order doesn't matter).
"""

import pytest
from sted import SemanticJsonTreeConsistencyEvaluator


class TestOrderSensitiveFieldDetection:
    """Test _is_order_sensitive_field() method."""

    def test_no_order_sensitive_fields(self):
        """When order_sensitive_fields is empty, no field should match."""
        evaluator = SemanticJsonTreeConsistencyEvaluator(
            model_id='all-MiniLM-L6-v2',
            order_sensitive_fields=set()
        )
        assert evaluator._is_order_sensitive_field("trace") is False
        assert evaluator._is_order_sensitive_field("steps") is False
        assert evaluator._is_order_sensitive_field("items") is False

    def test_direct_field_name_match(self):
        """Direct field names should be detected."""
        evaluator = SemanticJsonTreeConsistencyEvaluator(
            model_id='all-MiniLM-L6-v2',
            order_sensitive_fields={"trace", "steps"}
        )
        assert evaluator._is_order_sensitive_field("trace") is True
        assert evaluator._is_order_sensitive_field("steps") is True
        assert evaluator._is_order_sensitive_field("items") is False

    def test_nested_path_match(self):
        """Nested paths containing order-sensitive field should match."""
        evaluator = SemanticJsonTreeConsistencyEvaluator(
            model_id='all-MiniLM-L6-v2',
            order_sensitive_fields={"trace", "calls"}
        )
        assert evaluator._is_order_sensitive_field("root.trace") is True
        assert evaluator._is_order_sensitive_field("root.data.trace") is True
        assert evaluator._is_order_sensitive_field("root.calls[0]") is True
        assert evaluator._is_order_sensitive_field("root.agent.calls") is True

    def test_array_index_in_path(self):
        """Paths with array indices should correctly identify fields."""
        evaluator = SemanticJsonTreeConsistencyEvaluator(
            model_id='all-MiniLM-L6-v2',
            order_sensitive_fields={"steps"}
        )
        assert evaluator._is_order_sensitive_field("steps[0]") is True
        assert evaluator._is_order_sensitive_field("root.steps[5]") is True
        assert evaluator._is_order_sensitive_field("data[0].steps") is True


class TestOrderSensitiveArrayComparison:
    """Test array comparison with order sensitivity."""

    @pytest.fixture
    def order_sensitive_evaluator(self):
        """Evaluator with order-sensitive fields configured."""
        return SemanticJsonTreeConsistencyEvaluator(
            model_id='all-MiniLM-L6-v2',
            order_sensitive_fields={"steps", "trace", "sequence"}
        )

    @pytest.fixture
    def order_insensitive_evaluator(self):
        """Evaluator with no order-sensitive fields."""
        return SemanticJsonTreeConsistencyEvaluator(
            model_id='all-MiniLM-L6-v2',
            order_sensitive_fields=set()
        )

    def test_identical_arrays_same_order(self, order_sensitive_evaluator):
        """Identical arrays in same order should have similarity 1.0."""
        json1 = {"steps": ["step1", "step2", "step3"]}
        json2 = {"steps": ["step1", "step2", "step3"]}

        similarity = order_sensitive_evaluator.calculate_tree_edit_distance_opt(
            json1, json2, variation_type="combined"
        )
        assert similarity == 1.0

    def test_order_matters_for_sensitive_fields(self, order_sensitive_evaluator):
        """Different order should result in lower similarity for order-sensitive fields.

        Note: At the tree edit distance level, arrays are represented as tree nodes with
        children, and matching uses Hungarian algorithm. Order sensitivity is applied
        at the direct array comparison level (_compare_arrays_unordered/_compare_arrays_ordered).
        """
        # Test via direct array comparison which does respect order
        arr1 = ["A", "B", "C"]
        arr2 = ["C", "B", "A"]  # Reversed order

        cost = order_sensitive_evaluator._compare_arrays_ordered(arr1, arr2, "steps", "steps")
        # When order matters, reversed arrays should have higher cost (> 0)
        # Only middle element matches by position
        assert cost > 0

        # Compare with unordered comparison through the delegating method
        cost_delegated = order_sensitive_evaluator._compare_arrays_unordered(arr1, arr2, "steps", "steps")
        # Should delegate to ordered comparison and produce same result
        assert cost_delegated == cost

    def test_order_ignored_for_non_sensitive_fields(self, order_insensitive_evaluator):
        """Same elements in different order should have high similarity for non-order-sensitive fields."""
        json1 = {"items": ["A", "B", "C"]}
        json2 = {"items": ["C", "B", "A"]}  # Reversed order

        similarity = order_insensitive_evaluator.calculate_tree_edit_distance_opt(
            json1, json2, variation_type="combined"
        )
        # When order doesn't matter, identical elements should match perfectly
        assert similarity == 1.0

    def test_order_sensitive_with_different_content(self, order_sensitive_evaluator):
        """Test order-sensitive comparison with genuinely different content."""
        json1 = {"steps": ["login", "click_button", "logout"]}
        json2 = {"steps": ["login", "fill_form", "submit"]}

        similarity = order_sensitive_evaluator.calculate_tree_edit_distance_opt(
            json1, json2, variation_type="combined"
        )
        # First element matches, others differ
        assert 0 < similarity < 1.0

    def test_length_difference_penalty(self, order_sensitive_evaluator):
        """Arrays with different lengths should be penalized."""
        json1 = {"steps": ["A", "B"]}
        json2 = {"steps": ["A", "B", "C", "D"]}

        similarity = order_sensitive_evaluator.calculate_tree_edit_distance_opt(
            json1, json2, variation_type="combined"
        )
        # Length difference should reduce similarity
        assert similarity < 1.0


class TestOrderSensitiveWithNestedObjects:
    """Test order-sensitive comparison with nested objects in arrays."""

    @pytest.fixture
    def evaluator(self):
        return SemanticJsonTreeConsistencyEvaluator(
            model_id='all-MiniLM-L6-v2',
            order_sensitive_fields={"calls", "trace"}
        )

    def test_nested_objects_same_order(self, evaluator):
        """Nested objects in same order should have high similarity."""
        json1 = {
            "trace": [
                {"action": "start", "tool": "search"},
                {"action": "end", "tool": "search"}
            ]
        }
        json2 = {
            "trace": [
                {"action": "start", "tool": "search"},
                {"action": "end", "tool": "search"}
            ]
        }

        similarity = evaluator.calculate_tree_edit_distance_opt(
            json1, json2, variation_type="combined"
        )
        assert similarity == 1.0

    def test_nested_objects_swapped_order(self, evaluator):
        """Swapped nested objects should have lower cost in ordered comparison."""
        # Test direct array comparison with nested objects
        arr1 = [
            {"action": "start", "tool": "search"},
            {"action": "end", "tool": "browse"}
        ]
        arr2 = [
            {"action": "end", "tool": "browse"},
            {"action": "start", "tool": "search"}
        ]

        # Ordered comparison should penalize position mismatch
        cost_ordered = evaluator._compare_arrays_ordered(arr1, arr2, "trace", "trace")
        # Unordered comparison should find optimal matching
        cost_unordered_via_delegate = evaluator._compare_arrays_unordered(arr1, arr2, "trace", "trace")

        # Since trace is order-sensitive, both should use ordered comparison
        assert cost_ordered == cost_unordered_via_delegate
        # Swapped order means position 0 and 1 don't match their counterparts
        assert cost_ordered > 0

    def test_agent_trace_comparison(self, evaluator):
        """Real-world test: agent trace with ordered function calls."""
        json1 = {
            "calls": [
                {"function": "get_weather", "args": {"city": "NYC"}},
                {"function": "send_email", "args": {"to": "user@example.com"}},
                {"function": "log_action", "args": {"message": "done"}}
            ]
        }
        json2 = {
            "calls": [
                {"function": "get_weather", "args": {"city": "NYC"}},
                {"function": "send_email", "args": {"to": "user@example.com"}},
                {"function": "log_action", "args": {"message": "completed"}}
            ]
        }

        similarity = evaluator.calculate_tree_edit_distance_opt(
            json1, json2, variation_type="combined"
        )
        # Very similar traces - only the message differs
        assert similarity > 0.8


class TestOrderSensitiveVsInsensitiveComparison:
    """Compare behavior between order-sensitive and order-insensitive array comparison methods."""

    def test_shuffled_array_comparison(self):
        """Same elements shuffled should produce different results based on order sensitivity."""
        arr1 = [1, 2, 3, 4, 5]
        arr2 = [5, 4, 3, 2, 1]  # Reversed

        # Order-sensitive evaluator
        sensitive_eval = SemanticJsonTreeConsistencyEvaluator(
            model_id='all-MiniLM-L6-v2',
            order_sensitive_fields={"items"}
        )

        # Order-insensitive evaluator
        insensitive_eval = SemanticJsonTreeConsistencyEvaluator(
            model_id='all-MiniLM-L6-v2',
            order_sensitive_fields=set()
        )

        # Compare using direct array methods
        sensitive_cost = sensitive_eval._compare_arrays_unordered(arr1, arr2, "items", "items")
        insensitive_cost = insensitive_eval._compare_arrays_unordered(arr1, arr2, "items", "items")

        # Order-insensitive should have lower cost (elements match optimally)
        # Order-sensitive should have higher cost (position mismatches)
        assert insensitive_cost < sensitive_cost

    def test_partially_matching_order(self):
        """Test arrays with some elements in correct position."""
        arr1 = ["A", "B", "C", "D"]
        arr2 = ["A", "C", "B", "D"]  # Middle two swapped

        sensitive_eval = SemanticJsonTreeConsistencyEvaluator(
            model_id='all-MiniLM-L6-v2',
            order_sensitive_fields={"data"}
        )

        insensitive_eval = SemanticJsonTreeConsistencyEvaluator(
            model_id='all-MiniLM-L6-v2',
            order_sensitive_fields=set()
        )

        # Compare using direct array methods
        sensitive_cost = sensitive_eval._compare_arrays_unordered(arr1, arr2, "data", "data")
        insensitive_cost = insensitive_eval._compare_arrays_unordered(arr1, arr2, "data", "data")

        # Order-insensitive should be 0 cost (perfect match via optimal matching)
        assert insensitive_cost == 0
        # Order-sensitive should have cost > 0 (2 out of 4 in wrong position)
        assert sensitive_cost > 0


class TestEdgeCases:
    """Test edge cases for order-sensitive comparison."""

    @pytest.fixture
    def evaluator(self):
        return SemanticJsonTreeConsistencyEvaluator(
            model_id='all-MiniLM-L6-v2',
            order_sensitive_fields={"steps"}
        )

    def test_empty_arrays(self, evaluator):
        """Empty arrays should be identical."""
        json1 = {"steps": []}
        json2 = {"steps": []}

        similarity = evaluator.calculate_tree_edit_distance_opt(
            json1, json2, variation_type="combined"
        )
        assert similarity == 1.0

    def test_one_empty_array(self, evaluator):
        """One empty array vs non-empty should have low similarity."""
        json1 = {"steps": []}
        json2 = {"steps": ["A", "B", "C"]}

        similarity = evaluator.calculate_tree_edit_distance_opt(
            json1, json2, variation_type="combined"
        )
        assert similarity < 1.0

    def test_single_element_arrays(self, evaluator):
        """Single element arrays comparison."""
        json1 = {"steps": ["only_step"]}
        json2 = {"steps": ["only_step"]}

        similarity = evaluator.calculate_tree_edit_distance_opt(
            json1, json2, variation_type="combined"
        )
        assert similarity == 1.0

    def test_mixed_types_in_array(self, evaluator):
        """Arrays with mixed types."""
        json1 = {"steps": ["step1", 123, True]}
        json2 = {"steps": ["step1", 123, True]}

        similarity = evaluator.calculate_tree_edit_distance_opt(
            json1, json2, variation_type="combined"
        )
        assert similarity == 1.0


class TestCompareArraysOrderedDirectly:
    """Test _compare_arrays_ordered method directly."""

    def test_direct_ordered_comparison(self):
        """Test calling _compare_arrays_ordered directly."""
        evaluator = SemanticJsonTreeConsistencyEvaluator(
            model_id='all-MiniLM-L6-v2'
        )

        arr1 = ["A", "B", "C"]
        arr2 = ["A", "B", "C"]

        cost = evaluator._compare_arrays_ordered(arr1, arr2, "test", "test")
        assert cost == 0  # Identical arrays should have 0 cost

    def test_direct_ordered_reversed(self):
        """Test ordered comparison with reversed array."""
        evaluator = SemanticJsonTreeConsistencyEvaluator(
            model_id='all-MiniLM-L6-v2'
        )

        arr1 = ["A", "B", "C"]
        arr2 = ["C", "B", "A"]

        cost = evaluator._compare_arrays_ordered(arr1, arr2, "test", "test")
        # Only middle element matches by position
        assert cost > 0

    def test_direct_ordered_length_diff(self):
        """Test ordered comparison with different lengths."""
        evaluator = SemanticJsonTreeConsistencyEvaluator(
            model_id='all-MiniLM-L6-v2'
        )

        arr1 = ["A", "B"]
        arr2 = ["A", "B", "C", "D"]

        cost = evaluator._compare_arrays_ordered(arr1, arr2, "test", "test")
        # First two match, but length diff adds cost
        assert cost > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
