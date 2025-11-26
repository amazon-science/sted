#!/usr/bin/env python3
"""
Test script to verify public release is working correctly.
Tests all core functionality after cleanup.
"""

import sys
import warnings

def test_imports():
    """Test that all public imports work"""
    print("Testing imports...")
    from sted import (
        STED,
        SemanticJsonTreeConsistencyEvaluator,
        StructuralConsistencyAnalyzer,
        JsonNode,
        StringSimilarityCache,
        collect_all_values,
        count_json_elements
    )
    print("✓ All imports successful")
    return True

def test_json_node():
    """Test JsonNode tree construction"""
    print("Testing JsonNode...")
    from sted import JsonNode
    
    data = {'name': 'John', 'items': [1, 2, 3]}
    node = JsonNode.from_dict(data)
    reconstructed = node.reconstruct_json()
    
    assert reconstructed == data, f"Reconstruction failed: {reconstructed}"
    print("✓ JsonNode working")
    return True

def test_similarity_cache():
    """Test similarity cache"""
    print("Testing StringSimilarityCache...")
    from sted import StringSimilarityCache
    
    cache = StringSimilarityCache()
    cache.set("hello", "world", 0.5)
    assert cache.get("hello", "world") == 0.5
    assert cache.get("world", "hello") == 0.5  # Symmetric
    assert cache.get("foo", "bar") is None
    print("✓ StringSimilarityCache working")
    return True

def test_utils():
    """Test utility functions"""
    print("Testing utils...")
    from sted import collect_all_values, count_json_elements
    
    data = {'a': 1, 'b': {'c': 'hello'}, 'd': [1, 2]}
    
    values = collect_all_values(data)
    assert 'hello' in values
    
    count = count_json_elements(data)
    assert count > 0
    print(f"✓ Utils working (collected {len(values)} values, {count} elements)")
    return True

def test_evaluator_init():
    """Test evaluator initialization with local model"""
    print("Testing SemanticJsonTreeConsistencyEvaluator init (local model)...")
    from sted import SemanticJsonTreeConsistencyEvaluator
    
    # Use local sentence transformer (no AWS needed)
    evaluator = SemanticJsonTreeConsistencyEvaluator(
        model_id='all-MiniLM-L6-v2'
    )
    assert evaluator.embedding_model is not None
    print("✓ Evaluator initialized with local model")
    return True

def test_similarity_methods():
    """Test available similarity methods"""
    print("Testing similarity methods...")
    from sted import SemanticJsonTreeConsistencyEvaluator
    
    evaluator = SemanticJsonTreeConsistencyEvaluator(model_id='all-MiniLM-L6-v2')
    
    expected_methods = ['ted', 'sted', 'bertscore', 'deepdiff', 'deepdiff_opt']
    for method in expected_methods:
        assert method in evaluator.calculate_similarity_method, f"Missing method: {method}"
    
    # Verify experimental methods are removed
    removed_methods = ['gnn', 'llm_judge']
    for method in removed_methods:
        assert method not in evaluator.calculate_similarity_method, f"Experimental method not removed: {method}"
    
    print(f"✓ All {len(expected_methods)} methods available, experimental methods removed")
    return True

def test_sted_calculation():
    """Test STED similarity calculation"""
    print("Testing STED calculation...")
    from sted import SemanticJsonTreeConsistencyEvaluator
    
    evaluator = SemanticJsonTreeConsistencyEvaluator(model_id='all-MiniLM-L6-v2')
    
    json1 = {'name': 'John', 'age': 30}
    json2 = {'name': 'John', 'age': 30}
    json3 = {'name': 'Jane', 'age': 25}
    
    # Identical should be 1.0
    sim_identical = evaluator.calculate_tree_edit_distance_opt(json1, json2, variation_type="combined")
    assert sim_identical == 1.0, f"Identical JSONs should have similarity 1.0, got {sim_identical}"
    
    # Different should be < 1.0
    sim_different = evaluator.calculate_tree_edit_distance_opt(json1, json3, variation_type="combined")
    assert 0 < sim_different < 1.0, f"Different JSONs should have similarity between 0 and 1, got {sim_different}"
    
    print(f"✓ STED calculation working (identical={sim_identical:.2f}, different={sim_different:.2f})")
    return True

def test_variation_types():
    """Test different variation types"""
    print("Testing variation types...")
    from sted import SemanticJsonTreeConsistencyEvaluator
    
    evaluator = SemanticJsonTreeConsistencyEvaluator(model_id='all-MiniLM-L6-v2')
    
    json1 = {'user_name': 'Alice', 'city': 'New York'}
    json2 = {'userName': 'Alice', 'location': 'NYC'}
    
    for vtype in ['structural', 'content', 'combined']:
        sim = evaluator.calculate_tree_edit_distance_opt(json1, json2, variation_type=vtype)
        assert 0 <= sim <= 1.0, f"Invalid similarity for {vtype}: {sim}"
        print(f"  {vtype}: {sim:.4f}")
    
    print("✓ All variation types working")
    return True

def test_deepdiff_methods():
    """Test DeepDiff similarity methods"""
    print("Testing DeepDiff methods...")
    from sted import SemanticJsonTreeConsistencyEvaluator
    
    evaluator = SemanticJsonTreeConsistencyEvaluator(model_id='all-MiniLM-L6-v2')
    
    json1 = {'a': 1, 'b': 'hello'}
    json2 = {'a': 2, 'b': 'world'}
    
    sim1 = evaluator.calculate_similarity_with_deepdiff(json1, json2)
    sim2 = evaluator.calculate_similarity_with_deepdiff_opt(json1, json2, variation_type="combined")
    
    assert 0 <= sim1 <= 1.0
    assert 0 <= sim2 <= 1.0
    print(f"✓ DeepDiff methods working (basic={sim1:.2f}, opt={sim2:.2f})")
    return True

def test_batch_consistency():
    """Test batch consistency evaluation"""
    print("Testing batch consistency...")
    from sted import SemanticJsonTreeConsistencyEvaluator, StructuralConsistencyAnalyzer
    
    evaluator = SemanticJsonTreeConsistencyEvaluator(model_id='all-MiniLM-L6-v2')
    analyzer = StructuralConsistencyAnalyzer(evaluator)
    
    outputs = [
        {'name': 'Alice', 'age': 25},
        {'name': 'Alice', 'age': 25},
        {'name': 'Alice', 'age': 26},
    ]
    
    result = analyzer.evaluate_structural_consistency(outputs, method_name="sted", variation_type="combined")
    
    assert 'consistency_metrics' in result
    assert 'supporting_stats' in result
    print(f"✓ Batch consistency working (coefficient={result['consistency_metrics']['consistency_coefficient']:.4f})")
    return True

def test_analyzer_with_ground_truth():
    """Test consistency evaluation with ground truth"""
    print("Testing analyzer with ground truth...")
    from sted import SemanticJsonTreeConsistencyEvaluator, StructuralConsistencyAnalyzer
    
    evaluator = SemanticJsonTreeConsistencyEvaluator(model_id='all-MiniLM-L6-v2')
    analyzer = StructuralConsistencyAnalyzer(evaluator)
    
    gt = {'name': 'Alice', 'age': 25}
    outputs = [
        {'name': 'Alice', 'age': 25},
        {'name': 'Alice', 'age': 26},
    ]
    
    result = analyzer.evaluate_structural_consistency(outputs, gt=gt, method_name="sted")
    
    assert result['has_ground_truth'] == True
    assert 'consistency_metrics' in result
    print(f"✓ Analyzer with ground truth working")
    return True

def test_field_level_consistency():
    """Test field-level consistency evaluation"""
    print("Testing field-level consistency...")
    from sted import SemanticJsonTreeConsistencyEvaluator, StructuralConsistencyAnalyzer
    
    evaluator = SemanticJsonTreeConsistencyEvaluator(model_id='all-MiniLM-L6-v2')
    analyzer = StructuralConsistencyAnalyzer(evaluator)
    
    outputs = [
        {'name': 'Alice', 'age': 25, 'city': 'NYC'},
        {'name': 'Alice', 'age': 25, 'city': 'New York'},
        {'name': 'Alice', 'age': 26, 'city': 'NYC'},
    ]
    
    result = analyzer.evaluate_field_level_consistency(outputs)
    
    assert 'field_level_metrics' in result
    assert 'overall_field_consistency' in result
    print(f"✓ Field-level consistency working (overall={result['overall_field_consistency']:.4f})")
    return True

def test_collect_string_pairs():
    """Test string pair collection"""
    print("Testing collect_all_string_pairs...")
    from sted import SemanticJsonTreeConsistencyEvaluator, StructuralConsistencyAnalyzer
    
    evaluator = SemanticJsonTreeConsistencyEvaluator(model_id='all-MiniLM-L6-v2')
    analyzer = StructuralConsistencyAnalyzer(evaluator)
    
    outputs = [{'a': 'hello'}, {'a': 'world'}]
    pairs = analyzer.collect_all_string_pairs(outputs)
    
    assert len(pairs) > 0
    assert all(isinstance(p, tuple) and len(p) == 2 for p in pairs)
    print(f"✓ String pair collection working ({len(pairs)} pairs)")
    return True

def test_no_experimental_imports():
    """Verify experimental modules are not in public API"""
    print("Testing experimental modules are not in public API...")
    from sted import __all__
    
    experimental = ['gnn', 'pdc_metric', 'probabilistic_consistency', 'calculate_gnn_similarity', 'llm_judge']
    for exp in experimental:
        assert exp not in __all__, f"{exp} should not be in public API"
    
    # Verify evaluator doesn't have experimental methods
    from sted import SemanticJsonTreeConsistencyEvaluator
    evaluator = SemanticJsonTreeConsistencyEvaluator(model_id='all-MiniLM-L6-v2')
    
    assert 'gnn' not in evaluator.calculate_similarity_method
    assert 'llm_judge' not in evaluator.calculate_similarity_method
    
    print("✓ Experimental modules correctly removed from public API")
    return True

def run_all_tests():
    """Run all tests"""
    print("=" * 60)
    print("STED Public Release Test Suite")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_json_node,
        test_similarity_cache,
        test_utils,
        test_evaluator_init,
        test_similarity_methods,
        test_sted_calculation,
        test_variation_types,
        test_deepdiff_methods,
        test_batch_consistency,
        test_analyzer_with_ground_truth,
        test_field_level_consistency,
        test_collect_string_pairs,
        test_no_experimental_imports,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ {test.__name__} failed with error: {e}")
            failed += 1
        print()
    
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0

if __name__ == "__main__":
    # Suppress warnings for cleaner output
    warnings.filterwarnings('ignore')
    
    success = run_all_tests()
    sys.exit(0 if success else 1)
