#!/usr/bin/env python3
"""
Basic usage examples for STED Consistency Library
"""


from sted.semantic_json_tree_consistency import SemanticJsonTreeConsistencyEvaluator
from sted.structural_consistency_analyzer import StructuralConsistencyAnalyzer

# Example 1: Basic similarity calculation
def example_basic_similarity():
    """Calculate similarity between two JSON structures"""
    print("=" * 60)
    print("Example 1: Basic Similarity Calculation")
    print("=" * 60)
    
    evaluator = SemanticJsonTreeConsistencyEvaluator(
        model_id='amazon.titan-embed-text-v2:0'
    )
    
    json1 = {'name': 'John', 'age': 30, 'city': 'New York'}
    json2 = {'name': 'John', 'age': 30, 'location': 'NYC'}
    
    similarity = evaluator.calculate_tree_edit_distance_opt(
        json1, json2, 
        variation_type="combined"
    )
    
    print(f"JSON1: {json1}")
    print(f"JSON2: {json2}")
    print(f"Similarity: {similarity:.4f}\n")


# Example 2: Structural consistency focus
def example_structural_consistency():
    """Focus on structural similarity"""
    print("=" * 60)
    print("Example 2: Structural Consistency")
    print("=" * 60)
    
    evaluator = SemanticJsonTreeConsistencyEvaluator(
        model_id='amazon.titan-embed-text-v2:0'
    )
    
    json1 = {'user': {'name': 'Alice', 'age': 25}}
    json2 = {'user': {'name': 'Bob', 'age': 30}}
    
    similarity = evaluator.calculate_tree_edit_distance_opt(
        json1, json2,
        variation_type="structural"
    )
    
    print(f"JSON1: {json1}")
    print(f"JSON2: {json2}")
    print(f"Structural Similarity: {similarity:.4f}\n")


# Example 3: Content consistency focus
def example_content_consistency():
    """Focus on content/semantic similarity"""
    print("=" * 60)
    print("Example 3: Content Consistency")
    print("=" * 60)
    
    evaluator = SemanticJsonTreeConsistencyEvaluator(
        model_id='amazon.titan-embed-text-v2:0'
    )
    
    json1 = {'description': 'A red car'}
    json2 = {'description': 'A crimson automobile'}
    
    similarity = evaluator.calculate_tree_edit_distance_opt(
        json1, json2,
        variation_type="content"
    )
    
    print(f"JSON1: {json1}")
    print(f"JSON2: {json2}")
    print(f"Content Similarity: {similarity:.4f}\n")


# Example 4: Batch consistency evaluation
def example_batch_consistency():
    """Evaluate consistency across multiple JSON outputs"""
    print("=" * 60)
    print("Example 4: Batch Consistency Evaluation")
    print("=" * 60)
    
    evaluator = SemanticJsonTreeConsistencyEvaluator(
        model_id='amazon.titan-embed-text-v2:0'
    )
    analyzer = StructuralConsistencyAnalyzer(evaluator)
    
    # Multiple LLM outputs for the same prompt
    json_outputs = [
        {'name': 'Alice', 'age': 25, 'city': 'New York'},
        {'name': 'Alice', 'age': 25, 'city': 'NYC'},
        {'name': 'Alice', 'age': 25, 'location': 'New York City'},
    ]
    
    result = analyzer.evaluate_structural_consistency(
        json_outputs,
        method_name="ted",
        variation_type="combined"
    )
    
    print(f"Number of outputs: {len(json_outputs)}")
    print(f"Mean similarity: {result['supporting_stats']['mean_similarity']:.4f}")
    print(f"Consistency coefficient: {result['consistency_metrics']['consistency_coefficient']:.4f}")
    print(f"Stability score: {result['consistency_metrics']['stability_score']:.4f}\n")


if __name__ == "__main__":
    example_basic_similarity()
    example_structural_consistency()
    example_content_consistency()
    example_batch_consistency()
