"""
Integration of Semantic JSON Tree Consistency Evaluation

This module demonstrates how to use the semantic JSON tree consistency evaluation
to analyze the structural consistency of JSON outputs with semantic understanding.
"""

import json
from typing import List, Dict, Any, Union
import datetime
import numpy as np

# Import both consistency modules
from json_tree_consistency import evaluate_json_structural_consistency
from semantic_json_tree_consistency import evaluate_semantic_json_consistency


def compare_consistency_methods(
    json_outputs: List[Union[str, Dict]],
    array_order_matters: bool = False,
    required_fields: List[str] = None,
    semantic_threshold: float = 0.7
) -> Dict[str, Any]:
    """
    Compare the results of both consistency evaluation methods.
    
    Args:
        json_outputs: List of JSON outputs to evaluate
        array_order_matters: Whether array order matters for comparison
        required_fields: List of required field paths
        semantic_threshold: Minimum semantic similarity threshold
        
    Returns:
        Dictionary with comparison results
    """
    # Run standard tree consistency evaluation
    standard_result = evaluate_json_structural_consistency(
        json_outputs,
        array_order_matters=array_order_matters,
        required_fields=required_fields
    )
    
    # Run semantic tree consistency evaluation
    semantic_result = evaluate_semantic_json_consistency(
        json_outputs,
        array_order_matters=array_order_matters,
        required_fields=required_fields,
        use_semantic_similarity=True,
        semantic_threshold=semantic_threshold
    )
    
    # Calculate improvement
    standard_score = standard_result.get('structural_consistency_score', 0)
    semantic_score = semantic_result.get('structural_consistency_score', 0)
    improvement = semantic_score - standard_score
    percent_improvement = (improvement / standard_score * 100) if standard_score > 0 else 0
    
    # Prepare comparison report
    comparison = {
        "timestamp": datetime.datetime.now().isoformat(),
        "num_outputs_analyzed": len(json_outputs),
        "standard_consistency_score": standard_score,
        "semantic_consistency_score": semantic_score,
        "absolute_improvement": improvement,
        "percent_improvement": percent_improvement,
        "semantic_threshold_used": semantic_threshold,
        "array_order_matters": array_order_matters,
        "required_fields": required_fields,
        "standard_perfect_consistency": standard_result.get('perfect_consistency', False),
        "semantic_perfect_consistency": semantic_result.get('perfect_consistency', False),
        "standard_operation_counts": standard_result.get('operation_counts', {}),
        "semantic_operation_counts": semantic_result.get('operation_counts', {})
    }
    
    return comparison


def analyze_json_dataset(
    json_file_path: str,
    sample_size: int = 10,
    array_order_matters: bool = False,
    semantic_threshold: float = 0.7
) -> Dict[str, Any]:
    """
    Analyze a dataset of JSON objects for structural consistency.
    
    Args:
        json_file_path: Path to JSON file containing an array of objects
        sample_size: Number of objects to sample for analysis
        array_order_matters: Whether array order matters for comparison
        semantic_threshold: Minimum semantic similarity threshold
        
    Returns:
        Dictionary with analysis results
    """
    # Load JSON dataset
    try:
        with open(json_file_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        return {"error": f"Failed to load JSON file: {str(e)}"}
    
    # Ensure we have a list of objects
    if not isinstance(data, list):
        if isinstance(data, dict) and any(isinstance(data.get(k), list) for k in data):
            # Find the first list in the dictionary
            for k, v in data.items():
                if isinstance(v, list) and len(v) > 0:
                    data = v
                    break
        else:
            return {"error": "JSON file does not contain an array of objects"}
    
    # Ensure we have objects
    data = [item for item in data if isinstance(item, dict)]
    if not data:
        return {"error": "No valid JSON objects found in the dataset"}
    
    # Sample the dataset
    if len(data) > sample_size:
        import random
        random.seed(42)  # For reproducibility
        sampled_data = random.sample(data, sample_size)
    else:
        sampled_data = data
    
    # Compare consistency methods
    comparison = compare_consistency_methods(
        sampled_data,
        array_order_matters=array_order_matters,
        semantic_threshold=semantic_threshold
    )
    
    # Add dataset info
    comparison["dataset_info"] = {
        "file_path": json_file_path,
        "total_objects": len(data),
        "sampled_objects": len(sampled_data),
        "average_object_size": np.mean([len(json.dumps(obj)) for obj in sampled_data])
    }
    
    return comparison


if __name__ == "__main__":
    # Example usage
    json1 = {
        "user_name": "John Doe",
        "user_age": 30,
        "email_address": "john.doe@example.com",
        "home_address": {
            "street_name": "123 Main St",
            "city_name": "New York",
            "postal_code": "10001"
        },
        "interests": ["reading", "swimming", "coding"],
        "is_active": True,
        "account_balance": 1500.50
    }
    
    json2 = {
        "name": "John Doe",  # Semantically similar to "user_name"
        "age": 31,  # Semantically similar to "user_age"
        "email": "john.doe@example.com",  # Semantically similar to "email_address"
        "address": {  # Semantically similar to "home_address"
            "street": "123 Main Street",  # Semantically similar to "street_name"
            "city": "New York",  # Semantically similar to "city_name"
            "zip": "10001"  # Semantically similar to "postal_code"
        },
        "hobbies": ["coding", "reading", "running"],  # Semantically similar to "interests"
        "active": True,  # Semantically similar to "is_active"
        "balance": 1499.00  # Semantically similar to "account_balance"
    }
    
    json3 = {
        "firstName": "John",
        "lastName": "Doe",
        "contact": {
            "email": "john.doe@example.com",
            "phone": "123-456-7890"
        }
    }
    
    print("=== JSON Tree Consistency Comparison ===\n")
    
    # Compare both methods
    result = compare_consistency_methods(
        [json1, json2, json3],
        array_order_matters=False,
        semantic_threshold=0.6
    )
    
    print(f"Standard Consistency Score: {result['standard_consistency_score']:.4f}")
    print(f"Semantic Consistency Score: {result['semantic_consistency_score']:.4f}")
    print(f"Improvement: {result['absolute_improvement']:.4f} ({result['percent_improvement']:.1f}%)")
    
    # Try with a real dataset if available
    try:
        print("\n=== Analyzing ShareGPT Dataset Sample ===\n")
        dataset_result = analyze_json_dataset(
            "extracted_sharegpt_data/all_conversations.json",
            sample_size=5,
            semantic_threshold=0.6
        )
        
        if "error" in dataset_result:
            print(f"Error: {dataset_result['error']}")
        else:
            print(f"Dataset: {dataset_result['dataset_info']['file_path']}")
            print(f"Objects analyzed: {dataset_result['dataset_info']['sampled_objects']}")
            print(f"Standard Consistency Score: {dataset_result['standard_consistency_score']:.4f}")
            print(f"Semantic Consistency Score: {dataset_result['semantic_consistency_score']:.4f}")
            print(f"Improvement: {dataset_result['absolute_improvement']:.4f} ({dataset_result['percent_improvement']:.1f}%)")
    except Exception as e:
        print(f"Failed to analyze dataset: {e}")