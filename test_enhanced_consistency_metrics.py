"""
Test script for enhanced consistency metrics in semantic JSON tree consistency

This script demonstrates the comprehensive consistency metrics for evaluating
structural consistency across multiple JSON outputs.
"""

import json
import numpy as np
from semantic_json_tree_consistency import SemanticJsonTreeConsistencyEvaluator, evaluate_semantic_json_consistency

def print_separator():
    print("\n" + "="*60 + "\n")

def print_metrics(metrics, indent=0):
    """Pretty print nested metrics dictionary."""
    indent_str = " " * indent
    for key, value in metrics.items():
        if isinstance(value, dict):
            print(f"{indent_str}{key}:")
            print_metrics(value, indent + 2)
        else:
            if isinstance(value, float):
                print(f"{indent_str}{key}: {value:.4f}")
            else:
                print(f"{indent_str}{key}: {value}")

def generate_json_variations(base_json, num_variations=5, variation_level=0.2):
    """Generate variations of a base JSON with controlled consistency."""
    variations = [base_json]
    
    for i in range(1, num_variations):
        # Create a copy of the base JSON
        variation = json.loads(json.dumps(base_json))
        
        # Apply random variations
        _apply_variations(variation, level=variation_level * i)
        variations.append(variation)
    
    return variations

def _apply_variations(json_obj, level=0.2, path=""):
    """Apply random variations to a JSON object."""
    import random
    
    if isinstance(json_obj, dict):
        # Randomly modify some keys
        keys = list(json_obj.keys())
        num_to_modify = max(1, int(len(keys) * level))
        keys_to_modify = random.sample(keys, min(num_to_modify, len(keys)))
        
        for key in keys_to_modify:
            # Different variation strategies
            strategy = random.choice(["rename", "modify", "remove", "add"])
            
            if strategy == "rename" and random.random() < level:
                # Rename key
                new_key = key + "_modified"
                json_obj[new_key] = json_obj[key]
                del json_obj[key]
            
            elif strategy == "modify":
                # Modify value
                _apply_variations(json_obj[key], level, path + "." + key)
            
            elif strategy == "remove" and random.random() < level * 0.5:
                # Remove key (less likely)
                del json_obj[key]
            
            elif strategy == "add" and random.random() < level * 0.7:
                # Add new key
                json_obj[key + "_new"] = "new_value"
        
    elif isinstance(json_obj, list) and json_obj:
        # Modify list elements
        for i in range(len(json_obj)):
            if random.random() < level:
                if isinstance(json_obj[i], (dict, list)):
                    _apply_variations(json_obj[i], level, f"{path}[{i}]")
                else:
                    # Modify primitive value
                    if isinstance(json_obj[i], str):
                        json_obj[i] = json_obj[i] + " (modified)"
                    elif isinstance(json_obj[i], (int, float)):
                        json_obj[i] = json_obj[i] * (1 + level)
        
        # Possibly add or remove elements
        if random.random() < level and json_obj:
            if random.choice([True, False]):
                # Add element
                if isinstance(json_obj[0], dict):
                    json_obj.append({})
                elif isinstance(json_obj[0], str):
                    json_obj.append("new_item")
                elif isinstance(json_obj[0], (int, float)):
                    json_obj.append(0)
            else:
                # Remove element
                json_obj.pop()
    
    elif isinstance(json_obj, str) and random.random() < level:
        # Modify string
        json_obj = json_obj + " (modified)"
    
    elif isinstance(json_obj, (int, float)) and random.random() < level:
        # Modify number
        json_obj = json_obj * (1 + level)

def test_consistency_metrics():
    """Test the enhanced consistency metrics."""
    print("Testing enhanced consistency metrics...")
    
    # Create evaluator
    evaluator = SemanticJsonTreeConsistencyEvaluator(
        use_semantic_similarity=True,
        string_method='semantic'
    )
    
    # Base JSON for testing
    base_json = {
        "user": {
            "name": "John Doe",
            "age": 30,
            "email": "john.doe@example.com",
            "address": {
                "street": "123 Main St",
                "city": "New York",
                "zip": "10001"
            },
            "preferences": {
                "theme": "dark",
                "notifications": True,
                "language": "en-US"
            }
        },
        "products": [
            {
                "id": "p1",
                "name": "Product 1",
                "price": 99.99,
                "inStock": True
            },
            {
                "id": "p2",
                "name": "Product 2",
                "price": 149.99,
                "inStock": False
            }
        ],
        "metadata": {
            "version": "1.0",
            "lastUpdated": "2025-06-20T12:00:00Z"
        }
    }
    
    print("\nTest Case 1: High Consistency (Minor Variations)")
    high_consistency_jsons = generate_json_variations(base_json, num_variations=5, variation_level=0.05)
    high_consistency_result = evaluator.evaluate_structural_consistency(high_consistency_jsons)
    
    print("High Consistency Metrics:")
    print_metrics(high_consistency_result["consistency_metrics"])
    print("\nStatistical Metrics:")
    print_metrics(high_consistency_result["statistical_metrics"])
    
    print("\nTest Case 2: Medium Consistency (Moderate Variations)")
    medium_consistency_jsons = generate_json_variations(base_json, num_variations=5, variation_level=0.2)
    medium_consistency_result = evaluator.evaluate_structural_consistency(medium_consistency_jsons)
    
    print("Medium Consistency Metrics:")
    print_metrics(medium_consistency_result["consistency_metrics"])
    print("\nStatistical Metrics:")
    print_metrics(medium_consistency_result["statistical_metrics"])
    
    print("\nTest Case 3: Low Consistency (Major Variations)")
    low_consistency_jsons = generate_json_variations(base_json, num_variations=5, variation_level=0.5)
    low_consistency_result = evaluator.evaluate_structural_consistency(low_consistency_jsons)
    
    print("Low Consistency Metrics:")
    print_metrics(low_consistency_result["consistency_metrics"])
    print("\nStatistical Metrics:")
    print_metrics(low_consistency_result["statistical_metrics"])
    
    # Compare the three cases
    print("\nComparison of Consistency Metrics:")
    print(f"{'Metric':<25} {'High':<10} {'Medium':<10} {'Low':<10}")
    print("-" * 55)
    
    metrics = [
        "mean_similarity", 
        "std_deviation", 
        "consistency_coefficient",
        "similarity_range"
    ]
    
    for metric in metrics:
        high = high_consistency_result["consistency_metrics"][metric]
        medium = medium_consistency_result["consistency_metrics"][metric]
        low = low_consistency_result["consistency_metrics"][metric]
        print(f"{metric:<25} {high:<10.4f} {medium:<10.4f} {low:<10.4f}")
    
    # Compare statistical metrics
    print("\nComparison of Statistical Metrics:")
    
    # Quartiles
    print("\nQuartiles:")
    quartile_metrics = ["median", "iqr"]
    for metric in quartile_metrics:
        high = high_consistency_result["statistical_metrics"]["quartiles"][metric]
        medium = medium_consistency_result["statistical_metrics"]["quartiles"][metric]
        low = low_consistency_result["statistical_metrics"]["quartiles"][metric]
        print(f"{metric:<25} {high:<10.4f} {medium:<10.4f} {low:<10.4f}")
    
    # Entropy and Gini
    print("\nEntropy and Gini:")
    other_metrics = ["entropy", "gini_coefficient"]
    for metric in other_metrics:
        high = high_consistency_result["statistical_metrics"][metric]
        medium = medium_consistency_result["statistical_metrics"][metric]
        low = low_consistency_result["statistical_metrics"][metric]
        print(f"{metric:<25} {high:<10.4f} {medium:<10.4f} {low:<10.4f}")
    
    # Check for outliers
    print("\nOutliers Detected:")
    print(f"High Consistency: {len(high_consistency_result['outliers'])}")
    print(f"Medium Consistency: {len(medium_consistency_result['outliers'])}")
    print(f"Low Consistency: {len(low_consistency_result['outliers'])}")
    
    if high_consistency_result['outliers']:
        print("\nSample outlier from high consistency set:")
        print_metrics(high_consistency_result['outliers'][0])

if __name__ == "__main__":
    print_separator()
    print("ENHANCED CONSISTENCY METRICS TEST")
    print_separator()
    
    try:
        test_consistency_metrics()
    except Exception as e:
        import traceback
        print(f"Error during testing: {e}")
        traceback.print_exc()
    
    print_separator()
    print("Test completed!")
    print_separator()