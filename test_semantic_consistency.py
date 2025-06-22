"""
Test script for Semantic JSON Tree Consistency Evaluation

This script demonstrates the usage of the semantic JSON tree consistency evaluator
and compares it with the standard tree consistency evaluator.
"""

import json
import time
from typing import Dict, List, Any

# Import the evaluation functions
try:
    from json_tree_consistency import evaluate_json_structural_consistency
    from semantic_json_tree_consistency import evaluate_semantic_json_consistency
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure both json_tree_consistency.py and semantic_json_tree_consistency.py are in the current directory.")
    import sys
    sys.exit(1)


def print_separator():
    print("\n" + "="*60 + "\n")


def test_with_examples():
    """Test with example JSON objects that have semantically similar structures."""
    print("Testing with semantically similar JSON structures...")
    
    # Example 1: User profile with different key names but similar semantics
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
    
    # Example 2: Different structure but some semantic overlap
    json3 = {
        "firstName": "John",
        "lastName": "Doe",
        "contact": {
            "email": "john.doe@example.com",
            "phone": "123-456-7890"
        }
    }
    
    # Run standard evaluation
    print("\nStandard Tree Consistency Evaluation:")
    start_time = time.time()
    standard_result = evaluate_json_structural_consistency(
        [json1, json2, json3],
        array_order_matters=False
    )
    standard_time = time.time() - start_time
    
    print(f"Consistency Score: {standard_result['structural_consistency_score']:.4f}")
    print(f"Perfect Consistency: {standard_result['perfect_consistency']}")
    print(f"Execution Time: {standard_time:.4f} seconds")
    
    # Run semantic evaluation
    print("\nSemantic Tree Consistency Evaluation:")
    start_time = time.time()
    semantic_result = evaluate_semantic_json_consistency(
        [json1, json2, json3],
        array_order_matters=False,
        use_semantic_similarity=True,
        semantic_threshold=0.6
    )
    semantic_time = time.time() - start_time
    
    print(f"Consistency Score: {semantic_result['structural_consistency_score']:.4f}")
    print(f"Perfect Consistency: {semantic_result['perfect_consistency']}")
    print(f"Execution Time: {semantic_time:.4f} seconds")
    
    # Calculate improvement
    improvement = semantic_result['structural_consistency_score'] - standard_result['structural_consistency_score']
    percent_improvement = (improvement / standard_result['structural_consistency_score'] * 100) if standard_result['structural_consistency_score'] > 0 else 0
    
    print("\nComparison:")
    print(f"Absolute Improvement: {improvement:.4f}")
    print(f"Percent Improvement: {percent_improvement:.1f}%")
    print(f"Time Difference: {semantic_time - standard_time:.4f} seconds")


def test_with_product_data():
    """Test with product data that has different schemas but similar semantics."""
    print("Testing with product data examples...")
    
    # Product data with different schemas
    product1 = {
        "product_id": "P12345",
        "product_name": "Wireless Headphones",
        "product_description": "High-quality wireless headphones with noise cancellation",
        "product_price": 99.99,
        "product_category": "Electronics",
        "product_specifications": {
            "color": "Black",
            "weight": "250g",
            "battery_life": "20 hours"
        },
        "in_stock": True,
        "shipping_options": ["Standard", "Express", "Next Day"]
    }
    
    product2 = {
        "id": "P12345",
        "title": "Wireless Headphones",
        "description": "High-quality wireless headphones with noise cancellation",
        "price": 99.99,
        "category": "Electronics",
        "specs": {
            "color": "Black",
            "weight": "250g",
            "battery": "20 hours"
        },
        "available": True,
        "shipping": ["Standard", "Express", "Next Day"]
    }
    
    product3 = {
        "sku": "P12345",
        "name": "Wireless Headphones",
        "details": "High-quality wireless headphones with noise cancellation",
        "retail_price": 99.99,
        "dept": "Electronics",
        "attributes": {
            "color": "Black",
            "weight_grams": 250,
            "battery_duration_hours": 20
        },
        "stock_status": "In Stock",
        "delivery_methods": ["Standard", "Express", "Next Day"]
    }
    
    # Run standard evaluation
    print("\nStandard Tree Consistency Evaluation:")
    standard_result = evaluate_json_structural_consistency(
        [product1, product2, product3],
        array_order_matters=False
    )
    
    print(f"Consistency Score: {standard_result['structural_consistency_score']:.4f}")
    print(f"Perfect Consistency: {standard_result['perfect_consistency']}")
    
    # Run semantic evaluation
    print("\nSemantic Tree Consistency Evaluation:")
    semantic_result = evaluate_semantic_json_consistency(
        [product1, product2, product3],
        array_order_matters=False,
        use_semantic_similarity=True,
        semantic_threshold=0.6
    )
    
    print(f"Consistency Score: {semantic_result['structural_consistency_score']:.4f}")
    print(f"Perfect Consistency: {semantic_result['perfect_consistency']}")
    
    # Calculate improvement
    improvement = semantic_result['structural_consistency_score'] - standard_result['structural_consistency_score']
    percent_improvement = (improvement / standard_result['structural_consistency_score'] * 100) if standard_result['structural_consistency_score'] > 0 else 0
    
    print("\nComparison:")
    print(f"Absolute Improvement: {improvement:.4f}")
    print(f"Percent Improvement: {percent_improvement:.1f}%")


if __name__ == "__main__":
    print_separator()
    print("SEMANTIC JSON TREE CONSISTENCY EVALUATION TEST")
    print_separator()
    
    # Check if sentence-transformers is available
    try:
        import sentence_transformers
        print(f"sentence-transformers version: {sentence_transformers.__version__}")
    except ImportError:
        print("Warning: sentence-transformers not installed. Semantic similarity will be disabled.")
    
    # Check if scipy is available
    try:
        import scipy
        print(f"scipy version: {scipy.__version__}")
    except ImportError:
        print("Warning: scipy not installed. Using fallback implementation for linear assignment.")
    
    # Check if zss is available
    try:
        import zss
        print(f"zss is available")
    except ImportError:
        print("Warning: zss not installed. Using custom tree edit distance implementation.")
    
    try:
        print("\nRunning example tests...")
        test_with_examples()
        print_separator()
        print("Running product data tests...")
        test_with_product_data()
    except Exception as e:
        import traceback
        print(f"Error during testing: {e}")
        traceback.print_exc()
    
    print_separator()
    print("Test completed!")
    print_separator()