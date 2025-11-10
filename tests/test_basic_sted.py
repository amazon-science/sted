#!/usr/bin/env python3
"""Test basic STED functionality as described in README"""


from sted.semantic_json_tree_consistency import SemanticJsonTreeConsistencyEvaluator

print("=" * 60)
print("Testing Basic STED Similarity Calculation")
print("=" * 60)

# Initialize evaluator with embedding model
print("\n1. Initializing evaluator...")
try:
    evaluator = SemanticJsonTreeConsistencyEvaluator(
        model_id='amazon.titan-embed-text-v2:0',
    )
    print("✓ Evaluator initialized successfully")
except Exception as e:
    print(f"✗ Failed to initialize evaluator: {e}")
    exit(1)

# Compare two JSON structures
print("\n2. Comparing JSON structures...")
json1 = {'name': 'John', 'age': 30, 'city': 'New York'}
json2 = {'name': 'John', 'age': 30, 'location': 'NYC'}

print(f"JSON1: {json1}")
print(f"JSON2: {json2}")

try:
    # Calculate similarity using STED
    similarity = evaluator.calculate_tree_edit_distance_opt(
        json1, json2, 
        variation_type="combined"
    )
    
    print(f"\n✓ Calculation successful!")
    print(f"Similarity score: {similarity:.4f}")
    
except Exception as e:
    print(f"✗ Failed to calculate similarity: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n" + "=" * 60)
print("✓ Basic STED test completed successfully!")
print("=" * 60)
