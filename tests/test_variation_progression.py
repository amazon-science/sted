#!/usr/bin/env python3
"""Test variation progression analysis"""

import json

print("=" * 60)
print("Testing Variation Progression Analysis")
print("=" * 60)

# Check if synthetic datasets exist
print("\n1. Checking synthetic datasets...")
dataset_files = [
    "synthetic_dataset/expression_variation_dataset_2025-08-25_07-02-22-full-dataset.json",
    "synthetic_dataset/semantic_variation_dataset_2025-08-25_04-23-54-full-dataset.json",
    "synthetic_dataset/schema_variation_dataset_2025-08-28_14-02-39-full-dataset.json"
]

for file_path in dataset_files:
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)
        print(f"✓ {os.path.basename(file_path)}")
        print(f"  Samples: {len(data)}")
        if data:
            sample = data[0]
            print(f"  Keys: {list(sample.keys())[:5]}...")
    else:
        print(f"✗ {file_path} not found")

# Check if progression results exist
print("\n2. Checking progression results...")
result_files = [
    "expression_variation_progression_results.json",
    "semantic_variation_progression_results.json"
]

for file_path in result_files:
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)
        print(f"✓ {file_path}")
        print(f"  Variation ratios: {len(data)}")
    else:
        print(f"⚠ {file_path} not found (needs to be generated)")

print("\n" + "=" * 60)
print("✓ Variation progression test completed!")
print("=" * 60)
