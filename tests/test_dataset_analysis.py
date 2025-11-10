#!/usr/bin/env python3
"""Test dataset analysis functionality"""

import json

print("=" * 60)
print("Testing Dataset Analysis")
print("=" * 60)

# Test 1: Check if data directories exist
print("\n1. Checking data directories...")
data_dirs = [
    "sharegpt_data/sharegpt-structured-output-json",
    "sharegpt_data/sharegpt-quizz-generation-json-output"
]

for dir_path in data_dirs:
    if os.path.exists(dir_path):
        files = [f for f in os.listdir(dir_path) if f.endswith('.json') and f.startswith('conversation_')]
        print(f"✓ {dir_path}: {len(files)} conversation files")
    else:
        print(f"✗ {dir_path}: Not found")

# Test 2: Load and validate a sample conversation
print("\n2. Loading sample conversation...")
sample_file = "sharegpt_data/sharegpt-structured-output-json/conversation_1.json"
try:
    with open(sample_file, 'r') as f:
        data = json.load(f)
    print(f"✓ Successfully loaded {sample_file}")
    print(f"  Keys: {list(data.keys())}")
    if 'conversations' in data:
        print(f"  Conversations: {len(data['conversations'])}")
except Exception as e:
    print(f"✗ Failed to load sample: {e}")

# Test 3: Check synthetic dataset directory
print("\n3. Checking synthetic dataset...")
if os.path.exists("synthetic_dataset"):
    files = [f for f in os.listdir("synthetic_dataset") if f.endswith('.json')]
    print(f"✓ synthetic_dataset exists with {len(files)} files")
    for f in files[:3]:
        print(f"  - {f}")
else:
    print("✗ synthetic_dataset directory not found")

print("\n" + "=" * 60)
print("✓ Dataset analysis test completed!")
print("=" * 60)
