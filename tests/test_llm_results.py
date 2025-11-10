#!/usr/bin/env python3
"""Test LLM generation results structure"""

import json

print("=" * 60)
print("Testing LLM Generation Results")
print("=" * 60)

# Check LLM generation directories
print("\n1. Checking LLM generation directories...")
llm_gen_dir = "llm_gen_results"

if os.path.exists(llm_gen_dir):
    model_dirs = [d for d in os.listdir(llm_gen_dir) if os.path.isdir(os.path.join(llm_gen_dir, d))]
    print(f"✓ Found {len(model_dirs)} model directories:")
    for model_dir in sorted(model_dirs)[:5]:
        model_path = os.path.join(llm_gen_dir, model_dir)
        result_dirs = [d for d in os.listdir(model_path) if os.path.isdir(os.path.join(model_path, d))]
        print(f"  - {model_dir}: {len(result_dirs)} result sets")
else:
    print(f"✗ {llm_gen_dir} not found")
    exit(0)

# Check a sample result file
print("\n2. Checking sample result structure...")
sample_found = False
for model_dir in model_dirs:
    model_path = os.path.join(llm_gen_dir, model_dir)
    result_dirs = [d for d in os.listdir(model_path) if os.path.isdir(os.path.join(model_path, d))]
    if result_dirs:
        sample_result_dir = os.path.join(model_path, result_dirs[0])
        result_file = os.path.join(sample_result_dir, "generation_results.json")
        if os.path.exists(result_file):
            with open(result_file, 'r') as f:
                data = json.load(f)
            print(f"✓ Sample result file: {result_file}")
            print(f"  Keys: {list(data.keys())}")
            if 'results' in data:
                print(f"  Results count: {len(data['results'])}")
            sample_found = True
            break

if not sample_found:
    print("⚠ No sample result files found")

# Check consistency metrics
print("\n3. Checking consistency metrics...")
metric_files = [
    "structural_consistency_metrics_results_v1.json",
    "content_consistency_metrics_results_v1.json"
]

for file_path in metric_files:
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)
        print(f"✓ {file_path}")
        print(f"  Models: {len(data)}")
    else:
        print(f"⚠ {file_path} not found (needs to be generated)")

print("\n" + "=" * 60)
print("✓ LLM results test completed!")
print("=" * 60)
