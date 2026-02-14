"""
Analyze the actual nested JSON content within dataset string fields.
The datasets contain JSON-serialized strings that need to be parsed.
"""

import json
import numpy as np
from pathlib import Path
from collections import Counter

DATASET_DIR = Path("/Users/guanghu/Documents/genai/projects/sted-internal/research/datasets")


def extract_json_from_strings(obj):
    """Extract JSON objects from string fields."""
    jsons = []

    if isinstance(obj, dict):
        for key, value in obj.items():
            if isinstance(value, str):
                # Try to parse as JSON
                try:
                    parsed = json.loads(value)
                    if isinstance(parsed, (dict, list)):
                        jsons.append(parsed)
                except:
                    pass
            jsons.extend(extract_json_from_strings(value))
    elif isinstance(obj, list):
        for item in obj:
            jsons.extend(extract_json_from_strings(item))

    return jsons


def count_json_depth(obj, current_depth=0):
    """Count max depth of JSON."""
    if isinstance(obj, dict):
        if not obj:
            return current_depth
        return max(count_json_depth(v, current_depth + 1) for v in obj.values())
    elif isinstance(obj, list):
        if not obj:
            return current_depth
        return max(count_json_depth(item, current_depth + 1) for item in obj)
    return current_depth


def count_keys(obj):
    """Count total keys."""
    if isinstance(obj, dict):
        count = len(obj)
        for v in obj.values():
            count += count_keys(v)
        return count
    elif isinstance(obj, list):
        return sum(count_keys(item) for item in obj)
    return 0


def analyze_xlam_dataset():
    """Analyze the XLAM function calling dataset by extracting nested JSON."""

    dataset_path = DATASET_DIR / "Salesforce_xlam-function-calling-60k.json"

    print("="*80)
    print("ANALYZING XLAM DATASET - NESTED JSON CONTENT")
    print("="*80)

    with open(dataset_path) as f:
        data = json.load(f)

    # Sample for analysis
    sample_size = 1000
    np.random.seed(42)
    sample = [data[i] for i in np.random.choice(len(data), min(sample_size, len(data)), replace=False)]

    print(f"\nSampled {len(sample)} examples from {len(data)} total")

    # Extract nested JSONs from string fields
    all_nested_jsons = []
    for item in sample:
        nested = extract_json_from_strings(item)
        all_nested_jsons.extend(nested)

    print(f"Extracted {len(all_nested_jsons)} nested JSON objects from string fields")

    if not all_nested_jsons:
        print("No nested JSON found!")
        return

    # Analyze nested structures
    depths = [count_json_depth(obj) for obj in all_nested_jsons]
    key_counts = [count_keys(obj) for obj in all_nested_jsons]

    print(f"\nNested JSON Statistics:")
    print(f"  Depth: mean={np.mean(depths):.2f}, median={np.median(depths):.0f}, range={np.min(depths)}-{np.max(depths)}")
    print(f"  Keys: mean={np.mean(key_counts):.2f}, median={np.median(key_counts):.0f}, range={np.min(key_counts)}-{np.max(key_counts)}")

    # Show examples
    print(f"\nExample nested JSON structures:")
    print("-"*80)

    for i, obj in enumerate(all_nested_jsons[:5]):
        print(f"\nExample {i+1}:")
        print(f"  Type: {type(obj).__name__}")
        print(f"  Depth: {count_json_depth(obj)}")
        print(f"  Keys: {count_keys(obj)}")
        print(f"  Sample: {json.dumps(obj, indent=2)[:300]}...")

    # Analyze tools field specifically (function schemas)
    print(f"\n" + "="*80)
    print("ANALYZING FUNCTION SCHEMAS (tools field)")
    print("="*80)

    tool_schemas = []
    for item in sample:
        try:
            if 'tools' in item and isinstance(item['tools'], str):
                tools = json.loads(item['tools'])
                if isinstance(tools, list):
                    tool_schemas.extend(tools)
        except:
            pass

    print(f"\nExtracted {len(tool_schemas)} function schemas")

    if tool_schemas:
        depths = [count_json_depth(t) for t in tool_schemas]
        keys = [count_keys(t) for t in tool_schemas]

        print(f"  Depth: mean={np.mean(depths):.2f}, range={np.min(depths)}-{np.max(depths)}")
        print(f"  Keys: mean={np.mean(keys):.2f}, range={np.min(keys)}-{np.max(keys)}")

        # Check key names
        all_keys = Counter()
        for schema in tool_schemas:
            if isinstance(schema, dict):
                all_keys.update(schema.keys())

        print(f"  Common keys: {dict(all_keys.most_common(10))}")

        print(f"\nExample function schema:")
        if tool_schemas:
            print(json.dumps(tool_schemas[0], indent=2)[:800])

    # Analyze answers field (function calls)
    print(f"\n" + "="*80)
    print("ANALYZING FUNCTION CALLS (answers field)")
    print("="*80)

    function_calls = []
    for item in sample:
        try:
            if 'answers' in item and isinstance(item['answers'], str):
                answers = json.loads(item['answers'])
                if isinstance(answers, list):
                    function_calls.extend(answers)
        except:
            pass

    print(f"\nExtracted {len(function_calls)} function calls")

    if function_calls:
        depths = [count_json_depth(c) for c in function_calls]
        keys = [count_keys(c) for c in function_calls]

        print(f"  Depth: mean={np.mean(depths):.2f}, range={np.min(depths)}-{np.max(depths)}")
        print(f"  Keys: mean={np.mean(keys):.2f}, range={np.min(keys)}-{np.max(keys)}")

        print(f"\nExample function call:")
        if function_calls:
            print(json.dumps(function_calls[0], indent=2))

    # Suitability assessment
    print(f"\n" + "="*80)
    print("SUITABILITY FOR COMBINED SIMILARITY TESTING")
    print("="*80)

    print("\nFunction Schemas (tools):")
    if tool_schemas:
        schema_depth = np.mean([count_json_depth(t) for t in tool_schemas])
        schema_keys = np.mean([count_keys(t) for t in tool_schemas])

        if schema_depth >= 2 and schema_keys >= 5:
            print("  ✓ GOOD: Complex nested structures (depth≥2, keys≥5)")
            print("  ✓ Can test structure-guided matching")
        else:
            print(f"  ⚠ LIMITED: depth={schema_depth:.1f}, keys={schema_keys:.1f}")

    print("\nFunction Calls (answers):")
    if function_calls:
        call_depth = np.mean([count_json_depth(c) for c in function_calls])
        call_keys = np.mean([count_keys(c) for c in function_calls])

        if call_depth >= 2 and call_keys >= 3:
            print("  ✓ GOOD: Structured function calls")
            print("  ✓ Can test parameter variations")
        else:
            print(f"  ⚠ LIMITED: depth={call_depth:.1f}, keys={call_keys:.1f}")

    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    print("""
For evaluating the new combined similarity approach:

1. FUNCTION SCHEMA COMPARISON:
   - Compare different function schemas with same/similar purposes
   - Test structural matching of nested parameter definitions
   - Good for testing complex structural alignment

2. FUNCTION CALL COMPARISON:
   - Compare function calls with different argument values
   - Test value swap scenarios (swapped argument names)
   - Good for testing content similarity within structure

3. CROSS-DATASET COMPARISON:
   - Compare function calls across different datasets
   - Test how combined similarity handles structural variations

4. SYNTHETIC VARIATIONS:
   - Create synthetic variations by:
     a) Swapping parameter values
     b) Renaming parameters
     c) Adding/removing optional parameters
   - Perfect for controlled testing of combined similarity
""")


if __name__ == "__main__":
    analyze_xlam_dataset()
