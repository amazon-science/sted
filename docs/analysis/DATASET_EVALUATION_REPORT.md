# Dataset Suitability Evaluation for Combined Similarity Testing

## Executive Summary

**Status**: ✅ **Datasets ARE suitable** (after extracting nested JSON content)

The downloaded datasets contain JSON-serialized strings in their fields that, when parsed, reveal complex nested structures suitable for evaluating the new structure-guided combined similarity approach.

## Dataset Analysis Results

### Initial Assessment (Top-level Structure)
**Result**: All datasets appeared unsuitable (depth=1, few keys)

**Issue**: The analysis only looked at the top-level wrapper structure, not the actual JSON content stored as strings inside fields like "tools" and "answers".

### Corrected Assessment (Nested JSON Content)

#### Best Dataset: **Salesforce/xlam-function-calling-60k**

**Total Size**: 60,000 examples

**Nested JSON Content**:
- **Function Schemas (tools field)**:
  - Depth: mean=3.0 (consistent)
  - Keys: mean=11, range=6-95
  - Structure: `{name, description, parameters{...}}`
  - **Suitability**: ✅ **EXCELLENT** for structural testing

- **Function Calls (answers field)**:
  - Depth: mean=2.13, range=1-7
  - Keys: mean=3.66, range=2-9
  - Structure: `{name, arguments{...}}`
  - **Suitability**: ✅ **GOOD** for content variation testing

**Example Function Schema**:
```json
{
  "name": "getmusic",
  "description": "Fetches all TikTok videos using a music track",
  "parameters": {
    "getmusic": {
      "description": "The music track identifier",
      "type": "str",
      "default": "6818239458366753542"
    }
  }
}
```

**Example Function Call**:
```json
{
  "name": "getmusic",
  "arguments": {
    "getmusic": "Shape of You"
  }
}
```

## Suitability for Testing New Combined Similarity Approach

### ✅ Excellent For:

#### 1. **Structural Matching Testing**
- Function schemas have consistent depth (3 levels)
- Complex nested parameter definitions
- Good for testing how structural matching handles nested objects

**Test Case Example**:
```python
schema1 = {"name": "getmusic", "parameters": {"track": {...}}}
schema2 = {"name": "getmusic", "parameters": {"song": {...}}}
# Test: Does combined similarity respect parameter name differences?
```

#### 2. **Value Swap Testing**
- Function calls with multiple arguments
- Can create synthetic swaps to test the short string fix

**Test Case Example**:
```python
call1 = {"name": "transfer", "arguments": {"from": "A", "to": "B"}}
call2 = {"name": "transfer", "arguments": {"from": "B", "to": "A"}}
# Test: Does combined similarity correctly show ~0.75 for value swap?
```

#### 3. **Parameter Variations**
- Many functions with similar structures but different argument values
- Good for testing content similarity within fixed structure

**Test Case Example**:
```python
call1 = {"name": "getmusic", "arguments": {"track": "X"}}
call2 = {"name": "getmusic", "arguments": {"track": "Y"}}
# Test: Content difference with perfect structure
```

#### 4. **Cross-Function Comparison**
- Different functions with overlapping parameter patterns
- Tests how combined handles structural similarity with different names

**Test Case Example**:
```python
func1 = {"name": "getmusic", "arguments": {"id": "123"}}
func2 = {"name": "getvideo", "arguments": {"id": "456"}}
# Test: Similar structure, different function names
```

### ⚠️ Limitations:

1. **Limited Short Strings** (14.3% < 4 chars)
   - Fewer opportunities to test short string fix extensively
   - Mitigation: Create synthetic short string test cases

2. **Consistent Depth** (function schemas all depth=3)
   - Less variability in structural complexity
   - Mitigation: Test across function calls (depth=1-7)

3. **Domain-Specific** (function calling patterns)
   - May not represent all JSON use cases
   - Mitigation: Use multiple datasets and synthetic variations

## Recommended Testing Strategy

### Phase 1: Baseline Comparison (100-1000 examples)
1. Sample function schema pairs
2. Calculate structural, content, and combined similarities
3. Verify combined is bounded between structural and content
4. **Expected**: 95%+ of cases should show proper bounding

### Phase 2: Value Swap Testing (synthetic)
1. Create function calls with swapped argument values
2. Test combined similarity matches expected ~0.75
3. Verify short string fix works (single-char parameter names)
4. **Expected**: Combined ≈ 0.75 for perfect structure + swapped values

### Phase 3: Structural Variation Testing
1. Compare functions with similar parameters but different names
2. Test parameter addition/removal
3. Verify structural matching guides content comparison
4. **Expected**: Combined should be closer to structural than content

### Phase 4: Real-World Consistency
1. For same function name, calculate consistency across variations
2. Measure how combined similarity changes with parameter modifications
3. Compare against human judgment of "similarity"
4. **Expected**: Combined should align with intuition better than OLD approach

## Evaluation Metrics

### Success Criteria

| Metric | Target | Method |
|--------|--------|--------|
| Bounding Rate | >95% | % of cases where min(S,C) ≤ Combined ≤ max(S,C) |
| Value Swap Detection | 0.70-0.80 | Combined similarity for swapped values |
| Short String Accuracy | >90% | Correct handling of strings < 4 chars |
| Structural Guidance | >80% | Cases where combined follows structural matching |
| Human Alignment | >0.7 | Correlation with human similarity judgments |

### Comparison Points

Compare NEW vs OLD approach:
- Paradox cases (combined < both S and C)
- Value swap cases
- Short string cases
- Overall consistency scores

## Code Implementation

```python
# Extract nested JSON from dataset
def extract_nested_json(dataset_item):
    """Extract JSON objects from string fields."""
    nested_jsons = []
    for key, value in dataset_item.items():
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
                if isinstance(parsed, (dict, list)):
                    nested_jsons.append((key, parsed))
            except:
                pass
    return nested_jsons

# Evaluation loop
from sted import STED

evaluator = STED()

for item in dataset:
    nested = extract_nested_json(item)

    # Extract function calls
    if 'answers' in dict(nested):
        calls = nested['answers']

        # Compare pairs of function calls
        for i, call1 in enumerate(calls):
            for call2 in calls[i+1:]:
                s = evaluator.calculate_tree_edit_distance(
                    call1, call2, variation_type="structural"
                )
                c = evaluator.calculate_tree_edit_distance(
                    call1, call2, variation_type="content"
                )
                combined = evaluator.calculate_tree_edit_distance(
                    call1, call2, variation_type="combined"
                )

                # Verify bounding
                assert min(s, c) <= combined <= max(s, c), "Bounding violated!"
```

## Conclusion

**Verdict**: ✅ **The Salesforce xlam-function-calling-60k dataset IS suitable** for evaluating the new structure-guided combined similarity approach.

**Key Strengths**:
1. Large sample size (60K examples)
2. Complex nested structures (depth 2-3)
3. Rich parameter variations
4. Real-world function calling patterns

**Next Steps**:
1. Implement extraction pipeline for nested JSON
2. Run Phase 1 baseline comparison (1000 examples)
3. Create synthetic test cases for value swaps
4. Measure success metrics and compare OLD vs NEW

**Estimated Evaluation Time**: 2-3 days for comprehensive testing

## Additional Recommendations

### Supplement with Synthetic Data
Create targeted synthetic test cases for:
- Extreme value swaps (all parameters swapped)
- Short string variations (single-char keys/values)
- Depth variations (1-6 levels deep)
- Type diversity (mix of strings, numbers, booleans, nulls)

### Consider Berkeley Function Calling Leaderboard (BFCL)
- Download available in the repository
- Provides diverse function calling patterns
- Can serve as additional validation dataset

### Create Evaluation Dashboard
Build a simple dashboard to visualize:
- Distribution of combined vs structural vs content scores
- Paradox case detection (combined < both)
- Value swap results
- Success metric tracking
