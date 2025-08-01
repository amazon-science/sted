# STED: Human-Aligned Consistency Evaluation Experiments

## Research Framework

### Consistency Definitions

**LLM-Aligned Inconsistency**: Surface-level changes that LLMs might produce across runs but humans would consider semantically equivalent:
- Word/expression changes with same meaning ("active" vs "enabled")
- Output format changes (nested vs flat JSON)
- Order changes in key-value pairs and array elements
- Data type changes (string "123" vs number 123)

**Human-Aligned Consistency**: Consistency evaluation that matches human intuitive judgment:
- Order changes in dictionaries and arrays are typically **not inconsistencies**
- Semantic equivalence is preserved despite surface variations
- Focus on meaning preservation rather than exact matching

### Synthetic Human Judgment Approach

**Note**: This experiment uses **synthetic data as a proxy for human judgment**. Rather than collecting expensive human annotations, we generate controlled datasets with known consistency patterns that reflect how humans would intuitively judge consistency. This approach allows us to:

- **Control for specific consistency patterns** (order changes, schema variations, semantic equivalence)
- **Generate large-scale evaluation datasets** without annotation costs
- **Test theoretical predictions** about human-aligned consistency
- **Validate method behavior** against expected human intuitions

The synthetic datasets are designed based on established cognitive principles of human consistency judgment, making them a reliable proxy for actual human annotations in this controlled experimental setting.

**Advantages of Synthetic Human Judgment Approach**:
- **Scalability**: Generate thousands of samples without annotation costs
- **Controllability**: Precise control over consistency patterns and edge cases
- **Reproducibility**: Deterministic datasets enable exact replication
- **Cost-effectiveness**: Eliminates expensive human annotation studies
- **Theoretical validation**: Direct testing of method behavior against known patterns
- **Rapid iteration**: Quick generation of new test cases for method refinement

### Research Hypothesis

**STED (Semantic Tree Edit Distance) should demonstrate human-aligned consistency evaluation by:**
- Remaining stable (low standard deviation) on semantically equivalent variations
- Showing instability (high standard deviation) only on semantically different content
- Outperforming traditional methods (DeepDiff, BERTScore) in human-alignment

## Experimental Design

### Dataset Categories

#### 1. **Order Change Dataset**
**Description**: JSON objects with identical content but different key/element ordering
```json
// Original
{"name": "John", "age": 30, "skills": ["Python", "Java", "SQL"]}

// Order Changed
{"age": 30, "skills": ["Java", "SQL", "Python"], "name": "John"}
```
**Synthetic Human Judgment**: Consistent (order doesn't affect meaning)
**Expected STED**: Stable (low std deviation)

#### 2. **Schema Change Dataset**
**Description**: Same semantic content in different structural formats (structural inconsistency)
```json
// Nested Format
{"user": {"personal": {"name": "John", "age": 30}, "skills": ["Python", "Java"]}}

// Flat Format  
{"user_name": "John", "user_age": 30, "user_skills": ["Python", "Java"]}
```
**Synthetic Human Judgment**: Inconsistent (different structural levels, despite same content)
**Expected STED**: Unstable (high std deviation due to structural changes)

**Note**: STED should only be stable when key-value pairs remain at the same structural level. When nesting levels change (e.g., `user.personal.name` vs `user_name`), this represents a structural inconsistency that should be detected.

#### 3. **Words/Expression Change - Same Meaning Dataset**
**Description**: Synonymous terms and equivalent expressions
```json
// Original
{"status": "active", "role": "administrator", "location": "New York"}

// Synonymous
{"status": "enabled", "role": "admin", "location": "NYC"}
```
**Synthetic Human Judgment**: Consistent (synonymous terms)
**Expected STED**: Stable (semantic similarity preserved)

#### 4. **Words/Expression Change - Different Meaning Dataset**
**Description**: Similar surface forms but different semantic content
```json
// Original
{"status": "active", "role": "administrator", "salary": 75000}

// Different Meaning
{"status": "inactive", "role": "user", "salary": 45000}
```
**Synthetic Human Judgment**: Inconsistent (different meanings)
**Expected STED**: Unstable (high std deviation)

### Evaluation Metrics

#### Primary Metrics

1. **Consistency Coefficient** (Most Important)
   - **Formula**: `mean([sample_mean * (1 - min((sample_std/sample_mean)^1.5, 1.0)) for each sample])`
   - **Interpretation**: Combines accuracy and stability; higher values = better overall consistency
   - **Range**: 0-1 scale where 1.0 = perfect consistency
   - **Advantage**: Penalizes both low accuracy and high variability

2. **Normalized Coefficient of Variation** (Stability Focus)
   - **Formula**: `mean([sample_std / sample_mean for each sample])`
   - **Interpretation**: Scale-independent measure of relative variability; lower = more stable
   - **Range**: 0+ where 0 = perfect stability
   - **Advantage**: Comparable across different similarity ranges

3. **Stability Score** (Interpretability)
   - **Formula**: `1.0 / (1.0 + mean_of_stds)`
   - **Interpretation**: Intuitive stability measure; higher = more stable
   - **Range**: 0-1 scale where 1.0 = perfect stability
   - **Advantage**: Easy to interpret and compare

#### Secondary Metrics

4. **Mean of Standard Deviations**: `mean_of_stds` - raw stability measure
5. **Range-Normalized Standard Deviation**: Scale-invariant stability measure  
6. **Inter-Quartile Range (IQR)**: Distribution spread analysis
7. **Temperature Correlation**: Validation of temperature-stability relationships

### Expected Results Matrix

| Dataset Type | STED (Proposed) | BERTScore | DeepDiff |
|--------------|----------------|-----------|----------|
| **Order Change** | ✅ Stable (Low Std) | ❌ Unstable (High Std) | ❌ Unstable (High Std) |
| **Schema Change** | ✅ Unstable (High Std) | ❌ Unstable (High Std) | ❌ Unstable (High Std) |
| **Same Meaning Words** | ✅ Stable (Low Std) | ✅ Stable (Low Std) | ❌ Unstable (High Std) |
| **Different Meaning Words** | ✅ Unstable (High Std) | ✅ Unstable (High Std) | ✅ Unstable (High Std) |

**Key Insight**: STED correctly identifies structural inconsistencies (Schema Change) as unstable, while being robust to meaningless variations (Order Change). This demonstrates superior discrimination between structural changes that matter vs. surface changes that don't.

### Validation Hypotheses

#### H1: STED Human-Alignment Superiority
**STED will show the most human-aligned consistency patterns:**
- Low variability on order changes (humans don't care about order)
- **High variability on schema changes (structural inconsistency should be detected)**
- Low variability on synonymous content (humans understand semantics)
- High variability on genuinely different content

**Key Distinction**: STED correctly identifies that changing structural levels (e.g., `user.name` → `user_name`) represents a meaningful inconsistency, even when semantic content is preserved.

#### H2: Traditional Method Limitations
**DeepDiff will be most unstable:**
- High variability on all surface changes (exact matching limitation)
- Cannot distinguish between meaningful and meaningless variations

**BERTScore will be moderately stable:**
- Stable on semantic content but unstable on structural changes
- Order-sensitive for arrays, structure-sensitive for objects

#### H3: LLM Output Evaluation
**On real LLM outputs (Claude 3.5 Sonnet, Haiku, etc.):**
- STED will show most stable consistency evaluation
- DeepDiff will show highest instability (sensitive to all surface changes)
- BERTScore will be intermediate (semantic understanding but structure-sensitive)

## Experimental Implementation

### Phase 1: Synthetic Data Generation

#### Dataset Creation
```python
# Generate 4 synthetic datasets with controlled variations
datasets = {
    "order_change": generate_order_variations(base_samples, n_variations=10),
    "schema_change": generate_schema_variations(base_samples, n_variations=10), 
    "same_meaning": generate_synonym_variations(base_samples, n_variations=10),
    "different_meaning": generate_semantic_variations(base_samples, n_variations=10)
}
```

#### Sample Structure
```python
{
    "sample_id": "sample_001",
    "dataset_type": "order_change",
    "ground_truth": {...},
    "variations": [
        {"variation_id": 1, "content": {...}},
        {"variation_id": 2, "content": {...}},
        # ... 10 variations total
    ],
    "human_judgment": "consistent",  # or "inconsistent"
    "expected_sted_stability": "stable"  # or "unstable"
}
```

### Phase 2: Method Comparison

#### Evaluation Pipeline
```python
for dataset_name, dataset in datasets.items():
    for method in ["sted", "bertscore", "deepdiff"]:
        # Calculate pairwise similarities for each sample
        sample_similarities = []
        for sample in dataset:
            similarities = calculate_pairwise_similarities(
                sample["variations"], method=method
            )
            sample_similarities.append(similarities)
        
        # Calculate consistency stability (mean of stds)
        consistency_stability = np.mean([
            np.std(similarities) for similarities in sample_similarities
        ])
        
        # Calculate human-aligned accuracy
        accuracy_scores = []
        for sample in dataset:
            for variation in sample["variations"]:
                accuracy = calculate_similarity(
                    variation["content"], sample["ground_truth"], method=method
                )
                accuracy_scores.append(accuracy)
        
        human_aligned_accuracy = np.mean(accuracy_scores)
        
        results[dataset_name][method] = {
            "consistency_stability": consistency_stability,
            "human_aligned_accuracy": human_aligned_accuracy,
            "coefficient_of_variation": consistency_stability / human_aligned_accuracy
        }
```

### Phase 3: LLM Output Validation

#### Real-World Testing
```python
# Generate outputs using different LLM models
models = [
    "claude-3-5-sonnet-20241022-v2:0",
    "claude-3-5-sonnet-20240620-v1:0", 
    "claude-3-5-haiku-20241022-v1:0"
]

for model in models:
    # Generate multiple outputs for same prompts
    outputs = generate_llm_outputs(
        prompts=test_prompts,
        model=model,
        n_runs=10,
        temperature=0.7
    )
    
    # Evaluate consistency using all methods
    for method in ["sted", "bertscore", "deepdiff"]:
        stability_scores = evaluate_consistency(outputs, method=method)
        
        llm_results[model][method] = {
            "mean_stability": np.mean(stability_scores),
            "stability_variance": np.var(stability_scores),
            "human_correlation": correlate_with_human_judgment(stability_scores)
        }
```

## Success Criteria

### Quantitative Validation

1. **STED Human-Alignment**: 
   - Low std deviation on order/schema/synonym datasets (< 0.1)
   - High std deviation only on different-meaning dataset (> 0.3)

2. **Traditional Method Limitations**:
   - DeepDiff: High std deviation on all datasets (> 0.4)
   - BERTScore: Mixed performance, high std on structural changes

3. **Statistical Significance**:
   - ANOVA F-test showing significant differences between methods
   - Post-hoc tests confirming STED superiority in human-alignment

### Qualitative Validation

1. **Human Annotation Study**:
   - Collect human consistency judgments on subset of data
   - Calculate correlation between method scores and human judgments
   - STED should show highest correlation with human intuition

2. **Error Analysis**:
   - Identify cases where traditional methods fail but STED succeeds
   - Demonstrate semantic understanding capabilities
   - Show robustness to surface-level variations

## Expected Impact

### Research Contributions

1. **Novel Evaluation Paradigm**: Human-aligned consistency evaluation framework
2. **Methodological Advancement**: STED algorithm with semantic understanding
3. **Empirical Validation**: Comprehensive comparison on controlled datasets
4. **Practical Applications**: Better LLM output evaluation for production systems

### Practical Benefits

1. **Reduced False Negatives**: Fewer cases where humans see consistency but metrics don't
2. **Improved Model Selection**: Better guidance for choosing LLM parameters
3. **Production Reliability**: More accurate assessment of LLM output stability
4. **Human-Centric AI**: Evaluation methods aligned with human cognitive patterns

## Implementation Timeline

- **Week 1-2**: Synthetic dataset generation and validation
- **Week 3-4**: STED algorithm implementation and optimization  
- **Week 5-6**: Comparative evaluation on synthetic datasets
- **Week 7-8**: LLM output generation and real-world validation
- **Week 9-10**: Human annotation study and correlation analysis
- **Week 11-12**: Results analysis, paper writing, and submission

This experimental framework provides rigorous validation of STED's human-aligned consistency evaluation capabilities while clearly demonstrating the limitations of existing approaches.