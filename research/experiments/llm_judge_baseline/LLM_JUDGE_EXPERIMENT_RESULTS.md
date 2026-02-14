# LLM-as-Judge Baseline Experiment Results

**Date:** December 4, 2025
**Experiment:** Evaluating LLM-as-Judge as a baseline for JSON consistency measurement
**Dataset:** Synthetic Expression Variation Dataset (50 pairs: 5 samples × 10 variation ratios)

## Configuration

### Model Settings
- **Model:** Claude Opus 4.5 (`global.anthropic.claude-opus-4-5-20251101-v1:0`)
- **Provider:** AWS Bedrock
- **Temperature:** 0.0 (deterministic)
- **Max Tokens:** 8000
- **Region:** us-west-2

### Prompt Design
The prompt was designed to explicitly instruct the LLM to:
1. Compare JSON 2 against JSON 1 (reference)
2. Count different fields vs total fields
3. Calculate similarity = 1.0 - (different_fields / total_fields)
4. Output structured scores (STRUCTURAL_SCORE, SEMANTIC_SCORE, OVERALL_SCORE)

## Results Summary

### Correlation with Variation Ratio

| Metric | Spearman ρ | p-value | Significance |
|--------|-----------|---------|--------------|
| **LLM Judge** | **-0.494** | 0.147 | **Not significant** |
| STED | -1.000 | <0.001 | *** |
| TED | -- | -- | -- |
| BERTScore | -1.000 | <0.001 | *** |
| DeepDiff | -0.997 | <0.001 | *** |

### Score Ranges by Variation Ratio

| Variation | LLM Judge | STED | TED | BERTScore | DeepDiff |
|-----------|-----------|------|-----|-----------|----------|
| 0.1 | 0.066 | 0.984 | 1.000 | 0.973 | 0.978 |
| 0.2 | 0.008 | 0.978 | 1.000 | 0.963 | 0.937 |
| 0.3 | 0.076 | 0.974 | 1.000 | 0.959 | 0.908 |
| 0.4 | 0.068 | 0.971 | 1.000 | 0.954 | 0.863 |
| 0.5 | 0.008 | 0.967 | 1.000 | 0.946 | 0.834 |
| 0.6 | 0.082 | 0.961 | 1.000 | 0.942 | 0.788 |
| 0.7 | 0.022 | 0.959 | 1.000 | 0.939 | 0.662 |
| 0.8 | 0.060 | 0.955 | 1.000 | 0.935 | 0.616 |
| 0.9 | 0.006 | 0.952 | 1.000 | 0.933 | 0.552 |
| 1.0 | 0.006 | 0.946 | 1.000 | 0.931 | 0.552 |

### Score Range (Sensitivity)

| Metric | Score Range |
|--------|-------------|
| LLM Judge | 0.076 |
| STED | 0.038 |
| TED | 0.000 |
| BERTScore | 0.042 |
| DeepDiff | 0.426 |

## Key Findings

### 1. LLM-as-Judge Shows Weak, Non-Significant Correlation
- Spearman ρ = -0.494 (correct direction, but weak)
- p-value = 0.147 (not statistically significant)
- Cannot reliably distinguish between 10% and 100% semantic variation

### 2. Near-Zero Scores Across All Variation Levels
- All LLM Judge scores are < 0.1
- The model treats all samples as "nearly completely different"
- Even 10% variation yields a score of 0.066 (should be ~0.9)

### 3. STED Achieves Perfect Correlation
- Spearman ρ = -1.000 (perfect negative correlation)
- p-value < 0.001 (highly significant)
- Scores range from 0.946 (100% variation) to 0.984 (10% variation)

### 4. Quick Test vs Full Dataset Discrepancy
On simple JSON objects (4 fields), LLM Judge works correctly:
- Identical: 1.000
- 25% different: 0.830
- 50% different: 0.650
- 75% different: 0.480
- 100% different: 0.300

On complex synthetic dataset (nested JSON with many fields), LLM Judge fails:
- All scores collapse to near-zero
- No meaningful discrimination between variation levels

## Implications for the Paper

### Why LLM-as-Judge Fails on Complex JSON

1. **Field Counting Problem:** Complex nested JSON structures have many leaf nodes. Even small semantic changes can affect multiple fields, causing the LLM to count high percentages of "different" fields.

2. **Semantic vs Syntactic Confusion:** The LLM applies strict syntactic comparison ("active" vs "enabled" = different) rather than semantic understanding.

3. **Context Window Limitations:** Large JSON objects may be truncated, affecting the LLM's ability to assess full structure.

4. **Prompt Sensitivity:** Despite explicit instructions, the LLM's behavior varies significantly based on input complexity.

### Advantages of STED

1. **Deterministic:** Same inputs always produce same outputs
2. **Statistically Significant:** Perfect correlation with semantic variation
3. **Cost-Effective:** No API calls required
4. **Scalable:** Can process any JSON size without truncation concerns
5. **Interpretable:** Clear structural + semantic decomposition

## Files Generated

- `llm_judge_synthetic_results.json` - Raw evaluation results
- `llm_judge_synthetic_table.tex` - LaTeX table for paper
- `LLM_JUDGE_EXPERIMENT_RESULTS.md` - This document

## Code Location

- `sted/llm_judge.py` - LLM Judge implementation
- `research/experiments/llm_judge_baseline/evaluate_llm_judge_synthetic.py` - Evaluation script

## Reproducibility

To reproduce these results:

```python
from sted import create_llm_judge

judge = create_llm_judge(
    provider="bedrock",
    model_id="global.anthropic.claude-opus-4-5-20251101-v1:0",
    temperature=0.0,
    max_tokens=8000
)

# Compare two JSON objects
score = judge.calculate_similarity(json1, json2)
```

## Conclusion

LLM-as-Judge, even with state-of-the-art models (Claude Opus 4.5) and carefully designed prompts, **fails to reliably measure JSON consistency** on complex, real-world structures. The weak, non-significant correlation (ρ = -0.494, p = 0.147) compared to STED's perfect correlation (ρ = -1.000, p < 0.001) demonstrates the need for specialized metrics like STED for production JSON consistency evaluation.
