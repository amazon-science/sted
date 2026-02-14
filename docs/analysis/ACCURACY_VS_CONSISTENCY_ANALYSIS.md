# Accuracy vs Consistency Analysis

## Overview

This analysis examines the relationship between **accuracy** (STED similarity to ground truth) and **consistency** (c_mean) across 18 LLMs.

**Key Distinction:**
- **Validity**: Whether response was parseable/valid (infrastructure-related)
- **Accuracy**: How close generated output is to ground truth (measured via STED similarity)

## Experiment Setup

| Parameter | Value |
|-----------|-------|
| Dataset | Toucan (1006 samples) |
| Models | 18 FINAL_MODELS |
| Temperature | 0.0 |
| Runs per sample | 10 |
| Total samples | 16,415 |

## Results

### Observed Correlations

| Metric Pair | Pearson r | Interpretation |
|-------------|-----------|----------------|
| Accuracy vs c_mean | **0.894** | Strong positive |
| Accuracy vs ranking_score | 0.679 | Moderate positive |
| Accuracy vs stability_score | 0.197 | Weak |
| Accuracy vs validity_rate | 0.063 | No correlation |

### Key Observations

1. **Validity ≠ Accuracy**: Validity-accuracy correlation is r=0.063, meaning parseable output does not imply correct output.

2. **c_mean correlates with accuracy**: Models with higher c_mean tend to have higher accuracy, but this does not prove causation.

3. **Ranking score matters more than stability**: Models with high stability but low ranking (e.g., Llama-3.3-70B: stability=0.951, ranking=0.391) still have low accuracy.

## Limitations

- **Correlation ≠ Causation**: We observe that high-consistency models also have high accuracy, but cannot prove that improving consistency causes improved accuracy.
- **Confounding factors**: Better models may naturally exhibit both higher accuracy AND higher consistency.
- **No intervention data**: We did not measure the same model before/after applying a consistency intervention.

## Files

- **Script**: `scripts/analysis/evaluate_accuracy_vs_consistency.py`
- **Results**: `results/accuracy_analysis/accuracy_vs_consistency_t0.json`
