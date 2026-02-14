# Causal Validation via Bidirectional Intervention: ShareGPT Results

**Date**: February 7, 2026

## Overview

This document presents causal validation results from the ShareGPT dataset experiments, extending the Toucan tool-calling results presented in Section 5 of the KDD paper. The ShareGPT experiments validate Simpson's Paradox findings on open-domain text generation tasks.

## Experiment Setup

- **Dataset**: ShareGPT (open-domain conversational prompts)
- **Model**: Claude-Sonnet-4 (us.anthropic.claude-sonnet-4-20250514-v1:0)
- **Temperatures**: 0.0, 0.5, 1.0
- **Runs per sample**: 10
- **Total samples**: 412 intervention experiments
- **Features tested**: `has_should`, `has_must`, `has_if`
- **Rewriting method**: Rule-based (deterministic)

### Bidirectional Intervention Design
- **ADD**: Add feature to prompts lacking it (e.g., add "should" to prompts without "should")
- **REMOVE**: Remove feature from prompts with it (e.g., remove "should" from prompts containing it)

## Results Summary

### Aggregate Effects by Feature

| Feature | n | Mean Delta | Std | t-stat | p-value |
|---------|---|------------|-----|--------|---------|
| has_if | 4 | +4.59% | 4.44% | 1.79 | 0.171 |
| has_must | 168 | +1.42% | 10.37% | 1.77 | 0.078 |
| has_should | 240 | +0.51% | 8.51% | 0.93 | 0.351 |

**Key Finding**: Aggregate effects appear small and non-significant, but this masks important conditional effects.

### Simpson's Paradox: Conditional Effects

The critical finding is that interventions have **opposite effects** depending on baseline consistency:

| Feature | Low Baseline (<0.8) | High Baseline (>=0.8) | Difference | p-value |
|---------|---------------------|----------------------|------------|---------|
| has_should | +1.06% (n=209) | -3.19% (n=31) | +4.25% | **0.0093** |
| has_must | +1.82% (n=156) | -3.73% (n=12) | +5.55% | 0.0749 |
| has_if | +5.25% (n=3) | +2.60% (n=1) | +2.65% | 1.0 |

**Key Insight**:
- For **has_should** (p=0.0093): Statistically significant Simpson's Paradox
  - Interventions **improve** consistency for difficult prompts (low baseline)
  - Interventions **harm** consistency for easy prompts (high baseline)

### Correlation: Delta vs Original Consistency

| Feature | Pearson r | p-value |
|---------|-----------|---------|
| has_must | -0.341 | **<0.0001** |
| has_should | -0.146 | **0.0236** |
| has_if | -0.484 | 0.516 |

**Interpretation**: Negative correlation confirms that interventions help difficult prompts more than easy ones.

### By Intervention Type

| Feature | ADD Effect | n | REMOVE Effect | n |
|---------|------------|---|---------------|---|
| has_if | +4.59% | 4 | - | 0 |
| has_must | +1.01% | 114 | +2.29% | 54 |
| has_should | +1.58% | 90 | -0.12% | 150 |

## Comparison with Toucan (Tool Calling) Results

The paper reports Toucan results with Simpson's Paradox effects up to +34.7% improvement for difficult prompts. The ShareGPT results show similar but smaller effects:

| Metric | Toucan (Paper) | ShareGPT |
|--------|----------------|----------|
| Max improvement (low baseline) | +34.7% | +5.25% |
| Max degradation (high baseline) | -4.2% | -3.73% |
| Simpson's Paradox significance | p<0.001 | p=0.009 |

**Why smaller effects in ShareGPT?**
1. ShareGPT is open-domain text generation (less structured than tool calling)
2. Consistency measured via word overlap (Jaccard) vs semantic tree similarity
3. Fewer model variations tested (1 model vs 4)

## Key Insights

### 1. Simpson's Paradox Generalizes
The conditional effects observed in Toucan (tool calling) **also appear in ShareGPT (text generation)**:
- Interventions that help difficult prompts can harm easy prompts
- Aggregate null effects mask large conditional effects

### 2. Practical Implications
- **Don't blindly apply prompt engineering** - what works for low-consistency prompts may hurt high-consistency ones
- **Monitor baseline consistency** before applying interventions
- **Feature importance varies by domain** - `has_should` is significant in ShareGPT; `must` features dominate in Toucan

### 3. Theoretical Validation
The negative correlation between delta and original consistency confirms the theoretical prediction that consistency gains are bounded and follow diminishing returns.

## Data Location

- Raw results: `/tmp/causal_intervention_sharegpt/sharegpt_results.jsonl` (EC2)
- S3: `s3://bedrock-bda-us-west-2-876d1ca6-1c81-4950-9dfe-322168df390b/causal_results/sharegpt_results.jsonl`
- Local copy: `/tmp/sharegpt_causal_results.jsonl`

## Scripts

- Intervention experiment: `scripts/experiments/causal_intervention_scaled.py`
- Analysis: `/tmp/analyze_causal.py`

## References

- KDD 2026 Paper, Section 5: "Causal Validation via Bidirectional Intervention"
- Related: `CONSISTENCY_FACTORS_ANALYSIS.md`, `PROMPT_TOOL_FEATURES_CONSISTENCY.md`
