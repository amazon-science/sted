# STED Analysis Documentation

This directory contains comprehensive analysis of STED (Semantic Tree Edit Distance) for evaluating LLM output consistency and accuracy.

## Overview

The core research question: **Does higher STED consistency correlate with higher accuracy?**

**Answer: Yes, with r=0.89 at the model level and significant correlations at the sample level for 85% of models.**

---

## Analysis Documents

### 1. [Consistency-Accuracy Correlation](CONSISTENCY_ACCURACY_CORRELATION.md)

**The main analysis document** containing:
- Model-level correlation (r=0.89)
- Sample-level correlation by model
- Results for all variation types (structural, content, combined)
- Statistical analysis with Pearson and Spearman correlations

### 2. [Centroid Selection Strategy](CONSISTENCY_STRATEGY_EXPERIMENT.md)

Detailed analysis of using consistency for output selection:
- Centroid selection improves accuracy by 0.94% at T=1.0
- 75% of models improved with centroid selection
- Comparison across variation types

---

## Key Findings Summary

### Model-Level Correlation

| Metric Pair | Pearson r |
|-------------|-----------|
| accuracy vs c_mean | **0.8938** |
| accuracy vs ranking_score | 0.6785 |
| accuracy vs stability | 0.1970 |

### Sample-Level Correlation (T=1.0, Combined)

| Statistic | Value |
|-----------|-------|
| Average Spearman r | 0.45 |
| Models with significant (p<0.05) | 17/20 (85%) |
| Models with r > 0.4 | 14/20 (70%) |

### Centroid Selection Results

| Variation Type | T=1.0 Improvement | Models Improved |
|----------------|-------------------|-----------------|
| Combined | +0.94% | 75% |
| Content | +0.87% | 80% |
| Structural | +0.61% | 70% |

---

## Results Files

| File | Description |
|------|-------------|
| `results/accuracy_analysis/consistency_accuracy_correlation_toucan.json` | Sample-level correlations |
| `results/accuracy_analysis/accuracy_vs_consistency_t0.json` | Model-level correlations |
| `results/accuracy_analysis/centroid_selection_multi_temp_toucan.json` | Centroid selection results |
| `results/accuracy_analysis/accuracy_multi_temp.json` | Accuracy comparison at T=0.0 vs T=1.0 |

---

## Analysis Scripts

| Script | Purpose |
|--------|---------|
| `scripts/analysis/evaluate_consistency_accuracy_correlation.py` | Sample-level correlation |
| `scripts/analysis/evaluate_accuracy_vs_consistency.py` | Model-level correlation |
| `scripts/analysis/evaluate_centroid_selection_multi_temp.py` | Centroid selection |
| `scripts/analysis/evaluate_accuracy_multi_temp.py` | Temperature comparison |

---

## Practical Applications

1. **Model Ranking without Ground Truth**: Use c_mean as proxy for accuracy
2. **Output Selection**: Generate multiple outputs, select the centroid
3. **Quality Estimation**: Higher consistency indicates higher expected accuracy
4. **Confidence Scoring**: Low consistency may indicate uncertain or incorrect output

---

## Models Evaluated

20 LLMs across multiple families:
- **Claude**: 3.5-Haiku, 3.5-Sonnet, 3.7-Sonnet, Sonnet-4, Sonnet-4.5, Opus-4, Opus-4.5, Haiku-4.5
- **GPT**: 4.1-Mini, OSS-120B
- **Qwen**: 3-32B, 3-235B-A22B
- **Other**: Mimo-V2-Flash, Minimax-M2, Gemini-2.5-Flash-Lite, Nova-2-Lite, Mistral-Large-3-675B, Grok-4.1-Fast, Llama-3.3-70B, NemoTron-3-Nano-30B
