# Key Factors Affecting LLM Consistency

**Date**: 2026-01-30
**Datasets**: Toucan (225,344 samples), ShareGPT (20,320 samples)
**Models**: 21 (Toucan), 24 (ShareGPT)
**Method**: STED Consistency Metrics with MiniLM embeddings

## Executive Summary

This analysis identifies the key factors affecting LLM output consistency across two benchmark datasets. The **sample characteristics** (42.8% of variance) dominate over **model choice** (10%) and **temperature** (<0.1%). Empty responses and semantic spread (d_std) are the strongest predictors of inconsistency.

## Dataset Overview

| Metric | Toucan | ShareGPT |
|--------|--------|----------|
| Total samples | 225,344 | 20,320 |
| Unique prompts | 1,006 | 80 |
| Models tested | 21 | 24 |
| Mean consistency | 0.246 | 0.383 |
| Std consistency | 0.317 | 0.273 |
| Mean validity rate | 87.2% | 95.6% |

**Key observation**: ShareGPT shows 56% higher consistency than Toucan, likely due to simpler task structure.

---

## Key Findings

### Finding 1: Sample Characteristics Dominate

**Variance decomposition shows sample effects are the primary driver:**

| Source | Toucan | ShareGPT |
|--------|--------|----------|
| Between samples | 42.8% | 27.7% |
| Between models | 10.0% | 12.4% |
| Between temperatures | 0.02% | 0.06% |

**Implication**: The inherent difficulty/ambiguity of the task matters more than which model you choose or what temperature you use.

---

### Finding 2: Empty Responses Devastate Consistency

**Toucan Dataset:**
- Samples with empty responses: 19.5%
- Consistency (with empty): **0.083**
- Consistency (no empty): **0.286**
- Effect size (Cohen's d): **0.78** (large)

**ShareGPT Dataset:**
- Samples with empty responses: 10.2%
- Consistency (with empty): **0.330**
- Consistency (no empty): **0.389**
- Effect size (Cohen's d): **0.21** (small)

**Statistical significance**: p < 0.001 for both datasets

**Implication**: Empty/failed responses are a major source of inconsistency, especially for complex tasks (Toucan).

---

### Finding 3: Semantic Spread (d_std) is Highly Predictive

Lower d_std means responses are more semantically similar to each other.

| d_std Range | Toucan Consistency | ShareGPT Consistency |
|-------------|-------------------|---------------------|
| 0 - 0.1 | 0.186 | 0.417 |
| 0.1 - 0.2 | 0.106 | 0.147 |
| 0.2 - 0.3 | 0.051 | 0.062 |
| 0.3 - 0.4 | 0.010 | 0.009 |
| 0.4+ | 0.001 | 0.001 |

**Correlation**: r = -0.23 (Toucan), r = -0.31 (ShareGPT), both p < 0.001

**Implication**: When responses spread far apart semantically, consistency drops dramatically.

---

### Finding 4: Temperature Effect is Minimal

| Dataset | T=0.0 | T=1.0 | Drop |
|---------|-------|-------|------|
| Toucan | 0.252 | 0.240 | 4.9% |
| ShareGPT | 0.385 | 0.369 | 4.3% |

**Correlation with consistency**: r = -0.01 (Toucan), r = -0.02 (ShareGPT)

**Implication**: Temperature has a statistically significant but practically negligible effect on consistency. This challenges the common assumption that lower temperatures guarantee more consistent outputs.

---

### Finding 5: Model Family Differences

**Top Performers by Family (Consistency):**

| Rank | Toucan | ShareGPT |
|------|--------|----------|
| 1 | Amazon (0.331) | Meta (0.538) |
| 2 | Minimax (0.295) | Amazon (0.450) |
| 3 | xAI (0.298) | Alibaba (0.418) |
| 4 | Alibaba (0.294) | Anthropic (0.400) |
| 5 | Anthropic (0.262) | Google (0.388) |

**Lowest Performers:**
- Toucan: Meta (0.059), Mistral (0.110), OpenAI (0.184)
- ShareGPT: Nvidia (0.175), Mistral (0.319), Minimax (0.324)

**ANOVA result**: Model family effect is highly significant (p < 0.001) for both datasets.

---

### Finding 6: Model-Temperature Interaction

Some models are more temperature-sensitive than others:

**Most Temperature-Sensitive (Toucan):**
1. Minimax-M2: -17.7% drop from T=0 to T=1
2. Mimo-V2-Flash: -15.6%
3. Claude-3.5-Haiku: -13.6%

**Most Temperature-Stable:**
1. GPT-4.1-Mini: 0% change
2. Gemini-2.5-Flash-Lite: +5.1% (improves!)
3. Nova-2-Lite: -1.1%

**Implication**: Temperature tuning matters for some models but not others.

---

### Finding 7: Stability Score Correlation

| Dataset | Stable (=1.0) | Unstable (<1.0) | Correlation |
|---------|---------------|-----------------|-------------|
| Toucan | 57.9% | 42.1% | r = 0.35 |
| ShareGPT | 10.0% | 90.0% | r = 0.32 |

Stable samples (all responses valid) have higher consistency, but the relationship is moderate.

---

## Correlation Summary

| Factor | Correlation with Consistency | Effect |
|--------|------------------------------|--------|
| mean_similarity | r = +0.76*** | Strong positive |
| c_adj | r = +0.74*** | Strong positive |
| stability_score | r = +0.33*** | Moderate positive |
| validity_rate | r = +0.26*** | Moderate positive |
| empty_ratio | r = -0.26*** | Moderate negative |
| d_std | r = -0.24*** | Moderate negative |
| normalized_cv | r = -0.24*** | Moderate negative |
| temperature | r = -0.01*** | Negligible |

***: p < 0.001

---

## Practical Recommendations

### For Practitioners

1. **Focus on task design over model selection**: Sample characteristics explain 4x more variance than model choice
2. **Handle empty responses**: Implement retry logic or fallback mechanisms
3. **Don't over-optimize temperature**: The effect is minimal (<5%)
4. **Monitor semantic spread**: High d_std is an early warning sign of inconsistency

### For Researchers

1. **Report sample-level variance**: Aggregate metrics hide important variation
2. **Control for empty responses**: They confound consistency measurements
3. **Test model-temperature interactions**: Effects vary significantly by model
4. **Use both datasets**: Toucan (complex) and ShareGPT (simple) show different patterns

---

## Limitations

1. **Embedding model dependency**: Results use MiniLM; different embeddings may yield different patterns
2. **Task type confounding**: Toucan (tool calling) vs ShareGPT (text generation) have structural differences
3. **Model version variation**: Some models tested at different versions
4. **No prompt-level features**: Analysis doesn't include prompt complexity metrics

---

## Key Takeaways

1. **Sample > Model > Temperature** in terms of variance explained
2. **Empty responses** are the biggest consistency killer (d = 0.78 for Toucan)
3. **Semantic spread (d_std)** is a reliable predictor of inconsistency
4. **Temperature effect is overrated** - only 4-5% drop from T=0 to T=1
5. **Model families differ significantly** - choose based on your task type
6. **Some models improve with temperature** - test before assuming lower is better

---

## Appendix: Statistical Tests

| Test | Dataset | Result | Interpretation |
|------|---------|--------|----------------|
| Kruskal-Wallis (model family) | Toucan | H=20232, p<0.001 | Model family matters |
| Kruskal-Wallis (model family) | ShareGPT | H=1864, p<0.001 | Model family matters |
| T-test (empty vs no empty) | Toucan | t=-125, p<0.001 | Empty responses hurt |
| T-test (empty vs no empty) | ShareGPT | t=-9.3, p<0.001 | Empty responses hurt |
| Pearson (temp vs consistency) | Toucan | r=-0.01, p<0.001 | Minimal effect |
| Pearson (temp vs consistency) | ShareGPT | r=-0.02, p=0.002 | Minimal effect |
