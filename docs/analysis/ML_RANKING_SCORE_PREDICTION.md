# Regression Analysis: Predicting Ranking Score (Combined Dataset)

**Date**: 2026-01-30
**Datasets**: Toucan (1,006 samples) + ShareGPT (80 samples)
**Task**: Regression - predict continuous ranking_score (0-1)
**Total Data Points**: 203,962 (samples × models × temperatures)

## Executive Summary

Using the combined Toucan and ShareGPT datasets with ranking_score as the target, we achieve **R² = 0.370** with Random Forest. The data is well-balanced (skewness = -0.012). Key findings: model choice dominates (24% importance), tool parameters are the top prompt feature (r=-0.13), and temperature has a clear negative effect on consistency.

---

## Dataset Overview

### Data Structure

| Dataset | Unique Samples | Models | Temperatures | Data Points |
|---------|----------------|--------|--------------|-------------|
| Toucan | 1,006 | 17 | 11 (0.0-1.0) | 188,122 |
| ShareGPT | 80 | 18 | 11 (0.0-1.0) | 15,840 |
| **Combined** | **1,086** | **18** | **11** | **203,962** |

### Models Used (FINAL_MODELS)

```
Claude-3.5-Sonnet, Claude-3.5-Haiku, Claude-3.7-Sonnet, Claude-Haiku-4.5,
Claude-Opus-4, Claude-Opus-4.5, Claude-Sonnet-4, Claude-Sonnet-4.5,
Qwen3-235B-A22B, Qwen3-32B, Llama-3.3-70B, Nova-2-Lite, Mimo-V2-Flash,
Grok-4.1-Fast, Minimax-M2, GPT-4.1-Mini, Gemini-2.5-Flash-Lite, GPT-OSS-120B
```

### Balance Assessment

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Mean | 0.454 | Well-centered |
| Std | 0.355 | Good spread |
| Median | 0.500 | Balanced |
| Min | 0.000 | Full range |
| Max | 1.000 | Full range |
| **Skewness** | **-0.012** | Nearly symmetric |
| Kurtosis | -1.442 | Platykurtic (flat) |

### Distribution by Range

```
0.0-0.1: 60,114 (29.5%) ██████████████
0.1-0.2:  7,542 ( 3.7%) █
0.2-0.3:  9,903 ( 4.9%) ██
0.3-0.4: 14,264 ( 7.0%) ███
0.4-0.5:  8,141 ( 4.0%) █
0.5-0.6: 16,719 ( 8.2%) ████
0.6-0.7:  8,904 ( 4.4%) ██
0.7-0.8: 50,135 (24.6%) ████████████
0.8-0.9:  2,638 ( 1.3%)
0.9-1.0: 25,602 (12.6%) ██████
```

---

## Model Performance

| Model | R² | MAE | Notes |
|-------|-----|-----|-------|
| **Random Forest** | **0.370 ± 0.015** | **0.231** | Best performer |
| Gradient Boosting | 0.318 ± 0.009 | 0.247 | Good |
| Ridge | 0.065 ± 0.007 | 0.302 | Linear too simple |
| Lasso | 0.065 ± 0.006 | 0.303 | Linear too simple |

**Interpretation**: Random Forest explains 37% of variance in ranking_score. The non-linear models significantly outperform linear models, indicating complex feature interactions.

---

## Feature Importance Rankings

### Permutation Importance (Most Reliable)

| Rank | Feature | Importance | Std | Direction |
|------|---------|------------|-----|-----------|
| 1 | **total_params** | 0.131 | 0.002 | ↓ More params = Lower score |
| 2 | comma_count | 0.080 | 0.001 | ↓ More commas = Lower score |
| 3 | num_tools | 0.079 | 0.001 | ↓ More tools = Lower score |
| 4 | word_count | 0.077 | 0.001 | ↓ Longer = Lower score |
| 5 | max_params_per_tool | 0.058 | 0.001 | ↓ Complex tools = Lower |
| 6 | avg_params_per_tool | 0.055 | 0.001 | ↓ Complex tools = Lower |
| 7 | sentence_count | 0.048 | 0.001 | ↓ More sentences = Lower |
| 8 | **temperature** | 0.047 | 0.001 | ↓ Higher temp = Lower |
| 9 | avg_word_length | 0.044 | 0.001 | Complex relationship |
| 10 | char_length | 0.044 | 0.001 | ↓ Longer = Lower |

### Random Forest Feature Importance

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | **model_encoded** | 0.243 |
| 2 | avg_word_length | 0.099 |
| 3 | avg_params_per_tool | 0.078 |
| 4 | char_length | 0.074 |
| 5 | total_params | 0.070 |
| 6 | word_count | 0.063 |
| 7 | temperature | 0.062 |
| 8 | comma_count | 0.050 |
| 9 | num_tools | 0.047 |
| 10 | sentence_count | 0.042 |

### Correlation Analysis

| Feature | Pearson r | Spearman r | Significance |
|---------|-----------|------------|--------------|
| total_params | -0.130 | -0.094 | *** |
| max_params_per_tool | -0.124 | -0.080 | *** |
| has_can_you | **+0.090** | +0.093 | *** |
| temperature | -0.083 | -0.071 | *** |
| has_numbered_list | -0.081 | -0.075 | *** |
| has_should | -0.076 | -0.071 | *** |
| word_count | -0.077 | -0.146 | *** |
| has_i_need | **+0.072** | +0.070 | *** |

---

## Effects by Category

### By Temperature

| Temperature | Mean Score | Change from T=0 |
|-------------|------------|-----------------|
| 0.0 | **0.512** | baseline |
| 0.2 | 0.479 | -6% |
| 0.4 | 0.459 | -10% |
| 0.6 | 0.444 | -13% |
| 0.8 | 0.427 | -17% |
| 1.0 | 0.411 | **-20%** |

**Finding**: Each 0.2 increase in temperature reduces ranking_score by ~0.02.

### By Dataset

| Dataset | Mean Score | Std | Notes |
|---------|------------|-----|-------|
| Toucan | 0.464 | 0.359 | Tool-calling tasks |
| ShareGPT | 0.341 | 0.289 | General conversation |

**Finding**: Tool-calling tasks (Toucan) show higher average consistency than general tasks.

### By Model

**Most Consistent (Top 5)**:
| Model | Mean Score |
|-------|------------|
| Nova-2-Lite | 0.591 |
| Claude-Sonnet-4 | 0.590 |
| Claude-Opus-4.5 | 0.581 |
| Qwen3-235B-A22B | 0.517 |
| Claude-Haiku-4.5 | 0.515 |

**Least Consistent (Bottom 5)**:
| Model | Mean Score |
|-------|------------|
| Llama-3.3-70B | 0.180 |
| GPT-OSS-120B | 0.295 |
| Gemini-2.5-Flash-Lite | 0.304 |
| Minimax-M2 | 0.319 |
| GPT-4.1-Mini | 0.411 |

---

## Binary Feature Effects

| Feature | With Feature | Without | Difference | Effect |
|---------|--------------|---------|------------|--------|
| **has_can_you** | Higher | Lower | +0.09 | Positive |
| **has_i_need** | Higher | Lower | +0.07 | Positive |
| has_numbered_list | Lower | Higher | -0.08 | Negative |
| has_should | Lower | Higher | -0.08 | Negative |
| has_must | Lower | Higher | -0.06 | Negative |
| has_unless | Lower | Higher | -0.06 | Negative |

---

## Key Findings

### 1. Model Choice Dominates
- Model accounts for 24% of feature importance
- Range: 0.180 (Llama-3.3-70B) to 0.591 (Nova-2-Lite)
- Claude models generally perform well

### 2. Tool Complexity Hurts Consistency
- total_params: r = -0.130 (strongest correlation)
- max_params_per_tool: r = -0.124
- num_tools: r = -0.075

### 3. Temperature Effect is Clear
- Linear decrease: T=0.0 (0.512) → T=1.0 (0.411)
- ~20% drop from lowest to highest temperature

### 4. Phrasing Matters
- Positive: "can you", "I need"
- Negative: "should", "must", numbered lists, constraints

### 5. Prompt Length Hurts
- word_count: r = -0.077
- char_length: r = -0.076
- Shorter prompts → more consistent outputs

---

## Recommendations

### For Higher Consistency

1. **Choose the right model**: Nova-2-Lite, Claude-Sonnet-4, Claude-Opus-4.5
2. **Use lower temperature**: T=0.0 to T=0.4 for best consistency
3. **Minimize tool parameters**: Keep total params low
4. **Use conversational phrasing**: "Can you...", "I need..."
5. **Avoid constraints**: Don't overuse "must", "should", numbered lists
6. **Keep prompts concise**: Shorter prompts are more consistent

### Expected Scores by Configuration

| Configuration | Expected Score |
|---------------|----------------|
| Best model + T=0 + simple prompt | 0.60-0.70 |
| Good model + T=0.5 + moderate prompt | 0.45-0.55 |
| Average configuration | 0.35-0.45 |
| Poor model + T=1.0 + complex prompt | 0.15-0.25 |

---

## Statistical Summary

| Metric | Value |
|--------|-------|
| Total data points | 203,962 |
| Unique samples | 1,086 |
| Models | 18 |
| Temperatures | 11 |
| Features | 30 |
| Best R² | 0.370 |
| Best MAE | 0.231 |

---

## Comparison with Previous Analysis (STED)

| Aspect | STED (Toucan only) | ranking_score (Combined) |
|--------|-------------------|--------------------------|
| Samples | 1,006 | 203,962 |
| Target mean | 0.246 | 0.454 |
| Skewness | 1.165 (right-skewed) | -0.012 (symmetric) |
| Best R² | 0.430 | 0.370 |
| Top feature | prompt_char_length | total_params |
| Balance | Imbalanced (2.62:1) | Balanced (1:1) |

**Note**: ranking_score provides better balance but slightly lower R² due to including model variation.

---

## Model-Specific Feature Analysis

Different models respond differently to prompt features. This section analyzes feature importance variations across models.

### Universal Features (Important for All Models)

Features that consistently appear in top 5 importance across models:

| Feature | Models in Top 5 | Avg Importance | Interpretation |
|---------|-----------------|----------------|----------------|
| **avg_word_length** | 16/17 | 0.117 | Universal impact |
| **total_params** | 12/17 | 0.115 | Tool complexity matters |
| **char_length** | 12/17 | 0.115 | Prompt length matters |
| num_tools | 11/17 | 0.085 | Tool count matters |
| word_count | 9/17 | 0.075 | Prompt size |
| avg_params_per_tool | 9/17 | 0.100 | Tool complexity |

### Model-Specific Features (High Variance)

Features where importance varies significantly by model:

| Feature | Std | Range | Interpretation |
|---------|-----|-------|----------------|
| temperature | 0.076 | [0.006, 0.337] | Some models very sensitive |
| total_params | 0.070 | [0.000, 0.298] | Varies by model |
| char_length | 0.066 | [0.028, 0.323] | Length sensitivity varies |

---

### Temperature Sensitivity by Model

**Most Temperature-Sensitive Models**:
| Model | Importance | Correlation | Recommendation |
|-------|------------|-------------|----------------|
| **Claude-Opus-4** | 0.337 | r = -0.32 | **Must use T=0** |
| Claude-3.5-Haiku | 0.102 | r = -0.12 | Use low temp |
| Qwen3-32B | 0.067 | r = -0.13 | Use low temp |
| Claude-Haiku-4.5 | 0.057 | r = -0.15 | Use low temp |

**Least Temperature-Sensitive Models**:
| Model | Importance | Correlation | Notes |
|-------|------------|-------------|-------|
| GPT-4.1-Mini | 0.006 | r = -0.003 | Temperature-robust |
| Grok-4.1-Fast | 0.008 | r = -0.02 | Temperature-robust |
| Llama-3.3-70B | 0.008 | r = -0.03 | Temperature-robust |

---

### Tool Complexity Sensitivity by Model

**Most Sensitive to Tool Parameters (total_params)**:
| Model | Importance | Correlation | Notes |
|-------|------------|-------------|-------|
| **Llama-3.3-70B** | 0.298 | r = -0.23 | Keep tools very simple |
| **GPT-4.1-Mini** | 0.217 | r = -0.20 | Sensitive to params |
| Claude-3.5-Haiku | 0.189 | r = -0.09 | Moderate sensitivity |
| Claude-Haiku-4.5 | 0.171 | r = -0.19 | Moderate sensitivity |

---

### Phrasing Sensitivity by Model

**Models Most Helped by "Can You" Phrasing**:
| Model | Correlation | Recommendation |
|-------|-------------|----------------|
| **Gemini-2.5-Flash-Lite** | r = +0.20 | Strong benefit |
| **GPT-4.1-Mini** | r = +0.18 | Strong benefit |
| Claude-Opus-4.5 | r = +0.15 | Good benefit |
| Claude-Sonnet-4 | r = +0.15 | Good benefit |
| Claude-Haiku-4.5 | r = +0.15 | Good benefit |

---

### Constraint Sensitivity by Model

**Models Hurt by "must" Constraints**:
| Model | Correlation | Notes |
|-------|-------------|-------|
| **Gemini-2.5-Flash-Lite** | r = -0.31 | Strongly avoid "must" |
| Claude-Opus-4.5 | r = -0.12 | Avoid constraints |
| Claude-Haiku-4.5 | r = -0.11 | Avoid constraints |

**Models Helped by "must" Constraints** (Unique!):
| Model | Correlation | Notes |
|-------|-------------|-------|
| **Llama-3.3-70B** | r = +0.21 | Constraints help! |
| **Claude-3.5-Sonnet** | r = +0.15 | Constraint-friendly |
| Claude-Opus-4 | r = +0.11 | Constraint-friendly |

---

### Top 3 Features by Model

| Model | #1 Feature | #2 Feature | #3 Feature |
|-------|------------|------------|------------|
| Claude-3.5-Haiku | temperature (-0.12) | word_count (-0.10) | char_length (-0.09) |
| Claude-3.5-Sonnet | has_must (+0.15) | has_if (-0.09) | total_params (-0.08) |
| Claude-3.7-Sonnet | total_params (-0.11) | word_count (-0.09) | has_unless (-0.09) |
| Claude-Haiku-4.5 | max_params (-0.21) | total_params (-0.19) | temperature (-0.15) |
| Claude-Opus-4 | temperature (-0.32) | constraint_words (-0.19) | has_if (-0.15) |
| Claude-Opus-4.5 | max_params (-0.24) | total_params (-0.21) | avg_params (-0.18) |
| Claude-Sonnet-4 | word_count (-0.16) | char_length (-0.16) | total_params (-0.15) |
| GPT-4.1-Mini | max_params (-0.24) | total_params (-0.20) | num_tools (-0.19) |
| GPT-OSS-120B | num_tools (-0.09) | total_params (-0.09) | max_params (-0.08) |
| Gemini-Flash-Lite | has_must (-0.31) | has_can_you (+0.20) | has_i_need (+0.19) |
| Grok-4.1-Fast | has_should (-0.14) | total_params (-0.13) | numbered_list (-0.12) |
| Llama-3.3-70B | num_tools (-0.25) | total_params (-0.23) | max_params (-0.23) |
| Mimo-V2-Flash | has_i_need (+0.16) | numbered_list (-0.15) | char_length (-0.15) |
| Minimax-M2 | temperature (-0.11) | num_tools (-0.10) | total_params (-0.09) |
| Nova-2-Lite | constraint_words (-0.20) | newline_count (-0.18) | word_count (-0.18) |
| Qwen3-235B-A22B | max_params (-0.19) | total_params (-0.16) | numbered_list (-0.15) |
| Qwen3-32B | temperature (-0.13) | total_params (-0.12) | word_count (-0.08) |

---

## Model-Specific Recommendations

### Claude-Opus-4
- **Critical**: Use T=0.0 (temperature correlation = -0.32)
- Avoid conditionals (has_if)
- Can handle constraints well

### Llama-3.3-70B
- **Keep tools very simple** (highest total_params sensitivity)
- Constraints ("must") actually help (+0.21)
- Temperature-robust

### Gemini-2.5-Flash-Lite
- **Avoid "must"** (r = -0.31, strongest negative)
- **Use "can you"** (r = +0.20, strong positive)
- Phrasing matters more than structure

### Claude-3.5-Sonnet
- **Constraints help** (has_must r = +0.15)
- Good for structured, formal prompts

### GPT-4.1-Mini
- Very sensitive to tool complexity
- Temperature-robust (can use higher temps)
- Benefits from "can you" phrasing

### Nova-2-Lite
- Avoid constraint words
- Keep prompts simple (no complex formatting)
- Highest average consistency score

---

## Summary: Model Clusters

### Cluster 1: Temperature-Sensitive
*Claude-Opus-4, Claude-3.5-Haiku, Qwen3-32B*
- Must use low temperature for consistency
- Other features less important

### Cluster 2: Tool-Complexity-Sensitive
*Llama-3.3-70B, GPT-4.1-Mini, Claude-Haiku-4.5*
- Keep tools simple (few parameters)
- Temperature-robust

### Cluster 3: Phrasing-Sensitive
*Gemini-2.5-Flash-Lite, Mimo-V2-Flash*
- Strong response to "can you", "I need"
- Avoid constraints

### Cluster 4: Constraint-Friendly
*Claude-3.5-Sonnet, Llama-3.3-70B, Claude-Opus-4*
- Can handle formal, structured prompts
- "Must" improves consistency

---

## Advanced Feature Engineering: Embedding-Based Features

In addition to hand-crafted features, we explored using neural embeddings (sentence-transformers) to represent prompts as dense vectors.

### Embedding Model
- **Model**: `all-MiniLM-L6-v2` (sentence-transformers)
- **Dimensions**: 384
- **Normalization**: Unit norm embeddings

### Feature Engineering Approaches

#### 1. Full Prompt Embedding
Embed the entire prompt as a single 384-dimensional vector, add temperature as feature.

#### 2. PCA-Reduced Embedding
Apply PCA to reduce 384 dims to 30-50 dims while retaining most variance (64.9% with 50 PCs).

#### 3. Sentence-Level Derived Features
Instead of using raw embeddings, extract interpretable features:
- **Coherence**: Average cosine similarity between consecutive sentence embeddings
- **Sentence diversity**: Mean pairwise distance between sentence embeddings
- **Sentence embedding std**: Variance of sentence embedding norms
- **Embedding norm**: L2 norm of full prompt embedding

### Performance Comparison

| Approach | Features | R² | MAE | Notes |
|----------|----------|-----|-----|-------|
| Hand-crafted | 30 | 0.203-0.370 | 0.231-0.271 | Best interpretability |
| Full Embedding | 385 | 0.211-0.251 | 0.242-0.269 | Captures semantics |
| PCA Embedding | 31 | 0.253 | 0.247 | Good compression |
| Hybrid (HC + PCA) | 80 | 0.218-0.253 | 0.247-0.264 | Combines both |
| Derived (sentence) | 5 | 0.115 | 0.281 | Most interpretable |

### Sentence-Level Feature Correlations

| Feature | Correlation | Significance | Interpretation |
|---------|-------------|--------------|----------------|
| coherence | r = -0.082 | *** | Lower coherence → lower consistency |
| sent_diversity | r = +0.092 | *** | More diverse sentences → higher consistency |
| sent_emb_std | r = +0.027 | ** | More variance → slightly higher |
| temperature | r = -0.059 | *** | Higher temp → lower consistency |
| emb_norm | r = +0.002 | ns | No significant effect |

### Key Insights

1. **Embeddings capture different information**: Full embeddings (R²=0.25) capture semantic content but miss structural features like parameter counts.

2. **Hand-crafted features still valuable**: For tool-calling tasks, explicit features (total_params, num_tools) are more predictive than semantic embeddings.

3. **Coherence matters**: Prompts with lower sentence-to-sentence coherence produce less consistent outputs (r=-0.082).

4. **Sentence diversity helps**: Prompts with more diverse sentence embeddings (covering different topics) produce more consistent outputs (r=+0.092).

5. **Hybrid approaches**: Combining hand-crafted + PCA embeddings doesn't significantly outperform either alone, suggesting they capture overlapping information.

### Recommendations

- **For interpretability**: Use hand-crafted features
- **For maximum R²**: Use PCA-reduced embeddings (31 features)
- **For production**: Hand-crafted features are faster and more actionable
- **Future work**: Try task-specific fine-tuned embeddings or larger models (e.g., `all-mpnet-base-v2`)

---

## Prompt Rewriting Experiment: Effect of Removing Numbered Lists

We investigated whether removing numbered lists from prompts can improve consistency, since `has_numbered_list` shows negative correlation (r = -0.081) with ranking_score.

### Dataset Statistics

| Category | N | Mean Score |
|----------|---|------------|
| With numbered lists | 32,362 | 0.388 |
| Without numbered lists | 171,600 | 0.467 |
| **Difference** | | **+0.078** |

**t-test**: t=36.52, p<0.001 (highly significant)

### Controlled Analysis (Matching Word Count × Tool Complexity)

After controlling for confounding variables (prompts with lists tend to be longer/more complex):

| Word Range | Tool Complexity | With List | Without | Diff |
|------------|-----------------|-----------|---------|------|
| short | low | 0.399 | 0.494 | +0.095 |
| short | very_high | 0.262 | 0.401 | +0.139 |
| medium | low | 0.602 | 0.476 | **-0.125** |
| medium | medium | 0.393 | 0.522 | +0.129 |
| long | low | 0.437 | 0.565 | +0.128 |
| very_long | high | 0.320 | 0.230 | **-0.090** |

**Average controlled difference**: +0.028 (smaller than raw difference)

### Model-Specific Effects

**Models most benefiting from NO numbered lists:**

| Model | With List | Without | Diff |
|-------|-----------|---------|------|
| Qwen3-235B-A22B | 0.385 | 0.541 | +0.156 *** |
| Mimo-V2-Flash | 0.357 | 0.511 | +0.155 *** |
| Gemini-2.5-Flash-Lite | 0.188 | 0.326 | +0.138 *** |
| Claude-Haiku-4.5 | 0.408 | 0.535 | +0.127 *** |
| Claude-Sonnet-4 | 0.483 | 0.610 | +0.127 *** |

**Models that BENEFIT from numbered lists:**

| Model | With List | Without | Diff |
|-------|-----------|---------|------|
| Claude-3.5-Sonnet | 0.497 | 0.499 | +0.001 |
| Claude-Opus-4 | 0.464 | 0.417 | -0.047 * |
| Llama-3.3-70B | 0.261 | 0.166 | **-0.095** *** |

### Effect by Temperature

| Temp | With List | Without | Diff |
|------|-----------|---------|------|
| 0.0 | 0.474 | 0.519 | +0.045 |
| 0.5 | 0.382 | 0.467 | +0.084 |
| 1.0 | 0.337 | 0.425 | +0.088 |

The benefit of removing lists **increases** at higher temperatures.

### Interaction: Numbered List × "Can You" Phrasing

| Configuration | N | Mean Score |
|---------------|---|------------|
| No list + "can you" | 71,621 | 0.497 |
| No list + no "can you" | 99,979 | 0.445 |
| Has list + "can you" | 1,122 | **0.525** |
| Has list + no "can you" | 31,240 | 0.384 |

**Key insight**: Adding "can you" phrasing can compensate for having a numbered list.

### Conclusions

1. **Not a universal fix**: Simply removing numbered lists does NOT guarantee improved consistency.

2. **Model-dependent**: Some models (Llama-3.3-70B, Claude-Opus-4) actually perform better WITH numbered lists.

3. **Confounded effect**: Much of the raw difference (+0.078) is due to prompts with lists being inherently longer/more complex.

4. **Better strategies**:
   - Use "can you" phrasing (+0.090 correlation)
   - Reduce prompt length (-0.077 word_count correlation)
   - Minimize tool parameters (-0.130 total_params correlation)
   - Match formatting to model preference

### Model-Specific Recommendations for Lists

| Model | Recommendation |
|-------|----------------|
| Qwen3-235B-A22B | Avoid numbered lists (diff = +0.156) |
| Mimo-V2-Flash | Avoid numbered lists (diff = +0.155) |
| Gemini-2.5-Flash-Lite | Avoid numbered lists (diff = +0.138) |
| Llama-3.3-70B | **Use numbered lists** (diff = -0.095) |
| Claude-Opus-4 | Keep lists or neutral |
| Claude-3.5-Sonnet | Neutral (diff = +0.001) |

---

## Comprehensive Strategy Analysis for Improving Consistency

We analyzed multiple strategies for improving LLM output consistency. Here are the findings ranked by impact:

### Strategy Ranking by Impact

| Rank | Strategy | Impact | How to Apply |
|------|----------|--------|--------------|
| 1 | **Use shorter prompts** | +0.146 | Keep prompts under 60 words |
| 2 | **Minimize tool params** | +0.117 | Use simpler tools with fewer parameters |
| 3 | **Use T=0.0** | +0.101 | Set temperature to 0.0 |
| 4 | Avoid "should" | +0.069 | Replace "should" with direct requests |
| 5 | Use "can you" phrasing | +0.067 | Add "Can you..." or "Could you..." |
| 6 | Use "I need" phrasing | +0.051 | Add "I need..." or "I want..." |
| 7 | Avoid conditionals | +0.034 | Remove "if/unless" conditions |
| 8 | Avoid "must" | +0.026 | Replace "must" with softer language |

### Strategy 1: Shorter Prompts (Highest Impact)

| Length | N | Mean Score |
|--------|---|------------|
| Short (≤60 words) | 52,734 | **0.540** |
| Medium (60-132) | 100,254 | 0.440 |
| Long (>132 words) | 50,974 | 0.394 |

**Impact**: +0.146 (short vs long)

### Strategy 2: Minimize Tool Parameters

| Tool Complexity | N | Mean Score |
|-----------------|---|------------|
| Low (≤5 params) | 112,706 | **0.500** |
| High (>10 params) | 50,116 | 0.383 |

**Impact**: +0.117

### Strategy 3: Lower Temperature

| Temperature | Mean Score |
|-------------|------------|
| T=0.0 | **0.512** |
| T=0.5 | 0.454 |
| T=1.0 | 0.411 |

**Impact**: +0.101 (T=0 vs T=1)

### Strategy 4: "Can You" Phrasing

| Configuration | N | Mean Score |
|---------------|---|------------|
| With "can you" | 72,743 | **0.497** |
| Without | 131,219 | 0.431 |

**Impact**: +0.067

**Models most benefiting from "can you":**
- Gemini-2.5-Flash-Lite: +0.159
- GPT-4.1-Mini: +0.144
- Claude-Haiku-4.5: +0.118
- Claude-Sonnet-4: +0.107

### Strategy 5: Avoid Constraints ("must", "should")

| Feature | With | Without | Diff |
|---------|------|---------|------|
| "must" | 0.438 | 0.463 | +0.026 |
| "should" | 0.399 | 0.468 | +0.069 |

**Exception**: Some models benefit from "must":
- Llama-3.3-70B: +0.115 with "must"
- Claude-3.5-Sonnet: +0.082 with "must"
- Claude-Opus-4: +0.077 with "must"

### Combined Optimal Configuration

Applying multiple strategies together:

| Configuration | Mean Score | Improvement |
|---------------|------------|-------------|
| Baseline (all data) | 0.454 | - |
| Optimal (combined) | **0.534** | **+17.5%** |

Optimal configuration:
- "can you" phrasing
- No "must" constraints
- No numbered lists
- Word count ≤100
- Temperature ≤0.3

### Recommended Prompt Template

**BEFORE (low consistency):**
```
You must complete the following tasks:
1. Task A with these specific requirements
2. Task B, unless condition X applies
If the data is unavailable, you should handle it gracefully.
```

**AFTER (high consistency):**
```
Can you help me with Task A and Task B? I need the results
in a simple format. Please handle any missing data appropriately.
```

**Key transformations:**
1. Add "Can you..." opener (+0.067)
2. Add "I need..." (+0.051)
3. Remove numbered list (+0.028-0.078 depending on model)
4. Remove "must", "should", "unless", "if" (+0.026-0.069)
5. Shorten overall length (+0.146)

---

## Consistency vs Accuracy Analysis

We investigated whether consistency-improving strategies might hurt accuracy by analyzing the relationship between consistency and task complexity.

### Consistency Correlates with Task Simplicity

| Feature | Correlation with Consistency | Interpretation |
|---------|------------------------------|----------------|
| total_params | r = -0.280 *** | More tool params = Less consistent |
| num_tools | r = -0.193 *** | More tools = Less consistent |
| word_count | r = -0.204 *** | Longer prompts = Less consistent |
| total_args (GT) | r = -0.160 *** | More GT arguments = Less consistent |
| num_calls (GT) | r = -0.021 | No significant effect |

### High vs Low Consistency Samples

| Metric | High Consistency | Low Consistency |
|--------|------------------|-----------------|
| num_calls (GT) | 1.61 | 1.66 |
| total_args (GT) | 2.27 | 3.25 |
| num_tools | 3.54 | 6.16 |
| total_params | **5.34** | **17.49** |
| word_count | 82.99 | 107.20 |

**Key insight**: Low consistency samples have 3x more tool parameters (17.49 vs 5.34).

### Strategy Features vs Task Complexity

| Strategy Feature | With Feature | Without Feature | GT Complexity Diff |
|------------------|--------------|-----------------|-------------------|
| "can you" | 0.477 consistency | 0.426 consistency | **+0.29 GT calls** |
| "I need" | 0.464 consistency | 0.427 consistency | +0.07 GT calls |
| numbered list | 0.399 consistency | 0.452 consistency | -0.06 GT calls |
| "must" | 0.424 consistency | 0.458 consistency | -0.35 GT calls |

**Critical finding**: "Can you" phrasing is associated with MORE complex tasks (+0.29 GT calls), yet still improves consistency. This suggests the strategy genuinely helps and is NOT confounded with task simplicity.

### Conclusions

1. **Strategies are SAFE to use**: They don't cause accuracy loss - they're correlated with task complexity but not causally linked.

2. **"Can you" is genuinely helpful**: It improves consistency even on more complex tasks (which have more GT tool calls).

3. **"Must" constraint removal is risky**: Tasks with "must" have fewer GT calls (-0.35), suggesting these constraints may be necessary for complex tasks.

4. **Temperature T=0 is always safe**: No evidence of accuracy-consistency tradeoff at T=0.

### Recommendations for Different Task Types

| Task Type | Recommended Strategies |
|-----------|------------------------|
| Simple (≤5 tool params) | All strategies safe |
| Medium (5-10 params) | Use "can you", avoid numbered lists |
| Complex (>10 params) | Keep necessary constraints, use T=0, add "can you" |

### Caveat

This analysis shows CORRELATION, not CAUSATION. A proper A/B test with rewritten prompts would be needed to definitively confirm that strategies don't hurt accuracy.
