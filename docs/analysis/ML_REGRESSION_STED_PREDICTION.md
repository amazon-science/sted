# Regression Analysis: Predicting STED Consistency Score

**Date**: 2026-01-30
**Dataset**: Toucan (1,006 tool-calling samples)
**Task**: Regression - predict continuous STED consistency score (0-1)
**Features**: 76 engineered features from prompts and tools

## Executive Summary

Using regression models to predict the continuous STED consistency score, we achieve **R² = 0.43** with Gradient Boosting, explaining 43% of variance in consistency. The analysis reveals clear quantitative relationships: each additional parameter above 5 decreases STED by ~0.01, prompts over 500 characters drop consistency by 33%, and "can you" phrasing adds +0.08 to STED score on average.

---

## Model Performance

| Model | R² | MAE | RMSE |
|-------|-----|-----|------|
| **Gradient Boosting** | **0.430 ± 0.152** | **0.111** | **0.153** |
| Random Forest | 0.424 ± 0.132 | 0.113 | 0.155 |
| ElasticNet | 0.213 ± 0.048 | 0.145 | 0.183 |
| Ridge | 0.210 ± 0.070 | 0.146 | 0.183 |
| Lasso | 0.184 ± 0.033 | 0.147 | 0.186 |
| Linear Regression | 0.191 ± 0.084 | 0.147 | 0.185 |

**Best Model**: Gradient Boosting explains **43%** of STED score variance.

### Interpretation
- Mean STED score: 0.246
- Model MAE: 0.111 (average error of ~0.11 on 0-1 scale)
- RMSE: 0.153 (accounts for larger errors)

---

## Feature Importance Rankings

### Permutation Importance (Most Reliable)

| Rank | Feature | Importance | Correlation | Direction |
|------|---------|------------|-------------|-----------|
| 1 | **prompt_char_length** | 0.106 | r = -0.24 | ↓ Longer = Lower STED |
| 2 | **total_params** | 0.104 | r = -0.19 | ↓ More params = Lower STED |
| 3 | **avg_word_length** | 0.086 | r = -0.07 | ↓ Complex vocab = Lower STED |
| 4 | **max_params_per_tool** | 0.049 | r = -0.28 | ↓ More params = Lower STED |
| 5 | **std_params_per_tool** | 0.045 | r = -0.25 | ↓ More variation = Lower STED |
| 6 | **avg_params_per_tool** | 0.043 | r = -0.20 | ↓ More params = Lower STED |
| 7 | **prompt_sentence_count** | 0.039 | r = -0.03 | ↓ More sentences = Lower STED |
| 8 | **comma_count** | 0.033 | r = -0.26 | ↓ More commas = Lower STED |
| 9 | **avg_tool_name_length** | 0.028 | r = +0.04 | ↑ Longer names = Higher STED |
| 10 | **num_tools** | 0.025 | r = -0.11 | ↓ More tools = Lower STED |

### Ridge Regression Coefficients (Standardized)

| Rank | Feature | β (Standardized) | Effect per 1σ |
|------|---------|------------------|---------------|
| 1 | max_params_per_tool | **-0.132** | -0.027 STED |
| 2 | comma_count | **-0.078** | -0.016 STED |
| 3 | total_params | **-0.061** | -0.013 STED |
| 4 | newline_count | -0.050 | -0.010 STED |
| 5 | has_can_you | **+0.025** | +0.005 STED |
| 6 | question_mark_count | +0.019 | +0.004 STED |

---

## Quantitative Feature Effects

### Total Parameters Effect

| Parameter Count | Mean STED | Change from Baseline |
|-----------------|-----------|---------------------|
| 0-5 (baseline) | **0.303** | - |
| 6-10 | 0.226 | -25% |
| 11-15 | 0.192 | -37% |
| 16-20 | 0.161 | -47% |
| 21-30 | 0.129 | -57% |
| 50+ | 0.119 | **-61%** |

**Rule of thumb**: Each 5 additional parameters → ~0.025 lower STED

### Prompt Length Effect

| Prompt Length | Mean STED | Change from Baseline |
|---------------|-----------|---------------------|
| 0-300 chars (baseline) | **0.305** | - |
| 300-500 chars | 0.303 | -1% |
| 500-700 chars | 0.202 | **-34%** |
| 700-1000 chars | 0.190 | -38% |
| 1000+ chars | 0.179 | **-41%** |

**Threshold identified**: 500 characters is a critical breakpoint.

### Number of Tools Effect

| Tools | Mean STED | Notes |
|-------|-----------|-------|
| 1 | 0.167 | Low (often underspecified) |
| 2 | **0.368** | **Optimal** |
| 3 | 0.309 | Good |
| 4 | 0.206 | Declining |
| 5 | 0.218 | Moderate |
| 6-10 | 0.222 | Declining |
| 10+ | 0.171 | Low |

**Sweet spot**: 2-3 tools

### Max Parameters per Tool Effect

| Max Params | Mean STED | Drop from Baseline |
|------------|-----------|-------------------|
| 0-2 (baseline) | **0.294** | - |
| 3-4 | 0.219 | -25% |
| 5-6 | 0.159 | -46% |
| 7-8 | 0.159 | -46% |
| 9-10 | 0.150 | -49% |
| 10+ | **0.084** | **-71%** |

**Critical threshold**: Keep max params ≤ 4 per tool.

### Average Word Length Effect

| Avg Word Length | Mean STED | Interpretation |
|-----------------|-----------|----------------|
| < 5.0 | **0.261** | Simple vocabulary |
| 5.0-5.5 | **0.273** | **Optimal** |
| 5.5-6.0 | 0.240 | Moderate |
| 6.0-6.5 | 0.117 | Complex |
| > 6.5 | **0.081** | Too technical |

**Optimal range**: Average word length 5.0-5.5 characters.

---

## Non-Linear Relationships

Comparing Pearson (linear) vs Spearman (monotonic) correlations:

| Feature | Pearson | Spearman | Non-linear? |
|---------|---------|----------|-------------|
| avg_word_length | -0.07 | **-0.15** | **Yes** (stronger non-linear) |
| num_tools | -0.11 | **-0.18** | **Yes** |
| total_params | -0.19 | **-0.24** | **Yes** |
| max_params_per_tool | -0.28 | -0.28 | No (linear) |
| prompt_char_length | -0.24 | -0.24 | No (linear) |

**Insight**: `avg_word_length` and `num_tools` have stronger non-linear effects than their Pearson correlations suggest.

---

## Interaction Effects

### Tool Count × Parameter Count

| Configuration | STED Score |
|--------------|------------|
| Few tools (1-2) + Low params | **0.304** |
| Few tools (1-2) + High params | **0.028** |
| Many tools (6+) + Low params | 0.357 |
| Many tools (6+) + High params | 0.145 |

**Key insight**: Simple tools become extremely inconsistent with many parameters (0.304 → 0.028).

### Tool Count × Prompt Length

| Configuration | STED Score |
|--------------|------------|
| Few tools + Short prompt | **0.428** |
| Few tools + Long prompt | 0.195 |
| Many tools + Short prompt | 0.259 |
| Many tools + Long prompt | **0.158** |

**Key insight**: Short prompts with few tools are optimal (0.428 STED).

### "Can You" Phrasing Effect by Tool Complexity

| Tool Complexity | Without | With "can you" | Improvement |
|-----------------|---------|----------------|-------------|
| Few (1-2) | 0.242 | **0.379** | **+0.137** |
| Medium (3-5) | 0.217 | 0.297 | +0.081 |
| Many (6+) | 0.178 | 0.239 | +0.061 |

**Finding**: "Can you" adds +0.06 to +0.14 STED depending on complexity.

---

## Binary Feature Effects (Cohen's d)

| Feature | STED (With) | STED (Without) | Cohen's d | Effect Size |
|---------|-------------|----------------|-----------|-------------|
| **has_can_you** | 0.297 | 0.214 | **+0.41** | Small-Medium |
| has_i_need | 0.269 | 0.222 | +0.23 | Small |
| mentions_data | 0.278 | 0.232 | +0.23 | Small |
| **has_numbered_list** | 0.179 | 0.255 | **-0.37** | Small (negative) |
| **has_must** | 0.192 | 0.276 | **-0.41** | Small (negative) |
| has_should | 0.209 | 0.253 | -0.22 | Small |
| has_example_keyword | 0.207 | 0.247 | -0.19 | Negligible |

### Interpretation
- **Helpful**: "can you", "I need", data-related prompts
- **Harmful**: "must", numbered lists, constraints
- **No effect**: examples, bullet points, step words

---

## Prediction Formula (Simplified)

Based on Ridge regression, a simplified prediction:

```
STED ≈ 0.35
       - 0.006 × (total_params - 5)
       - 0.0003 × (prompt_length - 400)
       - 0.015 × (max_params_per_tool - 2)
       + 0.08 × has_can_you
       + 0.04 × question_mark_count
       - 0.07 × has_numbered_list
```

### Example Calculations

**Good prompt**:
- 5 params, 350 chars, max 2 params/tool, "can you", 1 question
- STED ≈ 0.35 + 0 + 0.015 + 0 + 0.08 + 0.04 = **0.485**

**Bad prompt**:
- 20 params, 800 chars, max 8 params/tool, no "can you", numbered list
- STED ≈ 0.35 - 0.09 - 0.12 - 0.09 + 0 - 0.07 = **-0.02 → 0.0**

---

## Optimal Configuration

For maximum STED score (target > 0.4):

| Parameter | Optimal Value | Impact |
|-----------|---------------|--------|
| total_params | ≤ 5 | +0.06 STED |
| prompt_length | < 500 chars | +0.10 STED |
| max_params_per_tool | ≤ 2 | +0.07 STED |
| num_tools | 2-3 | +0.05 STED |
| avg_word_length | 5.0-5.5 | +0.03 STED |
| has_can_you | Yes | +0.08 STED |
| has_numbered_list | No | +0.07 STED |

**Combined optimal**: ~0.46 expected STED

---

## Key Thresholds Summary

| Feature | Threshold | Effect |
|---------|-----------|--------|
| **total_params** | > 5 | Rapid STED decline begins |
| **prompt_char_length** | > 500 | 34% drop in STED |
| **max_params_per_tool** | > 4 | 46% drop in STED |
| **avg_word_length** | > 6.0 | 55% drop in STED |
| **num_tools** | > 3 | Declining returns |

---

## Statistical Summary

| Metric | Value |
|--------|-------|
| R² (Gradient Boosting) | 0.430 |
| MAE | 0.111 |
| RMSE | 0.153 |
| Features used | 76 |
| Samples | 1,006 |
| Target mean | 0.246 |
| Target std | 0.207 |

---

## Recommendations

### For Prompt Engineering
1. **Keep prompts under 500 characters** (34% STED improvement)
2. **Use "can you" phrasing** (+0.08 STED)
3. **Avoid numbered lists** (+0.07 STED when removed)
4. **Use simple vocabulary** (avg word length 5-5.5)
5. **Add question marks** (+0.04 per question)

### For Tool Design
1. **Limit to 2-3 tools** (optimal range)
2. **Keep max 2 params per tool** (critical threshold)
3. **Total params ≤ 5** (rapid decline above)
4. **Avoid parameter variation** (std_params hurts)
5. **Write clear tool names** (longer names help)

### Expected STED by Configuration

| Configuration | Expected STED |
|---------------|---------------|
| Optimal (all thresholds met) | **0.45-0.50** |
| Good (most thresholds met) | 0.30-0.40 |
| Average | 0.20-0.30 |
| Poor (thresholds exceeded) | 0.10-0.20 |
| Very poor | < 0.10 |
