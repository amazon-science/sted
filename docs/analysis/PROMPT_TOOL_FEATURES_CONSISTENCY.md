# Prompt and Tool Features Affecting Consistency

**Date**: 2026-01-30
**Dataset**: Toucan (1,006 tool-calling samples)
**Analysis**: Correlation between prompt/tool characteristics and STED consistency scores

## Executive Summary

This analysis investigates how prompt structure (examples, lists, complexity) and tool characteristics (count, parameters) affect LLM output consistency. The key finding is that **tool parameter complexity** (r = -0.28) is the strongest predictor of inconsistency, followed by **prompt length** (r = -0.24). Interestingly, examples in prompts show no significant effect, while numbered lists are associated with *lower* consistency.

---

## Key Findings

### 1. Tool Complexity is the Strongest Factor

| Factor | Correlation | P-value | Effect |
|--------|-------------|---------|--------|
| **max_params_per_tool** | r = -0.278 | p < 0.001 | Strongest negative |
| total_params | r = -0.194 | p < 0.001 | Strong negative |
| avg_params_per_tool | r = -0.199 | p < 0.001 | Strong negative |
| num_tools | r = -0.113 | p < 0.001 | Moderate negative |

**Interpretation**: Tools with many parameters create more opportunities for variation in how the model fills them.

### Consistency by Max Parameters per Tool

| Max Params | Consistency | Sample Count |
|------------|-------------|--------------|
| 0-2 | **0.294** | 602 |
| 3-4 | 0.219 | 170 |
| 5-6 | 0.159 | 83 |
| 7-8 | 0.159 | 19 |
| 9-10 | 0.150 | 69 |
| 10+ | **0.084** | 55 |

**Key insight**: Consistency drops by **71%** when max params increases from 0-2 to 10+.

---

### 2. Number of Tools Matters

| Tool Count | Consistency | Total Params (avg) |
|------------|-------------|--------------------|
| 1 tool | **0.368** | 4.1 |
| 2 tools | 0.309 | 7.8 |
| 3 tools | 0.206 | 5.2 |
| 4 tools | 0.218 | 8.7 |
| 5 tools | 0.251 | 8.7 |
| 6-10 tools | 0.190 | 15.0 |
| 10-20 tools | 0.153 | 34.2 |
| 20+ tools | 0.220 | 61.4 |

**Optimal threshold**: Tools ≤3 show significantly higher consistency (+0.085) than tools >3.

---

### 3. Prompt Structure Effects

| Feature | Correlation | P-value | Significant? |
|---------|-------------|---------|--------------|
| prompt_length | r = -0.235 | p < 0.001 | Yes*** |
| word_count | r = -0.230 | p < 0.001 | Yes*** |
| comma_count | r = -0.259 | p < 0.001 | Yes*** |
| question_count | r = +0.135 | p < 0.001 | Yes*** |
| has_numbered_list | r = -0.118 | p < 0.001 | Yes*** |
| has_bullet_points | r = +0.021 | p = 0.50 | No |
| has_example | r = -0.045 | p = 0.16 | No |

---

### 4. Examples Do NOT Help Consistency

**Surprising finding**: Including examples in prompts shows no significant correlation with consistency.

| Condition | Consistency | N |
|-----------|-------------|---|
| With example | 0.198 ± 0.133 | 35 |
| Without example | 0.248 ± 0.209 | 971 |

T-test: p = 0.156 (not significant)

**Possible explanation**: Examples may increase prompt complexity, offsetting any clarifying benefit.

---

### 5. Numbered Lists Decrease Consistency

**Counter-intuitive finding**: Prompts with numbered lists show *lower* consistency.

| Condition | Consistency | N |
|-----------|-------------|---|
| With numbered list | **0.179** ± 0.133 | 118 |
| Without numbered list | **0.255** ± 0.214 | 888 |

- T-test: t = -3.76, p < 0.001
- Cohen's d = 0.43 (medium effect)

**Hypothesis**: Numbered lists often indicate multi-step tasks with more complexity and decision points.

---

### 6. Question Count Improves Consistency

| Question Count | Consistency | N |
|----------------|-------------|---|
| 0 questions | 0.209 | 327 |
| 1 question | **0.262** | 653 |
| 2 questions | **0.321** | 23 |
| 3+ questions | **0.379** | 3 |

**Interpretation**: Explicit questions may help focus the model on specific outputs.

---

## Interaction Effects

### Tool Count × Parameter Count

| Configuration | Consistency |
|--------------|-------------|
| Simple tools (1-2), Few params (0-5) | **0.304** |
| Simple tools (1-2), Many params (16+) | 0.252 |
| Complex tools (6+), Few params (0-5) | 0.357 |
| Complex tools (6+), Many params (16+) | **0.175** |

**Key insight**: The worst case is many tools with many parameters.

---

## Regression Analysis

Multivariate regression with standardized coefficients:

| Feature | Standardized β | Direction |
|---------|---------------|-----------|
| max_params_per_tool | **-0.055** | Decreases consistency |
| word_count | **-0.053** | Decreases consistency |
| prompt_length | +0.044 | Increases consistency* |
| comma_count | -0.040 | Decreases consistency |
| num_tools | -0.017 | Decreases consistency |
| question_count | +0.012 | Increases consistency |

*Note: prompt_length shows positive coefficient when controlling for other factors, suggesting that longer prompts with simple structure can be beneficial.

**R² = 0.14** - These features explain 14% of consistency variance.

---

## Profile of High vs Low Consistency Samples

### Top 50 Most Consistent Samples
- Average tools: **3.1**
- Average params: **4.4**
- Has example: 0%
- Has numbered list: 0%
- Question count: 0.94

### Bottom 50 Least Consistent Samples
- Average tools: **7.8**
- Average params: **39.6**
- Has example: 4%
- Has numbered list: 8%
- Question count: 0.80

---

## Practical Recommendations

### For Tool Designers

1. **Limit parameters per tool to ≤4**: Consistency drops sharply above this threshold
2. **Keep total tool count ≤3**: Sweet spot for tool-calling consistency
3. **Prefer simple parameter types**: Avoid deeply nested objects
4. **Write clear tool descriptions**: avg_description_length correlates with consistency (r = -0.14)

### For Prompt Engineers

1. **Use explicit questions**: Question marks correlate with +13.5% consistency
2. **Avoid numbered lists for complex tasks**: They don't help and may hurt
3. **Don't assume examples help**: No significant benefit observed
4. **Reduce comma density**: Simpler syntax improves consistency
5. **Focus on clarity over length**: Longer prompts with simple structure are fine

### Optimal Configuration

```
Highest consistency profile:
- 1-3 tools
- ≤5 total parameters
- Max 2 params per tool
- Clear question format
- No numbered lists
- Minimal examples
```

---

## Statistical Summary

| Test | Result | Interpretation |
|------|--------|----------------|
| Pearson (max_params vs consistency) | r = -0.28, p < 0.001 | Strong negative |
| T-test (numbered list effect) | t = -3.76, p < 0.001 | Significant |
| T-test (example effect) | t = -1.42, p = 0.16 | Not significant |
| Regression R² | 0.14 | Moderate explanatory power |
| Kruskal-Wallis (tool count) | p < 0.001 | Significant group differences |

---

## Limitations

1. **Toucan-specific**: Results may not generalize to non-tool-calling tasks
2. **Correlation not causation**: Feature relationships may be confounded
3. **Limited example samples**: Only 35 samples had explicit examples
4. **Embedding dependency**: Results based on MiniLM embeddings

---

## Appendix: Feature Definitions

| Feature | Definition |
|---------|------------|
| num_tools | Number of tools available to the model |
| total_params | Sum of all parameters across all tools |
| max_params_per_tool | Maximum parameter count for any single tool |
| avg_params_per_tool | Mean parameter count per tool |
| has_example | Prompt contains "example", "e.g.", "for instance" |
| has_numbered_list | Prompt contains "1.", "2)", etc. |
| has_bullet_points | Prompt contains "-", "•", "*" list markers |
| question_count | Number of "?" in prompt |
| comma_count | Number of "," in prompt (complexity proxy) |
| has_step_words | Contains "first", "then", "next", "finally" |
| has_constraint | Contains "must", "should", "required", "ensure" |
