# ML-Based Consistency Prediction Analysis

**Date**: 2026-01-30
**Dataset**: Toucan (1,006 tool-calling samples)
**Task**: Binary classification (Consistent vs Inconsistent)
**Features**: 76 engineered features from prompts and tools

## Executive Summary

Using machine learning to predict LLM output consistency, we achieve **79.2% ROC-AUC** with Random Forest classifier. The analysis reveals that **tool complexity** (num_tools, total_params) is the strongest predictor, followed by **prompt characteristics** (word count, avg word length). A surprising finding is that **"can you" phrasing improves consistency by 10-27%** depending on task complexity.

---

## Model Performance

| Classifier | Accuracy | ROC-AUC |
|------------|----------|---------|
| Random Forest | **0.714 ± 0.005** | **0.792 ± 0.010** |
| Gradient Boosting | 0.704 ± 0.013 | 0.780 ± 0.007 |
| Logistic Regression | 0.634 ± 0.029 | 0.701 ± 0.035 |

**Best Model**: Random Forest with 79.2% AUC demonstrates that consistency is significantly predictable from prompt/tool features.

---

## Feature Importance Rankings

### By Random Forest (Built-in Importance)

| Rank | Feature | Importance | Category |
|------|---------|------------|----------|
| 1 | avg_word_length | 0.073 | Prompt |
| 2 | total_params | 0.066 | Tool |
| 3 | avg_tool_name_length | 0.054 | Tool |
| 4 | avg_tool_desc_length | 0.051 | Tool |
| 5 | std_params_per_tool | 0.049 | Tool |
| 6 | num_tools | 0.047 | Tool |
| 7 | avg_params_per_tool | 0.047 | Tool |
| 8 | prompt_word_count | 0.045 | Prompt |
| 9 | prompt_char_length | 0.044 | Prompt |
| 10 | max_params_per_tool | 0.042 | Tool |

### By Permutation Importance (More Reliable)

| Rank | Feature | Importance | Direction |
|------|---------|------------|-----------|
| 1 | **num_tools** | 0.021 | ↓ More tools = LESS consistent |
| 2 | **avg_params_per_tool** | 0.014 | ↓ More params = LESS consistent |
| 3 | **total_params** | 0.013 | ↓ More params = LESS consistent |
| 4 | avg_tool_name_length | 0.011 | ↑ Longer names = MORE consistent |
| 5 | std_params_per_tool | 0.009 | ↓ More variation = LESS consistent |
| 6 | prompt_word_count | 0.005 | ↓ Longer prompts = LESS consistent |
| 7 | avg_word_length | 0.005 | ↓ Longer words = LESS consistent |
| 8 | **has_can_you** | 0.004 | ↑ "Can you" = MORE consistent |

---

## Decision Tree Rules (Interpretable)

The decision tree with 70.1% accuracy reveals key thresholds:

```
IF total_params <= 7.5:
    IF avg_word_length <= 6.0:
        IF prompt_char_length <= 468:
            → CONSISTENT (high confidence)
        ELSE:
            → INCONSISTENT
    ELSE (avg_word_length > 6.0):
        → INCONSISTENT

ELSE (total_params > 7.5):
    IF avg_word_length <= 5.91:
        IF avg_tool_desc_length > 45.65:
            IF total_params <= 18.5:
                → CONSISTENT
            ELSE:
                → INCONSISTENT
        ELSE:
            → INCONSISTENT
    ELSE:
        → INCONSISTENT
```

### Key Thresholds Discovered

| Feature | Threshold | Rule |
|---------|-----------|------|
| total_params | ≤ 7.5 | Primary split - critical threshold |
| avg_word_length | ≤ 6.0 | Secondary - simpler language helps |
| prompt_char_length | ≤ 468 | Shorter prompts more consistent |
| avg_tool_desc_length | > 45.65 | Better descriptions help |

---

## Feature Interaction Effects

### Tool Complexity × Parameter Count

| Configuration | Consistency Rate |
|--------------|------------------|
| Simple (1-2 tools) + Few params (0-5) | **64.0%** |
| Simple (1-2 tools) + Many params (16+) | **0.0%** |
| Complex (6+ tools) + Few params (0-5) | 64.9% |
| Complex (6+ tools) + Many params (16+) | **22.4%** |

**Key insight**: Even simple tools become inconsistent with many parameters.

### Tool Complexity × Prompt Length

| Configuration | Consistency Rate |
|--------------|------------------|
| Simple tools + Short prompt | **75.3%** |
| Simple tools + Long prompt | 49.7% |
| Complex tools + Short prompt | 44.6% |
| Complex tools + Long prompt | **27.1%** |

**Key insight**: Long prompts hurt more with complex tools.

---

## Surprising Finding: "Can You" Effect

The phrasing "can you" in prompts significantly improves consistency:

| Tool Complexity | With "can you" | Without | Improvement |
|-----------------|----------------|---------|-------------|
| Simple (1-2 tools) | 78.2% | 51.5% | **+26.7%** |
| Medium (3-5 tools) | 59.2% | 47.1% | **+12.1%** |
| Complex (6+ tools) | 44.0% | 33.7% | **+10.3%** |

**Hypothesis**: Polite question framing may help the model focus on the specific request.

---

## Optimal Configuration Profile

Based on the analysis, the highest consistency rate (78%) is achieved with:

```
✓ 1-2 tools (simple)
✓ ≤5 total parameters
✓ "Can you" phrasing
✓ Short prompt (<468 characters)
✓ Simple language (avg word length ≤6)
✓ Clear tool descriptions (>45 chars)
```

### Worst Configuration (22% consistency):

```
✗ 6+ tools (complex)
✗ 16+ total parameters
✗ No polite phrasing
✗ Long prompt
✗ Complex vocabulary
```

---

## Feature Categories Summary

### Features that INCREASE Consistency

| Feature | Effect Size | Significance |
|---------|-------------|--------------|
| has_can_you | +13.3% | High |
| avg_tool_name_length | +0.58 | Moderate |
| has_help_me | +1.8% | Low |

### Features that DECREASE Consistency

| Feature | Effect Size | Significance |
|---------|-------------|--------------|
| total_params | -7.03 | **Very High** |
| num_tools | -1.72 | **High** |
| prompt_word_count | -14.35 | **High** |
| avg_params_per_tool | -0.70 | Moderate |
| conjunction_count | -0.75 | Moderate |
| std_params_per_tool | -0.52 | Moderate |

### Features with NO Effect

- has_example (examples don't help!)
- has_bullet_points
- has_numbered_list
- has_code_block
- has_inline_code

---

## Practical Recommendations

### For Maximum Consistency

1. **Limit parameters**: Keep total_params ≤ 7 (primary threshold)
2. **Use polite phrasing**: Include "can you" or "help me"
3. **Keep prompts short**: Under 468 characters ideal
4. **Use simple vocabulary**: Average word length ≤ 6 characters
5. **Write clear tool descriptions**: At least 45+ characters each
6. **Limit tool count**: 1-2 tools when possible

### Feature Engineering Guidelines

When designing prompts for consistent LLM outputs:

```python
# Good prompt pattern
prompt = "Can you help me [specific task] using [tool name]?"

# Avoid
prompt = "First, I need you to [task1], then [task2],
         additionally [task3], and furthermore [task4]..."
```

---

## Statistical Validation

| Metric | Value | Interpretation |
|--------|-------|----------------|
| ROC-AUC | 0.792 | Good discriminative power |
| Cross-validation folds | 5 | Robust estimate |
| Feature count | 76 | Comprehensive coverage |
| Sample size | 1,006 | Adequate for ML |
| Class balance | 50/50 | No rebalancing needed |

---

## Limitations

1. **Toucan-specific**: Results based on tool-calling benchmark only
2. **Embedding dependency**: Consistency measured via MiniLM
3. **Feature engineering bias**: Manual feature selection may miss important patterns
4. **No causal validation**: Correlations may not be causal

---

## Appendix: All 76 Features

### Prompt Features (35)
- Length: prompt_char_length, prompt_word_count, prompt_sentence_count, avg_word_length, max_word_length
- Punctuation: comma_count, period_count, question_mark_count, exclamation_count, colon_count, semicolon_count, parentheses_count, quote_count, bracket_count, punctuation_density
- Structure: has_bullet_points, has_numbered_list, has_lettered_list, has_any_list, newline_count, paragraph_count
- Examples: has_example_keyword, has_such_as, has_like_this, has_code_block, has_inline_code, quoted_string_count, has_sample_data
- Task: has_step_words, has_sequence_words, has_multiple_tasks, conjunction_count
- Constraints: has_must, has_should, has_need_to, has_required, has_ensure, constraint_word_count
- Conditionals: has_if, has_when, has_unless, has_otherwise, conditional_count
- Specificity: number_count, has_specific_values, has_exact_match, has_range
- Vagueness: vague_word_count, vague_word_ratio
- Question type: starts_with_question, is_imperative, has_help_me, has_can_you, has_i_need
- Domain: mentions_api, mentions_data, mentions_file, mentions_time, mentions_user

### Tool Features (17)
- Count: num_tools, tools_with_no_params, tools_with_many_params
- Parameters: total_params, max_params_per_tool, min_params_per_tool, avg_params_per_tool, std_params_per_tool, total_required_params, has_nested_params
- Types: string_params, number_params, boolean_params, array_params, object_params
- Descriptions: avg_tool_name_length, avg_tool_desc_length
