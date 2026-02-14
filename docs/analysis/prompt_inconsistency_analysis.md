# Prompt Features and LLM Inconsistency Analysis

**Date**: 2026-01-30
**Dataset**: Toucan (1,006 samples, 21 models)
**Consistency Metric**: STED (Semantic-Structural Consistency)

## Executive Summary

This analysis investigates the relationship between input prompt characteristics and LLM output inconsistency. By linking STED consistency scores with prompt features, we identify statistically significant predictors of inconsistency and provide actionable insights for prompt engineering.

## Methodology

1. **Data Sources**:
   - STED consistency results: `results/toucan/minilm-ec2/combined_consistency_metrics_results.json`
   - Original prompts: `llm_gen_results/toucan/*/all_results.json`

2. **Feature Extraction**: For each prompt, we extracted:
   - Lexical features (length, word count, punctuation)
   - Vagueness indicators (hedge words, ambiguous quantifiers)
   - Structural features (question count, constraint words)
   - Task complexity indicators (tool count, multi-step markers)

3. **Statistical Analysis**: Pearson correlation with consistency coefficient, aggregated across all models and temperatures.

## Key Findings

### 1. Statistically Significant Correlations

| Feature | Correlation | P-value | Interpretation |
|---------|-------------|---------|----------------|
| Comma count | r = -0.259 | p < 0.001 | More complex syntax → MORE inconsistent |
| Character length | r = -0.235 | p < 0.001 | Longer prompts → MORE inconsistent |
| Word count | r = -0.230 | p < 0.001 | More words → MORE inconsistent |
| Question count | r = +0.135 | p < 0.001 | Clear questions → MORE consistent |
| Tool count | r = -0.112 | p < 0.001 | More tools → MORE inconsistent |
| Constraint count | r = -0.101 | p < 0.01 | More constraints → MORE inconsistent |
| Step words | r = -0.088 | p < 0.01 | Multi-step tasks → MORE inconsistent |

### 2. Counterintuitive Findings

**Hypothesis rejected**: Vague words do NOT cause inconsistency

- Vague word ratio had a *positive* correlation with consistency (r = +0.09, p < 0.01)
- Hedge words showed no significant correlation (r = -0.02, p = 0.50)
- Open-ended questions showed no significant correlation (r = -0.02, p = 0.61)

**Explanation**: Vague words often appear in simpler prompts. The complexity of the task (measured by length, tool count, constraint count) is the true driver of inconsistency, not lexical vagueness.

### 3. Comparative Analysis: Low vs High Consistency Prompts

| Metric | Low Consistency (Bottom 25%) | High Consistency (Top 25%) |
|--------|------------------------------|----------------------------|
| Avg character length | 674.6 ± 358.8 | 492.3 ± 276.3 |
| Avg word count | 102.2 ± 50.1 | 78.4 ± 40.0 |
| Avg question marks | 0.69 ± 0.54 | 0.83 ± 0.51 |
| Avg commas | 6.6 ± 5.0 | 3.9 ± 3.5 |
| Avg tool count | 6.57 ± 5.22 | 4.85 ± 5.67 |

### 4. Domain/Topic Analysis

| Domain | Avg Consistency | Sample Size | Notes |
|--------|-----------------|-------------|-------|
| Weather | 0.342 ± 0.247 | 374 | HIGH CONSISTENCY - Simple, well-defined API calls |
| Data Analysis | 0.266 ± 0.237 | 345 | Moderate |
| API | 0.216 ± 0.190 | 160 | Moderate |
| Creative | 0.208 ± 0.159 | 43 | Inherently open-ended |
| Math | 0.195 ± 0.153 | 140 | LOW CONSISTENCY |
| Code | 0.194 ± 0.162 | 143 | LOW CONSISTENCY |
| Search | 0.194 ± 0.163 | 146 | LOW CONSISTENCY |
| Database | 0.188 ± 0.166 | 173 | LOW CONSISTENCY - Multiple valid query paths |
| Document | 0.180 ± 0.135 | 251 | LOW CONSISTENCY |
| Scheduling | 0.169 ± 0.157 | 158 | LOW CONSISTENCY - Complex constraint satisfaction |

### 5. Tool Complexity Analysis

| Number of Tools | Avg Consistency | Sample Size |
|-----------------|-----------------|-------------|
| 1 tool | 0.167 ± 0.103 | 141 |
| 2 tools | 0.368 ± 0.256 | 193 |
| 3 tools | 0.309 ± 0.213 | 144 |
| 4 tools | 0.206 ± 0.167 | 141 |
| 5+ tools | ~0.19 ± 0.17 | 327 |

**Insight**: 2-3 tools is the "sweet spot" for consistency. Single-tool tasks may be underspecified, while 4+ tools introduce selection ambiguity.

### 6. Tool-Specific Consistency

**Most Inconsistent Tools**:
- `think-tool-server-think`: 0.109 ± 0.038
- `flux-imagegen-server-*`: 0.165 ± 0.057
- `drawing-tool-*`: 0.183 ± 0.126

**Most Consistent Tools**:
- `weather-forecast-service-get_live_temp`: 0.510 ± 0.260
- `weather-api-server-getLiveTemperature`: 0.440 ± 0.249

## Root Causes of Inconsistency

Based on the analysis, the primary drivers of inconsistency are:

1. **Task Decomposition Ambiguity**: Complex prompts with multiple sub-tasks have many valid execution paths. The model may choose different orderings or approaches across runs.

2. **Tool Selection Non-determinism**: When multiple tools could satisfy a request, the model's tool selection varies, leading to different outputs.

3. **Reasoning Chain Length**: More reasoning steps accumulate variance. Each decision point introduces potential divergence.

4. **Underspecified Output Format**: Tasks without clear output format constraints allow for valid but inconsistent responses.

5. **Constraint Satisfaction Complexity**: Scheduling and database tasks require satisfying multiple constraints simultaneously, with multiple valid solutions.

## Recommendations

### For Prompt Engineering

1. **Keep prompts concise**: Aim for <500 characters when possible
2. **Reduce syntactic complexity**: Fewer commas, simpler sentence structure
3. **Use explicit questions**: Clear question marks improve consistency
4. **Limit tool scope**: 2-3 tools per task is optimal
5. **Specify output format**: Provide explicit format requirements

### For Future Research

1. **Semantic Task Graph Extraction**: Develop methods to identify sub-tasks and dependencies automatically
2. **Output Space Entropy Estimation**: Quantify how many valid responses exist for a given prompt
3. **Instruction Specificity Scoring**: Create metrics for constraint density vs output freedom
4. **Causal Intervention Studies**: Systematically modify prompts to isolate causal factors

## Example Prompts

### Most Inconsistent (Consistency = 0.000)

```
I need to set up a 30-minute virtual meeting with our client in New York to
finalize a contract amendment. The meeting must start during the client's
business hours (08:00-18:00 EST), finish before our internal audit cutoff
of 17:00 EST, fit into the available 15-minute blocks of the only conference...
```

**Characteristics**: Long, complex constraints, scheduling domain, multiple requirements

### Most Consistent (Consistency = 0.776)

```
I need to accurately determine the current temperature at a specific location
by cross-referencing data from two different weather APIs. Can you fetch the
live temperature for coordinates 40.7128° N, 74.0060° W from two separate
weather services so I can compare the results for consistency?
```

**Characteristics**: Clear question, specific coordinates, simple task, weather domain

## Appendix: Feature Definitions

| Feature | Definition |
|---------|------------|
| `char_length` | Total characters in prompt |
| `word_count` | Total words in prompt |
| `sentence_count` | Count of `.!?` terminators |
| `vague_word_count` | Count of words like "some", "few", "good", "maybe" |
| `vague_word_ratio` | vague_word_count / word_count |
| `hedge_word_count` | Count of words like "perhaps", "might", "could" |
| `is_open_question` | Binary: contains patterns like "what do you think" |
| `question_count` | Count of `?` characters |
| `constraint_count` | Count of patterns like "must", "required", "need to" |
| `comma_count` | Count of `,` characters |
| `number_count` | Count of numeric values |
| `step_words` | Count of words like "first", "then", "next" |
| `conjunction_count` | Count of words like "and", "also", "additionally" |
| `tool_count` | Number of tools available for the task |
