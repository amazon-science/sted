# KDD 2026 Paper: Data Source Mapping

This document maps every data point in the paper to its source experiment result file.

**Paper**: "What Makes LLM Structured Outputs Inconsistent? A Feature Importance Study of 18 Models and 199K+ Outputs"

**Last Updated**: 2026-02-09

---

## Base Paths

```
RESULTS_BASE = /Users/guanghu/Documents/genai/projects/sted-internal/results
KDD_TABLES = ${RESULTS_BASE}/kdd_paper_tables
FACTOR_ANALYSIS = ${RESULTS_BASE}/factor_analysis
CAUSAL = ${RESULTS_BASE}/causal_intervention_toucan
```

---

## Abstract Claims

| Claim | Value | Source File | Field/Key |
|-------|-------|-------------|-----------|
| Number of models | 18 | `scripts/analysis/generate_kdd_table_data.py` | `FINAL_MODELS` list |
| Evaluation instances | 199,188 | `${KDD_TABLES}/table2_model_comparison.json` | `n_samples` |
| Pooled R² | 0.10 | `${KDD_TABLES}/model_specific_r2_analysis.json` | `pooled_r2` |
| Per-model R² | 0.67 | `${KDD_TABLES}/model_specific_r2_analysis.json` | `per_model_r2.mean` |
| Cross-model SHAP ρ | 0.56 | `${KDD_TABLES}/model_specific_r2_analysis.json` | `cross_model_shap_correlation.mean` |
| Schema complexity SHAP | 19% | `${KDD_TABLES}/table4_shap_importance.json` | `results.schema_complexity.percentage` |
| Temperature SHAP | 12% | `${KDD_TABLES}/table4_shap_importance.json` | `results.temperature.percentage` |
| Query length SHAP | 7% | `${KDD_TABLES}/table4_shap_importance.json` | `results.query_length.percentage` |
| Interaction ratio | 2.6× | `${KDD_TABLES}/table5_interaction_effects.json` | `results.degradation_ratio` |
| Accuracy-consistency r | +0.33 | `${KDD_TABLES}/validity_consistency_correlation.json` | `correlation` |
| LOMO R² | -0.009±0.18 | `${KDD_TABLES}/table8_lomo_cv.json` | `summary.mean_R2`, `summary.std_R2` |
| Causal max improvement | +9.7% | `${CAUSAL}/combined_analysis.json` | `simpson_paradox.has_should.low_mean` |

---

## Table 1: Factor Correlations with Consistency

**Source**: `${FACTOR_ANALYSIS}/correlations.csv`

| Factor | Paper Pearson r | Paper Spearman ρ | Data Field |
|--------|-----------------|------------------|------------|
| Schema Complexity | -0.163 | -0.160 | `schema_complexity` row |
| Schema Depth | -0.148 | -0.152 | `schema_depth` row |
| Schema Breadth | -0.143 | -0.145 | `schema_breadth` row |
| Total Parameters | -0.135 | -0.131 | `total_params` row |
| Max Params/Tool | -0.126 | -0.122 | `max_params_per_tool` row |
| Param Type Diversity | -0.125 | -0.118 | `param_type_diversity` row |
| Avg Params/Tool | -0.107 | -0.095 | `avg_params_per_tool` row |
| Num Conjunctions | -0.092 | -0.107 | `num_conjunctions` row |
| Query Complexity | -0.089 | -0.098 | `query_complexity_score` row |
| Query Length | -0.080 | -0.112 | `query_length` row |

**Note**: Paper values are rounded from raw data. Differences of ±0.01 are due to rounding.

---

## Table 2: Predictive Model Comparison (5-fold CV)

**Source**: `${KDD_TABLES}/table2_model_comparison.json`

| Model | Paper R² | Paper Std | Data Field |
|-------|----------|-----------|------------|
| RF (+ model identity) | 0.693 | 0.005 | `results["RF (one-hot enc.)"]` |
| Gradient Boosting | 0.114 | 0.003 | `results["Gradient Boosting"]` |
| Random Forest | 0.103 | 0.004 | `model_specific_r2_analysis.json → pooled_r2` |
| Ridge Regression | 0.048 | 0.003 | `results["Ridge Regression"]` |

---

## Table 3: Per-Model R² from Controllable Factors

**Source**: `${KDD_TABLES}/table2_model_comparison_by_model.json`

| Model | Paper Toucan R² | Data Field |
|-------|-----------------|------------|
| Claude-3.5-Sonnet | 0.759 | `per_model_results["Claude-3.5-Sonnet"].methods.RF.R2` |
| Qwen3-235B-A22B | 0.747 | `per_model_results["Qwen3-235B-A22B"].methods.RF.R2` |
| Llama-3.3-70B | 0.745 | `per_model_results["Llama-3.3-70B"].methods.RF.R2` |
| GPT-4.1-Mini | 0.721 | `per_model_results["GPT-4.1-Mini"].methods.RF.R2` |
| Gemini-2.5-Flash | 0.450 | `per_model_results["Gemini-2.5-Flash-Lite"].methods.RF.R2` |
| Mean (all 18) | 0.670 | `${KDD_TABLES}/model_specific_r2_analysis.json → per_model_r2.mean` |
| Pooled | 0.103 | `${KDD_TABLES}/model_specific_r2_analysis.json → pooled_r2` |
| Gap | 0.567 | `${KDD_TABLES}/model_specific_r2_analysis.json → r2_gap` |

---

## Table 4: Per-Model SHAP Analysis

**Source**: `${KDD_TABLES}/table4_shap_importance.json` (pooled) and per-model SHAP files

| Model | Top SHAP Feature | SHAP Value | Avg ρ |
|-------|------------------|------------|-------|
| GPT-4.1-Mini | schema_depth | 0.133 | 0.07 |
| Llama-3.3-70B | schema_breadth | 0.112 | 0.53 |
| Gemini-2.5-Flash-Lite | query_length | 0.133 | 0.66 |
| Minimax-M2 | num_tools | 0.092 | 0.56 |
| GPT-OSS-120B | avg_params_per_tool | 0.059 | 0.62 |

**Full 18-model table**: `${KDD_TABLES}/table4_shap_importance.json` (Appendix I)

---

## Table 5: Temperature × Schema Complexity Interaction

**Source**: `${KDD_TABLES}/table5_interaction_effects.json`

| Schema Complexity | Low Temp (0-0.3) | Med Temp (0.3-0.6) | High Temp (0.6-1.0) |
|-------------------|------------------|--------------------|--------------------|
| Simple (Q1) | 0.783 | 0.755 | 0.727 |
| Medium (Q2-Q3) | 0.724 | 0.694 | 0.656 |
| Complex (Q4) | 0.645 | 0.585 | 0.525 |

**Data Fields**:
- Simple: `results.pivot_table["Low (0-0.3)"].Q1`, etc.
- Complex: `results.pivot_table["Low (0-0.3)"].Q4`, etc.
- Simple degradation: `results.simple_schema.degradation_pct` (7.2%)
- Complex degradation: `results.complex_schema.degradation_pct` (18.6%)
- Ratio: `results.degradation_ratio` (2.59×)

---

## Table 6: Model Family Consistency Statistics

**Source**: Computed from `${FACTOR_ANALYSIS}/factor_analysis_data.csv`

| Family | Mean S_α | Min | Max | Range |
|--------|----------|-----|-----|-------|
| Claude (n=8) | 0.720 | 0.65 | 0.78 | 0.13 |
| Qwen (n=2) | 0.680 | 0.64 | 0.72 | 0.08 |
| GPT (n=2) | 0.650 | 0.60 | 0.70 | 0.10 |
| Llama (n=1) | 0.620 | --- | --- | --- |
| Other OSS (n=5) | 0.580 | 0.48 | 0.68 | 0.20 |

---

## Table 7: Causal Ceiling Effect Results

### Consistency Data
**Source**: `${CAUSAL}/combined_analysis.json`

| Feature | Low Baseline Δ | High Baseline Δ | p-value |
|---------|----------------|-----------------|---------|
| has_should | +9.7% | -3.4% | <0.0001 |
| has_must | +9.7% | -1.0% | <0.0001 |
| has_if | +7.2% | -9.6% | <0.0001 |
| has_can_you | +3.9% | -18.1% | <0.0001 |
| has_please | +1.7% | -20.4% | <0.0001 |

**Data Fields**:
- Low baseline: `simpson_paradox.{feature}.low_mean`
- High baseline: `simpson_paradox.{feature}.high_mean`
- p-value: `simpson_paradox.{feature}.p_value`

### Accuracy Data (Table 10 in Paper - Directive Features Only)
**Source**: `${CAUSAL}/directive_intervention_results.json`

| Feature | Low Baseline Δ Acc | High Baseline Δ Acc | n_low | n_high |
|---------|-------------------|---------------------|-------|--------|
| has_must | +6.0% | -0.1% | 187 | 472 |
| has_should | +5.8% | -0.3% | 164 | 377 |

**Note**: Table 10 only includes directive features (has_must, has_should) which have complete accuracy data. Other features (has_if, has_please) only have consistency data, shown in Appendix.

**Data Fields (directive_intervention_results.json)**:
- Structure: `{metadata, results (list of 1200), analysis}`
- Each result contains:
  - `original_accuracy`: Accuracy before intervention
  - `rewritten_accuracy`: Accuracy after intervention
  - `delta_accuracy`: `rewritten_accuracy - original_accuracy`
  - `directive_word`: "must" or "should" (only these two available)
  - `intervention_type`: "add_directive" or "remove_directive"
  - `original_consistency`: Baseline consistency for stratification

**Calculation Method**:
```python
# Stratify by baseline consistency
low_mask = data['original_consistency'] < 0.8
high_mask = data['original_consistency'] >= 0.8

# For has_must:
must_data = [r for r in results if r['directive_word'] == 'must']
low_acc = np.mean([r['delta_accuracy'] for r in must_data if r['original_consistency'] < 0.8])
high_acc = np.mean([r['delta_accuracy'] for r in must_data if r['original_consistency'] >= 0.8])
```

---

## Table 8: LOMO Cross-Validation Summary

**Source**: `${KDD_TABLES}/table8_lomo_cv.json`

| Transfer Type | Mean R² | Std |
|---------------|---------|-----|
| Within Claude family | 0.12 | 0.05 |
| Within Qwen family | 0.08 | 0.03 |
| Cross-family (Claude→Other) | -0.15 | 0.12 |
| Cross-family (GPT→Other) | -0.32 | 0.18 |
| All pairs (overall) | -0.009 | 0.18 |

**Data Fields**:
- Overall mean: `summary.mean_R2`
- Overall std: `summary.std_R2`
- Per-model: `per_model_results.{model}.R2`

---

## Appendix Data Sources

### Appendix A: Weighted Analysis
**Source**: Computed on-demand from `${FACTOR_ANALYSIS}/factor_analysis_data.csv`

### Appendix B: Sensitivity Analysis
**Source**: `${KDD_TABLES}/table1_factor_correlations_by_model.json`

### Appendix C: Experimental Details
**Source**: `scripts/analysis/generate_kdd_table_data.py` → `FINAL_MODELS`

### Appendix D: Causal Validation Details
**Source**: `${CAUSAL}/combined_analysis.json` → `feature_counts`

### Appendix I: Per-Model SHAP (Full)
**Source**: `${FACTOR_ANALYSIS}/global_shap_importance.csv` and per-model files

### Appendix J: ShareGPT Cross-Validation
**Source**: `${CAUSAL}/sharegpt_results.jsonl`

### Appendix N: VIF Analysis
**Source**: `${KDD_TABLES}/vif_analysis.json`

### Appendix O: Decision Point Hypothesis
**Source**: `${KDD_TABLES}/decision_point_hypothesis.json`

---

## Scripts for Regenerating Data

| Table | Script |
|-------|--------|
| All KDD tables | `scripts/analysis/generate_kdd_table_data.py` |
| Correlations | `scripts/analysis/comprehensive_factor_analysis.py` |
| SHAP importance | `scripts/analysis/shap_feature_importance.py` |
| Per-model R² | `scripts/analysis/per_model_feature_importance.py` |
| VIF analysis | `scripts/analysis/compute_vif_analysis.py` |
| Decision point | `scripts/analysis/test_decision_point_hypothesis.py` |
| ShareGPT analysis | `scripts/analysis/sharegpt_factor_analysis.py` |

---

## Data Freshness

| Source File | Last Modified | Generated By |
|-------------|---------------|--------------|
| `factor_analysis_data.csv` | 2025-01-30 | `comprehensive_factor_analysis.py` |
| `table2_model_comparison.json` | 2025-02-09 | `generate_kdd_table_data.py` |
| `table4_shap_importance.json` | 2025-02-09 | `generate_kdd_table_data.py` |
| `table5_interaction_effects.json` | 2025-02-09 | `generate_kdd_table_data.py` |
| `table8_lomo_cv.json` | 2025-02-09 | `generate_kdd_table_data.py` |
| `combined_analysis.json` | 2025-02-08 | Causal intervention experiments |

---

## Validation Status

All paper claims verified against data on 2026-02-09:
- ✓ Table 1 correlations (within ±0.01 rounding)
- ✓ Table 2 model comparison R² values
- ✓ Table 3 per-model R² values
- ✓ Table 4 SHAP values
- ✓ Table 5 interaction effects
- ✓ Table 7 causal ceiling effects (consistency AND accuracy) - UPDATED
- ✓ Table 8 LOMO results
- ✓ Abstract statistics
- ✓ Conclusion statistics

---

## Design Decisions (Resolved)

| Issue | Resolution | Notes |
|-------|------------|-------|
| has_if/has_please accuracy data unavailable | **Removed from Table 10** | Only directive features (has_must, has_should) shown in main table. Consistency results for has_if/has_please referenced in footnote to Appendix. |

**Rationale**: The source file `directive_intervention_results.json` only contains accuracy data for `directive_word` values "must" and "should". Rather than showing incomplete data or marking cells as N/A, we simplified Table 10 to focus on directive features which show the strongest effects. Other features' consistency results appear in Appendix~\ref{app:add_remove}.
