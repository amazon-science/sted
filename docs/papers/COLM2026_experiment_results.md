# COLM 2026: Predicting LLM Structured Output Consistency from Prompt Linguistics

## Experiment Results Summary

**Date**: 2026-02-12 (v3: with embedding baselines)
**Status**: Paper-ready results with proper methodology and embedding comparison

---

## 1. Methodology Improvements (v2)

The v1 results had several methodological issues that have been corrected:

| Issue | v1 (broken) | v2 (fixed) |
|-------|-------------|------------|
| Data leakage | Standard KFold | GroupKFold by sample_idx |
| Target aggregation | Raw per-temperature c_mean | Per-sample mean across temperatures |
| Zero-variance features | 67 features (6 constant) | 61 features (removed constants) |
| ML model | Random Forest (default) | GBM (tuned: lr=0.05, depth=5, n=200) |
| Metrics | R² only | R², Pearson, Spearman, MAE |
| Baselines | None | Random, length-only, schema-only |

### Removed Zero-Variance Features (6)
`surface_polite_negative`, `surface_polite_impersonal`, `semantic_undefined_terms`, `semantic_coreference_chains`, `schema_has_array_params`, `schema_has_object_params`

---

## 2. Main Results

### Table 1: Prediction Performance by Configuration

| Configuration | R² (avg) | Pearson | Spearman | p-value |
|---------------|----------|---------|----------|---------|
| Random baseline | 0.000 | 0.000 | 0.000 | - |
| Prompt length only | 0.022 | 0.151 | - | - |
| Schema features only (4) | 0.043 | 0.249 | - | - |
| Universal (61 feats, GBM) | **0.072** | **0.279** | **0.277** | - |
| Per-model (61 feats, GBM) | **0.108** | **0.368** | **0.369** | <0.001 |
| LOMO generalization | **0.128** | **0.591** | **0.577** | - |

### Table 2: Per-Model Predictor Performance (GBM, GroupKFold)

| Model | R² | Pearson | Spearman |
|-------|-----|---------|----------|
| Gemini-2.5-Flash-Lite | **0.289** | **0.542** | **0.528** |
| GPT-4.1-Mini | **0.266** | **0.522** | **0.514** |
| NemoTron-3-Nano | **0.208** | **0.476** | **0.453** |
| Llama-3.3-70B | 0.164 | 0.433 | 0.423 |
| Claude-3.5-Sonnet | 0.151 | 0.418 | 0.389 |
| Claude-Haiku-4.5 | 0.146 | 0.409 | 0.434 |
| Minimax-M2 | 0.143 | 0.402 | 0.390 |
| Claude-3.5-Haiku | 0.131 | 0.395 | 0.367 |
| Grok-4.1-Fast | 0.127 | 0.394 | 0.400 |
| Claude-Opus-4.5 | 0.099 | 0.359 | 0.374 |
| Qwen3-235B-A22B | 0.098 | 0.369 | 0.396 |
| Claude-3.7-Sonnet | 0.090 | 0.359 | 0.366 |
| GPT-OSS-120B | 0.090 | 0.339 | 0.341 |
| Claude-Sonnet-4.5 | 0.086 | 0.341 | 0.365 |
| Mimo-V2-Flash:free | 0.076 | 0.347 | 0.342 |
| Mimo-V2-Flash | 0.049 | 0.321 | 0.328 |
| Mistral-Large-3-675B | 0.040 | 0.291 | 0.288 |
| Qwen3-32B | 0.016 | 0.253 | 0.262 |
| Claude-Sonnet-4 | 0.016 | 0.275 | 0.305 |
| Nova-2-Lite | -0.009 | 0.233 | 0.221 |
| Claude-Opus-4 | -0.017 | 0.249 | 0.262 |

**Average**: R²=0.108, Pearson=0.368, Spearman=0.369
**90% of models show positive R²**
**100% of models show positive Pearson correlation**

---

## 3. Statistical Significance

### Primary Tests
| Test | Statistic | p-value | Significant? |
|------|-----------|---------|--------------|
| One-sample t-test (R² > 0) | t=6.052 | **0.000006** | YES *** |
| All features vs Schema-only (R²) | t=4.900 | **0.000086** | YES *** |
| All features vs Schema-only (Pearson) | t=8.356 | **<0.000001** | YES *** |

### Effect Sizes
| Comparison | Cohen's d | Interpretation |
|------------|-----------|----------------|
| R² vs random baseline | 1.868 | **Large** |
| Pearson vs 0 | 6.193 | **Large** |
| All features vs schema-only | 0.932 | **Large** |

### Confidence Intervals (Bootstrap, n=1000)
| Metric | 95% CI |
|--------|--------|
| Per-model R² (mean) | [0.078, 0.143] |

---

## 4. Cross-Model Transfer Analysis

### Transfer Matrix Summary
| Metric | Value |
|--------|-------|
| Same-model Pearson (train=test) | 0.971 |
| Cross-model Pearson (avg) | 0.399 |
| Transfer ratio | 0.41 |

### Model Family Transfer
Transfer works well within families:
| Train → Test | Pearson |
|-------------|---------|
| Mimo-V2-Flash → Mimo-V2-Flash:free | 0.959 |
| Claude-3.7-Sonnet → Claude-3.5-Haiku | 0.847 |
| Claude-Sonnet-4.5 → Claude-Haiku-4.5 | 0.808 |

Transfer fails across families:
| Train → Test | Pearson |
|-------------|---------|
| GPT-4.1-Mini → Claude-3.7-Sonnet | -0.160 |
| Claude-3.7-Sonnet → GPT-4.1-Mini | -0.163 |
| Gemini-2.5-Flash-Lite → Claude-3.7-Sonnet | -0.148 |

### Leave-One-Model-Out (LOMO) Generalization
Train on 20 models, test on held-out:

| Model | R² | Pearson | Spearman |
|-------|-----|---------|----------|
| Minimax-M2 | 0.459 | 0.760 | 0.727 |
| Claude-Haiku-4.5 | 0.446 | 0.757 | 0.771 |
| Qwen3-32B | 0.432 | 0.696 | 0.664 |
| Claude-Opus-4.5 | 0.416 | 0.732 | 0.742 |
| Grok-4.1-Fast | 0.408 | 0.696 | 0.739 |
| ... | ... | ... | ... |
| **Average** | **0.128** | **0.591** | **0.577** |

**Key insight**: LOMO achieves much higher Pearson (0.591) than per-model CV (0.368), suggesting that multi-model training learns generalizable patterns about which prompts are inherently more consistent.

---

## 5. Feature Importance (Permutation Importance)

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | schema_num_tools | 0.0897 |
| 2 | semantic_lexical_ambiguity | 0.0497 |
| 3 | semantic_syntactic_ambiguity | 0.0348 |
| 4 | surface_politeness_score | 0.0209 |
| 5 | surface_prompt_length | 0.0198 |
| 6 | pragmatic_specificity_score | 0.0191 |
| 7 | semantic_ambiguity_score | 0.0182 |
| 8 | pragmatic_context_dependency | 0.0147 |
| 9 | pragmatic_question_type | 0.0127 |
| 10 | surface_modal_deontic_strong | 0.0124 |

### Feature Category Ablation
| Configuration | Features | R² | Pearson | Spearman |
|---------------|----------|-----|---------|----------|
| All features | 61 | 0.069 | 0.275 | 0.270 |
| Schema only | 4 | 0.022 | 0.174 | 0.194 |
| Semantic only | 17 | 0.014 | 0.190 | 0.176 |
| Surface only | 23 | 0.007 | 0.170 | 0.173 |
| Pragmatic only | 17 | -0.011 | 0.129 | 0.129 |
| Remove schema | 57 | 0.040 | 0.229 | 0.225 |
| Remove semantic | 44 | 0.048 | 0.244 | 0.249 |
| Remove surface | 38 | 0.055 | 0.258 | 0.256 |
| Remove pragmatic | 44 | 0.057 | 0.260 | 0.258 |

**Key findings**:
1. All 4 categories contribute (removing any reduces performance)
2. Schema and semantic features contribute most
3. Full model significantly outperforms any single category (p < 0.001)

---

## 6. Embedding Baseline Comparison (Experiment 5)

Critical experiment comparing 61 hand-crafted linguistic features against sentence embedding baselines (all-MiniLM-L6-v2, 384-dim).

### Table 4: Feature Representation Comparison (21 Models, GBM, GroupKFold)

| Configuration | Dims | R² (avg) | Pearson | Spearman |
|---------------|------|----------|---------|----------|
| Prompt length only | 1 | -0.110 | 0.071 | 0.090 |
| Schema features only | 4 | 0.046 | 0.251 | 0.280 |
| **Linguistic features (ours)** | **61** | **0.092** | **0.355** | **0.360** |
| Sentence embeddings | 384 | 0.131 | 0.384 | 0.389 |
| PCA-reduced embeddings | 50 | 0.126 | 0.387 | 0.395 |
| Embeddings + schema | 388 | 0.153 | 0.407 | 0.414 |
| **Ling + Emb combined** | **445** | **0.158** | **0.413** | **0.418** |
| **Ling + PCA-Emb (best)** | **111** | **0.173** | **0.435** | **0.445** |

### Statistical Tests (Embedding Comparison)

| Test | t-stat | p-value | Winner |
|------|--------|---------|--------|
| Linguistic vs Embedding (R²) | -4.612 | **0.000169** | Embedding |
| Combined vs Linguistic (R²) | 8.544 | **<0.000001** | Combined |
| Combined vs Embedding (R²) | 4.183 | **0.000459** | Combined |
| Linguistic vs Embedding (Pearson) | -2.929 | **0.008294** | Embedding |
| Combined vs Linguistic (Pearson) | 7.376 | **<0.000001** | Combined |
| Combined vs Embedding (Pearson) | 4.134 | **0.000514** | Combined |

### Win Rates (R²)
| Comparison | Win Rate |
|------------|----------|
| Linguistic > Embedding | 2/21 (10%) |
| Combined > Linguistic | 20/21 (95%) |
| Combined > Embedding | 19/21 (90%) |

### LOMO Generalization with Embeddings
| Configuration | LOMO Pearson |
|---------------|-------------|
| Linguistic (61) | 0.562 +/- 0.181 |
| Embedding (384) | 0.597 +/- 0.199 |
| Combined (445) | 0.599 +/- 0.196 |

### Key Findings (Embedding Comparison)

1. **Embeddings edge out linguistic features** in per-model R² (0.131 vs 0.092), but the gap is moderate
2. **Combined features significantly beat both** (p < 0.001 for both comparisons), showing linguistic features capture orthogonal information
3. **Ling + PCA-Emb is the best configuration** (R²=0.173, Pearson=0.435) - PCA reduces overfitting from 384 dims
4. **Linguistic features provide 6.2x dimensionality reduction** (61 vs 384) for only 30% less predictive power
5. **Linguistic features are interpretable** - schema_num_tools, semantic_ambiguity explain *why* consistency varies, embeddings do not
6. Llama-3.3-70B is one of only 2 models where linguistic features beat embeddings (R²=0.141 vs 0.090)

### Paper Framing

The embedding comparison supports a **complementarity narrative**: linguistic features and embeddings capture different aspects of prompt difficulty. The combined model (Ling+PCA-Emb) achieves the best performance (R²=0.173), and linguistic features provide unique interpretability that embeddings lack. This is a stronger argument than claiming linguistic features alone are sufficient.

---

## 7. Binary Classification

| Threshold | F1 | AUC | # Models |
|-----------|-----|-----|----------|
| c_mean > 0.7 | 0.601 | 0.684 | 21 |
| c_mean > 0.8 | **0.675** | **0.783** | 16 |
| c_mean > 0.9 | 0.674 | 0.785 | 15 |

**Practical application**: Can predict with AUC=0.78 whether a prompt will achieve high consistency.

---

## 8. Dataset Statistics

| Statistic | Value |
|-----------|-------|
| Total observations | 21 models × 1,006 samples × 11 temperatures |
| Unique prompts | 1,006 |
| Models evaluated | 21 |
| Temperatures | 11 (0.0 - 1.0) |
| Features | 61 (after removing 6 zero-variance) |
| Dataset | Toucan (tool-calling) |
| Runs per (sample, temperature) | 10 |

---

## 9. Paper-Ready Contributions

### Contribution 1: Linguistic features predict consistency (moderate effect)
- Per-model Pearson = 0.368 (avg), 0.542 (best)
- R² = 0.108 (avg), 0.289 (best)
- All effect sizes are **large** (Cohen's d > 0.8)
- Statistically significant (p < 0.001)

### Contribution 2: Models differ in linguistic sensitivity
- 100% of models show positive Pearson correlation
- But magnitude varies 4x (0.233 to 0.542)
- Within-family transfer works (Claude→Claude: ρ=0.84)
- Cross-family transfer fails (Claude→GPT: ρ=-0.16)

### Contribution 3: LOMO reveals generalizable patterns
- Training on 20 models generalizes to unseen model (Pearson=0.591)
- Suggests inherent prompt-level consistency patterns exist

### Contribution 4: Feature importance hierarchy
- Schema complexity > Semantic ambiguity > Surface features > Pragmatic load
- `schema_num_tools` is the dominant predictor
- All categories contribute (ablation shows significant drops)

### Contribution 5: Linguistic features complement embeddings
- Embeddings alone: R²=0.131, Pearson=0.384
- Linguistic alone: R²=0.092, Pearson=0.355 (6.2x fewer dimensions)
- Combined (Ling+PCA-Emb): R²=0.173, Pearson=0.435 (best)
- Combined significantly beats both (p < 0.001), confirming orthogonal information
- Linguistic features win on 2/21 models (Llama-3.3-70B, Claude-3.5-Sonnet)

### Contribution 6: Practical binary classifier
- AUC = 0.783 for predicting c_mean > 0.8
- Enables prompt screening before deployment

---

## 10. Files and Locations

### Scripts
- `scripts/experiments/colm_consistency_predictor/experiments/exp4_improved_predictor.py` - Main improved experiment
- `scripts/experiments/colm_consistency_predictor/experiments/exp5_embedding_baseline.py` - Embedding baseline comparison
- `scripts/experiments/colm_consistency_predictor/features/` - Feature extraction modules

### Data
- `experiments/colm_2026_consistency_predicto_20260210_154446/results/exp1_correlations/extracted_features.csv`
- `results/toucan_exact_final/combined_consistency_metrics_results.json`

### Results
- `experiments/colm_2026_consistency_predicto_20260210_154446/results/exp4_improved/`

---

## 11. Methodology

### GroupKFold Cross-Validation
```python
from sklearn.model_selection import GroupKFold

# Per-sample target: mean c_mean across all 11 temperatures
# GroupKFold by sample_idx prevents leakage
gkf = GroupKFold(n_splits=5)
for train_idx, test_idx in gkf.split(X, y, groups=sample_idx):
    model.fit(X[train_idx], y[train_idx])
    score = model.score(X[test_idx], y[test_idx])
```

### Per-Sample Aggregation
Instead of predicting c_mean per (sample, temperature), we average c_mean across temperatures per (sample, model) first. This gives one target per sample per model, avoiding temperature as a confounding variable.

### LOMO Protocol
For Leave-One-Model-Out: train GBM on all 20 models' data, test on held-out model. This evaluates whether prompt-level patterns generalize across model architectures.
