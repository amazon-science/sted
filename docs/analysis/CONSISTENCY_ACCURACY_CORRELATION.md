# Consistency-Accuracy Correlation Analysis

## Research Question

**Does higher STED consistency correlate with higher accuracy?**

**Answer: YES** - Strong correlation at model-level (r=0.89) and sample-level (85% of models show significant correlation).

---

## 1. Experiment Setup

### Method
- **Dataset**: Toucan (100 samples per model, 10 runs each)
- **Accuracy**: STED similarity between model output and ground truth
- **Consistency (c_mean)**: Mean pairwise STED similarity across 10 runs
- **Ranking Score**: Normalized consistency ranking (0-1, higher = more consistent)

### Models Tested (20 total)
Claude-Opus-4, Claude-Opus-4.5, Claude-Sonnet-4, Claude-Sonnet-4.5, Claude-Haiku-4.5, Claude-3.5-Haiku, Claude-3.5-Sonnet, Claude-3.7-Sonnet, GPT-4.1-Mini, GPT-OSS-120B, Qwen3-32B, Qwen3-235B-A22B, Mimo-V2-Flash, Minimax-M2, Gemini-2.5-Flash-Lite, Nova-2-Lite, Mistral-Large-3-675B, Grok-4.1-Fast, NemoTron-3-Nano-30B, Llama-3.3-70B

---

## 2. Results at T=0.0 (Deterministic)

| Model | Baseline Acc | Improved Acc | Acc Δ% | c_mean | Ranking |
|-------|-------------|--------------|--------|--------|---------|
| Mistral-Large-3-675B | 0.6599 | 0.7016 | **+6.33%** | 0.81 | 0.24 |
| Minimax-M2 | 0.8004 | 0.8264 | **+3.25%** | 0.90 | 0.45 |
| NemoTron-3-Nano-30B | 0.7595 | 0.7731 | **+1.79%** | 0.91 | 0.60 |
| Qwen3-235B-A22B | 0.7753 | 0.7852 | **+1.27%** | 0.94 | 0.70 |
| Mimo-V2-Flash | 0.8219 | 0.8253 | **+0.41%** | 0.96 | 0.73 |
| Nova-2-Lite | 0.7939 | 0.7964 | **+0.31%** | 0.99 | 0.90 |
| Claude-3.5-Haiku | 0.6203 | 0.6213 | **+0.17%** | 1.00 | 0.98 |
| Claude-3.7-Sonnet | 0.6212 | 0.6219 | **+0.12%** | 1.00 | 0.98 |
| GPT-4.1-Mini | 0.7593 | 0.7599 | **+0.09%** | 0.94 | 0.64 |
| Claude-Opus-4 | 0.7995 | 0.7995 | 0.00% | 1.00 | 1.00 |
| Claude-Sonnet-4.5 | 0.7612 | 0.7612 | 0.00% | 1.00 | 1.00 |
| Gemini-2.5-Flash-Lite | 0.7260 | 0.7256 | -0.05% | 1.00 | 0.97 |
| Claude-Opus-4.5 | 0.7691 | 0.7679 | -0.15% | 0.99 | 0.92 |
| Claude-Haiku-4.5 | 0.8163 | 0.8150 | -0.16% | 0.99 | 0.96 |
| Llama-3.3-70B | 0.4970 | 0.4948 | -0.43% | 0.99 | 0.91 |
| Qwen3-32B | 0.7321 | 0.7279 | -0.58% | 0.97 | 0.82 |
| Claude-Sonnet-4 | 0.8273 | 0.8199 | -0.89% | 0.99 | 0.92 |
| Claude-3.5-Sonnet | 0.5798 | 0.5743 | -0.95% | 0.99 | 0.90 |
| GPT-OSS-120B | 0.5697 | 0.5627 | -1.22% | 0.94 | 0.67 |
| Grok-4.1-Fast | 0.6800 | 0.6629 | -2.51% | 0.88 | 0.44 |

### T=0.0 Summary
| Metric | Value |
|--------|-------|
| Average Acc Improvement | **+0.34%** |
| Models Improved | **9/20 (45%)** |
| Mean Ranking | **0.79** |

---

## 3. Results at T=1.0 (High Variation)

| Model | Baseline Acc | Improved Acc | Acc Δ% | c_mean | Ranking |
|-------|-------------|--------------|--------|--------|---------|
| Mistral-Large-3-675B | 0.6451 | 0.6942 | **+7.62%** | 0.78 | 0.15 |
| Minimax-M2 | 0.7845 | 0.8271 | **+5.44%** | 0.87 | 0.35 |
| NemoTron-3-Nano-30B | 0.7336 | 0.7580 | **+3.33%** | 0.86 | 0.47 |
| Qwen3-32B | 0.7108 | 0.7333 | **+3.17%** | 0.91 | 0.55 |
| Mimo-V2-Flash | 0.8150 | 0.8320 | **+2.08%** | 0.89 | 0.55 |
| GPT-OSS-120B | 0.5679 | 0.5790 | **+1.95%** | 0.95 | 0.60 |
| Gemini-2.5-Flash-Lite | 0.7299 | 0.7403 | **+1.43%** | 0.89 | 0.47 |
| Claude-3.5-Sonnet | 0.5765 | 0.5841 | **+1.31%** | 0.97 | 0.82 |
| Claude-Sonnet-4 | 0.8257 | 0.8359 | **+1.23%** | 0.95 | 0.72 |
| Claude-Opus-4 | 0.7933 | 0.8015 | **+1.04%** | 0.96 | 0.75 |
| Claude-3.5-Haiku | 0.6279 | 0.6315 | **+0.58%** | 0.99 | 0.89 |
| Claude-Opus-4.5 | 0.7765 | 0.7809 | **+0.57%** | 0.95 | 0.69 |
| Qwen3-235B-A22B | 0.7712 | 0.7754 | **+0.54%** | 0.90 | 0.57 |
| Claude-Sonnet-4.5 | 0.7537 | 0.7569 | **+0.42%** | 0.95 | 0.68 |
| Claude-Haiku-4.5 | 0.7936 | 0.7953 | **+0.21%** | 0.92 | 0.54 |
| GPT-4.1-Mini | 0.7638 | 0.7631 | -0.10% | 0.93 | 0.63 |
| Claude-3.7-Sonnet | 0.6083 | 0.6043 | -0.66% | 0.99 | 0.88 |
| Llama-3.3-70B | 0.5193 | 0.5135 | -1.10% | 0.98 | 0.87 |
| Grok-4.1-Fast | 0.6839 | 0.6742 | -1.42% | 0.87 | 0.40 |
| Nova-2-Lite | 0.7896 | 0.7766 | -1.66% | 0.96 | 0.80 |

### T=1.0 Summary
| Metric | Value |
|--------|-------|
| Average Acc Improvement | **+1.30%** |
| Models Improved | **15/20 (75%)** |
| Mean Ranking | **0.62** |

---

## 4. Model-Level Correlation

**Question**: Do models with higher consistency also achieve higher accuracy?

| Metric Pair | Pearson r | Interpretation |
|-------------|-----------|----------------|
| **Accuracy vs c_mean** | **0.8938** | Very strong positive |
| Accuracy vs ranking_score | 0.6785 | Strong positive |
| Accuracy vs stability | 0.1970 | Weak |

### Model Accuracy and Consistency Rankings

| Model | Accuracy | c_mean | Acc Rank | c_mean Rank |
|-------|----------|--------|----------|-------------|
| Claude-Sonnet-4 | 0.8814 | 0.74 | 1 | 1 |
| Mimo-V2-Flash | 0.8761 | 0.74 | 2 | 2 |
| Claude-Opus-4 | 0.8685 | 0.73 | 3 | 3 |
| Nova-2-Lite | 0.8524 | 0.72 | 4 | 4 |
| Minimax-M2 | 0.8441 | 0.72 | 5 | 5 |
| Gemini-2.5-Flash-Lite | 0.8399 | 0.73 | 6 | 6 |
| Claude-Sonnet-4.5 | 0.8275 | 0.71 | 7 | 7 |
| Qwen3-32B | 0.8203 | 0.68 | 8 | 12 |
| Qwen3-235B-A22B | 0.8160 | 0.70 | 9 | 9 |
| Claude-Opus-4.5 | 0.8092 | 0.69 | 10 | 10 |
| Claude-Haiku-4.5 | 0.8088 | 0.69 | 11 | 11 |
| GPT-4.1-Mini | 0.8081 | 0.70 | 12 | 8 |
| Grok-4.1-Fast | 0.7964 | 0.68 | 13 | 13 |
| Claude-3.5-Haiku | 0.7416 | 0.60 | 14 | 14 |
| Claude-3.5-Sonnet | 0.7389 | 0.60 | 15 | 15 |
| Claude-3.7-Sonnet | 0.7261 | 0.58 | 16 | 16 |
| Llama-3.3-70B | 0.7102 | 0.57 | 17 | 17 |
| GPT-OSS-120B | 0.6558 | 0.53 | 18 | 18 |
| NemoTron-3-Nano-30B | N/A | 0.91 | - | - |
| Mistral-Large-3-675B | N/A | 0.81 | - | - |

**Conclusion**: Models with higher c_mean consistently achieve higher accuracy (r=0.89).

---

## 5. Sample-Level Correlation

**Question**: Within each model, do more consistent samples have higher accuracy?

### Spearman Correlation at T=1.0 (Combined Variation)

| Model | Spearman r | p-value | Significant? |
|-------|------------|---------|--------------|
| Mimo-V2-Flash | **0.72** | <0.001 | Yes |
| GPT-4.1-Mini | **0.69** | <0.001 | Yes |
| Minimax-M2 | **0.67** | <0.001 | Yes |
| Mistral-Large-3-675B | **0.66** | <0.001 | Yes |
| Grok-4.1-Fast | **0.65** | <0.001 | Yes |
| NemoTron-3-Nano-30B | **0.62** | <0.001 | Yes |
| Claude-Opus-4.5 | 0.54 | <0.001 | Yes |
| Claude-Haiku-4.5 | 0.51 | <0.001 | Yes |
| Qwen3-235B-A22B | 0.50 | <0.001 | Yes |
| Gemini-2.5-Flash-Lite | 0.49 | <0.001 | Yes |
| Claude-Sonnet-4 | 0.48 | <0.001 | Yes |
| Claude-Opus-4 | 0.46 | <0.001 | Yes |
| Nova-2-Lite | 0.45 | <0.001 | Yes |
| Qwen3-32B | 0.44 | <0.001 | Yes |
| Claude-Sonnet-4.5 | 0.31 | 0.002 | Yes |
| Claude-3.7-Sonnet | 0.24 | 0.018 | Yes |
| Claude-3.5-Haiku | 0.21 | 0.039 | Yes |
| Llama-3.3-70B | 0.19 | 0.313 | No |
| GPT-OSS-120B | 0.12 | 0.216 | No |
| Claude-3.5-Sonnet | 0.07 | 0.512 | No |

### Sample-Level Summary
| Metric | Value |
|--------|-------|
| Models with significant correlation | **17/20 (85%)** |
| Average Spearman r | **0.45** |
| Highest correlation | Mimo-V2-Flash: **0.72** |

---

## 6. Key Findings

| Finding | Value |
|---------|-------|
| Model-level correlation (c_mean vs accuracy) | **r = 0.89** |
| Sample-level: models with significant correlation | **85% (17/20)** |
| Average Spearman r (sample-level) | **0.45** |
| Highest sample-level correlation | Mimo-V2-Flash: **0.72** |

---

## 7. Practical Implications

1. **c_mean predicts accuracy**: Strong correlation (r=0.89) at model level
2. **Higher consistency = higher accuracy**: Both at model and sample level
3. **Most models show significant correlation**: 17/20 (85%) have p < 0.05
4. **T=1.0 shows more variation**: c_mean drops from 0.96 to 0.92, ranking from 0.79 to 0.62

---

## 8. Files

| File | Description |
|------|-------------|
| `results/accuracy_analysis/centroid_selection_multi_temp_toucan.json` | Selection strategy results |
| `results/accuracy_analysis/consistency_accuracy_correlation_toucan.json` | Sample-level correlation |
| `results/accuracy_analysis/accuracy_vs_consistency_t0.json` | Model-level correlation |
