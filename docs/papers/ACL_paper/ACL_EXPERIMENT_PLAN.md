# ACL 2026 Comprehensive Experiment Plan

## Paper: Linguistic Features and Consistency in LLM Tool Calling

### 1. Research Questions

**RQ1 (Modal Semantics):** How does modal verb strength (deontic vs epistemic continuum) affect LLM tool calling consistency?

**RQ2 (Politeness Theory):** How do politeness strategies (Brown & Levinson framework) influence consistency?

**RQ3 (Speech Acts):** Does speech act directness (Searle's taxonomy) correlate with consistency?

**RQ4 (Cross-Model):** Are linguistic effects consistent across different LLM architectures?

---

### 2. Experimental Design

#### 2.1 Models (6 diverse architectures)

| Model | Provider | Family | Parameters |
|-------|----------|--------|------------|
| Claude-Sonnet-4 | Bedrock | Anthropic Claude 4 | ~175B |
| Claude-3.5-Sonnet | Bedrock | Anthropic Claude 3.5 | ~175B |
| Llama-3.3-70B | Bedrock | Meta Llama 3.3 | 70B |
| Qwen3-235B | Bedrock | Alibaba Qwen 3 | 235B (MoE) |
| GPT-4.1-Mini | OpenRouter | OpenAI GPT-4.1 | ~8B |
| Gemini-2.5-Flash-Lite | OpenRouter | Google Gemini 2.5 | ~8B |

#### 2.2 Linguistic Variations (18 total)

**Modal Verbs (6 variations):**
| Variation | Modal | Strength | Type | Force Score |
|-----------|-------|----------|------|-------------|
| modal_must | must | Strong | Deontic | 0.95 |
| modal_need_to | need to | Strong | Deontic | 0.85 |
| modal_should | should | Medium | Deontic | 0.70 |
| modal_would | would like to | Medium | Epistemic | 0.55 |
| modal_could | could | Weak | Epistemic | 0.40 |
| modal_might | might want to | Weakest | Epistemic | 0.25 |

**Politeness Strategies (6 variations):**
| Variation | Strategy | Face Threat |
|-----------|----------|-------------|
| polite_bald | Bald on-record | 1.0 (highest) |
| polite_please | Positive politeness | 0.7 |
| polite_can_you | Conventional indirect | 0.5 |
| polite_could_you | More polite indirect | 0.4 |
| polite_grateful | Positive strong | 0.3 |
| polite_would_mind | Negative strong | 0.2 (lowest) |

**Speech Acts (3 variations):**
| Variation | Directness | Illocutionary Force |
|-----------|------------|---------------------|
| speech_directive | Direct imperative | 1.0 |
| speech_indirect | Conventional indirect | 0.6 |
| speech_hint | Non-conventional (hint) | 0.3 |

**Hedging (2 variations):**
| Variation | Type |
|-----------|------|
| hedge_conditional | "if possible" |
| hedge_temporal | "when you have a chance" |

**Baseline (1):**
| Variation | Description |
|-----------|-------------|
| baseline | Original prompt (no modification) |

#### 2.3 Experimental Parameters

| Parameter | Value | Justification |
|-----------|-------|---------------|
| Base prompts | 50 | Statistical power (n=50 per condition) |
| Runs per condition | 15 | Robust consistency estimation |
| Temperatures | 0.0, 0.3, 0.5, 0.7, 1.0 | Full temperature range |
| Total conditions | 50 × 18 × 5 = 4,500 | Per model |
| Total API calls | 4,500 × 15 = 67,500 | Per model |
| Grand total | 67,500 × 6 = 405,000 | All models |

#### 2.4 Dataset

- **Source:** Toucan benchmark (tool calling dataset)
- **Filtering criteria:**
  - ASCII prompts only
  - Single tool call expected
  - Prompt length: 50-500 characters
- **Available samples:** ~800 suitable prompts

---

### 3. Metrics and Statistical Analysis

#### 3.1 Primary Metric: Consistency Score (c_mean)

```
c_mean = mean(Jaccard(tools_i, tools_j)) for all pairs (i,j)
```

Where `Jaccard(A, B) = |A ∩ B| / |A ∪ B|` over tool name sets.

#### 3.2 Statistical Tests

| Test | Purpose | Threshold |
|------|---------|-----------|
| Bootstrap CI (95%) | Confidence intervals | 1000 resamples |
| Mann-Whitney U | Non-parametric comparison | p < 0.05 |
| Cohen's d | Effect size | \|d\| > 0.5 = large |
| Spearman ρ | Correlation with linguistic features | p < 0.05 |

#### 3.3 Hypotheses

**H1:** Modal strength negatively correlates with consistency (ρ < 0)
- Strong modals (must, need to) → lower consistency
- Weak modals (could, might) → higher consistency

**H2:** Face threat negatively correlates with consistency (ρ < 0)
- Bald on-record → lowest consistency
- Negative politeness → highest consistency

**H3:** Direct speech acts produce lower consistency than indirect

---

### 4. Implementation

#### 4.1 Scripts

```
scripts/experiments/acl_linguistic_variations/
├── run_comprehensive.py      # Main experiment runner
├── run_acl_ec2.sh           # EC2 batch runner
├── analyze_comprehensive.py  # Statistical analysis
└── generate_variations.py    # Linguistic transformations
```

#### 4.2 Running Experiments

**Local (single model):**
```bash
python run_comprehensive.py \
    --model "us.anthropic.claude-sonnet-4-20250514-v1:0" \
    --num-prompts 50 \
    --num-runs 15 \
    --temperatures 0.0 0.3 0.5 0.7 1.0
```

**EC2 (all models):**
```bash
./run_acl_ec2.sh
```

#### 4.3 Output Structure

```
results/acl_linguistic/comprehensive/
├── claude-sonnet-4_results.json
├── claude-3-5-sonnet_results.json
├── llama-3-3-70b_results.json
├── qwen3-235b-a22b_results.json
├── gpt-4-1-mini_results.json
├── gemini-2-5-flash-lite_results.json
├── combined_analysis.json
└── latex_tables.tex
```

---

### 5. Expected Results

Based on pilot study (960 samples, Claude-Sonnet-4):

#### 5.1 Modal Strength Effect
| Modal | Expected c_mean | Observed (pilot) |
|-------|-----------------|------------------|
| must | ~0.87 | 0.874 |
| should | ~0.90 | 0.898 |
| could | ~0.92 | 0.917 |
| might | ~0.94 | 0.939 |

**Expected correlation:** ρ ≈ -0.95 (p < 0.01)

#### 5.2 Politeness Effect
| Strategy | Expected c_mean | Observed (pilot) |
|----------|-----------------|------------------|
| bald | ~0.90 | 0.908 |
| please | ~0.97 | 0.974 |
| can_you | ~0.89 | 0.895 |
| would_mind | ~0.91 | N/A |

---

### 6. Paper Sections

1. **Introduction** - LLM consistency problem, linguistic framing
2. **Background** - Modal semantics, politeness theory, speech acts
3. **Methodology** - Experimental design, metrics
4. **Results** - Main findings with statistical significance
5. **Analysis** - Linguistic interpretation
6. **Discussion** - Implications for prompt engineering
7. **Conclusion** - Summary and future work

---

### 7. Timeline

| Phase | Duration | Tasks |
|-------|----------|-------|
| Week 1 | 3 days | Run all 6 models on EC2 |
| Week 2 | 2 days | Statistical analysis |
| Week 3 | 3 days | Paper writing and revision |
| Week 4 | 2 days | Final review and submission |

---

### 8. Estimated Costs

| Resource | Quantity | Unit Cost | Total |
|----------|----------|-----------|-------|
| Bedrock API (Claude) | ~150K calls | $0.003/call | ~$450 |
| Bedrock API (Llama) | ~70K calls | $0.001/call | ~$70 |
| Bedrock API (Qwen) | ~70K calls | $0.002/call | ~$140 |
| OpenRouter (GPT/Gemini) | ~140K calls | $0.0005/call | ~$70 |
| EC2 instance | 24 hours | $0.50/hour | ~$12 |
| **Total** | | | **~$750** |

---

### 9. Risk Mitigation

| Risk | Mitigation |
|------|------------|
| API rate limits | Use adaptive retry, parallel regions |
| Model unavailability | Have backup models ready |
| Inconsistent results | Increase sample size, bootstrap CI |
| Statistical power | 50 prompts × 15 runs = 750 per condition |
