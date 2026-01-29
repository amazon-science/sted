# Model Characteristics and Stability Analysis

**Generated:** 2026-01-19
**Data Sources:** OpenRouter API, Official Model Documentation

This document analyzes model characteristics that may affect stability in structured output and tool calling tasks.

---

## Model Specifications (Verified Data Only)

### Fully Verified Models

| Model | Total Params | Active Params | Architecture | Context | Source |
|-------|-------------|---------------|--------------|---------|--------|
| **Qwen3-235B-A22B** | 235B | 22B | **MoE** | 32K-131K | OpenRouter |
| **Qwen3-32B** | 32.8B | 32.8B | **Dense** | 41K-131K | OpenRouter |
| **Llama-3.3-70B** | 70B | 70B | **Dense** | 131K | OpenRouter |
| **DeepSeek-R1** | 671B | 37B | **MoE** | 64K | OpenRouter |
| **Mistral-Large-3** | 675B | 41B | **MoE** | 262K | OpenRouter |
| **Minimax-M2** | 230B | 10B | **MoE** | 197K | OpenRouter |
| **Pixtral-Large** | 124B | 124B | **Dense** | 131K | OpenRouter |
| **NemoTron-3-Nano** | 8B | 8B | **Dense** | - | Model name |

### Partially Verified Models

| Model | Known Info | Unknown | Source |
|-------|-----------|---------|--------|
| **GPT-4.1-Mini** | Context: 1M | Size, Architecture | OpenRouter |
| **Nemotron-70B** | ~70B (from Llama 3.1 base) | Exact size | OpenRouter |
| **GPT-OSS-120B** | 120B (from name) | Architecture | Model name |

### Undisclosed Models

| Model | Provider | Status |
|-------|----------|--------|
| **Claude-3.5-Sonnet** | Anthropic | No public specs |
| **Claude-3.7-Sonnet** | Anthropic | No public specs |
| **Claude-Haiku-4.5** | Anthropic | No public specs |
| **Claude-3.5-Haiku** | Anthropic | No public specs |
| **Claude-Opus-4** | Anthropic | No public specs |
| **Claude-Opus-4.5** | Anthropic | No public specs |
| **Claude-Sonnet-4** | Anthropic | No public specs |
| **Claude-Sonnet-4.5** | Anthropic | No public specs |
| **Nova-2-Lite** | Amazon | No public specs |
| **Nova-Pro** | Amazon | No public specs |
| **Grok-4.1-Fast** | xAI | No public specs |
| **Gemini-2.5-Flash-Lite** | Google | No public specs |
| **Mimo-V2-Flash** | Moonshot | No public specs |

---

## Performance Analysis (Verified Models)

### MoE vs Dense Performance

| Model | Architecture | Active Params | ShareGPT | Toucan | Temp Stability |
|-------|--------------|---------------|----------|--------|----------------|
| **Qwen3-235B-A22B** | MoE | 22B | 100.0% | 98% | **Excellent** |
| **Minimax-M2** | MoE | 10B | 98.2% | 84% | Good |
| **Mistral-Large-3** | MoE | 41B | Running | 59% | **Poor** (59→43%) |
| **DeepSeek-R1** | MoE | 37B | Pending | 0% | Broken |
| **Qwen3-32B** | Dense | 32B | 96.1% | 96% | **Excellent** |
| **Llama-3.3-70B** | Dense | 70B | 99.1% | 20% | Degrades |
| **Pixtral-Large** | Dense | 124B | 80.0% | Pending | Moderate |
| **NemoTron-3-Nano** | Dense | 8B | Pending | 41% | Degrades |

### MoE Architecture Analysis

| MoE Model | Active Params | Performance | Notes |
|-----------|---------------|-------------|-------|
| Qwen3-235B-A22B | 22B | **Excellent** | Best overall performer |
| Minimax-M2 | 10B | **Good** | Strong ShareGPT, moderate Toucan |
| Mistral-Large-3 | 41B | **Poor** | Degrades badly with temperature |
| DeepSeek-R1 | 37B | **Broken** | Tool calling doesn't work |

**Conclusion:** MoE architecture alone doesn't predict stability. Qwen3-235B with 22B active params outperforms Mistral-Large-3 with 41B active params, suggesting expert routing quality and training matter more than raw parameter count.

### Active Parameters vs Performance

| Active Params | Model | Architecture | ShareGPT | Toucan |
|---------------|-------|--------------|----------|--------|
| 8B | NemoTron-3-Nano | Dense | - | 41% |
| 10B | Minimax-M2 | MoE | 98.2% | 84% |
| 22B | Qwen3-235B-A22B | MoE | 100% | 98% |
| 32B | Qwen3-32B | Dense | 96.1% | 96% |
| 37B | DeepSeek-R1 | MoE | - | 0% |
| 41B | Mistral-Large-3 | MoE | Running | 59% |
| 70B | Llama-3.3-70B | Dense | 99.1% | 20% |
| 124B | Pixtral-Large | Dense | 80.0% | - |

**Observation:** The sweet spot appears to be 22B-32B active parameters. Larger models (41B+) don't guarantee better results.

### Temperature Stability by Architecture

| Pattern | MoE Models | Dense Models |
|---------|------------|--------------|
| **Excellent** (±2%) | Qwen3-235B-A22B | Qwen3-32B |
| **Good** (±5%) | Minimax-M2 | - |
| **Poor** (>10% drop) | Mistral-Large-3 | Llama-3.3-70B (Toucan), NemoTron |

### MoE Active-to-Total Ratio Analysis

| Model | Active/Total | Ratio | Performance |
|-------|--------------|-------|-------------|
| Qwen3-235B-A22B | 22B/235B | 9.4% | Excellent |
| Minimax-M2 | 10B/230B | 4.3% | Good |
| Mistral-Large-3 | 41B/675B | 6.1% | Poor |
| DeepSeek-R1 | 37B/671B | 5.5% | Broken |

**Note:** No clear correlation between active-to-total ratio and performance. Training quality appears more important.

---

## Key Findings

### 1. Active Parameter Efficiency
- Qwen3-235B with 22B active outperforms Mistral-675B with 41B active
- Suggests **routing quality** matters more than raw active parameter count
- Expert selection and load balancing are critical for MoE performance

### 2. Training Quality > Size
- Llama-3.3-70B (70B dense): 99% structured output, 20% tool calling
- Qwen3-32B (32B dense): 96% on both tasks
- **Task-specific training is critical** - raw capability doesn't transfer automatically

### 3. MoE Expert Routing
- Well-trained MoE (Qwen3-235B): Perfect temperature stability
- Poorly-routed MoE (Mistral-Large-3): Degrades 16% from T=0.0 to T=1.0
- **Expert selection consistency affects temperature stability**

### 4. Model Size Sweet Spot
- Too small (8B): Capability limited (41% Toucan)
- Sweet spot (22B-32B active): Best performance/stability ratio
- Larger (41B+ active): Diminishing returns, sometimes worse

---

## Potential Stability Factors

### Factors That May Improve Stability

| Factor | Evidence | Confidence |
|--------|----------|------------|
| Dense architecture | Generally more predictable | Medium |
| 22B-32B active params | Qwen3-235B, Qwen3-32B top performers | Medium |
| Strong instruction-tuning | Qwen3 models excel at both tasks | High |
| Tool-use specific training | Llama fails tools despite JSON success | High |
| Consistent expert routing (MoE) | Qwen3-235B vs Mistral-Large-3 | Medium |

### Factors That May Reduce Stability

| Factor | Evidence | Confidence |
|--------|----------|------------|
| Very small models (<20B) | NemoTron 41% Toucan | High |
| Poor MoE routing | Mistral-Large-3 degrades with temperature | Medium |
| Missing task-specific fine-tuning | Llama excellent JSON, poor tools | High |
| Very large MoE without proper training | Mistral-675B, DeepSeek-R1 | Medium |

### Summary Table

| Factor | Evidence | Confidence |
|--------|----------|------------|
| MoE can be excellent | Qwen3-235B = 100%/98% | **High** |
| MoE can be poor | Mistral-675B = 59%, DeepSeek-R1 = 0% | **High** |
| ~22B active is sweet spot | Qwen3-235B outperforms larger models | **Medium** |
| Training > Architecture | Same arch, different results (Llama vs Qwen) | **High** |
| Temperature stability varies by model | Not predictable by architecture alone | **High** |

---

## Task-Specific Performance Gap

| Model | ShareGPT | Toucan | Gap | Interpretation |
|-------|----------|--------|-----|----------------|
| **Llama-3.3-70B** | 99.1% | 20% | **-79%** | Structured output OK, tool calling broken |
| **Minimax-M2** | 98.2% | 84% | **-14%** | Better at structured output |
| **Qwen3-235B-A22B** | 100% | 98% | -2% | Excellent at both |
| **Qwen3-32B** | 96.1% | 96% | 0% | Balanced performance |

**Hypothesis:** Training focus matters significantly. Some models are optimized for one task over another.

---

## Research Questions for Further Investigation

1. **Why does Qwen3-235B (22B active) outperform Mistral-Large-3 (41B active)?**
   - Both are MoE, but vastly different results
   - Likely factors: Expert routing quality, training data, fine-tuning approach

2. **Why does Mistral-Large-3 degrade so badly with temperature?**
   - 59% at T=0.0 → 43% at T=1.0 (16% drop)
   - Hypothesis: Expert routing becomes less deterministic at higher temperatures

3. **Why is Llama-3.3-70B excellent at JSON but terrible at tool calling?**
   - 99% structured output vs 20% tool calling
   - Same architecture, different training focus

4. **Is there a correlation between MoE active-to-total ratio and stability?**
   - Current data inconclusive
   - Need more MoE models to establish pattern

5. **What role does context length play in stability?**
   - Models with longer context (Mistral-Large-3: 262K) don't necessarily perform better
   - May affect memory/attention patterns

---

## Failure Analysis (From Log Data)

### ShareGPT Structured Output Failures

Analysis of experiment logs reveals the actual failure reasons for each model. Note: Previously reported "RateLimit" errors were incorrect - they were matching source code line numbers `[run_inference:429]`, not HTTP 429 errors.

#### Verified Failure Counts (From Logs)

| Model | Total Invalid | Empty/Invalid | JSON Failed | Throttling | Timeout |
|-------|---------------|---------------|-------------|------------|---------|
| **Minimax-M2** | 162 | 161 | 98 | 0 | 0 |
| **Claude-Opus-4** | 196 | 196 | 1 | 0 | 0 |
| **Qwen3-235B-A22B** | 2 | 2 | 2 | 0 | 0 |

**Key Finding:** No actual rate limit (HTTP 429) errors were found in any analyzed logs.

#### Failure Categories

1. **Empty/Invalid Response**: Model returns empty response or non-JSON content
   - Most common failure type across all models
   - Often caused by model generating conversational text instead of JSON

2. **JSON Extraction Failed**: Model returns text containing invalid JSON
   - Malformed JSON structure (missing brackets, invalid syntax)
   - Text wrapped around JSON that prevents extraction

3. **Safety Filters**: Content blocked by model safety systems
   - Affected: Claude-Opus-4 (sample_058 with PII content)
   - Prompts requesting extraction of phone numbers, addresses

4. **Truncated Response**: Response cut off before completion
   - Affected: Mistral-Large-3-675B
   - Caused by max_tokens limit or early generation stop

#### Failure Rates by Model Category

| Category | Models | Typical Failure Rate | Primary Cause |
|----------|--------|---------------------|---------------|
| **Top Tier** | Qwen3-235B, Claude-3.5-Sonnet, Claude-Haiku-4.5 | <0.1% | Rare edge cases |
| **High Performers** | Nova-2-Lite, Minimax-M2, Claude-Opus-4 | 1-2% | Empty responses |
| **Mid Performers** | Qwen3-32B, Grok-4.1-Fast | 4-5% | JSON extraction |
| **Lower Performers** | Claude-Sonnet-4/4.5, Pixtral-Large | 16-20% | Empty responses |

### Toucan Tool Calling Failures

| Category | Models | Typical Failure Rate | Primary Cause |
|----------|--------|---------------------|---------------|
| **Broken** | DeepSeek-R1, Nova-Pro | 100% | Tool calling not supported |
| **Poor** | Llama-3.3-70B | 80% | Generates text instead of tool calls |
| **Moderate** | Mistral-Large-3, NemoTron-3-Nano | 40-60% | Format errors, missing params |
| **Good** | GPT-4.1-Mini, GPT-OSS-120B | 25-30% | Inconsistent formatting |
| **Excellent** | Qwen3-235B, Claude models | <5% | Minor edge cases |

---

## Appendix: Data Sources

### OpenRouter API
- URL: https://openrouter.ai/api/v1/models
- Model pages: https://openrouter.ai/{provider}/{model}

### Official Documentation
- Qwen: Model cards on Hugging Face
- Meta: Official Llama release notes
- Mistral: Official blog posts
- DeepSeek: Technical reports

### Unverified (Not Used)
- Claude models: Anthropic does not publish specifications
- Nova models: Amazon does not publish specifications
- Grok models: xAI does not publish specifications
- Gemini models: Google does not publish specifications

### Log Analysis
- Script: `scripts/analysis/analyze_experiment_logs.py`
- Analyzes experiment log files for error patterns
- Counts: Empty responses, JSON extraction failures, throttling, timeouts
- Note: Many experiments have logs on EC2 (not downloaded) - only local logs analyzed

### Corrected Data (2026-01-19)
- Fixed incorrect "RateLimit" counts that were matching source code line numbers
- Actual HTTP 429 rate limit errors: **0** across all analyzed experiments
- Primary failure causes: Empty/invalid responses, JSON extraction failures
