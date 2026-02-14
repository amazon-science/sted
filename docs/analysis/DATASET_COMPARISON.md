# Dataset Comparison for STED Evaluation

This document provides a comprehensive comparison of datasets available for evaluating the STED (Semantic Tree Edit Distance) framework.

## Executive Summary

| Dataset | Size | Avg Depth | Max Depth | Avg Fields | Unique Keys | Domain |
|---------|------|-----------|-----------|------------|-------------|--------|
| **ShareGPT** | 80 | ~2-3 | ~5 | ~15 | varies | Structured output, Quiz |
| **Toucan-1.5M** | 119,287 | 7.24 | 12 | 152.5 | 385+ | Tool/Function calling |
| **Xlam** | 60,000 | 3.0 | 12 | 6.2 | 3,179 | Function calling |
| **Maxscha-large** | 1,000* | 2.38 | 15 | 13.1 | 2,121 | JSON generation |
| **Maxscha** | 1,000* | 2.41 | 11 | 13.9 | 2,360 | JSON generation |
| **Glaive** | 62 | 1.29 | 2 | 2.6 | 21 | Function calling |
| **Jiraya** | 999 | ~1.3 | 2 | 10.2 | 4 | HTML-to-JSON |

*Sampled from larger dataset

---

## Detailed Dataset Analysis

### 1. ShareGPT (Base Dataset)

**Source**: Hugging Face
- `Arun63/sharegpt-structured-output-json` (30 samples)
- `Arun63/sharegpt-quizz-generation-json-output` (50 samples)

**Role in STED**: Base templates for synthetic variation generation

**Characteristics**:
- Conversational format with system/human/GPT turns
- JSON outputs embedded in conversation responses
- Moderate nesting depth (2-3 levels typical)
- Diverse prompt types (structured output, quiz generation)

**Strengths**:
- Real-world conversational context
- Diverse output schemas
- Good for template-based variation testing

**Limitations**:
- Small sample size (80 total)
- Variable output quality
- Some parsing errors (~5 samples)

**Download**:
```bash
python scripts/data/download_sharegpt_data.py --output-dir sharegpt_data
```

---

### 2. Toucan-1.5M-structured-Qwen (NEW)

**Source**: `beyoru/Toucan-1.5M-structured-Qwen`

**Size**: 119,287 examples (~905 MB)

**Format**:
```json
{
  "uuid": "string",
  "subset_name": "string (e.g., single-turn-original)",
  "question": "string",
  "target_tools": "string",
  "tools": "JSON string with tool definitions",
  "messages": [{"content": "string", "role": "string"}],
  "text": "string (full conversation)"
}
```

**JSON Complexity (tools field)**:
| Metric | Value |
|--------|-------|
| Avg Depth | 7.24 |
| Max Depth | 12 |
| Min Depth | 6 |
| Avg Fields/Sample | 152.48 |
| Max Fields | 535 |
| Unique Keys (100 samples) | 385 |
| Avg Tools/Sample | 8.52 |

**Strengths**:
- **Largest complexity**: Highest average depth (7.24) and field count (152.5)
- **Multi-tool scenarios**: Average 8.5 tools per sample
- **Rich schema diversity**: 385+ unique keys
- **Substantial size**: 119K examples for statistical significance
- **Multi-lingual**: Contains prompts in multiple languages

**Limitations**:
- Large download size (~905 MB)
- Requires parsing nested JSON from string fields
- Tool-calling specific domain

**Recommended Use Cases**:
- Testing STED on highly complex nested structures
- Evaluating structural matching on deep hierarchies
- Multi-tool function calling consistency

**Download**:
```python
from datasets import load_dataset
ds = load_dataset("beyoru/Toucan-1.5M-structured-Qwen", split="train")
```

---

### 3. Salesforce xlam-function-calling-60k

**Source**: `Salesforce/xlam-function-calling-60k`

**Size**: 60,000 examples

**Format**:
```json
{
  "query": "User question",
  "tools": "[JSON string with function schemas]",
  "answers": "[JSON string with function calls]"
}
```

**JSON Complexity**:
| Metric | Value |
|--------|-------|
| Avg Depth | 3.0 (consistent) |
| Max Depth | 12 |
| Avg Fields | 6.2 |
| Unique Keys | 3,179 |

**Nested Content**:
- **Function Schemas (tools)**: depth=3, keys=6-95
- **Function Calls (answers)**: depth=1-7, keys=2-9

**Strengths**:
- Large sample size (60K)
- High schema variability (3,179 unique keys)
- >95% accuracy (human verified)
- Apache 2.0 license

**Limitations**:
- Consistent depth (less structural variety)
- Function-calling specific domain

**Recommended Use Cases**:
- Large-scale consistency evaluation
- Schema variability testing
- Cross-function comparison

---

### 4. Maxscha JSON Instruction Generation

**Source**:
- `Maxscha/json-instruct-generation-large` (50K full)
- `Maxscha/json-instruct-generation` (10K full)

**Sampled Size**: 1,000 each for analysis

**JSON Complexity**:
| Metric | Maxscha-large | Maxscha |
|--------|---------------|---------|
| Avg Depth | 2.38 | 2.41 |
| Max Depth | 15 | 11 |
| Avg Fields | 13.1 | 13.9 |
| Avg Nodes | 22.8 | 24.3 |
| Unique Keys | 2,121 | 2,360 |

**Type Distribution**:
| Type | Maxscha-large | Maxscha |
|------|---------------|---------|
| String | 10,246 (45%) | 10,880 (45%) |
| Object | 3,408 (15%) | 3,642 (15%) |
| Integer | 3,320 (14%) | 3,683 (15%) |
| Array | 2,905 (13%) | 3,146 (13%) |
| Float | 2,469 (11%) | 2,561 (11%) |
| Boolean | 271 (1%) | 310 (1%) |
| Null | 154 (<1%) | 107 (<1%) |

**Strengths**:
- Good depth variability (1-15 levels)
- Rich type diversity
- Instruction-to-JSON format
- General domain coverage

**Limitations**:
- Moderate schema complexity
- Less standardized structure

---

### 5. Glaive Function Calling

**Source**: `glaiveai/glaive-function-calling-v2`

**Parsed Size**: 62 examples (from larger dataset)

**JSON Complexity**:
| Metric | Value |
|--------|-------|
| Avg Depth | 1.29 |
| Max Depth | 2 |
| Avg Fields | 2.6 |
| Unique Keys | 21 |

**Type Distribution**:
| Type | Count | Percentage |
|------|-------|------------|
| String | 178 | 60% |
| Object | 77 | 26% |
| Array | 17 | 6% |
| Integer | 14 | 5% |
| Float | 9 | 3% |

**Strengths**:
- Simple, consistent structure
- Good for baseline testing
- Multi-domain function calls

**Limitations**:
- Very shallow depth
- Limited schema variability
- Small parsed sample size

---

### 6. Jiraya HTML-to-JSON Extraction

**Source**: `Jiraya/html_to_json_information_extraction_dataset`

**Size**: 999 examples (6,930 full)

**JSON Complexity**:
| Metric | Value |
|--------|-------|
| Avg Depth | ~1.3 |
| Max Depth | 2 |
| Avg Fields | 10.2 |
| Unique Keys | 4 |

**Strengths**:
- Information extraction domain
- Real-world web data
- Consistent output format

**Limitations**:
- Very limited schema variability (4 unique keys)
- Shallow structure
- Domain-specific

---

## Dataset Comparison Matrix

### By Structural Complexity

```
High Complexity                                    Low Complexity
    |                                                    |
    v                                                    v
Toucan > Maxscha > Xlam > ShareGPT > Jiraya > Glaive
(7.24)   (2.4)    (3.0)   (~2.5)     (1.3)    (1.3)
```

### By Schema Variability (Unique Keys)

```
High Variability                                   Low Variability
    |                                                    |
    v                                                    v
Xlam > Maxscha > Toucan > ShareGPT > Glaive > Jiraya
(3179)  (2360)   (385+)   (varies)    (21)      (4)
```

### By Sample Size

```
Large                                              Small
  |                                                  |
  v                                                  v
Toucan > Xlam > Maxscha > Jiraya > ShareGPT > Glaive
(119K)   (60K)   (1K*)     (999)     (80)       (62)
```

---

## Recommended Testing Strategy

### Cost Considerations

Each sample requires **100 LLM calls** (10 runs × 10 temperatures from 0.0 to 0.9). This makes full-scale evaluation expensive. Dataset selection must balance:
- Statistical significance (sample count)
- Structural complexity (depth, variability)
- Cost efficiency

### Selected Datasets for STED Evaluation

| Dataset | Samples | Calls Required | Rationale |
|---------|---------|----------------|-----------|
| **ShareGPT** | 80 | 8,000 | Base templates, proven workflow |
| **Toucan** | TBD (sample) | TBD | High complexity (depth 7.24), multi-tool |

### Excluded Datasets

| Dataset | Reason for Exclusion |
|---------|---------------------|
| **Maxscha** | Structured evaluation only; schema too predictable for consistency testing |
| **Jiraya** | Simple/stable schema (4 unique keys); insufficient variation to show consistency differences |
| **Glaive** | Too shallow (depth 1.3); limited structural variability |
| **Xlam** | Consistent depth (3.0); less interesting for structural matching evaluation |

### Recommended Approach

**Phase 1: ShareGPT (Current)**
- 80 samples × 100 calls = 8,000 LLM calls
- Validates STED methodology
- Establishes baseline metrics

**Phase 2: Toucan (High Complexity)**
- Sample 50-100 examples from 119K pool
- Focus on diverse `subset_name` categories
- Tests STED on:
  - Deep nesting (avg depth 7.24)
  - Multi-tool scenarios (avg 8.5 tools)
  - High field density (avg 152.5 fields)

### Sampling Strategy for Toucan

```python
# Sample diverse examples from Toucan
from datasets import load_dataset

ds = load_dataset("beyoru/Toucan-1.5M-structured-Qwen", split="train", streaming=True)

# Group by subset_name for diversity
subsets = {}
for sample in ds:
    subset = sample['subset_name']
    if subset not in subsets:
        subsets[subset] = []
    if len(subsets[subset]) < 20:  # 20 per subset
        subsets[subset].append(sample)
    if sum(len(v) for v in subsets.values()) >= 100:
        break
```

### Cost Estimate

| Dataset | Samples | Calls | Est. Cost (Claude Sonnet) |
|---------|---------|-------|---------------------------|
| ShareGPT | 80 | 8,000 | ~$8-16 |
| Toucan (sampled) | 100 | 10,000 | ~$10-20 |
| **Total** | 180 | 18,000 | ~$18-36 |

---

## Final Selection Rationale

### Why ShareGPT + Toucan?

| Criterion | ShareGPT | Toucan | Combined Coverage |
|-----------|----------|--------|-------------------|
| **Depth Range** | 1-5 | 6-12 | Full spectrum (1-12) |
| **Complexity** | Low-Medium | High | Low to High |
| **Schema Stability** | Variable | Variable | Good for consistency testing |
| **Tool Calling** | Some | All | Function calling focus |
| **Cost** | 8K calls | 10K calls | ~18K total (manageable) |

### Why NOT Others?

| Dataset | Issue for Consistency Testing |
|---------|------------------------------|
| **Maxscha** | Schema too predictable; outputs follow fixed patterns |
| **Jiraya** | Only 4 unique keys; schema too stable to show variance |
| **Glaive** | Depth 1-2 only; too shallow for structural analysis |
| **Xlam** | Consistent depth=3; less structural variation |

### Key Insight

For **consistency evaluation**, you need datasets where:
1. LLM outputs can **vary structurally** (not just values)
2. Schema complexity is **high enough** to stress-test STED
3. Multiple valid output structures exist for same prompt

ShareGPT provides baseline complexity; Toucan provides high complexity with structural variability.

---

## Dataset Download Commands

### Selected Datasets (for STED evaluation)

```bash
# ShareGPT (base templates) - Already in use
python scripts/data/download_sharegpt_data.py --output-dir sharegpt_data

# Toucan (high complexity) - Recommended addition
python -c "from datasets import load_dataset; ds = load_dataset('beyoru/Toucan-1.5M-structured-Qwen'); ds.save_to_disk('toucan_data')"
```

### Other Datasets (reference only)

```bash
# Xlam - consistent depth, less suitable
python -c "from datasets import load_dataset; ds = load_dataset('Salesforce/xlam-function-calling-60k'); ds.save_to_disk('xlam_data')"

# Maxscha - structured eval only
python -c "from datasets import load_dataset; ds = load_dataset('Maxscha/json-instruct-generation-large'); ds.save_to_disk('maxscha_data')"

# Glaive - too shallow
python -c "from datasets import load_dataset; ds = load_dataset('glaiveai/glaive-function-calling-v2'); ds.save_to_disk('glaive_data')"
```

---

## References

- [ShareGPT Structured Output](https://huggingface.co/datasets/Arun63/sharegpt-structured-output-json)
- [Toucan-1.5M-structured-Qwen](https://huggingface.co/datasets/beyoru/Toucan-1.5M-structured-Qwen)
- [Salesforce xlam-function-calling-60k](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k)
- [Maxscha JSON Instruct Generation](https://huggingface.co/datasets/Maxscha/json-instruct-generation-large)
- [Glaive Function Calling v2](https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2)
- [Jiraya HTML-to-JSON](https://huggingface.co/datasets/Jiraya/html_to_json_information_extraction_dataset)

---

*Last updated: December 2024*
