# Real-World JSON Dataset Collection Plan for STED Evaluation

**Purpose:** Collect large-scale, real-world JSON datasets for comprehensive STED evaluation to strengthen the paper for top-tier venues (NeurIPS, ICML, ICLR).

**Target:** 10,000+ JSON pairs across diverse domains and complexity levels.

---

## 1. Function Calling Datasets (High Priority)

### 1.1 Salesforce/xlam-function-calling-60k
- **Source:** https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k
- **Size:** 60,000 examples
- **Format:** JSON with query, tools, answers
- **Use Case:** Evaluate consistency of function call generation
- **Quality:** >95% accuracy (human verified)
- **License:** Apache 2.0

**How to use for STED:**
```python
from datasets import load_dataset
dataset = load_dataset("Salesforce/xlam-function-calling-60k")
# Each example has ground-truth function calls
# Generate multiple LLM outputs for same query, measure consistency
```

### 1.2 Glaive Function Calling v2
- **Source:** https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2
- **Size:** 113,000 examples
- **Format:** Parquet with system prompts and function calls
- **Use Case:** Multi-domain function calling consistency
- **License:** Apache 2.0

### 1.3 HydraLM Glaive Standardized
- **Source:** https://huggingface.co/datasets/HydraLM/glaive_function_calling_v1_standardized
- **Size:** 380,000 examples
- **Format:** Standardized function calling format
- **Use Case:** Large-scale consistency evaluation

---

## 2. Berkeley Function Calling Leaderboard (BFCL)

### 2.1 BFCL Benchmark Data
- **Source:** https://github.com/ShishirPatil/gorilla
- **Versions:** V1-V4 with increasing complexity
- **Features:**
  - V1: Single-turn function calling
  - V2: Enterprise real-world scenarios
  - V3: Multi-turn interactions
  - V4: Agentic evaluation with web search
- **Format:** JSON with function schemas and test cases
- **License:** Apache 2.0

**Download:**
```bash
git clone https://github.com/ShishirPatil/gorilla
cd gorilla/berkeley-function-call-leaderboard
```

---

## 3. Information Extraction Datasets

### 3.1 HTML to JSON Extraction
- **Source:** https://huggingface.co/datasets/Jiraya/html_to_json_information_extraction_dataset
- **Size:** 6,930 examples
- **Use Case:** Structured data extraction consistency

### 3.2 OmniStruct Benchmark
- **Paper:** arxiv:2511.18335
- **Tasks:** Information extraction, table generation, function calling
- **Note:** Check paper for dataset availability

---

## 4. JSON Instruction Generation

### 4.1 JSON Instruct Generation Large
- **Source:** https://huggingface.co/datasets/Maxscha/json-instruct-generation-large
- **Size:** 50,000 examples
- **Use Case:** Instruction-to-JSON generation consistency

### 4.2 JSON Instruct Generation
- **Source:** https://huggingface.co/datasets/Maxscha/json-instruct-generation
- **Size:** 10,000 examples

---

## 5. Proposed Experiment Design

### 5.1 Consistency Evaluation Protocol

For each dataset, we will:

1. **Sample Selection:** Select 1,000 representative examples per dataset
2. **Multi-LLM Generation:** Generate outputs from 4+ LLMs:
   - GPT-4o
   - Claude 3.5 Sonnet
   - Llama 3.1 70B
   - Gemini 1.5 Pro
3. **Multi-Run Consistency:** Generate 5 outputs per prompt per model
4. **Metrics Comparison:** Compare STED vs baselines on all pairs

### 5.2 Evaluation Dimensions

| Dimension | Description | Datasets |
|-----------|-------------|----------|
| **Cross-Model Consistency** | Same prompt, different models | All |
| **Within-Model Consistency** | Same prompt, same model, multiple runs | All |
| **Complexity Scaling** | How metrics perform vs JSON complexity | Function calling |
| **Semantic Preservation** | Detecting semantic equivalence | IE datasets |

### 5.3 Expected Dataset Summary

| Dataset | Size | Domain | Complexity |
|---------|------|--------|------------|
| xlam-function-calling | 60K | API calls | Medium |
| glaive-function-calling-v2 | 113K | Multi-domain | Medium |
| BFCL v4 | ~5K | Enterprise | High |
| json-instruct-large | 50K | General | Low-Medium |
| html-to-json | 7K | Web extraction | High |
| **Total** | **~235K** | - | - |

---

## 6. Implementation Plan

### Phase 1: Data Collection (Week 1)
- [ ] Download all datasets from HuggingFace
- [ ] Clone BFCL repository
- [ ] Create unified data loader

### Phase 2: Data Preprocessing (Week 1-2)
- [ ] Standardize JSON formats
- [ ] Remove invalid/malformed examples
- [ ] Create train/test splits
- [ ] Sample 1K per dataset for experiments

### Phase 3: LLM Output Generation (Week 2-3)
- [ ] Set up API access for GPT-4o, Claude, Gemini
- [ ] Generate 5 outputs per prompt per model
- [ ] Store all outputs with metadata

### Phase 4: Evaluation (Week 3-4)
- [ ] Run STED on all pairs
- [ ] Run baseline metrics (TED, BERTScore, DeepDiff, LLM-Judge)
- [ ] Compute correlations and statistical tests

### Phase 5: Analysis (Week 4-5)
- [ ] Create comparison tables
- [ ] Generate visualizations
- [ ] Write results section

---

## 7. Code Structure

```
research/
├── datasets/
│   ├── download_datasets.py      # Download all datasets
│   ├── preprocess_datasets.py    # Standardize formats
│   └── data_loaders/
│       ├── function_calling.py
│       ├── bfcl.py
│       └── info_extraction.py
├── generation/
│   ├── generate_outputs.py       # Multi-LLM generation
│   └── models/
│       ├── openai_model.py
│       ├── claude_model.py
│       └── gemini_model.py
├── evaluation/
│   ├── run_all_metrics.py        # Comprehensive evaluation
│   └── analyze_results.py        # Statistical analysis
└── results/
    └── real_world_benchmark/
```

---

## 8. Expected Contributions

1. **Large-scale real-world evaluation** (235K+ examples)
2. **Multi-LLM consistency study** (4+ models)
3. **Domain-specific analysis** (function calling, IE, general JSON)
4. **Complexity analysis** (metric behavior vs JSON complexity)

This will significantly strengthen the paper for main conference submission.

---

## 9. References

- [xlam-function-calling-60k](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k)
- [glaive-function-calling-v2](https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2)
- [BFCL Benchmark](https://gorilla.cs.berkeley.edu/leaderboard.html)
- [OmniStruct](https://arxiv.org/abs/2511.18335)
