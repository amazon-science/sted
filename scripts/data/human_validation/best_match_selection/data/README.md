# Human Validation Datasets for Best-Match Selection

This directory contains human validation datasets for evaluating STED against baseline similarity methods.

## Task Description

**Best-Match Selection**: For each sample, compare the ground truth against multiple LLM-generated responses. Each similarity method (STED, BERTScore, DeepDiff, TED) picks its "most similar" response. Human annotators then select which response is actually most similar to the ground truth.

This approach provides an objective evaluation with high inter-annotator agreement, as it's essentially a "best-match retrieval" task.

## Terminology

- **Sample**: A unique prompt + ground truth pair that has multiple LLM-generated responses to compare
- **Candidate**: One of the LLM-generated responses for a sample (each method picks its "best" candidate)
- **Disagreement**: When at least 2 methods pick different candidates as "most similar" to ground truth
- **Validation Item**: A sample where methods disagreed, included in the dataset for human annotation

## Datasets

### 1. Toucan Dataset (Tool Calls) - Full Disagreement Set
- **File**: `toucan_validation_dataset.json`
- **Size**: 1.4 MB
- **Items**: 343 samples where methods disagreed
- **Format**: Tool call JSON structures
- **Model Filter**: Only includes responses from the 18 curated FINAL_MODELS

**Disagreement Statistics**:
| Metric | Value |
|--------|-------|
| Total samples processed | 990 |
| Samples with method agreement | 647 (65.4%) |
| Samples with method disagreement | 343 (34.6%) |

**Method Pick Distribution**:
| Method | A | B | C | D |
|--------|---|---|---|---|
| STED | 138 | 165 | 34 | 6 |
| DeepDiff | 166 | 145 | 27 | 5 |
| TED | 146 | 156 | 36 | 5 |
| BERTScore | 165 | 139 | 35 | 4 |

**Complexity Distribution**:
| Complexity | Count | Criteria |
|------------|-------|----------|
| Simple | 0 | depth <= 2, nodes <= 10 |
| Medium | 289 | depth 3-4, nodes 10-30 |
| Complex | 54 | depth >= 5 or nodes >= 30 |

**Candidates per Item**: mean=2.4, min=2, max=4

*Note: Toucan tool calls naturally have depth 3+ structures, resulting in no "simple" samples.*

### 1b. Toucan Dataset (Subset for Comparison)
- **File**: `toucan_validation_dataset_79.json`
- **Size**: ~240 KB
- **Items**: 79 samples (subset to match ShareGPT size)
- **Purpose**: Enables fair comparison with ShareGPT (similar sample count)
- **Strategy**: Keep ALL complex samples, randomly sample medium

**Complexity Distribution**:
| Complexity | Count | Note |
|------------|-------|------|
| Simple | 0 | None available |
| Medium | 57 | Randomly sampled |
| Complex | 22 | All included (100%) |

### 2. ShareGPT Dataset (Structured Outputs)
- **File**: `sharegpt_validation_dataset.json`
- **Size**: 1.08 MB
- **Items**: 79 samples where methods disagreed
- **Format**: Structured JSON outputs (e.g., `{"modelReport": {...}}`)

**Disagreement Statistics**:
| Metric | Value |
|--------|-------|
| Total samples in dataset | 80 |
| Samples with method disagreement | 79 |
| Disagreement rate | 98.8% |

**Complexity Distribution**:
| Complexity | Count | Criteria |
|------------|-------|----------|
| Simple | 5 | depth <= 2, nodes <= 10 |
| Medium | 5 | depth 3-4, nodes 10-30 |
| Complex | 69 | depth >= 5 or nodes >= 30 |

*Note: ShareGPT structured outputs are inherently complex (87% have depth>=5 or nodes>=30).*

**Method Pick Distribution**:
| Method | A | B | C | D |
|--------|---|---|---|---|
| STED | 26 | 26 | 20 | 7 |
| DeepDiff | 33 | 27 | 9 | 10 |
| TED | 28 | 27 | 17 | 7 |
| BERTScore | 16 | 29 | 30 | 4 |

**Candidates per Item**: mean=3.2, min=2, max=4

### 2b. ShareGPT Dataset (Final Models Only)
- **File**: `sharegpt_validation_final_models.json`
- **Size**: 1.22 MB
- **Items**: 76 samples where methods disagreed
- **Format**: Structured JSON outputs
- **Model Filter**: Only includes responses from the 18 curated FINAL_MODELS

**FINAL_MODELS List** (18 models):
```
Qwen3-235B-A22B, Claude-3.5-Sonnet, Claude-Haiku-4.5, Claude-3.7-Sonnet,
Claude-3.5-Haiku, Claude-Opus-4.5, Claude-Opus-4, Claude-Sonnet-4,
Claude-Sonnet-4.5, Qwen3-32B, Llama-3.3-70B, Nova-2-Lite, Mimo-V2-Flash,
Grok-4.1-Fast, Minimax-M2, GPT-4.1-Mini, Gemini-2.5-Flash-Lite, GPT-OSS-120B
```

**Disagreement Statistics**:
| Metric | Value |
|--------|-------|
| Total samples processed | 76 |
| Samples with method disagreement | 76 |
| Disagreement rate | 100% |

**Complexity Distribution**:
| Complexity | Count | Criteria |
|------------|-------|----------|
| Simple | 5 | depth <= 2, nodes <= 10 |
| Medium | 3 | depth 3-4, nodes 10-30 |
| Complex | 68 | depth >= 5 or nodes >= 30 |

*Note: This dataset filters to only the 18 final models used in the ICML paper visualization, ensuring consistency between human validation and model comparison results.*

**Generation Command**:
```bash
python scripts/data/human_validation/best_match_selection/generate_dataset.py \
  --results-dir llm_gen_results \
  --output sharegpt_validation_final_models.json \
  --n-items 100 --max-samples 150 \
  --dataset sharegpt --final-models-only
```

## Dataset Schema

Each dataset follows this JSON structure:

```json
{
  "metadata": {
    "created": "2026-01-28T...",
    "n_items": 79,
    "task_type": "ranking",
    "methods_compared": ["sted", "deepdiff", "ted", "bertscore"]
  },
  "annotation_guidelines": {
    "task": "Given the Ground Truth, choose which candidate response is MOST similar.",
    "criteria": [
      "Same tool names and function calls",
      "Same parameter names and values",
      "Same structure (order may differ)",
      "Could be used interchangeably in downstream system"
    ],
    "confidence_scale": {
      "1": "Very uncertain - all look equally similar/different",
      "2": "Somewhat uncertain",
      "3": "Moderately confident",
      "4": "Confident",
      "5": "Very confident - clearly best match"
    }
  },
  "items": [
    {
      "id": "rank_0000",
      "sample_id": "...",
      "ground_truth": { ... },
      "candidates": [
        {
          "label": "A",
          "response": { ... },
          "model": "model-name"
        },
        ...
      ],
      "metadata": {
        "dataset": "toucan",
        "depth": 4,
        "node_count": 25,
        "array_count": 3,
        "object_count": 8,
        "method_picks": {"sted": "A", "deepdiff": "B", ...},
        "n_candidates": 3,
        "candidate_models": ["Claude-3.5-Sonnet", "GPT-4.1-Mini", ...],
        "gt_metrics": {"depth": 4, "node_count": 25, "array_count": 3, "object_count": 8}
      },
      "annotation": {
        "choice": null,
        "confidence": null,
        "annotator_id": null,
        "timestamp": null
      }
    }
  ]
}
```

## Per-Sample Metadata Fields

Each sample includes the following metadata for deep analysis:

| Field | Description |
|-------|-------------|
| `dataset` | Source dataset ("toucan" or "sharegpt") |
| `depth` | Maximum nesting depth of the ground truth JSON |
| `node_count` | Total number of nodes in the ground truth JSON tree |
| `array_count` | Number of array elements in the ground truth |
| `object_count` | Number of object elements in the ground truth |
| `method_picks` | Which candidate each method selected as "most similar" |
| `n_candidates` | Number of candidate responses (2-4) |
| `candidate_models` | List of LLM models that generated the candidates |
| `gt_metrics` | Full structural metrics object for reference |

## Methods Compared

| Method | Description |
|--------|-------------|
| **STED** | Semantic Tree Edit Distance - combines structural and semantic similarity |
| **BERTScore** | Semantic similarity using BERT embeddings on serialized JSON |
| **DeepDiff** | Structural difference metric based on dictionary comparison |
| **TED** | Traditional Tree Edit Distance using ZSS algorithm |

## Generation Details

- **Generated**: 2026-01-28 (Toucan updated 2026-01-29)
- **Embedding Model**: all-MiniLM-L6-v2
- **Selection Criteria**: Only samples where at least 2 methods disagree on the best match
- **Source Data**:
  - Toucan: 990 samples processed from toucan_tool_calls_1006.json → 343 disagreements (34.6%)
  - ShareGPT: 80 samples processed from sharegpt structured outputs → 79 disagreements (98.8%)

## Sampling Strategy

### Sample Selection Pipeline
1. **Load samples** with ground truth and multiple LLM responses (minimum 4 models required)
2. **Compute similarity scores** for each (ground_truth, response) pair using all 4 methods
3. **Identify method picks**: Each method selects its "most similar" response
4. **Filter for disagreements**: Only include samples where at least 2 methods pick different responses
5. **Stratify by complexity**: Balance samples across simple/medium/complex categories

### Stratification Criteria
| Complexity | Criteria | Description |
|------------|----------|-------------|
| Simple | depth <= 2 AND node_count <= 10 | Flat structures with few nodes |
| Medium | depth 3-4, node_count 10-30 | Moderate nesting and size |
| Complex | depth >= 5 OR node_count >= 30 | Deep nesting or many nodes |

### Why Disagreement Filtering?
- **Higher signal**: When methods agree, human validation adds little information
- **Efficient annotation**: Focus human effort on cases where methods differ
- **Method comparison**: Directly measures which method aligns best with human judgment

## Response Sampling Strategy

**Temperature**: T=0.0 (deterministic outputs)
- Uses only T=0.0 responses for each model to ensure consistent, reproducible comparisons
- Falls back to T=0.1 if T=0.0 not available

**Model Selection**:
- Toucan (full dataset): 18 FINAL_MODELS (see list in Section 2b)
- ShareGPT: 25+ models (or 18 FINAL_MODELS with `--final-models-only`)

**Models Used**:
| Provider | Models |
|----------|--------|
| Anthropic | Claude-3.5-Haiku, Claude-3.5-Sonnet, Claude-3.7-Sonnet, Claude-Haiku-4.5, Claude-Sonnet-4, Claude-Sonnet-4.5, Claude-Opus-4, Claude-Opus-4.5 |
| OpenAI | GPT-4.1-Mini |
| Google | Gemini-2.5-Flash-Lite |
| xAI | Grok-4.1-Fast |
| Meta | Llama-3.3-70B |
| Mistral | Mistral-Large-3-675B |
| NVIDIA | Nemotron-Nano |
| Amazon | Nova-2-Lite, Nova-Pro |
| Alibaba | Qwen3-32B, Qwen3-235B |
| MiniMax | MiniMax-M2 |
| Mimo | Mimo-V2-Flash |
| OSS | GPT-OSS-120B |

**Minimum Responses**: >= 4 model responses per sample required for inclusion

## Annotation Interface

A web-based annotation interface is available in `human_validation_interface/` for collecting human judgments. The interface:
- Shows ground truth and candidate responses side-by-side
- Allows selection of the most similar response
- Supports confidence ratings (1-5)
- Saves progress to localStorage
- Computes win rates for each method

## Expected Analysis

After human annotation:
1. **Win Rate**: Percentage of times each method's pick matches human choice
2. **Agreement**: Inter-annotator agreement (if multiple annotators)
3. **Stratified Analysis**: Performance by complexity level
4. **Error Analysis**: Cases where STED disagrees with human judgment
