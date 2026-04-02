# STED and Consistency Scoring: A Framework for Evaluating LLM Structured Output Reliability

A comprehensive framework for evaluating and improving consistency in LLM-generated structured outputs. This framework combines STED (Semantic Tree Edit Distance), a novel similarity metric that balances semantic flexibility with structural strictness, with a consistency scoring framework that aggregates multiple STED measurements to quantify output reliability.

> 📄 **Paper**: Accepted at NeurIPS 2025 Workshop on Structured Probabilistic Inference & Generative Modeling

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Dataset](#dataset)
- [STED Effectiveness Verification](#sted-effectiveness-verification)
- [LLM Consistency Benchmarking](#llm-consistency-benchmarking)
- [Key Components](#key-components)
- [Results and Findings](#results-and-findings)
- [Contributing](#contributing)
- [Citation](#citation)

## Overview

Large Language Models (LLMs) are increasingly deployed for structured data generation tasks, yet their output consistency remains a critical challenge for production applications. This framework addresses this challenge through two key contributions:

1. **STED (Semantic Tree Edit Distance)**: A novel similarity metric that balances semantic flexibility with structural strictness when comparing JSON outputs
2. **Consistency Scoring Framework**: Aggregates multiple STED measurements across repeated generations to quantify output reliability

Through systematic experiments on synthetic datasets with controlled schema, expression, and semantic variations, STED achieves superior performance (0.86–0.90 similarity for semantic equivalents, 0.0 for structural breaks) compared to existing metrics including TED, BERTScore, and DeepDiff.

## Key Features

- **Semantic Tree Edit Distance (STED)**: Advanced similarity calculation combining structural and semantic analysis
- **Multiple Variation Types**: Support for schema, expression, and semantic variations
- **LLM Benchmarking**: Comprehensive evaluation of different LLMs across temperature settings
- **MCP Server Support**: Model Context Protocol server for real-time consistency evaluation in agentic systems
- **Synthetic Dataset Generation**: Automated generation of variation datasets for evaluation
- **Visualization Tools**: Rich plotting and analysis capabilities for results interpretation

## Installation

```bash
# Clone the repository
git clone https://github.com/amazon-science/sted.git
cd sted

# Install the library
pip install -e .

# Or with uv
uv pip install -e .
```

For development:
```bash
pip install -e ".[dev]"
# Or: uv sync
```

### Credentials Setup

**AWS Credentials** (for Bedrock embedding models):
```bash
aws configure
# Or set: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION
```

**OpenAI API Key** (optional, for OpenAI model evaluation):
```bash
export OPENAI_API_KEY=<your-openai-api-key>
```

### Troubleshooting

**NumPy/PyTorch Compatibility Error**

If you see `_ARRAY_API not found` or "module compiled using NumPy 1.x cannot be run in NumPy 2.x":

```bash
# Option 1: Downgrade NumPy (if using PyTorch < 2.4)
pip install "numpy<2"

# Option 2: Upgrade PyTorch (recommended)
pip install --upgrade "torch>=2.4"
```

## Quick Start

### Basic Similarity Calculation

```python
from sted.semantic_json_tree_consistency import SemanticJsonTreeConsistencyEvaluator

# Initialize evaluator
evaluator = SemanticJsonTreeConsistencyEvaluator(
    model_id='amazon.titan-embed-text-v2:0'
)

# Compare JSON structures
json1 = {'name': 'John', 'age': 30, 'city': 'New York'}
json2 = {'name': 'John', 'age': 30, 'location': 'NYC'}

# Calculate similarity
similarity = evaluator.calculate_tree_edit_distance_opt(
    json1, json2, 
    variation_type="combined"
)
print(f"Similarity: {similarity:.4f}")  # Output: 0.8650
```

### Batch Consistency Evaluation

```python
from sted.semantic_json_tree_consistency import SemanticJsonTreeConsistencyEvaluator
from sted.structural_consistency_analyzer import StructuralConsistencyAnalyzer

evaluator = SemanticJsonTreeConsistencyEvaluator(model_id='amazon.titan-embed-text-v2:0')
analyzer = StructuralConsistencyAnalyzer(evaluator)

# Multiple LLM outputs for the same prompt
json_outputs = [
    {'name': 'Alice', 'age': 25, 'city': 'New York'},
    {'name': 'Alice', 'age': 25, 'city': 'NYC'},
    {'name': 'Alice', 'age': 25, 'location': 'New York City'},
]

result = analyzer.evaluate_structural_consistency(
    json_outputs, method_name="ted", variation_type="combined"
)
print(f"Consistency: {result['consistency_metrics']['consistency_coefficient']:.4f}")
```

For more examples, see `examples/basic_usage.py` and [Library Usage Guide](./docs/guides/LIBRARY_USAGE.md).

## Dataset

The framework supports two datasets for evaluation:

### ShareGPT Dataset (Structured Output)

JSON structured output generation tasks:
- **sharegpt-structured-output-json**: 30 samples (report extraction, data parsing)
- **sharegpt-quizz-generation-json-output**: 50 samples (MCQ generation)
- **Total**: 80 samples (75 valid after parsing error exclusion)

```bash
# Download ShareGPT data
python scripts/data/download_sharegpt_data.py
```

### Toucan Dataset (Tool Calling)

Tool/function calling evaluation dataset:
- **Source**: [Nexusflow/Toucan](https://huggingface.co/datasets/Nexusflow/Toucan)
- **Content**: Multi-turn conversations with tool calls
- **Use case**: Evaluating LLM tool calling consistency

```bash
# Download Toucan data
python scripts/data/download_toucan_data.py
```

### Generate Synthetic Datasets

For STED effectiveness verification:

```bash
python scripts/data/generate_synthetic_datasets.py --base-dataset-dir sharegpt_data
```

Creates three variation types:
- **Schema Variation**: Field name changes, structure flattening/nesting
- **Expression Variation**: Different expressions with same semantic meaning
- **Semantic Variation**: Changes in semantic content

## STED Effectiveness Verification

### Similarity Progression Analysis

```bash
python scripts/dataset_analysis/analyze_semantic_expression_variation_progression.py \
  synthetic_dataset/expression_variation_dataset_*.json \
  synthetic_dataset/semantic_variation_dataset_*.json \
  --output-dir results/variation_progression
```

### Visualization

```bash
# Expression and Semantic Variation
python scripts/visualization/visualize_variation_progression.py

# Schema Variation
python scripts/visualization/visualize_schema_variation.py
```

## LLM Consistency Benchmarking

The framework supports two experiment modes for different datasets:

| Mode | Dataset | Use Case |
|------|---------|----------|
| `structured` | ShareGPT | JSON structured output generation (MCQ, reports, etc.) |
| `tool-calling` | Toucan | Tool/function calling evaluation |

### Model Setup

The framework supports models from multiple providers. Model configuration is centralized in `sted/model_config.py`:

```python
# sted/model_config.py
MODEL_REGISTRY = {
    # model_id -> (provider, display_name)
    "us.anthropic.claude-3-7-sonnet-20250219-v1:0": ("bedrock", "Claude-3.7-Sonnet"),
    "us.deepseek.v3-v1:0": ("bedrock", "DeepSeek-V3.1"),
    "openai/gpt-4o": ("openai", "GPT-4o"),
    "google/gemini-2.5-pro": ("openai", "Gemini-2.5-Pro"),
    # ... add more models here
}
```

**Provider Types:**

| Provider | Model ID Format | API Used | Credentials |
|----------|----------------|----------|-------------|
| `bedrock` | `us.<provider>.<model>-v1:0` | AWS Bedrock Converse API | AWS credentials |
| `openai` | `<provider>/<model>` | OpenAI-compatible API | `OPENAI_API_KEY`, `OPENAI_BASE_URL` |

**To add a new model:**

1. Add entry to `MODEL_REGISTRY` in `sted/model_config.py`:
   ```python
   "us.meta.llama3-3-70b-instruct-v1:0": ("bedrock", "Llama-3.3-70B"),
   ```

2. **Configure credentials** in `.env`:
   ```bash
   # For Bedrock models - only AWS credentials needed (via aws configure)
   
   # For OpenAI-compatible APIs - set these environment variables
   OPENAI_API_KEY=<your-api-key>
   OPENAI_BASE_URL=https://openrouter.ai/api/v1  # Optional, for OpenRouter
   ```

**Note**: When using Bedrock models, you don't need to set `OPENAI_API_KEY`. The script only requires OpenAI credentials when using models with the `"openai"` provider type.

### Step 1: Generate LLM Outputs

#### ShareGPT Experiment (Structured Output)

For JSON structured output generation tasks (MCQ generation, report extraction, etc.):

```bash
# Download ShareGPT data first
python scripts/data/download_sharegpt_data.py

# Run temperature experiment
python scripts/eval/run_temperature_experiment.py \
  --mode structured \
  --data-dir sharegpt_data \
  --output-dir llm_gen_results/sharegpt \
  --model-id us.anthropic.claude-3-5-haiku-20241022-v1:0 \
  --max-tokens 8000
```

**Key parameters:**
- `--mode structured`: Use structured output mode (default)
- `--data-dir`: Directory containing ShareGPT JSON files
- `--run-num`: Number of inference runs per sample (default: 10)
- `--sample-limit`: Maximum samples to process (default: all samples)
- `--include-schema`: Include JSON schema in prompt for better formatting

#### Toucan Experiment (Tool Calling)

For tool/function calling evaluation:

```bash
# Download Toucan data first
python scripts/data/download_toucan_data.py

# Run temperature experiment
python scripts/eval/run_temperature_experiment.py \
  --mode tool-calling \
  --dataset-path toucan_data/toucan_tool_calls.json \
  --output-dir llm_gen_results/toucan \
  --model-id us.anthropic.claude-3-5-haiku-20241022-v1:0 \
  --max-tokens 1024
```

**Key parameters:**
- `--mode tool-calling`: Use tool calling mode
- `--dataset-path`: Path to Toucan dataset JSON file
- `--run-num`: Number of inference runs per sample (default: 10)
- `--sample-limit`: Maximum samples to process (default: all samples)
- `--max-workers`: Parallel workers for inference (default: 10)
- `--start-idx`: Starting index in dataset (default: 0)

#### Running Individual Temperatures

For finer control, you can run `generate_structured_outputs.py` or `generate_tool_calls.py` directly:

```bash
# ShareGPT - single temperature
python scripts/eval/generate_structured_outputs.py \
  --data-dir sharegpt_data \
  --output-dir llm_gen_results/sharegpt/generations-model-timestamp \
  --temperature 0.5 \
  --model-id us.anthropic.claude-3-5-haiku-20241022-v1:0

# Toucan - single temperature
python scripts/eval/generate_tool_calls.py \
  --dataset-path toucan_data/toucan_tool_calls.json \
  --dataset-type toucan \
  --output-dir llm_gen_results/toucan/generations-model-timestamp \
  --temperature 0.5 \
  --model us.anthropic.claude-3-5-haiku-20241022-v1:0
```

### Step 2: Calculate Consistency Metrics

```bash
python scripts/eval/calculate_consistency_metrics.py
```

### Step 3: Visualize Results

```bash
python scripts/visualization/visualize_consistency_scores.py
```

For all scripts, see [Scripts Reference](./docs/guides/SCRIPTS_REFERENCE.md).

## Key Components

### STED Algorithm

STED extends classical tree edit distance with semantic awareness:

- **Semantic-Enhanced Tree Edit Distance**: Recognizes equivalent keys and values while preserving structural constraints
- **Order-Invariant Matching**: Uses Hungarian algorithm for optimal element pairing in O(n³) time
- **Multi-Level Similarity**: Integrates structural, key, value, and type similarities with configurable weights

The semantic update cost: `γ_upd(v1, v2) = w_s · γ_struct(v1, v2) + w_c · γ_content(v1, v2)`

### Variation Types

| Type | Description | Example |
|------|-------------|---------|
| Schema | Field name/structure changes | `"user_name"` → `"userName"` |
| Expression | Linguistic variations, same meaning | `"good book"` → `"nice book"` |
| Semantic | Content meaning changes | Should trigger alerts |

### Algorithm Complexity

- Tree construction: O(n)
- Embedding computation: O(k) with caching
- Optimized STED: O(n₁ × n₂ × (n₁ + n₂))
- Hungarian algorithm: O(max(n₁, n₂)³)

See [STED Complexity Analysis](./docs/api/sted_complexity_analysis.md) for details.

## Results and Findings

Evaluated **19 LLMs** across multiple temperature settings:

| Provider | Models |
|----------|--------|
| Anthropic | Claude 3 Haiku, Claude 3.5 Haiku, Claude 3.5 Sonnet, Claude 3.7 Sonnet |
| Amazon | Nova Lite, Nova Pro v1 |
| Meta | Llama 3.1 70B, Llama 3.1 405B, Llama 3.3 70B, Llama 4 Maverick |
| OpenAI | GPT-4o, GPT-4.1 Mini |
| Google | Gemini 2.5 Flash, Gemini 2.5 Flash Lite, Gemini 2.5 Pro |
| DeepSeek | DeepSeek V3, DeepSeek V3.1 |
| Alibaba | Qwen3 32B, Qwen3 235B A22B |

**Scale**: 19 models, 11 temperature settings (0.0–1.0), ~2.1M outputs across 1,006 tool-calling samples

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Citation

If you use this framework in your research, please cite:

```bibtex
@inproceedings{wang2025sted,
  title={STED and Consistency Scoring: A Framework for Evaluating LLM Structured Output Reliability},
  author={Wang, Guanghui and Yu, Jinze and Zhang, Xing and Jiang, Dayuan and Deb, Tomal and Liu, Xuefeng and He, Peiyang and Song, Yin},
  booktitle={NeurIPS 2025 Workshop on Structured Probabilistic Inference \& Generative Modeling},
  year={2025}
}
```
