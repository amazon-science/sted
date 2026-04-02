# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

STED (Semantic Tree Edit Distance) is a framework for evaluating consistency in LLM-generated structured outputs. It combines a novel similarity metric (STED) with a consistency scoring framework to quantify output reliability for JSON/tool-calling tasks.

## Build and Development Commands

```bash
# Install in development mode
pip install -e .
# Or with uv
uv pip install -e .

# Run tests
pytest tests/
pytest tests/test_basic_sted.py -v  # Single test file

# Run a specific test
pytest tests/test_basic_sted.py::test_function_name -v
```

## Key Scripts

### Data Collection
```bash
# Download datasets
python scripts/data/download/download_sharegpt_data.py
python scripts/data/download/download_toucan_data.py

# Generate synthetic datasets for STED validation
python scripts/data/generate_synthetic_datasets.py --base-dataset-dir sharegpt_data
```

### LLM Generation Experiments
```bash
# Run temperature experiment (generates outputs across temps 0.0-1.0)
python scripts/eval/run_temperature_experiment.py \
  --mode tool-calling \
  --dataset-path toucan_data/toucan_tool_calls.json \
  --output-dir llm_gen_results/toucan \
  --model-id us.anthropic.claude-3-5-haiku-20241022-v1:0

# Single temperature generation
python scripts/eval/generate_tool_calls.py \
  --dataset-path toucan_data/toucan_tool_calls.json \
  --temperature 0.7 \
  --model us.anthropic.claude-3-5-haiku-20241022-v1:0
```

### Consistency Calculation
```bash
# Calculate STED consistency metrics from generation results
python scripts/eval/calculate_consistency_metrics.py \
  --results-dir llm_gen_results/toucan \
  --output-dir results
```

## Architecture

### Core Library (`sted/`)

- **`semantic_json_tree_consistency.py`**: Main STED evaluator class (`SemanticJsonTreeConsistencyEvaluator`). Computes semantic tree edit distance between JSON structures using embedding-based similarity and Hungarian algorithm for optimal matching.

- **`structural_consistency_analyzer.py`**: `StructuralConsistencyAnalyzer` wraps STED to compute aggregate consistency metrics (c_mean, c_std) across multiple LLM outputs.

- **`model_config.py`**: Centralized model registry mapping model IDs to providers (bedrock/openai), display names, and rate limits. Add new models here.

- **`bedrock_utils.py`**: AWS Bedrock API utilities for both Converse API (Bedrock models) and OpenAI-compatible APIs (OpenRouter).

- **`json_tree_node.py`**: `JsonNode` class for tree representation of JSON structures.

### Provider Architecture

The framework supports two provider types configured in `MODEL_REGISTRY`:

| Provider | Model ID Format | API |
|----------|----------------|-----|
| `bedrock` | `us.<provider>.<model>-v1:0` | AWS Bedrock Converse API |
| `openai` | `<provider>/<model>` | OpenAI-compatible (OpenRouter) |

### Experiment Pipeline

1. **Generate**: `run_temperature_experiment.py` → calls `generate_tool_calls.py` per temperature
2. **Store**: Results saved to `llm_gen_results/{dataset}/{model}/temp_X_XX/all_results.json`
3. **Evaluate**: `calculate_consistency_metrics.py` computes STED metrics
4. **Analyze**: Scripts in `scripts/analysis/` for statistical analysis

### Research Experiments

Research experiments live in `scripts/experiments/`:
- `acl_linguistic_variations/`: Linguistic intervention experiments
- `colm_architecture/`: Model architecture analysis experiments

Pre-collected results (2.1M outputs, 19 models) are in `llm_gen_results/toucan/`.

## Environment Setup

```bash
# AWS credentials (for Bedrock models)
aws configure

# OpenAI-compatible APIs (for OpenRouter models)
export OPENAI_API_KEY=<your-key>
export OPENAI_BASE_URL=https://openrouter.ai/api/v1
```

## Data Locations

- `data/toucan/`: Toucan tool-calling dataset (1006 samples)
- `data/acl_stratified/`: Stratified samples for experiments
- `llm_gen_results/`: Pre-collected LLM generation results
- `results/`: Analysis outputs and metrics
- `experiments/`: Experiment records and configurations

---

## Experiment Management Rules

**IMPORTANT**: All experiments MUST follow these rules for reproducibility.

### Rule 1: Create Experiment Record First

Before running any experiment, create a record:

```bash
python experiments/scripts/manage_experiment.py create \
    --name "experiment_name" \
    --description "What this experiment does" \
    --script "scripts/experiments/path/to/script.py" \
    --data-s3 "s3://<your-s3-bucket>/data/toucan/" \
    --paper "COLM2026"
```

### Rule 2: Record All Run Details

Every experiment run MUST record:

| Field | Description | Example |
|-------|-------------|---------|
| `instance_id` | EC2 instance used | `i-0abc123def456` |
| `instance_type` | EC2 instance type | `t3.xlarge` |
| `data_source` | S3 URI of input data | `s3://<your-s3-bucket>/data/toucan/` |
| `script_path` | Script that was run | `scripts/experiments/colm_architecture/run_colm_experiments.py` |
| `status` | running/completed/failed | `completed` |

```bash
python experiments/scripts/manage_experiment.py record \
    --experiment experiment_id \
    --instance-id i-0abc123def456 \
    --instance-type t3.xlarge \
    --data-source s3://<your-s3-bucket>/data/toucan/ \
    --status completed
```

### Rule 3: Virtual Environment Per Experiment

Each experiment directory MUST have:
- `requirements.txt`: Frozen dependencies (`pip freeze > requirements.txt`)
- `run.sh`: Reproduction script that creates venv and installs deps

The `run.sh` template:
```bash
#!/bin/bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python script.py "$@"
```

### Rule 4: All Data in S3

- **Input data**: `s3://<your-s3-bucket>/data/{dataset}/`
- **Code snapshots**: `s3://<your-s3-bucket>/code/{experiment_id}/`
- **Results**: `s3://<your-s3-bucket>/experiments/{experiment_id}/results/`
- **Requirements**: `s3://<your-s3-bucket>/experiments/{experiment_id}/requirements.txt`

Sync results after completion:
```bash
python experiments/scripts/manage_experiment.py sync \
    --experiment experiment_id \
    --direction upload
```

### Rule 5: Experiment Directory Structure

```
experiments/{experiment_id}/
├── config.json          # Full experiment configuration
├── requirements.txt     # Python dependencies (frozen)
├── run.sh              # Reproduction script
└── results/            # Local results (synced to S3)
```

### Quick Reference

```bash
# List all experiments
python experiments/scripts/manage_experiment.py list

# Show experiment details
python experiments/scripts/manage_experiment.py show --experiment {id}

# Setup EC2 for experiment
python experiments/scripts/manage_experiment.py setup-ec2 --experiment {id}

# Sync results to S3
python experiments/scripts/manage_experiment.py sync --experiment {id} --direction upload

# Download results from S3
python experiments/scripts/manage_experiment.py sync --experiment {id} --direction download
```
