# LLM Benchmarking Results Summary

## Overview

This document summarizes the comprehensive benchmarking results for evaluating consistency of structured outputs across multiple Large Language Models (LLMs) using the STED (Semantic Tree Edit Distance) framework.

## Tested Models

A total of **10 different LLMs** were evaluated across various temperature settings:

### 1. Anthropic Models

#### Claude 3 Haiku
- **Model ID**: `anthropic.claude-3-haiku-20240307-v1`
- **Temperature Range**: 0.0 - 1.0 (11 settings)
- **Result Sets**: 11
- **Samples per Temperature**: 80

#### Claude 3.5 Haiku
- **Model ID**: `anthropic.claude-3-5-haiku-20241022-v1`
- **Temperature Range**: 0.0 - 0.95 (20 settings)
- **Result Sets**: 20
- **Samples per Temperature**: 80

#### Claude 3.7 Sonnet
- **Model ID**: `us.anthropic.claude-3-7-sonnet-20250219-v1`
- **Temperature Range**: 0.0 - 1.0 (11 settings)
- **Result Sets**: 11
- **Samples per Temperature**: 80

### 2. Amazon Models

#### Nova Pro v1
- **Model ID**: `us.amazon.nova-pro-v1`
- **Temperature Range**: 0.0 - 0.95 (20 settings)
- **Result Sets**: 20
- **Samples per Temperature**: 80

### 3. Meta Models

#### Llama 3.3 70B
- **Model ID**: `us.meta.llama3-3-70b-instruct-v1`
- **Temperature Range**: 0.0 - 0.95 (20 settings)
- **Result Sets**: 20
- **Samples per Temperature**: 80

### 4. OpenAI Models

#### GPT-4.1 Mini
- **Model ID**: `gpt-4.1-mini`
- **Temperature Range**: 0.0 - 1.0 (11 settings)
- **Result Sets**: 11
- **Samples per Temperature**: 80

### 5. Google Models

#### Gemini 2.5 Flash Lite
- **Model ID**: `gemini-2.5-flash-lite`
- **Temperature Range**: 0.0 - 1.0 (11 settings)
- **Result Sets**: 11
- **Samples per Temperature**: 80

### 6. DeepSeek Models

#### DeepSeek v3
- **Model ID**: `deepseek.v3-v1`
- **Temperature Range**: 0.0 - 1.0 (11 settings)
- **Result Sets**: 11
- **Samples per Temperature**: 80

### 7. Alibaba Qwen Models

#### Qwen3 32B
- **Model ID**: `qwen.qwen3-32b-v1`
- **Temperature Range**: 0.0 - 1.0 (11 settings)
- **Result Sets**: 11
- **Samples per Temperature**: 80

#### Qwen3 235B A22B
- **Model ID**: `qwen.qwen3-235b-a22b-2507-v1`
- **Temperature Range**: 0.0 - 1.0 (11 settings)
- **Result Sets**: 11
- **Samples per Temperature**: 80

## Experiment Design

### Dataset
- **Base Dataset**: ShareGPT structured output samples
- **Total Samples**: 80 samples per temperature setting
- **Sample Types**: 
  - 30 from sharegpt-structured-output-json
  - 50 from sharegpt-quizz-generation-json-output

### Temperature Settings
- **Standard Range**: 0.0, 0.1, 0.2, ..., 1.0 (11 points)
- **Fine-grained Range**: 0.0, 0.05, 0.10, 0.15, ..., 0.95 (20 points)

### Evaluation Metrics

Each result set contains:
1. **Individual Sample Results** (sample_*.json)
   - 80 JSON files with LLM-generated structured outputs
   - Each includes prompt, response, and metadata

2. **Aggregated Results** (all_results.json)
   - Combined results for all 80 samples
   - Includes generation metadata and statistics

3. **Consistency Metrics**
   - BERTScore Results (results_bertscore.json)
   - DeepDiff Results (results_deepdiff.json)
   - Tree Edit Distance Results (results_ted.json)

## Total Data Points

- **Models**: 10
- **Temperature Settings**: 127 total (varies by model)
- **Samples per Setting**: 80
- **Total Generations**: ~10,160 structured outputs
- **Evaluation Metrics per Generation**: 3 (BERTScore, DeepDiff, TED)
- **Total Metric Calculations**: ~30,480

## Analysis Results

The consistency analysis results are available in:
- results/structural_consistency_metrics_results_v1.json
- results/content_consistency_metrics_results_v1.json
- results/combined_consistency_metrics_results_v1.json
- results/consistency_score_by_consistency_type_with_errors.png

## Key Findings

1. **Temperature Impact**: Consistency generally decreases as temperature increases across all models
2. **Model Variations**: Different models show varying levels of consistency at the same temperature
3. **Structural vs Content**: Some models maintain structural consistency better than content consistency
4. **Optimal Temperature**: Most models show best consistency at temperature 0.0-0.2

## Last Updated

2025-11-08
