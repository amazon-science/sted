# Field-Aware Consistency Evaluation Framework for LLM Structured Outputs

A comprehensive framework for evaluating the consistency of structured outputs from Large Language Models (LLMs) using Semantic Tree Edit Distance (STED). This framework provides field-aware consistency evaluation that considers both structural and semantic similarities in JSON outputs.

## Overview

This framework implements STED (Semantic Tree Edit Distance), a novel approach that combines traditional tree edit distance algorithms with semantic similarity measures to evaluate the consistency of LLM-generated structured outputs. The framework supports multiple types of variations and provides comprehensive analysis tools for LLM benchmarking.

## Key Features

- **Semantic Tree Edit Distance (STED)**: Advanced similarity calculation combining structural and semantic analysis
- **Multiple Variation Types**: Support for schema, expression, and semantic variations
- **LLM Benchmarking**: Comprehensive evaluation of different LLMs across temperature settings
- **Synthetic Dataset Generation**: Automated generation of variation datasets for evaluation
- **Visualization Tools**: Rich plotting and analysis capabilities for results interpretation

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd field-aware-consistency-evaluation-framework

# Install dependencies using uv
uv sync
```

## Quick Start

### Basic Similarity Calculation with STED

```python
from src.semantic_json_tree_consistency import SemanticJsonTreeConsistencyEvaluator

# Initialize evaluator with embedding model
evaluator = SemanticJsonTreeConsistencyEvaluator(
    model_id='amazon.titan-embed-text-v2:0',
)

# Compare two JSON structures
json1 = {'name': 'John', 'age': 30, 'city': 'New York'}
json2 = {'name': 'John', 'age': 30, 'location': 'NYC'}

# Calculate similarity using STED
result = evaluator.calculate_tree_edit_distance_opt(
    json1, json2, 
    variation_type="combined"
)

print(f"Similarity score: {result['similarity']}")
print(f"Edit distance: {result['edit_distance']}")
```

## Dataset

The framework uses ShareGPT datasets for evaluation:
- **sharegpt-structured-output-json**: 30 samples with structured JSON responses
- **sharegpt-quizz-generation-json-output**: 50 samples with quiz generation outputs
- **Total**: 80 samples (75 valid samples after parsing error exclusion)

### Dataset Construction

Generate synthetic datasets with different variation types:

```bash
uv run scripts/generate_sythetic_datasets.py
```

This creates datasets with three variation types:
- **Schema Variation**: Field name changes, structure flattening/nesting
- **Expression Variation**: Different expressions with same semantic meaning
- **Semantic Variation**: Changes in semantic content

### Dataset Analysis

Analyze base dataset complexity (75 samples):
```bash
uv run scripts/dataset_analysis/analyze_base_dataset_stat.py
```

Analyze synthetic dataset complexity (2400 samples):
```bash
uv run scripts/dataset_analysis/analyze_synthetic_dataset_stat.py
```

## STED Effectiveness Verification

### Similarity Progression Analysis

Calculate similarities between variants and ground truth at different variation ratios:

```bash
uv run scripts/eval/analyze_similarity_variation_progression.py
```

This generates `{variation_type}_variation_progression_results.json` files for each variation type.

### Visualization

**Expression and Semantic Variation Analysis:**
```bash
uv run scripts/result_analysis/visualize_progression_expression_semantic_separate_charts.py
```
Output: `similarity_progression_combined.png`

**Schema Variation Analysis:**
```bash
uv run scripts/result_analysis/visualize_schema_variation_results.py
```
Output: `schema_variation_analysis_with_errors.png`

## LLM Consistency Benchmarking

Comprehensive benchmarking of LLMs using STED for structural, content, and overall consistency evaluation.

### Step 1: Generate LLM Structured Outputs

Generate structured responses across temperature range (0.0-1.0):

```bash
uv run eval/run_temperature_experiment.py \
  --data-dir sharegpt_data \
  --output-dir llm_gen_results \
  --run-num 10 \
  --model-id anthropic.claude-3-haiku-20240307-v1:0 \
  --force-regenerate \
  --max-tokens 2000 \
  --include-schema
```

**Output Structure:**
```
llm_gen_results/
└── generations-<model-name>/
    └── llm_gen_results_<model_id>_temp_<temperature>_<timestamp>/
        ├── generation_results.json
        └── individual_results/
```

### Step 2: Calculate Consistency Metrics

Calculate consistency scores for structural, content, and combined metrics:

```bash
uv run scripts/eval/calculate_consistency_metrics.py
```

Output: `{consistency_type}_consistency_metrics_results.json` files

### Step 3: Visualize Results

Generate comprehensive consistency analysis visualizations:

```bash
uv run scripts/result_analysis/analyze_consistency_score_llm_benckmarking.py
```

Output: `consistency_score_by_consistency_type_with_errors.png`

## Project Structure

```
field-aware-consistency-evaluation-framework/
├── src/                                    # Core implementation
│   ├── semantic_json_tree_consistency.py  # Main STED implementation
│   ├── json_tree_node.py                 # Tree node structures
│   ├── bedrock_utils.py                   # AWS Bedrock utilities
│   └── utils.py                           # Helper functions
├── scripts/                               # Analysis and generation scripts
│   ├── eval/                             # Evaluation scripts
│   ├── result_analysis/                  # Visualization scripts
│   └── dataset_analysis/                 # Dataset analysis tools
├── experiments/                          # Experiment results
│   ├── experiment-1/                    # STED effectiveness verification
│   └── experiment-2/                    # LLM consistency benchmarking
├── sharegpt_data/                        # Base datasets
├── synthetic_dataset/                    # Generated synthetic datasets
└── llm_gen_results/                      # LLM generation results
```

## Key Components

### STED Algorithm
- **Structural Analysis**: Tree edit distance with optimized matching
- **Semantic Analysis**: Embedding-based similarity using transformer models
- **Combined Scoring**: Weighted combination of structural and semantic similarities

### Variation Types
- **Schema**: Field name changes, structure modifications
- **Expression**: Linguistic variations with preserved semantics
- **Semantic**: Content meaning changes

### Evaluation Metrics
- **Structural Consistency**: Focus on JSON structure similarity
- **Content Consistency**: Emphasis on semantic content similarity
- **Combined Consistency**: Balanced structural and semantic evaluation

## Algorithm Complexity

Refer to [STED Computational Complexity Analysis](./sted_complexity_analysis.md) for detailed complexity analysis including:
- Tree construction: O(n)
- Embedding computation: O(k) with caching
- Optimized STED: O(n₁ × n₂ × (n₁ + n₂))
- Hungarian algorithm optimization: O(max(n₁, n₂)³)

## Results and Findings

The framework has been used to evaluate multiple LLMs including:
- Claude 3 Haiku
- Claude 3.5 Haiku  
- Claude 3.7 Sonnet
- Llama 3.3 70B
- Amazon Nova Pro

Key findings demonstrate STED's effectiveness in capturing both structural and semantic consistency in LLM outputs across different temperature settings.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

[Add your license information here]

## Citation

If you use this framework in your research, please cite:

```bibtex
[Add citation information when available]
```