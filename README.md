# Field-Aware Consistency Evaluation Framework for LLM Structured Outputs

A comprehensive framework for evaluating the consistency of structured outputs from Large Language Models (LLMs) using Semantic Tree Edit Distance (STED). This framework provides field-aware consistency evaluation that considers both structural and semantic similarities in JSON outputs.

## Overview

This framework implements STED (Semantic Tree Edit Distance), a novel approach that combines traditional tree edit distance algorithms with semantic similarity measures to evaluate the consistency of LLM-generated structured outputs. The framework supports multiple types of variations and provides comprehensive analysis tools for LLM benchmarking.

## Key Features

- **Semantic Tree Edit Distance (STED)**: Advanced similarity calculation combining structural and semantic analysis
- **Multiple Variation Types**: Support for schema, expression, and semantic variations
- **LLM Benchmarking**: Comprehensive evaluation of different LLMs across temperature settings
- **MCP Server Support**: Model Context Protocol server for real-time consistency evaluation in agentic systems
- **Synthetic Dataset Generation**: Automated generation of variation datasets for evaluation
- **Visualization Tools**: Rich plotting and analysis capabilities for results interpretation

## Installation

### As a Library

Install the STED consistency library for use in your own projects:

```bash
# Clone the repository
git clone <repository-url>
cd field-aware-consistency-evaluation-framework

# Install the library
pip install -e .

# Or with uv
uv pip install -e .
```

### For Development

```bash
# Install with development dependencies
pip install -e ".[dev]"

# Install dependencies using uv
uv sync
```

## Credentials Setup

### AWS Credentials

This framework uses AWS Bedrock for embedding models and LLM inference. Configure your AWS credentials:

```bash
# Configure AWS credentials
aws configure

# Or set environment variables
export AWS_ACCESS_KEY_ID=<your-access-key>
export AWS_SECRET_ACCESS_KEY=<your-secret-key>
export AWS_DEFAULT_REGION=us-east-1
```

Ensure your AWS account has access to:
- Amazon Bedrock models (e.g., `amazon.titan-embed-text-v2:0`)
- Required model permissions in your region

### OpenAI API Key

For OpenAI model evaluation, set your API key:

```bash
# Set environment variable
export OPENAI_API_KEY=<your-openai-api-key>

# Optional: Set custom base URL (for OpenAI-compatible APIs)
export OPENAI_BASE_URL=<your-base-url>
```

## Library Usage

### Quick Example

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

### More Examples

See `examples/basic_usage.py` for complete examples:

```bash
python examples/basic_usage.py
```

For detailed API documentation, see [Library Usage Guide](./LIBRARY_USAGE.md).

## Quick Start

### Basic Similarity Calculation with STED

```python
from sted.semantic_json_tree_consistency import SemanticJsonTreeConsistencyEvaluator

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

### Batch Consistency Calculation

Evaluate consistency across multiple JSON outputs:

```python
from sted.semantic_json_tree_consistency import SemanticJsonTreeConsistencyEvaluator
from sted.structural_consistency_analyzer import StructuralConsistencyAnalyzer

# Initialize evaluator and analyzer
evaluator = SemanticJsonTreeConsistencyEvaluator(
    model_id='amazon.titan-embed-text-v2:0'
)
analyzer = StructuralConsistencyAnalyzer(evaluator)

# Multiple LLM outputs for the same prompt
json_outputs = [
    {'name': 'Alice', 'age': 25, 'city': 'New York'},
    {'name': 'Alice', 'age': 25, 'city': 'NYC'},
    {'name': 'Alice', 'age': 25, 'location': 'New York City'},
]

# Calculate consistency metrics
result = analyzer.evaluate_structural_consistency(
    json_outputs,
    method_name="ted",
    variation_type="combined"
)

print(f"Mean similarity: {result['supporting_stats']['mean_similarity']:.4f}")
print(f"Consistency coefficient: {result['consistency_metrics']['consistency_coefficient']:.4f}")
print(f"Stability score: {result['consistency_metrics']['stability_score']:.4f}")
```

## Dataset

The framework uses ShareGPT datasets for evaluation:
- **sharegpt-structured-output-json**: 30 samples with structured JSON responses
- **sharegpt-quizz-generation-json-output**: 50 samples with quiz generation outputs
- **Total**: 80 samples (75 valid samples after parsing error exclusion)

### Dataset Construction

Generate synthetic datasets with different variation types:

```bash
python scripts/data/generate_synthetic_datasets.py
```

This creates datasets with three variation types:
- **Schema Variation**: Field name changes, structure flattening/nesting
- **Expression Variation**: Different expressions with same semantic meaning
- **Semantic Variation**: Changes in semantic content

### Dataset Analysis

Analyze base dataset complexity (75 samples):
```bash
python scripts/dataset_analysis/analyze_base_dataset_stat.py --output-dir results/base_dataset_analysis
```

Analyze synthetic dataset complexity (2400 samples):
```bash
python scripts/dataset_analysis/analyze_synthetic_dataset_stat.py --output-dir results/synthetic_dataset_analysis
```

## STED Effectiveness Verification

### Similarity Progression Analysis

Calculate similarities between variants and ground truth at different variation ratios:

```bash
python scripts/dataset_analysis/analyze_semantic_expression_variation_progression.py \
  synthetic_dataset/expression_variation_dataset_*.json \
  synthetic_dataset/semantic_variation_dataset_*.json \
  --output-dir results/variation_progression
```

This generates `{variation_type}_variation_progression_results.json` and `{variation_type}_variation_progression_analysis.png` files for each variation type.

### Visualization

**Expression and Semantic Variation Analysis:**
```bash
python scripts/visualization/visualize_progression_expression_semantic_separate_charts.py
```
Output: `similarity_progression_combined.png`

**Schema Variation Analysis:**
```bash
python scripts/visualization/visualize_schema_variation_results.py
```
Output: `schema_variation_analysis_with_errors.png`

## LLM Consistency Benchmarking

Comprehensive benchmarking of LLMs using STED for structural, content, and overall consistency evaluation.

### Step 1: Generate LLM Structured Outputs

Generate structured responses across temperature range (0.0-1.0):

```bash
python scripts/eval/run_temperature_experiment.py \
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
python scripts/eval/calculate_consistency_metrics.py
```

Output: `{consistency_type}_consistency_metrics_results.json` files

### Step 3: Visualize Results

Generate comprehensive consistency analysis visualizations:

```bash
python scripts/visualization/visualize_consistency_scores.py
```

Output: `consistency_score_by_consistency_type_with_errors.png`

![LLM Consistency Scores](results_archive/v1_2025-11-08/llm_consistency/consistency_score_by_consistency_type_with_errors.png)

## Project Structure

```
field-aware-consistency-evaluation-framework/
├── sted/                                  # Product: Core library (pip installable)
│   ├── __init__.py                       # Package initialization
│   ├── semantic_json_tree_consistency.py  # Main STED implementation
│   ├── structural_consistency_analyzer.py # Batch consistency evaluation
│   ├── json_tree_node.py                 # Tree node structures
│   ├── bedrock_utils.py                   # AWS Bedrock utilities
│   ├── probabilistic_consistency.py       # Probabilistic consistency metrics
│   ├── pdc_metric.py                      # PDC metric implementation
│   ├── evaluator_config.py               # Configuration management
│   ├── similarity_cache.py               # Embedding cache
│   ├── gnn.py                            # Graph neural network utilities
│   └── utils.py                           # Helper functions
├── tests/                                # Product: Unit tests
│   ├── test_basic_sted.py               # Basic STED functionality tests
│   ├── test_dataset_analysis.py         # Dataset validation tests
│   └── test_llm_results.py              # LLM results structure tests
├── examples/                             # Product: Usage examples
│   └── basic_usage.py                   # Basic STED usage examples
├── benchmarks/                           # Product: Performance benchmarks
├── mcp_dev/                              # Product: MCP server (FastMCP-based)
│   ├── server.py                        # MCP server implementation
│   ├── test_client.py                   # MCP client for testing
│   ├── README.md                        # MCP usage guide
│   ├── AGENTCORE_DEPLOYMENT_GUIDE.md   # AWS deployment guide
│   └── prepare_agentcore_deployment.sh # Deployment script
├── scripts/                              # Shared: Utilities and tools
│   ├── data/                            # Data preparation scripts
│   ├── eval/                            # LLM evaluation scripts
│   ├── visualization/                   # Visualization scripts
│   ├── dataset_analysis/                # Dataset analysis tools
│   └── analysis/                        # Consistency analysis tools
├── docs/                                 # Shared: Documentation
│   ├── api/                             # API documentation
│   ├── LLM_BENCHMARKING_RESULTS.md     # Benchmarking results
│   └── README.md                        # Documentation index
├── research/                             # Research: Experiments and papers
│   ├── experiments/                     # Experiment results
│   │   ├── experiment-1/               # STED effectiveness verification
│   │   └── experiment-2/               # LLM consistency benchmarking
│   ├── notebooks/                       # Jupyter notebooks for analysis
│   ├── papers/                          # Research papers and notes
│   └── datasets/                        # Research datasets (gitignored)
├── results/                              # Generated results and metrics (gitignored)
├── results_archive/                      # Archived experiment results
├── sharegpt_data/                        # Base datasets (gitignored)
├── synthetic_dataset/                    # Generated synthetic datasets (gitignored)
├── llm_gen_results/                      # LLM generation results (gitignored)
├── README.md                             # Main documentation
├── LIBRARY_USAGE.md                      # Library usage guide
├── SCRIPTS_REFERENCE.md                  # Scripts documentation
├── pyproject.toml                        # Project configuration
└── uv.lock                               # Dependency lock file
```

## Scripts Reference

For detailed information about all scripts, see [Scripts Reference](./SCRIPTS_REFERENCE.md).

### Quick Reference

**Data Preparation** (`scripts/data/`):
- `download_sharegpt_data.py` - Download ShareGPT datasets
- `generate_synthetic_datasets.py` - Generate synthetic variation datasets

**Evaluation** (`scripts/eval/`):
- `run_temperature_experiment.py` - Run LLM experiments across temperatures
- `calculate_consistency_metrics.py` - Calculate consistency metrics

**Visualization** (`scripts/visualization/`):
- `visualize_consistency_scores.py` - Visualize LLM benchmarking results
- `visualize_variation_progression.py` - Visualize variation analysis

See [SCRIPTS_REFERENCE.md](./SCRIPTS_REFERENCE.md) for complete documentation of all 26 scripts.

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

Refer to [STED Computational Complexity Analysis](./docs/api/sted_complexity_analysis.md) for detailed complexity analysis including:
- Tree construction: O(n)
- Embedding computation: O(k) with caching
- Optimized STED: O(n₁ × n₂ × (n₁ + n₂))
- Hungarian algorithm optimization: O(max(n₁, n₂)³)

## Results and Findings

The framework has been used to evaluate **10 different LLMs** across multiple temperature settings:

**Anthropic Models:**
- Claude 3 Haiku
- Claude 3.5 Haiku  
- Claude 3.7 Sonnet

**Amazon Models:**
- Amazon Nova Pro v1

**Meta Models:**
- Llama 3.3 70B

**OpenAI Models:**
- GPT-4.1 Mini

**Google Models:**
- Gemini 2.5 Flash Lite

**DeepSeek Models:**
- DeepSeek v3

**Alibaba Qwen Models:**
- Qwen3 32B
- Qwen3 235B A22B

**Total Evaluation:**
- 10 models tested
- 127 temperature settings (varies by model)
- 80 samples per temperature
- ~10,160 total structured outputs generated
- ~30,480 consistency metric calculations

For detailed benchmarking results, see [LLM Benchmarking Results](./docs/LLM_BENCHMARKING_RESULTS.md).

Key findings demonstrate STED's effectiveness in capturing both structural and semantic consistency in LLM outputs across different temperature settings.

## MCP Server for Agentic Systems

The framework includes a Model Context Protocol (MCP) server built with FastMCP for real-time consistency evaluation in agentic systems.

### Features

- **evaluate_consistency**: Compare two JSON structures using STED
- **evaluate_batch_consistency**: Evaluate consistency across multiple JSON structures
- **evaluate_tool_calls**: Evaluate agent tool call consistency
- **FastMCP-powered**: Automatic schema generation, type safety, and clean decorator-based API

### Quick Start

```bash
# Test the MCP server
cd mcp_dev
python test_client.py
```

### Integration

Add to your MCP client configuration:

```json
{
  "mcpServers": {
    "sted-evaluator": {
      "command": "python",
      "args": ["server.py"],
      "cwd": "/path/to/field-aware-consistency-evaluation-framework/mcp_dev"
    }
  }
}
```

### Example Usage

```json
{
  "method": "tools/call",
  "params": {
    "name": "evaluate_tool_calls",
    "arguments": {
      "tool_calls": [
        {"tool": "search", "parameters": {"query": "test"}},
        {"tool": "search", "parameters": {"query": "test"}}
      ],
      "variation_type": "combined"
    }
  }
}
```

See [mcp_dev/README.md](./mcp_dev/README.md) for detailed documentation.

## Future Features & TODO List

### 🏆 Model Benchmarking Expansion
- [ ] **Extended Model Coverage**: Benchmark additional state-of-the-art models
  - GPT-4 (OpenAI)
  - GPT-4o (OpenAI)
  - GPT-5 (OpenAI)
  - Gemini-2.5-Pro (Google)
  - Qwen3 (Alibaba)
- [ ] **Cross-Provider Comparison**: Comprehensive analysis across different model providers
- [ ] **Model Size Impact Analysis**: Evaluate consistency patterns across different model sizes
- [ ] **Fine-tuned Model Evaluation**: Test consistency of domain-specific fine-tuned models

### 🎯 Field-Level Analysis
- [ ] **Field-Level Consistency Evaluation**: Implement granular analysis to measure consistency at individual field level
- [ ] **Field Importance Weighting**: Add capability to assign different weights to fields based on their importance
- [ ] **Field-Specific Metrics**: Develop specialized metrics for different field types (text, numeric, categorical)
- [ ] **Cross-Field Dependency Analysis**: Analyze how inconsistencies in one field affect others

### 🔍 Inconsistency Root Cause Analysis
- [ ] **Inconsistency Factor Identification**: Implement algorithms to identify primary factors causing inconsistencies
  - Temperature sensitivity analysis per field
  - Prompt complexity impact assessment
  - Model-specific inconsistency patterns
- [ ] **Inconsistency Categorization**: Classify inconsistencies by type (semantic, structural, formatting)
- [ ] **Statistical Correlation Analysis**: Identify correlations between input characteristics and output inconsistencies
- [ ] **Attention Mechanism Analysis**: Integrate attention weights to understand model focus areas

### 🛠️ Consistency Improvement Methods
- [ ] **Adaptive Prompting**: Develop dynamic prompt adjustment based on detected inconsistencies
- [ ] **Consistency-Aware Sampling**: Implement sampling strategies that optimize for consistency
- [ ] **Multi-Pass Refinement**: Add iterative refinement process to improve consistency
- [ ] **Ensemble Consistency**: Combine multiple model outputs to achieve higher consistency
- [ ] **Template-Based Generation**: Provide structured templates to guide consistent output generation

### 📊 Advanced Analytics & Visualization
- [ ] **Interactive Consistency Dashboard**: Web-based interface for real-time consistency monitoring
- [ ] **Consistency Heatmaps**: Visual representation of consistency across different dimensions
- [ ] **Trend Analysis**: Track consistency improvements over time and model versions
- [ ] **Comparative Analysis Tools**: Side-by-side comparison of different models and configurations

### 🔧 Framework Enhancements
- [ ] **Multi-Provider API Support**: Integrate with OpenAI, Google, Anthropic, and other APIs
- [x] **MCP Server Support**: Implement Model Context Protocol server to enable STED evaluation as tools for agentic AI systems
- [ ] **Real-Time Evaluation**: Support for streaming evaluation of live model outputs
- [ ] **Custom Similarity Functions**: Allow users to define domain-specific similarity measures
- [ ] **Batch Processing Optimization**: Improve performance for large-scale evaluations
- [ ] **Configuration Management**: YAML/JSON-based configuration for different evaluation scenarios

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