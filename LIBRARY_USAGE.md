# STED Consistency Library - Usage Guide

## Installation

### Install from source

```bash
# Clone the repository
git clone <repository-url>
cd field-aware-consistency-evaluation-framework

# Install the library
pip install -e .

# Or with uv
uv pip install -e .
```

### Install with development dependencies

```bash
pip install -e ".[dev]"
```

## Quick Start

### Basic Usage

```python
from sted.semantic_json_tree_consistency import SemanticJsonTreeConsistencyEvaluator

# Initialize evaluator
evaluator = SemanticJsonTreeConsistencyEvaluator(
    model_id='amazon.titan-embed-text-v2:0'
)

# Compare two JSON structures
json1 = {'name': 'John', 'age': 30, 'city': 'New York'}
json2 = {'name': 'John', 'age': 30, 'location': 'NYC'}

# Calculate similarity
similarity = evaluator.calculate_tree_edit_distance_opt(
    json1, json2, 
    variation_type="combined"
)

print(f"Similarity: {similarity:.4f}")
```

## API Reference

### SemanticJsonTreeConsistencyEvaluator

Main class for calculating STED similarity between JSON structures.

#### Constructor

```python
SemanticJsonTreeConsistencyEvaluator(
    model_id='amazon.titan-embed-text-v2:0',
    sort_arrays=False,
    sort_keys=False
)
```

**Parameters:**
- `model_id` (str): Embedding model ID for semantic similarity
- `sort_arrays` (bool): Whether to sort arrays before comparison
- `sort_keys` (bool): Whether to sort dictionary keys

#### Methods

##### calculate_tree_edit_distance_opt()

Calculate optimized tree edit distance between two JSON objects.

```python
similarity = evaluator.calculate_tree_edit_distance_opt(
    json1: dict,
    json2: dict,
    variation_type: str = "combined"
) -> float
```

**Parameters:**
- `json1` (dict): First JSON object
- `json2` (dict): Second JSON object
- `variation_type` (str): Type of consistency to evaluate
  - `"structural"`: Focus on structure similarity
  - `"content"`: Focus on semantic content similarity
  - `"combined"`: Balanced structural and semantic evaluation

**Returns:**
- `float`: Similarity score between 0.0 and 1.0

## Variation Types

### Structural Consistency
Focuses on JSON structure similarity (field names, nesting, data types).

```python
similarity = evaluator.calculate_tree_edit_distance_opt(
    json1, json2,
    variation_type="structural"
)
```

### Content Consistency
Focuses on semantic content similarity using embeddings.

```python
similarity = evaluator.calculate_tree_edit_distance_opt(
    json1, json2,
    variation_type="content"
)
```

### Combined Consistency
Balanced evaluation of both structure and content.

```python
similarity = evaluator.calculate_tree_edit_distance_opt(
    json1, json2,
    variation_type="combined"
)
```

## Examples

See `examples/basic_usage.py` for complete examples:

```bash
python examples/basic_usage.py
```

## Advanced Usage

### Batch Consistency Evaluation

Evaluate consistency across multiple JSON outputs using `StructuralConsistencyAnalyzer`:

```python
from sted.semantic_json_tree_consistency import SemanticJsonTreeConsistencyEvaluator
from sted.structural_consistency_analyzer import StructuralConsistencyAnalyzer

evaluator = SemanticJsonTreeConsistencyEvaluator()
analyzer = StructuralConsistencyAnalyzer(evaluator)

# Multiple LLM outputs for the same prompt
json_outputs = [
    {'name': 'Alice', 'age': 25, 'city': 'New York'},
    {'name': 'Alice', 'age': 25, 'city': 'NYC'},
    {'name': 'Alice', 'age': 25, 'location': 'New York City'},
]

result = analyzer.evaluate_structural_consistency(
    json_outputs,
    method_name="ted",
    variation_type="combined"
)

print(f"Mean similarity: {result['supporting_stats']['mean_similarity']:.4f}")
print(f"Consistency coefficient: {result['consistency_metrics']['consistency_coefficient']:.4f}")
print(f"Stability score: {result['consistency_metrics']['stability_score']:.4f}")
```

### Pairwise Evaluation

```python
from sted.semantic_json_tree_consistency import SemanticJsonTreeConsistencyEvaluator

evaluator = SemanticJsonTreeConsistencyEvaluator()

json_pairs = [
    (json1_a, json2_a),
    (json1_b, json2_b),
    (json1_c, json2_c),
]

similarities = [
    evaluator.calculate_tree_edit_distance_opt(j1, j2)
    for j1, j2 in json_pairs
]

print(f"Average similarity: {sum(similarities) / len(similarities):.4f}")
```

### Custom Embedding Models

```python
# Use different embedding model
evaluator = SemanticJsonTreeConsistencyEvaluator(
    model_id='your-custom-model-id'
)
```

## AWS Configuration

The library uses AWS Bedrock for embeddings. Configure AWS credentials:

```bash
# Set environment variables
export AWS_REGION=us-east-1
export AWS_ACCESS_KEY_ID=your-key
export AWS_SECRET_ACCESS_KEY=your-secret

# Or use AWS CLI configuration
aws configure
```

## Performance Considerations

- **Caching**: The library caches embeddings to improve performance
- **Complexity**: O(n₁ × n₂ × (n₁ + n₂)) for tree comparison
- **Batch Processing**: Process multiple comparisons in parallel for better throughput

## Troubleshooting

### Import Errors

If you encounter import errors, ensure the package is installed:

```bash
pip install -e .
```

### AWS Credentials

If you get AWS credential errors:

```bash
aws configure
# Or set environment variables
export AWS_REGION=us-east-1
```

## Contributing

See the main README for contribution guidelines.

## License

[Add license information]
