# Semantic JSON Tree Consistency Evaluation

An advanced framework for evaluating structural and semantic consistency between JSON objects using tree edit distance algorithms with semantic understanding.

## Overview

The Semantic JSON Tree Consistency Evaluation framework provides sophisticated tools for comparing JSON structures beyond simple string matching. It converts JSON objects into tree representations and calculates edit distances between them, taking into account both structural similarities and semantic meaning.

Key capabilities include:

- **Semantic Understanding**: Recognizes when fields have different names but similar meanings
- **Tree-Based Comparison**: Uses tree edit distance algorithms for accurate structural comparison
- **Enhanced String Comparison**: Intelligently handles long text values by breaking them into meaningful chunks
- **Comprehensive Metrics**: Provides detailed consistency metrics including statistical measures
- **Customizable Costs**: Allows fine-tuning of edit operation costs for different scenarios

## Key Features

### Semantic Similarity

- **Embedding-Based Comparison**: Uses sentence embeddings to detect semantic similarity between keys and values
- **Key Mapping**: Finds optimal mapping between differently named but semantically similar keys
- **Customizable Thresholds**: Adjustable semantic similarity thresholds for different use cases

### Advanced String Comparison

- **Content-Aware Chunking**: Intelligently splits long text based on content type (code, natural language)
- **Optimal Chunk Matching**: Uses Hungarian algorithm to find best matches between text chunks
- **Structure Recognition**: Detects and preserves structure in code blocks and formatted text

### Comprehensive Consistency Metrics

- **Basic Metrics**: Mean similarity, standard deviation, min/max values
- **Statistical Measures**: Quartiles, entropy, Gini coefficient
- **Outlier Detection**: Identifies anomalous pairs with Z-scores
- **Detailed Analysis**: Operation counts, frequently edited paths, most different pairs

## Installation

```bash
# Install required packages
pip install numpy scipy sentence-transformers
```

## Quick Start

```python
from semantic_json_tree_consistency import evaluate_semantic_json_consistency

# Example JSON objects with semantically similar structures
json1 = {
    "user_name": "John Doe",
    "user_age": 30,
    "email_address": "john.doe@example.com"
}

json2 = {
    "name": "John Doe",  # Semantically similar to "user_name"
    "age": 31,           # Semantically similar to "user_age"
    "email": "john.doe@example.com"  # Semantically similar to "email_address"
}

# Evaluate consistency with semantic understanding
result = evaluate_semantic_json_consistency(
    [json1, json2],
    array_order_matters=False,
    use_semantic_similarity=True,
    semantic_threshold=0.7
)

print(f"Consistency Score: {result['consistency_metrics']['mean_similarity']:.4f}")
print(f"Standard Deviation: {result['consistency_metrics']['std_deviation']:.4f}")
print(f"Consistency Coefficient: {result['consistency_metrics']['consistency_coefficient']:.4f}")
```

## Core Components

### JsonNode

Tree representation of JSON elements with type information and path tracking:

```python
node = JsonNode("user.name", "John Doe", "string")
```

### SemanticJsonTreeConsistencyEvaluator

The main evaluator class with configurable parameters:

```python
evaluator = SemanticJsonTreeConsistencyEvaluator(
    use_semantic_similarity=True,
    embedding_model="all-MiniLM-L6-v2",
    semantic_threshold=0.7,
    array_order_matters=False,
    string_method="semantic",
    number_tolerance=0.01
)
```

## Advanced Usage

### Comparing Complex JSON Structures

```python
# Compare complex nested structures
similarity, operations = evaluator.calculate_tree_edit_distance(complex_json1, complex_json2)

print(f"Structural Similarity: {similarity:.4f}")
print("Edit Operations:")
for op in operations[:5]:  # Show first 5 operations
    print(f"- {op['operation']} at {op['path']}")
```

### Evaluating Consistency Across Multiple JSONs

```python
# Evaluate consistency across multiple JSON outputs
json_outputs = [json1, json2, json3, json4, json5]
consistency_report = evaluator.evaluate_structural_consistency(json_outputs)

# Access comprehensive metrics
print(f"Mean Similarity: {consistency_report['consistency_metrics']['mean_similarity']:.4f}")
print(f"Standard Deviation: {consistency_report['consistency_metrics']['std_deviation']:.4f}")
print(f"Consistency Coefficient: {consistency_report['consistency_metrics']['consistency_coefficient']:.4f}")

# Access statistical metrics
print(f"Median Similarity: {consistency_report['statistical_metrics']['quartiles']['median']:.4f}")
print(f"IQR: {consistency_report['statistical_metrics']['quartiles']['iqr']:.4f}")
print(f"Entropy: {consistency_report['statistical_metrics']['entropy']:.4f}")
print(f"Gini Coefficient: {consistency_report['statistical_metrics']['gini_coefficient']:.4f}")

# Check for outliers
if consistency_report['outliers']:
    print(f"Found {len(consistency_report['outliers'])} outlier pairs")
    for outlier in consistency_report['outliers']:
        print(f"Pair {outlier['pair']} with similarity {outlier['similarity']:.4f} (Z-score: {outlier['z_score']:.2f})")
```

### Handling Long Text Values

The framework intelligently handles long text values by:

1. Detecting content type (code, natural language, structured data)
2. Splitting into appropriate chunks (sentences, paragraphs, code blocks)
3. Finding optimal matches between chunks
4. Calculating similarity based on matched chunks and coverage

```python
# Compare JSON objects with long text values
json1 = {"description": long_text1, "code": code_block1}
json2 = {"description": long_text2, "code": code_block2}

similarity, _ = evaluator.calculate_tree_edit_distance(json1, json2)
print(f"Similarity with long text handling: {similarity:.4f}")
```

## Configuration Parameters

### Main Parameters

- **use_semantic_similarity** (bool): Whether to use embedding-based semantic similarity
- **embedding_model** (str): Name of the sentence transformer model
- **semantic_threshold** (float): Minimum similarity to consider keys as matching
- **array_order_matters** (bool): Whether array element order affects similarity
- **string_method** (str): Method for string comparison ('levenshtein', 'semantic', 'exact', 'jaccard')
- **number_tolerance** (float): Relative tolerance for number comparison

### Advanced Parameters

- **path_weight_decay** (float): Weight decay factor for deeper paths
- **type_change_cost** (Dict): Custom costs for type changes
- **required_fields** (Set[str]): Set of required field paths
- **key_semantic_weight** (float): Weight for semantic similarity vs exact match for keys
- **exact_match_weight** (float): Weight for exact key matching

## Consistency Metrics

The framework provides comprehensive metrics for evaluating consistency:

### Basic Metrics

- **mean_similarity**: Average similarity across all pairwise comparisons
- **std_deviation**: Standard deviation of similarity scores
- **min_similarity** / **max_similarity**: Minimum and maximum similarity values
- **similarity_range**: Difference between max and min similarity
- **consistency_coefficient**: Combined metric that rewards high similarity and penalizes variance

### Statistical Metrics

- **quartiles**: Q1, median, Q3, and IQR values
- **entropy**: Measure of unpredictability in similarity distribution
- **gini_coefficient**: Measure of inequality in similarity distribution

### Outlier Analysis

- **outliers**: List of outlier pairs with similarity values and Z-scores
- **most_different_pairs**: Pairs with lowest similarity scores and their edit operations

## Requirements

- Python 3.8+
- Required packages:
  - numpy
  - scipy
  - sentence-transformers
  - zss (optional, for Zhang-Shasha algorithm)

## Integration with Field-Aware Consistency Evaluation

The Semantic JSON Tree Consistency framework can be integrated with the Field-Aware Consistency Evaluation framework for comprehensive evaluation of LLM outputs:

```python
from semantic_json_tree_consistency import SemanticJsonTreeConsistencyEvaluator
from src.consistency_eval import FieldAwareConsistencyCalculator

# Initialize both evaluators
tree_evaluator = SemanticJsonTreeConsistencyEvaluator(
    use_semantic_similarity=True,
    semantic_threshold=0.7
)

field_evaluator = FieldAwareConsistencyCalculator(
    bedrock_client=bedrock_client,
    eval_fields=["reason", "position", "modified_version"],
    result_field_name="corrections"
)

# Use tree evaluator for structural consistency
structural_report = tree_evaluator.evaluate_structural_consistency(json_outputs)

# Use field evaluator for field-specific consistency
field_report, metrics = field_evaluator.calculate_prompt_consistency(responses)

# Combine insights from both approaches
combined_analysis = {
    "structural_consistency": structural_report["consistency_metrics"]["mean_similarity"],
    "field_consistency": field_report["consistency_score"]["overall"],
    "structural_std_dev": structural_report["consistency_metrics"]["std_deviation"],
    "outliers_detected": len(structural_report["outliers"]) > 0
}
```