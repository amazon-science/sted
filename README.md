# Semantic JSON Tree Consistency Evaluation

A sophisticated framework for evaluating structural and semantic consistency between JSON objects using tree edit distance algorithms with semantic understanding.

## Overview

The Semantic JSON Tree Consistency Evaluation framework provides advanced tools for comparing JSON structures beyond simple string matching. It converts JSON objects into tree representations and calculates edit distances between them, taking into account both structural similarities and semantic meaning.

This framework is particularly useful for:
- Evaluating consistency of LLM-generated JSON outputs
- Comparing JSON structures with different naming conventions
- Measuring similarity between complex nested JSON objects
- Analyzing structural patterns across multiple JSON documents

## Key Features

- **Semantic Understanding**: Recognizes when fields have different names but similar meanings
- **Tree-Based Comparison**: Uses tree edit distance algorithms for accurate structural comparison
- **Enhanced String Comparison**: Intelligently handles long text values by breaking them into meaningful chunks
- **Comprehensive Metrics**: Provides detailed consistency metrics including statistical measures
- **Customizable Costs**: Allows fine-tuning of edit operation costs for different scenarios

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

## Semantic Similarity Features

### Key Mapping with Semantic Understanding

The framework can recognize when keys have different names but similar meanings:

```python
# Example with semantically similar keys
json1 = {"user_info": {"user_name": "John", "user_age": 30}}
json2 = {"profile": {"name": "John", "age": 30}}

# With semantic similarity enabled
result_semantic = evaluate_semantic_json_consistency(
    [json1, json2],
    use_semantic_similarity=True
)

# Without semantic similarity
result_standard = evaluate_semantic_json_consistency(
    [json1, json2],
    use_semantic_similarity=False
)

print(f"Semantic similarity: {result_semantic['consistency_metrics']['mean_similarity']:.4f}")
print(f"Standard similarity: {result_standard['consistency_metrics']['mean_similarity']:.4f}")
```

### Enhanced String Comparison for Long Text

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

## Tree Edit Distance Algorithm

The framework uses the Tree Edit Distance algorithm to compare JSON structures:

1. **Tree Conversion**: JSON objects are converted to tree structures with typed nodes
2. **Edit Operations**: Three basic operations are defined - insert, delete, and update
3. **Cost Functions**: Custom cost functions determine the cost of each operation
4. **Optimal Edit Script**: The algorithm finds the minimum-cost sequence of operations
5. **Similarity Score**: Final similarity is calculated as 1 - (normalized distance)

For optimal matching between trees, the framework uses:
- **Zhang-Shasha Algorithm**: An efficient algorithm for tree edit distance calculation
- **Hungarian Algorithm**: For optimal bipartite matching in array comparison
- **Semantic Matching**: For finding corresponding keys with different names

## Requirements

- Python 3.8+
- Required packages:
  - numpy
  - scipy
  - sentence-transformers
  - zss (optional, for Zhang-Shasha algorithm)

## Examples

### Comparing Product Data with Different Schemas

```python
# Product data with different schemas
product1 = {
    "product_id": "P12345",
    "product_name": "Wireless Headphones",
    "product_price": 99.99,
    "product_category": "Electronics",
    "product_specifications": {
        "color": "Black",
        "weight": "250g",
        "battery_life": "20 hours"
    }
}

product2 = {
    "id": "P12345",
    "title": "Wireless Headphones",
    "price": 99.99,
    "category": "Electronics",
    "specs": {
        "color": "Black",
        "weight": "250g",
        "battery": "20 hours"
    }
}

# Compare with semantic understanding
evaluator = SemanticJsonTreeConsistencyEvaluator(use_semantic_similarity=True)
similarity, operations = evaluator.calculate_tree_edit_distance(product1, product2)

print(f"Semantic similarity: {similarity:.4f}")
print("Key edit operations:")
for op in operations[:3]:
    print(f"- {op['operation']} at {op['path']}")
```

### Analyzing Multiple LLM Outputs

```python
# Multiple JSON outputs from an LLM
llm_outputs = [output1, output2, output3, output4, output5]

# Evaluate consistency
consistency_report = evaluator.evaluate_structural_consistency(llm_outputs)

# Print key metrics
print(f"Overall consistency: {consistency_report['consistency_metrics']['mean_similarity']:.4f}")
print(f"Consistency coefficient: {consistency_report['consistency_metrics']['consistency_coefficient']:.4f}")

# Identify problematic areas
if consistency_report['frequently_edited_paths']:
    print("\nMost frequently edited paths:")
    for item in consistency_report['frequently_edited_paths'][:3]:
        print(f"- {item['path']}: {item['edit_count']} edits")
```

## Command-line Tools

### Model Comparison

The `run_model_comparison.py` script allows you to compare the consistency of JSON outputs across different LLM models at a fixed temperature.

```bash
python run_model_comparison.py --data-dir <data_directory> --output-dir <output_directory> [options]
```

#### Options:
- `--data-dir`: Directory containing the data files (required)
- `--output-dir`: Directory to save results (default: "./model_comparison")
- `--temperature`: Fixed temperature to use for all models (default: 0.7)
- `--run-num`: Number of runs per model (default: 10)
- `--sample-limit`: Number of samples to process (default: 5)

#### Example:
```bash
python run_model_comparison.py --data-dir ./sharegpt_data --output-dir ./model_comparison --temperature 0.5 --run-num 5 --sample-limit 3
```

#### Output:
- Generates JSON outputs for each model
- Evaluates consistency using semantic tree comparison
- Creates visualizations comparing models:
  - Mean consistency scores
  - Consistency coefficient
  - Semantic improvement
  - Standard deviation
  - Empty response rates
  - Consistency vs. empty response correlation

### Temperature Experiment

The `run_temperature_experiment.py` script evaluates how temperature settings affect the consistency of JSON outputs for a single model.

```bash
python run_temperature_experiment.py --data-dir <data_directory> --output-dir <output_directory> [options]
```

#### Options:
- `--data-dir`: Directory containing the data files (required)
- `--output-dir`: Directory to save results (default: "./temperature_experiment")
- `--model-id`: Model ID to use (default: "us.anthropic.claude-3-5-sonnet-20241022-v2:0")
- `--run-num`: Number of runs per temperature (default: 10)
- `--sample-limit`: Number of samples to process (default: 5)

#### Example:
```bash
python run_temperature_experiment.py --data-dir ./sharegpt_data --output-dir ./temperature_results --model-id us.amazon.nova-pro-v1:0 --run-num 5
```

#### Output:
- Generates JSON outputs at different temperatures (0.1, 0.3, 0.5, 0.7, 1.0)
- Evaluates consistency using semantic tree comparison
- Creates visualizations showing temperature effects:
  - Mean consistency vs. temperature
  - Consistency coefficient vs. temperature
  - Standard deviation vs. temperature
  - Min-max range vs. temperature
  - Empty response rate vs. temperature

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request