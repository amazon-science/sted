# Modular LLM Generation and Evaluation

This repository provides a modular approach to LLM generation and evaluation, allowing you to:

1. Generate outputs from LLMs
2. Evaluate those outputs using various metrics
3. Run different evaluation experiments on the same generated outputs
4. Compare different evaluation methodologies

## Scripts

### 1. LLM Generation (`llm_gen_simple.py`)

This script focuses solely on generating outputs from LLMs and saving them for later evaluation.

```bash
python llm_gen_simple.py --data-dir extracted_sharegpt_data --output-dir ./generations
```

Key features:
- Parallel inference using ThreadPoolExecutor
- Optional JSON schema inclusion in prompts
- Saves both individual sample results and combined results

#### Arguments:

- `--model-id`: The ID of the Bedrock model to use (default: "us.anthropic.claude-3-5-sonnet-20241022-v2:0")
- `--data-dir`: The directory containing the data files (required)
- `--output-dir`: Directory to save generation results (default: "./generations")
- `--run-num`: Number of inference runs to perform (default: 5)
- `--temperature`: Temperature for sampling (default: 0.1)
- `--top-p`: Top-p (nucleus) sampling parameter (default: 0.9)
- `--top-k`: Top-k sampling parameter (default: 200)
- `--include-schema`: Include JSON schema in the prompt to guide the output structure
- `--sample-limit`: Limit the number of samples to process

### 2. Evaluation (`evaluate_generations.py`)

This script evaluates previously generated LLM outputs using various metrics and now includes overall metrics across all samples.

```bash
python evaluate_generations.py --input-file ./generations/llm_gen_results_claude-3-5-sonnet-20241022-v2_temp_0_10_20250622_123456/all_results.json --output-dir ./evaluation_results
```

Key features:
- Supports multiple evaluation metrics:
  - Semantic tree-based metrics (accuracy and cross-run consistency)
  - NLP-based metrics (BLEU, ROUGE, BERTScore)
- Calculates overall metrics across all samples
- Saves comprehensive evaluation results

#### Arguments:

- `--input-file`: Path to the JSON file containing the generated outputs (required)
- `--output-dir`: Directory to save the evaluation results (default: "./evaluation_results")
- `--metrics`: Which metrics to calculate (choices: "semantic", "nlp", "all"; default: "all")

### 3. Hungarian Algorithm Experiment (`run_hungarian_experiment.py`)

This script evaluates the effectiveness of the Hungarian algorithm for comparing arrays and long free text in the semantic tree evaluation approach.

```bash
python run_hungarian_experiment.py --data-dir extracted_sharegpt_data --output-dir ./hungarian_experiment
```

Key features:
- Compares evaluation with and without Hungarian algorithm
- Analyzes complex fields (arrays and long texts) in the data
- Creates visualizations showing improvements in accuracy and stability
- Provides detailed analysis of when Hungarian algorithm is most beneficial

#### Arguments:

- `--data-dir`: Directory containing the data files
- `--input-file`: Path to the JSON file containing the generated outputs (optional)
- `--output-dir`: Directory to save experiment results (default: "./hungarian_experiment")
- `--run-num`: Number of runs to perform (default: 5)
- `--include-schema`: Include JSON schema in the prompt

Note: The script uses a default sample limit of 10 when generating outputs.

### 4. String Method Comparison (`run_string_method_comparison.py`)

This script compares different string comparison methods in the semantic tree evaluation.

```bash
python run_string_method_comparison.py --data-dir extracted_sharegpt_data --output-dir ./string_method_experiment
```

Key features:
- Compares different string comparison methods (levenshtein, semantic, exact, jaccard)
- Evaluates impact on accuracy, stability, and cross-run consistency
- Creates visualizations showing the performance of each method
- Recommends optimal string method based on combined metrics

#### Arguments:

- `--data-dir`: Directory containing the data files
- `--input-file`: Path to the JSON file containing the generated outputs (optional)
- `--output-dir`: Directory to save experiment results (default: "./string_method_experiment")
- `--run-num`: Number of runs to perform (default: 5)
- `--include-schema`: Include JSON schema in the prompt
- `--string-methods`: List of string methods to test (choices: "levenshtein", "semantic", "exact", "jaccard"; default: all methods)

## Workflow Examples

### Example 1: Temperature Comparison

#### Step 1: Generate outputs with different temperatures

```bash
# Generate outputs with temperature 0.1
python llm_gen_simple.py --data-dir extracted_sharegpt_data --output-dir ./generations --temperature 0.1 --include-schema

# Generate outputs with temperature 0.7
python llm_gen_simple.py --data-dir extracted_sharegpt_data --output-dir ./generations --temperature 0.7 --include-schema
```

#### Step 2: Evaluate the generated outputs

```bash
# Evaluate outputs with temperature 0.1
python evaluate_generations.py --input-file ./generations/llm_gen_results_claude-3-5-sonnet-20241022-v2_temp_0_10_*/all_results.json --output-dir ./evaluation_results

# Evaluate outputs with temperature 0.7
python evaluate_generations.py --input-file ./generations/llm_gen_results_claude-3-5-sonnet-20241022-v2_temp_0_70_*/all_results.json --output-dir ./evaluation_results
```

#### Step 3: Compare different evaluation metrics

```bash
# Evaluate using only semantic metrics
python evaluate_generations.py --input-file ./generations/llm_gen_results_claude-3-5-sonnet-20241022-v2_temp_0_10_*/all_results.json --output-dir ./evaluation_results --metrics semantic

# Evaluate using only NLP metrics
python evaluate_generations.py --input-file ./generations/llm_gen_results_claude-3-5-sonnet-20241022-v2_temp_0_10_*/all_results.json --output-dir ./evaluation_results --metrics nlp
```

### Example 2: Hungarian Algorithm Evaluation

#### Step 1: Generate outputs

```bash
python llm_gen_simple.py --data-dir extracted_sharegpt_data --output-dir ./generations --include-schema --sample-limit 10
```

Note: The Hungarian experiment uses a default sample limit of 10 when generating outputs.

#### Step 2: Run Hungarian algorithm experiment

```bash
python run_hungarian_experiment.py --input-file ./generations/llm_gen_results_*/all_results.json --output-dir ./hungarian_experiment
```

#### Step 3: Analyze the results

Examine the visualizations in `./hungarian_experiment/visualizations/` to see the impact of the Hungarian algorithm on accuracy and stability.

### Example 3: String Method Comparison

#### Step 1: Generate outputs

```bash
python llm_gen_simple.py --data-dir extracted_sharegpt_data --output-dir ./generations --include-schema
```

#### Step 2: Compare different string methods

```bash
python run_string_method_comparison.py --input-file ./generations/llm_gen_results_*/all_results.json --output-dir ./string_method_experiment
```

Or specify specific string methods to test:

```bash
python run_string_method_comparison.py --input-file ./generations/llm_gen_results_*/all_results.json --output-dir ./string_method_experiment --string-methods levenshtein semantic
```

#### Step 3: Analyze the results

Examine the visualizations in `./string_method_experiment/visualizations/` to see which string method performs best.

## Metrics Explained

### Semantic Tree-Based Metrics

These metrics use tree edit distance with semantic understanding to evaluate:

1. **Ground Truth Accuracy**:
   - **Semantic similarity**: How similar the generated outputs are to the ground truth
   - **Stability across runs**: How consistent the similarity is across multiple runs

2. **Cross-Run Consistency**:
   - **Overall consistency score**: How consistent the outputs are across multiple runs
   - **Perfect consistency**: Whether all outputs are identical
   - **Consistency coefficient**: A measure of overall consistency quality

### NLP-Based Metrics

These metrics use standard NLP evaluation techniques:

1. **Ground Truth Accuracy**:
   - **BLEU score**: Measures n-gram precision
   - **ROUGE-L F1**: Measures longest common subsequence
   - **BERTScore**: Uses contextual embeddings for semantic similarity
   - **Jaccard similarity**: Measures overlap of keys

2. **Cross-Run Stability**:
   - **Overall stability**: How consistent the outputs are across multiple runs

### Hungarian Algorithm Enhancement

The Hungarian algorithm improves evaluation in two key ways:

1. **Array Matching**:
   - Finds optimal matching between array elements regardless of order
   - Particularly valuable for unordered lists or when order doesn't matter semantically

2. **Long Text Comparison**:
   - Breaks long texts into meaningful chunks
   - Finds optimal matching between chunks across texts
   - Captures structural similarity even when wording differs

### String Comparison Methods

Multiple string comparison methods are available:

1. **Levenshtein**: Character-based edit distance (default)
2. **Semantic**: Embedding-based semantic similarity
3. **Exact**: Binary exact matching
4. **Jaccard**: Token overlap similarity

## Benefits of This Modular Approach

1. **Separation of Concerns**: Generation and evaluation are separate processes
2. **Reusability**: Generate once, evaluate multiple times with different metrics
3. **Experimentation**: Easy to compare different evaluation approaches
4. **Efficiency**: No need to regenerate outputs for each evaluation experiment
5. **Comprehensive Analysis**: Multiple metrics provide a more complete picture of model performance
6. **Customizability**: Choose the evaluation methods most appropriate for your specific use case

## Experiments

This section outlines key experiments to demonstrate the value of tree-based semantic evaluation with stability consideration.

### 1. Temperature-Stability Correlation

**Purpose**: Determine the mathematical relationship between temperature settings and output stability.

**Implementation**:
```bash
python run_temperature_experiment.py --data-dir extracted_sharegpt_data \
                                   --output-dir ./temperature_experiment \
                                   --run-num 10 \
                                   --include-schema
```

**Expected Outcomes**:
- A mathematical formula describing the relationship between temperature and stability
- Visualization of the temperature-stability curve
- Identification of optimal temperature ranges for different use cases

### 2. Comparative Analysis of Evaluation Methods

**Purpose**: Compare tree-based semantic evaluation against traditional metrics.

**Implementation**:
```bash
# Generate outputs
python llm_gen_simple.py --data-dir extracted_sharegpt_data --output-dir ./generations --include-schema

# Evaluate using all metrics
python evaluate_generations.py --input-file ./generations/*/all_results.json --metrics all
```

**Expected Outcomes**:
- Demonstration that tree-based semantic metrics capture similarities that other metrics miss
- Quantification of the advantages over traditional metrics
- Identification of cases where different metrics disagree

### 3. Model Comparison

**Purpose**: Compare different models using tree-based semantic evaluation.

**Implementation**:
```bash
# Run with different models
python run_model_comparison.py --data-dir extracted_sharegpt_data \
                             --output-dir ./model_comparison \
                             --run-num 10 \
                             --include-schema
```

**Expected Outcomes**:
- Ranking of models by both accuracy and stability
- Identification of models with the best accuracy-stability balance
- Insights into model-specific strengths and weaknesses

### 4. Structural Complexity Analysis

**Purpose**: Analyze how structural complexity affects evaluation metrics.

**Implementation**:
```bash
# Generate test cases with varying complexity
python generate_complexity_test_cases.py --output-dir ./complexity_test_cases

# Run generation and evaluation
python run_complexity_experiment.py --test-cases ./complexity_test_cases \
                                  --output-dir ./complexity_experiment \
                                  --run-num 10
```

**Expected Outcomes**:
- Demonstration that tree-based semantic evaluation is more robust to structural complexity
- Quantification of how traditional metrics degrade with increasing complexity
- Guidelines for choosing evaluation metrics based on output complexity

### 5. Semantic Threshold Sensitivity

**Purpose**: Analyze the impact of different semantic thresholds on evaluation results.

**Implementation**:
```bash
# Generate outputs once
python llm_gen_simple.py --data-dir extracted_sharegpt_data --output-dir ./generations --include-schema

# Evaluate with different thresholds
python run_threshold_experiment.py --input-file ./generations/*/all_results.json \
                                 --output-dir ./threshold_experiment \
                                 --thresholds 0.5 0.6 0.7 0.8 0.9
```

**Expected Outcomes**:
- Optimal semantic threshold recommendations
- Understanding of how threshold affects precision vs. recall in semantic matching
- Guidelines for threshold selection based on use case

### 6. Hungarian Algorithm Effectiveness

**Purpose**: Evaluate the effectiveness of the Hungarian algorithm for comparing arrays and long free text.

**Implementation**:
```bash
python run_hungarian_experiment.py --data-dir extracted_sharegpt_data \
                                  --output-dir ./hungarian_experiment \
                                  --run-num 10 \
                                  --include-schema
```

**Key Features**:
- Compares evaluation with and without Hungarian algorithm
- Analyzes complex fields (arrays and long texts) in the data
- Creates visualizations showing improvements in accuracy and stability
- Calculates statistical significance of improvements

**Expected Outcomes**:
- Quantification of improvement from using the Hungarian algorithm
- Identification of cases where the Hungarian algorithm provides the most benefit
- Demonstration of robustness to array order and text structure variations
- Empirical evidence for the value of this algorithmic choice
- Visualization of accuracy vs. stability improvements

### 7. Run Count Analysis

**Purpose**: Determine the optimal number of runs for reliable stability measurement.

**Implementation**:
```bash
python run_count_experiment.py --data-dir extracted_sharegpt_data \
                              --output-dir ./run_count_experiment \
                              --max-runs 20 \
                              --include-schema
```

**Expected Outcomes**:
- Determination of minimum runs needed for reliable stability measurement
- Analysis of diminishing returns with increasing run counts
- Cost-benefit analysis for different run counts