# Scripts Reference

## Data Preparation (`scripts/data/`)

### download_sharegpt_data.py
Download and extract ShareGPT datasets for structured output evaluation.

**Usage:**
```bash
python scripts/data/download_sharegpt_data.py --output-dir sharegpt_data
```

**Parameters:**
- `--output-dir`: Directory to save downloaded datasets (default: `sharegpt_data`)

**Outputs:**
- Downloads two datasets:
  - `sharegpt-structured-output-json` (30 samples)
  - `sharegpt-quizz-generation-json-output` (50 samples)
- Creates individual conversation JSON files
- Creates `all_conversations.json` for each dataset

### generate_synthetic_datasets.py
Generate synthetic datasets with schema, expression, and semantic variations for STED evaluation.

## Evaluation & Generation (`scripts/eval/`)

### run_temperature_experiment.py
Run LLM generation experiments across different temperature settings to evaluate consistency.

### generate_structured_outputs.py
Generate structured JSON outputs from LLMs for consistency evaluation.

**Usage:**
```bash
python scripts/eval/generate_structured_outputs.py \
  --data-dir sharegpt_data \
  --output-dir llm_gen_results \
  --model-id anthropic.claude-3-haiku-20240307-v1:0 \
  --temperature 0.5 \
  --run-num 10 \
  --sample-limit 5 \
  --max-tokens 2000 \
  --include-schema
```

**Parameters:**
- `--data-dir`: Directory containing the data files (required)
- `--output-dir`: Directory to save generation results (default: `./generations`)
- `--model-id`: The ID of the model to use (default: `us.anthropic.claude-3-5-sonnet-20241022-v2:0`)
- `--temperature`: Temperature for sampling 0.0-2.0 (default: `0`)
- `--run-num`: Number of inference runs to perform (default: `5`)
- `--sample-limit`: Limit the number of samples to process (default: `-1` for all)
- `--max-tokens`: Maximum tokens for LLM generation (default: `8000`)
- `--max-context-tokens`: Maximum context tokens to use (default: `32767`)
- `--include-schema`: Include JSON schema in the prompt
- `--skip-long-samples`: Skip samples that exceed token limit
- `--max-workers`: Maximum parallel workers (default: auto-determined)
- `--top-p`: Top-p sampling parameter (default: `0.9`)
- `--top-k`: Top-k sampling parameter (default: `200`)
- `--verbose`: Enable verbose logging

**Outputs:**
- `llm_gen_results_{model}_{temp}_{timestamp}/` directory containing:
  - `all_results.json` - All generation results
  - `{sample_id}.json` - Individual sample results
  - `{sample_id}_modified_prompt.txt` - Modified prompts (if schema included)

### calculate_consistency_metrics.py
Calculate structural, content, and combined consistency metrics for LLM outputs using STED.

**Usage:**
```bash
python scripts/eval/calculate_consistency_metrics.py \
  --results-dir llm_gen_results \
  --output-dir results/consistency_metrics
```

**Parameters:**
- `--results-dir`: Directory containing LLM generation results (default: `llm_gen_results`)
- `--output-dir`: Directory to save output files (default: `results`)

**Outputs:**
- `structural_consistency_metrics_results.json` - Structural consistency metrics
- `content_consistency_metrics_results.json` - Content consistency metrics
- `combined_consistency_metrics_results.json` - Combined consistency metrics
- `{variation_type}_consistency_metrics_comparison.png` - Visualization for each type

## Analysis (`scripts/analysis/`)

### compare_consistency_approaches.py
Compare different consistency calculation approaches (STED, probabilistic, etc.).

### compare_stability_calculations.py
Compare different stability score calculation methods.

**Usage:**
```bash
python scripts/analysis/compare_stability_calculations.py --output-dir results/stability_analysis
```

**Parameters:**
- `--output-dir`: Directory to save output files (default: `results`)

**Outputs:**
- `stability_calculation_comparison.png` - Comparison visualization
- `continuous_stability_comparison.png` - Continuous comparison plot

### verify_consistency_metrics_theory.py
Verify theoretical properties of consistency metrics (metric space, statistical, information theory).

**Usage:**
```bash
python scripts/analysis/verify_consistency_metrics_theory.py --output-dir results/theory_verification
```

**Parameters:**
- `--output-dir`: Directory to save output files (default: `results`)

**Outputs:**
- `power_transformation_analysis.png` - Power transformation analysis visualization

## Dataset Analysis (`scripts/dataset_analysis/`)

### analyze_base_dataset_stat.py
Analyze statistical properties of the base ShareGPT dataset (75 samples).

**Usage:**
```bash
python scripts/dataset_analysis/analyze_base_dataset_stat.py \
  --dataset-dir synthetic_dataset \
  --output-dir results/base_dataset_analysis
```

**Parameters:**
- `--dataset-dir`: Directory containing synthetic datasets (default: `synthetic_dataset`)
- `--output-dir`: Directory to save output files (default: `results`)

**Outputs:**
- `75_base_samples_metrics.csv` - Detailed metrics for each sample
- `75_samples_summary_statistics.csv` - Summary statistics table
- `75_base_samples_analysis.png` - Comprehensive visualizations
- `base_dataset_analysis.md` - Complete analysis report

### analyze_synthetic_dataset_stat.py
Analyze statistical properties of synthetic variation datasets (2400 samples).

**Usage:**
```bash
python scripts/dataset_analysis/analyze_synthetic_dataset_stat.py \
  --dataset-dir synthetic_dataset \
  --output-dir results/synthetic_dataset_analysis
```

**Parameters:**
- `--dataset-dir`: Directory containing synthetic datasets (default: `synthetic_dataset`)
- `--output-dir`: Directory to save output files (default: `results`)

**Outputs:**
- `comprehensive_dataset_metrics.csv` - Detailed metrics for unique samples
- `dataset_summary_statistics.csv` - Summary statistics table
- `comprehensive_dataset_analysis.png` - Comprehensive visualizations
- `synthetic_dataset_analysis.md` - Complete analysis report

### analyze_synthetic_schema_variations.py
Analyze schema variation patterns in synthetic datasets.

**Usage:**
```bash
python scripts/dataset_analysis/analyze_synthetic_schema_variations.py \
  synthetic_dataset/schema_variation_dataset_*.json \
  --output-dir results/schema_variation
```

**Parameters:**
- `dataset_file`: Schema variation dataset file to analyze (positional, required)
- `--output-dir`: Directory to save output files (default: `results`)

**Outputs:**
- `schema_variation_analysis.png` - Visualization of schema variations
- `schema_variation_analysis_results.json` - Detailed analysis results

### analyze_semantic_expression_variation_progression.py
Analyze semantic and expression variation progression patterns.

**Usage:**
```bash
python scripts/dataset_analysis/analyze_semantic_expression_variation_progression.py \
  synthetic_dataset/expression_variation_dataset_*.json \
  --output-dir results/variation_progression
```

**Parameters:**
- `files`: Dataset files to analyze (positional, required)
- `--output-dir`: Directory to save output files (default: `results`)

**Outputs:**
- `{variation_type}_variation_progression_results.json` - Detailed progression data
- `{variation_type}_variation_progression_analysis.png` - Visualization

### analyze_variation_consistency.py
Analyze consistency patterns across different variation ratios in synthetic datasets.

**Usage:**
```bash
python scripts/dataset_analysis/analyze_variation_consistency.py \
  synthetic_dataset/expression_variation_dataset_*.json \
  --method sted \
  --variation-type combined \
  --output-dir results/variation_consistency
```

**Parameters:**
- `file`: Dataset file to analyze (positional, required)
- `--method`: Similarity calculation method (default: `sted`, choices: ted, sted, bertscore, deepdiff, gnn)
- `--variation-type`: Type of variation to analyze (default: `combined`, choices: structural, content, combined)
- `--output-dir`: Directory to save output files (default: `results`)

**Outputs:**
- `variation_consistency_{method}_{variation_type}.json` - Consistency analysis results

### calculate_variation_consistency_metrics.py
Calculate consistency metrics for synthetic variation datasets.

## Visualization (`scripts/visualization/`)

### visualize_consistency_scores.py
Visualize consistency scores across models and temperatures for LLM benchmarking.

**Usage:**
```bash
python scripts/visualization/visualize_consistency_scores.py \
  --combined results/combined_consistency_metrics_results.json \
  --content results/content_consistency_metrics_results.json \
  --structural results/structural_consistency_metrics_results.json \
  --output-dir results/visualizations
```

**Parameters:**
- `--combined`: Path to combined consistency metrics results JSON file (required)
- `--content`: Path to content consistency metrics results JSON file (required)
- `--structural`: Path to structural consistency metrics results JSON file (required)
- `--output-dir`: Directory to save output files (default: `results`)

**Outputs:**
- `consistency_score_by_consistency_type_with_errors.png` - Consistency score visualization
- `normalized_cv_by_consistency_type_with_errors.png` - Normalized CV visualization
- `detailed_consistency_statistics.csv` - Detailed statistics
- `enhanced_consistency_summary.csv` - Summary statistics

### visualize_variation_progression.py
Visualize how similarity changes with variation ratio for expression and semantic variations.

**Usage:**
```bash
python scripts/visualization/visualize_variation_progression.py \
  --semantic results/semantic_variation_progression_results.json \
  --expression results/expression_variation_progression_results.json \
  --output-dir results/visualizations
```

**Parameters:**
- `--semantic`: Path to semantic variation results JSON (default: `experiments/experiment-1/semantic_variation_progression_results.json`)
- `--expression`: Path to expression variation results JSON (default: `experiments/experiment-1/expression_variation_progression_results.json`)
- `--output-dir`: Directory to save output files (default: `results`)

**Outputs:**
- `similarity_progression_comparison.png` - Comparison visualization

### visualize_schema_variation.py
Visualize schema variation analysis results.

**Usage:**
```bash
python scripts/visualization/visualize_schema_variation.py \
  --input results/schema_variation_analysis_results.json \
  --output-dir results/visualizations
```

**Parameters:**
- `--input`: Input JSON file with analysis results (default: `schema_variation_analysis_results.json`)
- `--output-dir`: Directory to save output files (default: `results`)

**Outputs:**
- `schema_variation_analysis_with_errors.png` - Visualization with error bars
- `schema_variation_detailed_statistics.csv` - Detailed statistics

## Experiments (`scripts/experiments/`)

Research experiments for probabilistic consistency and sigma parameter tuning.

### evaluate_probabilistic_consistency.py
Evaluate probabilistic consistency approach on real LLM data.

### evaluate_probabilistic_fixed_sigma.py
Evaluate probabilistic consistency with fixed sigma value (σ₀=0.05).

### evaluate_final_probabilistic.py
Final evaluation comparing adaptive probabilistic vs power transformation.

### test_sigma_values.py
Test different sigma values for probabilistic consistency.

### quick_test_sigma.py
Quick analysis of existing results with different sigma values.

### validate_pdc_neurips.py
Validate PDC (Probabilistic Distance-based Consistency) metric for NeurIPS submission.

### validate_power_transform_theory.py
Validate power transformation theory for consistency metrics.

