# Structured Generation Evaluation

## Dataset
We use the dataset below:
[Arun63/sharegpt-structured-output-json](https://huggingface.co/datasets/Arun63/sharegpt-structured-output-json)
[Arun63/sharegpt-quizz-generation-json-output](https://huggingface.co/datasets/Arun63/sharegpt-quizz-generation-json-output/viewer/default/train?row=3&views%5B%5D=train)

Run the command below to prepare dataset
```bash
uv run download_sharegpt_data.py --output-dir sharegpt_data
```

## Generate structured data with LLM

### Inference with different temperature
```bash

```


### Temperature-Stability Correlation Experiment

The `run_temperature_experiment.py` script analyzes how temperature settings affect the variability and consistency of LLM-generated structured outputs. It runs comprehensive experiments across different temperature values and calculates correlation metrics.

#### Basic Usage
```bash
# Run experiment with default settings (temperatures 0.0-1.0)
uv run run_temperature_experiment.py --data-dir sharegpt_data --output-dir ./temperature_experiment

# Run with custom temperature range
uv run run_temperature_experiment.py --data-dir sharegpt_data --temperatures 0.0 0.3 0.6 0.9 --run-num 15

# Include JSON schema in prompts
uv run run_temperature_experiment.py --data-dir sharegpt_data --include-schema

# Use specific model
uv run run_temperature_experiment.py --data-dir sharegpt_data --model-id us.anthropic.claude-3-5-haiku-20241022-v1:0

# Run only generation phase (skip evaluation)
uv run run_temperature_experiment.py --data-dir sharegpt_data --exe-evaluation False

# Run evaluation on existing generations
uv run run_temperature_experiment.py --data-dir sharegpt_data --exe-evaluation True
```

#### Command Line Options
- `--data-dir`: Directory containing the data files (required)
- `--output-dir`: Directory to save experiment results (default: `./temperature_experiment`)
- `--run-num`: Number of generation runs per temperature (default: 10)
- `--temperatures`: Custom list of temperatures to test (default: 0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
- `--include-schema`: Include JSON schema in the generation prompt
- `--model-id`: Specify the LLM model to use (default: Claude 3.5 Sonnet)
- `--force-regenerate`: Force regeneration even if results already exist
- `--exe-evaluation`: Execute evaluation phase (similarity calculations and analysis) (default: True)

#### What the Script Does
1. **Generation Phase**: Runs LLM generation at each temperature setting with multiple iterations
2. **Evaluation Phase**: Calculates similarity metrics using three methods:
   - TED (Tree Edit Distance) for structural similarity
   - BERTScore for semantic similarity  
   - DeepDiff for content comparison
3. **Analysis Phase**: Computes correlations between temperature and variability metrics:
   - Standard deviation of mean similarities
   - Coefficient of variation
   - Stability scores
   - Linear and polynomial regression analysis
4. **Visualization Phase**: Creates comprehensive plots showing temperature vs variability relationships

#### Output Structure
```
temperature_experiment/
├── generations/                    # Generated outputs for each temperature
│   ├── llm_gen_results_temp_0_00_*/
│   ├── llm_gen_results_temp_0_20_*/
│   └── ...
├── visualizations/                 # Analysis plots and charts
│   ├── temperature_std_correlation_detailed.png
│   ├── temperature_std_means_comparison.png
│   ├── temperature_accuracy_vs_variability.png
│   └── *_correlation_matrix.png
└── temperature_std_correlation_results.json  # Complete analysis results
```

#### Key Metrics Analyzed
- **Standard Deviation of Means**: Primary metric for temperature-stability correlation
- **Mean Similarity**: Overall accuracy across generations
- **Coefficient of Variation**: Normalized variability measure
- **Stability Score**: Inverse relationship to variability

#### Example Background Execution
```bash
# Run experiment in background with logging
nohup uv run run_temperature_experiment.py --data-dir sharegpt_data --temperatures 0.0 0.3 0.6 0.9 --run-num 2 --model-id us.mistral.pixtral-large-2502-v1:0 --force-regenerate --sample-limit 2 > output.log 2> error.log &
```