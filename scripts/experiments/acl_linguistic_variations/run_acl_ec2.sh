#!/bin/bash
# ACL 2026 Linguistic Experiments - EC2 Runner
# Runs comprehensive experiments across multiple models on AWS EC2

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
RESULTS_DIR="$PROJECT_ROOT/results/acl_linguistic/comprehensive"

# Configuration
NUM_PROMPTS=50       # More prompts for statistical power
NUM_RUNS=15          # More runs per condition for significance
TEMPERATURES="0.0 0.3 0.5 0.7 1.0"  # Include T=0.0 for deterministic baseline

# Models to evaluate (6 key models for ACL paper)
BEDROCK_MODELS=(
    "us.anthropic.claude-sonnet-4-20250514-v1:0"
    "us.anthropic.claude-3-5-sonnet-20241022-v2:0"
    "us.meta.llama3-3-70b-instruct-v1:0"
    "qwen.qwen3-235b-a22b-2507-v1:0"
)

OPENROUTER_MODELS=(
    "openai/gpt-4.1-mini"
    "google/gemini-2.5-flash-lite"
)

echo "=============================================="
echo "ACL 2026 Linguistic Experiments"
echo "=============================================="
echo "Project Root: $PROJECT_ROOT"
echo "Results Dir: $RESULTS_DIR"
echo "Num Prompts: $NUM_PROMPTS"
echo "Num Runs: $NUM_RUNS"
echo "Temperatures: $TEMPERATURES"
echo ""

cd "$PROJECT_ROOT"

# Create results directory
mkdir -p "$RESULTS_DIR"

# Function to run experiment for a single model
run_model() {
    local model_id="$1"
    local model_name=$(echo "$model_id" | sed 's/.*\///' | sed 's/:.*$//' | tr '.' '-')
    local log_file="$RESULTS_DIR/${model_name}.log"

    echo "[$(date '+%H:%M:%S')] Starting: $model_name"

    PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH" python3 scripts/experiments/acl_linguistic_variations/run_comprehensive.py \
        --model "$model_id" \
        --num-prompts $NUM_PROMPTS \
        --num-runs $NUM_RUNS \
        --temperatures $TEMPERATURES \
        --output-dir "$RESULTS_DIR" \
        2>&1 | tee "$log_file"

    echo "[$(date '+%H:%M:%S')] Completed: $model_name"
}

# Run Bedrock models (parallel if on EC2 with sufficient resources)
echo ""
echo "=== Running Bedrock Models ==="
for model in "${BEDROCK_MODELS[@]}"; do
    run_model "$model"
done

# Run OpenRouter models (require API key)
if [ -n "$OPENROUTER_API_KEY" ]; then
    echo ""
    echo "=== Running OpenRouter Models ==="
    for model in "${OPENROUTER_MODELS[@]}"; do
        run_model "$model"
    done
else
    echo ""
    echo "WARNING: OPENROUTER_API_KEY not set, skipping OpenRouter models"
fi

# Generate combined analysis
echo ""
echo "=== Generating Combined Analysis ==="
PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH" python3 scripts/experiments/acl_linguistic_variations/analyze_comprehensive.py \
    --results-dir "$RESULTS_DIR" \
    --output "$RESULTS_DIR/combined_analysis.json"

echo ""
echo "=============================================="
echo "ALL EXPERIMENTS COMPLETED"
echo "=============================================="
echo "Results saved to: $RESULTS_DIR"
echo ""
ls -la "$RESULTS_DIR"
