#!/bin/bash
# Run Multi-LLM Benchmark in Background
# This script runs the experiment using nohup so it continues even if SSH disconnects

set -e

# Configuration
SAMPLE_SIZE=${SAMPLE_SIZE:-100}
RUNS_PER_MODEL=${RUNS_PER_MODEL:-5}
TEMPERATURE=${TEMPERATURE:-0.7}
MODELS=${MODELS:-"claude-sonnet claude-haiku llama-70b"}

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

# Create output directory with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="$SCRIPT_DIR/results/run_${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"

# Log file
LOG_FILE="$OUTPUT_DIR/experiment.log"

echo "=========================================="
echo "Multi-LLM Benchmark Runner"
echo "=========================================="
echo "Sample size: $SAMPLE_SIZE"
echo "Runs per model: $RUNS_PER_MODEL"
echo "Temperature: $TEMPERATURE"
echo "Models: $MODELS"
echo "Output: $OUTPUT_DIR"
echo "Log: $LOG_FILE"
echo "=========================================="

# Activate virtual environment if exists
if [ -f "$PROJECT_ROOT/../venv/bin/activate" ]; then
    source "$PROJECT_ROOT/../venv/bin/activate"
elif [ -f "$PROJECT_ROOT/.venv/bin/activate" ]; then
    source "$PROJECT_ROOT/.venv/bin/activate"
fi

# Change to project root
cd "$PROJECT_ROOT"

# Run experiment in background with nohup
echo "Starting experiment in background..."
echo "Use 'tail -f $LOG_FILE' to monitor progress"
echo ""

nohup python "$SCRIPT_DIR/run_multi_llm_experiment.py" \
    --models $MODELS \
    --sample-size $SAMPLE_SIZE \
    --runs-per-model $RUNS_PER_MODEL \
    --temperature $TEMPERATURE \
    --output-dir "$OUTPUT_DIR" \
    > "$LOG_FILE" 2>&1 &

PID=$!
echo $PID > "$OUTPUT_DIR/experiment.pid"

echo "Experiment started with PID: $PID"
echo "PID saved to: $OUTPUT_DIR/experiment.pid"
echo ""
echo "To monitor: tail -f $LOG_FILE"
echo "To check status: ps -p $PID"
echo "To stop: kill $PID"
echo ""
echo "When complete, run evaluation:"
echo "python $SCRIPT_DIR/evaluate_consistency.py --results-dir $OUTPUT_DIR"
