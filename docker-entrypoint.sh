#!/bin/bash
set -e

# STED Temperature Experiment - Single Model per EC2 Instance
#
# Environment Variables:
#   MODEL_ID        - Model ID from MODEL_REGISTRY (required)
#   MODE            - Experiment mode: tool-calling or structured (default: tool-calling)
#   TEMPERATURES    - Space-separated temperatures (default: 0.0 0.1 ... 1.0)
#   SAMPLES         - Number of samples (default: 1006 for tool-calling, 100 for structured)
#   RUNS            - Runs per temperature (default: 10)
#   MAX_WORKERS     - Parallel workers (default: auto from MODEL_REGISTRY)
#   MAX_TOKENS      - Max tokens for generation (default: 2048 for tool-calling, 3000 for structured)
#   INCLUDE_SCHEMA  - Include schema in prompt for structured mode (default: true)
#   AWS_*           - AWS credentials for Bedrock models
#   OPENAI_API_KEY  - API key for OpenAI/OpenRouter models

# Handle --list-models flag
if [ "${1:-}" == "--list-models" ]; then
    echo "Available Models (from sted/model_config.py MODEL_REGISTRY):"
    echo "============================================================"
    python3 -c "
from sted.model_config import MODEL_REGISTRY
print()
print('Bedrock Models:')
print('-' * 70)
for model_id, (provider, name, workers) in MODEL_REGISTRY.items():
    if provider == 'bedrock':
        print(f'  {name:30s} (workers: {workers:3d})  {model_id}')
print()
print('OpenAI/OpenRouter Models:')
print('-' * 70)
for model_id, (provider, name, workers) in MODEL_REGISTRY.items():
    if provider == 'openai':
        print(f'  {name:30s} (workers: {workers:3d})  {model_id}')
"
    exit 0
fi

# Check if MODEL_ID is set
if [ -z "$MODEL_ID" ]; then
    echo "Error: MODEL_ID environment variable is required"
    echo ""
    echo "Usage:"
    echo "  # Tool-calling mode (Toucan dataset)"
    echo "  docker run -e MODEL_ID=us.amazon.nova-2-lite-v1:0 sted-experiment"
    echo ""
    echo "  # Structured output mode (ShareGPT dataset)"
    echo "  docker run -e MODEL_ID=us.amazon.nova-2-lite-v1:0 -e MODE=structured sted-experiment"
    echo ""
    echo "To list available models:"
    echo "  docker run sted-experiment --list-models"
    exit 1
fi

# Set mode (default: tool-calling)
MODE="${MODE:-tool-calling}"

# Set defaults based on mode
if [ "$MODE" == "structured" ]; then
    SAMPLES="${SAMPLES:-100}"
    MAX_TOKENS="${MAX_TOKENS:-3000}"
    INCLUDE_SCHEMA="${INCLUDE_SCHEMA:-true}"
    DATA_DIR="${DATA_DIR:-/app/sharegpt_data}"
else
    SAMPLES="${SAMPLES:-1006}"
    MAX_TOKENS="${MAX_TOKENS:-2048}"
    DATASET="${DATASET:-toucan_data/toucan_tool_calls_1006.json}"
fi

TEMPERATURES="${TEMPERATURES:-0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0}"
RUNS="${RUNS:-10}"
OUTPUT_DIR="${OUTPUT_DIR:-/app/results}"

# Get model config from MODEL_REGISTRY
PROVIDER=$(python3 -c "from sted.model_config import get_provider; print(get_provider('$MODEL_ID'))")
DISPLAY_NAME=$(python3 -c "from sted.model_config import get_display_name; print(get_display_name('$MODEL_ID'))")

# Get max_workers from MODEL_REGISTRY if not specified
if [ -z "$MAX_WORKERS" ]; then
    MAX_WORKERS=$(python3 -c "from sted.model_config import get_max_workers; print(get_max_workers('$MODEL_ID'))")
fi

# Verify credentials
if [ "$PROVIDER" == "bedrock" ]; then
    if [ -z "$AWS_ACCESS_KEY_ID" ] && [ ! -f ~/.aws/credentials ]; then
        echo "Warning: AWS credentials not found. Using IAM role if available."
    fi
elif [ "$PROVIDER" == "openai" ]; then
    if [ -z "$OPENAI_API_KEY" ]; then
        echo "Error: OPENAI_API_KEY is required for OpenAI/OpenRouter models"
        exit 1
    fi
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Display configuration
echo "========================================================================"
echo "STED Temperature Experiment"
echo "========================================================================"
echo "Mode:         $MODE"
echo "Model:        $DISPLAY_NAME"
echo "Model ID:     $MODEL_ID"
echo "Provider:     $PROVIDER"
echo "Temperatures: $TEMPERATURES"
echo "Samples:      $SAMPLES"
echo "Runs:         $RUNS"
echo "Max Tokens:   $MAX_TOKENS"
echo "Workers:      $MAX_WORKERS"
echo "Output:       $OUTPUT_DIR"
if [ "$MODE" == "structured" ]; then
    echo "Data Dir:     $DATA_DIR"
    echo "Inc Schema:   $INCLUDE_SCHEMA"
else
    echo "Dataset:      $DATASET"
fi
echo "========================================================================"
echo ""

# Run the experiment based on mode
if [ "$MODE" == "structured" ]; then
    # Check if ShareGPT data exists
    if [ ! -d "$DATA_DIR" ] || [ -z "$(ls -A $DATA_DIR 2>/dev/null)" ]; then
        echo "Error: ShareGPT data not found at $DATA_DIR"
        echo "Please mount your ShareGPT data: -v /path/to/sharegpt_data:/app/sharegpt_data"
        exit 1
    fi

    # Build command
    CMD="python3 scripts/eval/run_temperature_experiment.py \
        --mode structured \
        --data-dir $DATA_DIR \
        --output-dir $OUTPUT_DIR \
        --model-id $MODEL_ID \
        --sample-limit $SAMPLES \
        --temperatures $TEMPERATURES \
        --run-num $RUNS \
        --max-tokens $MAX_TOKENS"

    if [ "$INCLUDE_SCHEMA" == "true" ]; then
        CMD="$CMD --include-schema"
    fi

    exec $CMD
else
    # Tool-calling mode
    exec python3 scripts/eval/run_temperature_experiment.py \
        --mode tool-calling \
        --dataset-path "$DATASET" \
        --output-dir "$OUTPUT_DIR" \
        --model-id "$MODEL_ID" \
        --sample-limit "$SAMPLES" \
        --temperatures $TEMPERATURES \
        --run-num "$RUNS" \
        --max-tokens "$MAX_TOKENS" \
        --max-workers "$MAX_WORKERS"
fi
