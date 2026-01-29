#!/bin/bash
# STED Experiment Runner Script
# This script helps run temperature experiments with different models
#
# Usage:
#   ./scripts/docker-run-experiment.sh --help
#   ./scripts/docker-run-experiment.sh --list-models
#   ./scripts/docker-run-experiment.sh --model us.amazon.nova-2-lite-v1:0
#   ./scripts/docker-run-experiment.sh --all-bedrock  # Run all Bedrock models sequentially

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed"
    exit 1
fi

# Build the image if it doesn't exist
build_image() {
    echo "Building STED experiment Docker image..."
    docker build -t sted-experiment:latest .
}

# Run experiment with a specific model
run_experiment() {
    local model_id="$1"
    shift
    local extra_args="$@"

    echo "Running experiment with model: $model_id"

    docker run --rm \
        -v "$PROJECT_ROOT/results:/app/results" \
        -v "$PROJECT_ROOT/logs:/app/logs" \
        -e AWS_ACCESS_KEY_ID \
        -e AWS_SECRET_ACCESS_KEY \
        -e AWS_SESSION_TOKEN \
        -e AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-us-west-2}" \
        -e OPENAI_API_KEY \
        -e OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://openrouter.ai/api/v1}" \
        sted-experiment:latest \
        --model "$model_id" $extra_args
}

# Run all Bedrock models (dynamically from MODEL_REGISTRY)
run_all_bedrock() {
    local extra_args="$@"

    # Get all Bedrock models from MODEL_REGISTRY
    BEDROCK_MODELS=$(docker run --rm sted-experiment:latest python3 -c "
from sted.model_config import MODEL_REGISTRY
for model_id, (provider, name, workers) in MODEL_REGISTRY.items():
    if provider == 'bedrock':
        print(model_id)
")

    for model in $BEDROCK_MODELS; do
        echo ""
        echo "========================================"
        echo "Starting experiment: $model"
        echo "========================================"
        run_experiment "$model" $extra_args || echo "Warning: $model experiment failed"
    done
}

# Run all OpenAI models (dynamically from MODEL_REGISTRY)
run_all_openai() {
    local extra_args="$@"

    # Get all OpenAI models from MODEL_REGISTRY
    OPENAI_MODELS=$(docker run --rm sted-experiment:latest python3 -c "
from sted.model_config import MODEL_REGISTRY
for model_id, (provider, name, workers) in MODEL_REGISTRY.items():
    if provider == 'openai':
        print(model_id)
")

    for model in $OPENAI_MODELS; do
        echo ""
        echo "========================================"
        echo "Starting experiment: $model"
        echo "========================================"
        run_experiment "$model" $extra_args || echo "Warning: $model experiment failed"
    done
}

# Show help
show_help() {
    cat << EOF
STED Docker Experiment Runner

Available models are defined in sted/model_config.py (MODEL_REGISTRY)

Usage: $0 [options]

Options:
    --help              Show this help message
    --build             Build the Docker image
    --list-models       List all available models from MODEL_REGISTRY
    --model MODEL_ID    Run experiment with specified model
    --all-bedrock       Run all Bedrock models from MODEL_REGISTRY sequentially
    --all-openai        Run all OpenAI/OpenRouter models from MODEL_REGISTRY sequentially
    --test              Run quick test (10 samples, 3 runs)

Additional arguments after --model or --all-* are passed to the container:
    --temperatures T1 T2 ...   Temperatures to test
    --samples N                Number of samples
    --runs N                   Runs per temperature
    --workers N                Parallel workers

Examples:
    # Build the image
    $0 --build

    # List all available models
    $0 --list-models

    # Run Nova 2 Lite with default settings
    $0 --model us.amazon.nova-2-lite-v1:0

    # Run quick test
    $0 --test

    # Run with custom temperatures
    $0 --model us.amazon.nova-2-lite-v1:0 --temperatures 0.0 0.5 1.0 --samples 100

    # Run all Bedrock models
    $0 --all-bedrock --samples 500

    # Run all OpenAI models
    $0 --all-openai --samples 500
EOF
}

# Parse arguments
case "${1:-}" in
    --help)
        show_help
        ;;
    --build)
        build_image
        ;;
    --list-models)
        # Ensure image exists
        if ! docker image inspect sted-experiment:latest &> /dev/null; then
            build_image
        fi
        docker run --rm sted-experiment:latest --list-models
        ;;
    --model)
        shift
        if [ -z "$1" ]; then
            echo "Error: --model requires a model ID"
            exit 1
        fi
        # Ensure image exists
        if ! docker image inspect sted-experiment:latest &> /dev/null; then
            build_image
        fi
        run_experiment "$@"
        ;;
    --all-bedrock)
        shift
        # Ensure image exists
        if ! docker image inspect sted-experiment:latest &> /dev/null; then
            build_image
        fi
        run_all_bedrock "$@"
        ;;
    --all-openai)
        shift
        # Ensure image exists
        if ! docker image inspect sted-experiment:latest &> /dev/null; then
            build_image
        fi
        run_all_openai "$@"
        ;;
    --test)
        # Ensure image exists
        if ! docker image inspect sted-experiment:latest &> /dev/null; then
            build_image
        fi
        run_experiment "us.amazon.nova-2-lite-v1:0" --samples 10 --runs 3 --temperatures 0.0 0.5 1.0
        ;;
    *)
        show_help
        ;;
esac
