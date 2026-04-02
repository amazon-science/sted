#!/bin/bash
# Experiment: COLM 2026 Consistency Predictor
# ID: colm_2026_consistency_predicto_20260210_154446
# Created: 2026-02-10T15:44:55.538309

set -e

# Setup virtual environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Set environment
export PYTHONPATH=${PYTHONPATH}:/Users/guanghu/Documents/genai/projects/sted-internal
export AWS_DEFAULT_REGION=${AWS_DEFAULT_REGION:-us-east-1}

# Run experiment
python scripts/experiments/colm_consistency_predictor/experiments/exp3_train_predictor.py "$@"

# Sync results to S3
aws s3 sync results/ s3://<your-s3-bucket>/experiments/colm_2026_consistency_predicto_20260210_154446/results/
