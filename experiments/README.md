# Experiment Management

This directory contains experiment records and management tools for reproducible research.

## Directory Structure

```
experiments/
├── README.md                    # This file
├── registry.json                # Index of all experiments
├── templates/
│   └── experiment_template.json # Template for new experiments
├── {experiment_id}/             # One directory per experiment
│   ├── config.json              # Experiment configuration
│   ├── requirements.txt         # Python dependencies
│   ├── run.sh                   # Reproduction script
│   └── results/                 # Local results (synced from S3)
└── scripts/
    └── manage_experiment.py     # Experiment management CLI
```

## Quick Start

### Create New Experiment

```bash
python experiments/scripts/manage_experiment.py create \
    --name "colm_architecture_analysis" \
    --description "COLM 2026 architecture analysis experiments" \
    --script "scripts/experiments/colm_architecture/run_colm_experiments.py"
```

### Record Experiment Run

```bash
python experiments/scripts/manage_experiment.py record \
    --experiment colm_architecture_analysis \
    --instance-id i-0abc123def456 \
    --instance-type t3.xlarge \
    --data-source s3://sted-experiment-data/toucan_data/ \
    --status completed
```

### Sync Results to S3

```bash
python experiments/scripts/manage_experiment.py sync \
    --experiment colm_architecture_analysis \
    --direction upload
```

### List All Experiments

```bash
python experiments/scripts/manage_experiment.py list
```

## S3 Structure

All experiment data is stored in S3:

```
s3://sted-experiment-data/
├── code/                        # Code snapshots
│   └── {experiment_id}/
│       └── code-{timestamp}.tar.gz
├── data/                        # Input datasets
│   ├── toucan/
│   └── sharegpt/
├── experiments/                 # Experiment configs and results
│   └── {experiment_id}/
│       ├── config.json
│       ├── requirements.txt
│       ├── runs/
│       │   └── {run_id}/
│       │       ├── run_config.json
│       │       └── results/
│       └── results/             # Final aggregated results
└── venvs/                       # Virtual environment snapshots
    └── {experiment_id}/
        └── requirements-{timestamp}.txt
```

## Experiment Lifecycle

1. **Create**: `manage_experiment.py create` - Initialize experiment directory and config
2. **Setup**: `manage_experiment.py setup-ec2` - Launch EC2 and setup environment
3. **Run**: Execute experiment script on EC2
4. **Record**: `manage_experiment.py record` - Log run details
5. **Sync**: `manage_experiment.py sync` - Upload results to S3
6. **Archive**: `manage_experiment.py archive` - Archive completed experiment
