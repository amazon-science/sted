#!/usr/bin/env python3
"""
Experiment Management CLI

Manages experiment lifecycle: create, setup, run, record, sync, archive.

Usage:
    python manage_experiment.py create --name "exp_name" --script "path/to/script.py"
    python manage_experiment.py record --experiment exp_name --instance-id i-xxx
    python manage_experiment.py sync --experiment exp_name --direction upload
    python manage_experiment.py list
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
import shutil
import hashlib

# Paths
SCRIPT_DIR = Path(__file__).parent
EXPERIMENTS_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = EXPERIMENTS_DIR.parent
TEMPLATES_DIR = EXPERIMENTS_DIR / "templates"
REGISTRY_FILE = EXPERIMENTS_DIR / "registry.json"

# S3 Configuration
S3_BUCKET = os.environ.get("STED_S3_BUCKET", "sted-experiment-data")


def load_registry() -> dict:
    """Load experiment registry."""
    if REGISTRY_FILE.exists():
        with open(REGISTRY_FILE) as f:
            return json.load(f)
    return {"experiments": {}, "updated_at": None}


def save_registry(registry: dict):
    """Save experiment registry."""
    registry["updated_at"] = datetime.now().isoformat()
    with open(REGISTRY_FILE, "w") as f:
        json.dump(registry, f, indent=2)


def generate_experiment_id(name: str) -> str:
    """Generate unique experiment ID."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name_slug = name.lower().replace(" ", "_").replace("-", "_")[:30]
    return f"{name_slug}_{timestamp}"


def get_git_info() -> dict:
    """Get current git info."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
        ).strip()
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=PROJECT_ROOT, text=True
        ).strip()
        return {"commit": commit, "branch": branch}
    except subprocess.CalledProcessError:
        return {"commit": "", "branch": ""}


def create_experiment(args):
    """Create a new experiment."""
    # Load template
    with open(TEMPLATES_DIR / "experiment_template.json") as f:
        config = json.load(f)

    # Generate ID
    exp_id = generate_experiment_id(args.name)
    exp_dir = EXPERIMENTS_DIR / exp_id

    # Fill in config
    config["experiment_id"] = exp_id
    config["name"] = args.name
    config["description"] = args.description or ""
    config["created_at"] = datetime.now().isoformat()
    config["updated_at"] = datetime.now().isoformat()
    config["paper"] = args.paper or ""

    # Git info
    git_info = get_git_info()
    config["code"]["commit"] = git_info["commit"]
    config["code"]["branch"] = git_info["branch"]
    config["code"]["script_path"] = args.script or ""

    # Data source
    if args.data_s3:
        config["data"]["s3_uri"] = args.data_s3
        config["data"]["source_type"] = "s3"

    # Infrastructure
    if args.instance_type:
        config["infrastructure"]["instance_type"] = args.instance_type

    # Create experiment directory
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "results").mkdir(exist_ok=True)

    # Save config
    with open(exp_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Generate requirements.txt from current environment
    try:
        reqs = subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze"], text=True
        )
        with open(exp_dir / "requirements.txt", "w") as f:
            f.write(reqs)
    except subprocess.CalledProcessError:
        print("Warning: Could not generate requirements.txt")

    # Create run script
    run_script = f"""#!/bin/bash
# Experiment: {args.name}
# ID: {exp_id}
# Created: {datetime.now().isoformat()}

set -e

# Setup virtual environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Set environment
export PYTHONPATH=${{PYTHONPATH}}:{PROJECT_ROOT}
export AWS_DEFAULT_REGION=${{AWS_DEFAULT_REGION:-us-east-1}}

# Run experiment
python {args.script or 'scripts/eval/run_temperature_experiment.py'} "$@"

# Sync results to S3
aws s3 sync results/ s3://{S3_BUCKET}/experiments/{exp_id}/results/
"""
    with open(exp_dir / "run.sh", "w") as f:
        f.write(run_script)
    os.chmod(exp_dir / "run.sh", 0o755)

    # Update registry
    registry = load_registry()
    registry["experiments"][exp_id] = {
        "name": args.name,
        "status": "created",
        "created_at": config["created_at"],
        "path": str(exp_dir.relative_to(PROJECT_ROOT)),
        "script": args.script
    }
    save_registry(registry)

    print(f"Created experiment: {exp_id}")
    print(f"Directory: {exp_dir}")
    print(f"Config: {exp_dir / 'config.json'}")
    print(f"\nNext steps:")
    print(f"  1. Edit {exp_dir / 'config.json'} to configure parameters")
    print(f"  2. Run: python manage_experiment.py setup-ec2 --experiment {exp_id}")
    print(f"  3. Or run locally: cd {exp_dir} && ./run.sh")

    return exp_id


def record_run(args):
    """Record an experiment run."""
    registry = load_registry()

    if args.experiment not in registry["experiments"]:
        # Try to find by name
        found = None
        for exp_id, exp_info in registry["experiments"].items():
            if exp_info["name"] == args.experiment:
                found = exp_id
                break
        if not found:
            print(f"Error: Experiment '{args.experiment}' not found")
            sys.exit(1)
        args.experiment = found

    exp_dir = EXPERIMENTS_DIR / args.experiment
    config_file = exp_dir / "config.json"

    if not config_file.exists():
        print(f"Error: Config not found: {config_file}")
        sys.exit(1)

    with open(config_file) as f:
        config = json.load(f)

    # Create run record
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_record = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "instance_id": args.instance_id,
        "instance_type": args.instance_type or config["infrastructure"]["instance_type"],
        "region": args.region or config["infrastructure"]["region"],
        "data_source": args.data_source or config["data"]["s3_uri"],
        "status": args.status or "running",
        "duration_hours": args.duration,
        "cost_estimate": args.cost,
        "notes": args.notes or "",
        "results_s3": f"s3://{S3_BUCKET}/experiments/{args.experiment}/runs/{run_id}/"
    }

    # Add to config
    config["runs"].append(run_record)
    config["updated_at"] = datetime.now().isoformat()
    if args.status:
        config["status"] = args.status

    with open(config_file, "w") as f:
        json.dump(config, f, indent=2)

    # Update registry
    registry["experiments"][args.experiment]["status"] = args.status or "running"
    registry["experiments"][args.experiment]["last_run"] = run_id
    save_registry(registry)

    print(f"Recorded run {run_id} for experiment {args.experiment}")
    print(f"  Instance: {args.instance_id} ({run_record['instance_type']})")
    print(f"  Status: {run_record['status']}")
    print(f"  Results S3: {run_record['results_s3']}")


def sync_experiment(args):
    """Sync experiment data to/from S3."""
    registry = load_registry()

    if args.experiment not in registry["experiments"]:
        print(f"Error: Experiment '{args.experiment}' not found")
        sys.exit(1)

    exp_dir = EXPERIMENTS_DIR / args.experiment
    s3_path = f"s3://{S3_BUCKET}/experiments/{args.experiment}/"

    if args.direction == "upload":
        print(f"Uploading {exp_dir} to {s3_path}")
        subprocess.run([
            "aws", "s3", "sync",
            str(exp_dir), s3_path,
            "--exclude", ".venv/*",
            "--exclude", "__pycache__/*",
            "--exclude", "*.pyc"
        ], check=True)
        print("Upload complete")

    elif args.direction == "download":
        print(f"Downloading {s3_path} to {exp_dir}")
        subprocess.run([
            "aws", "s3", "sync",
            s3_path, str(exp_dir)
        ], check=True)
        print("Download complete")


def list_experiments(args):
    """List all experiments."""
    registry = load_registry()

    if not registry["experiments"]:
        print("No experiments found.")
        return

    print(f"{'ID':<40} {'Name':<30} {'Status':<12} {'Created':<20}")
    print("-" * 105)

    for exp_id, info in sorted(registry["experiments"].items(),
                                key=lambda x: x[1].get("created_at", ""), reverse=True):
        created = info.get("created_at", "")[:19]
        print(f"{exp_id:<40} {info['name']:<30} {info['status']:<12} {created:<20}")


def setup_ec2(args):
    """Setup EC2 instance for experiment."""
    registry = load_registry()

    if args.experiment not in registry["experiments"]:
        print(f"Error: Experiment '{args.experiment}' not found")
        sys.exit(1)

    exp_dir = EXPERIMENTS_DIR / args.experiment
    config_file = exp_dir / "config.json"

    with open(config_file) as f:
        config = json.load(f)

    infra = config["infrastructure"]

    print(f"Setting up EC2 for experiment: {args.experiment}")
    print(f"  Instance type: {infra['instance_type']}")
    print(f"  Region: {infra['region']}")
    print(f"  AMI: {infra['ami_id']}")

    # Create instance
    cmd = [
        "aws", "ec2", "run-instances",
        "--image-id", infra["ami_id"],
        "--instance-type", infra["instance_type"],
        "--iam-instance-profile", f"Name={infra['iam_role']}",
        "--security-groups", infra["security_group"],
        "--block-device-mappings",
        f'[{{"DeviceName":"/dev/xvda","Ebs":{{"VolumeSize":{infra["disk_size_gb"]},"VolumeType":"gp3"}}}}]',
        "--tag-specifications",
        f'ResourceType=instance,Tags=[{{Key=Name,Value={args.experiment}}},{{Key=Experiment,Value={config["experiment_id"]}}}]',
        "--region", infra["region"],
        "--query", "Instances[0].InstanceId",
        "--output", "text"
    ]

    if args.key_name:
        cmd.extend(["--key-name", args.key_name])

    if args.dry_run:
        print("Dry run - would execute:")
        print(" ".join(cmd))
        return

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        instance_id = result.stdout.strip()
        print(f"Created instance: {instance_id}")

        # Wait for instance
        print("Waiting for instance to be ready...")
        subprocess.run([
            "aws", "ec2", "wait", "instance-running",
            "--instance-ids", instance_id,
            "--region", infra["region"]
        ], check=True)

        print(f"\nInstance ready: {instance_id}")
        print(f"\nNext steps:")
        print(f"  1. Deploy code: aws s3 sync . s3://{S3_BUCKET}/code/{args.experiment}/")
        print(f"  2. Run experiment on instance")
        print(f"  3. Record run: python manage_experiment.py record --experiment {args.experiment} --instance-id {instance_id}")

        # Record the run
        record_args = argparse.Namespace(
            experiment=args.experiment,
            instance_id=instance_id,
            instance_type=infra["instance_type"],
            region=infra["region"],
            data_source=config["data"]["s3_uri"],
            status="running",
            duration=None,
            cost=None,
            notes=f"Auto-created by setup-ec2"
        )
        record_run(record_args)

    except subprocess.CalledProcessError as e:
        print(f"Error creating instance: {e}")
        sys.exit(1)


def show_experiment(args):
    """Show experiment details."""
    registry = load_registry()

    if args.experiment not in registry["experiments"]:
        print(f"Error: Experiment '{args.experiment}' not found")
        sys.exit(1)

    exp_dir = EXPERIMENTS_DIR / args.experiment
    config_file = exp_dir / "config.json"

    with open(config_file) as f:
        config = json.load(f)

    print(f"Experiment: {config['name']}")
    print(f"ID: {config['experiment_id']}")
    print(f"Status: {config['status']}")
    print(f"Description: {config['description']}")
    print(f"\nCode:")
    print(f"  Script: {config['code']['script_path']}")
    print(f"  Branch: {config['code']['branch']}")
    print(f"  Commit: {config['code']['commit'][:8]}")
    print(f"\nData:")
    print(f"  S3 URI: {config['data']['s3_uri']}")
    print(f"\nInfrastructure:")
    print(f"  Instance: {config['infrastructure']['instance_type']}")
    print(f"  Region: {config['infrastructure']['region']}")
    print(f"\nRuns ({len(config['runs'])}):")
    for run in config["runs"][-5:]:  # Show last 5 runs
        print(f"  - {run['run_id']}: {run['status']} on {run['instance_id']}")


def main():
    parser = argparse.ArgumentParser(description="Experiment Management CLI")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Create command
    create_parser = subparsers.add_parser("create", help="Create new experiment")
    create_parser.add_argument("--name", "-n", required=True, help="Experiment name")
    create_parser.add_argument("--description", "-d", help="Description")
    create_parser.add_argument("--script", "-s", help="Main script path")
    create_parser.add_argument("--paper", "-p", help="Associated paper (e.g., COLM2026)")
    create_parser.add_argument("--data-s3", help="S3 URI for input data")
    create_parser.add_argument("--instance-type", help="EC2 instance type")

    # Record command
    record_parser = subparsers.add_parser("record", help="Record experiment run")
    record_parser.add_argument("--experiment", "-e", required=True, help="Experiment ID or name")
    record_parser.add_argument("--instance-id", "-i", required=True, help="EC2 instance ID")
    record_parser.add_argument("--instance-type", help="Instance type")
    record_parser.add_argument("--region", help="AWS region")
    record_parser.add_argument("--data-source", help="Data source S3 URI")
    record_parser.add_argument("--status", choices=["running", "completed", "failed", "cancelled"])
    record_parser.add_argument("--duration", type=float, help="Duration in hours")
    record_parser.add_argument("--cost", type=float, help="Estimated cost in USD")
    record_parser.add_argument("--notes", help="Additional notes")

    # Sync command
    sync_parser = subparsers.add_parser("sync", help="Sync experiment to/from S3")
    sync_parser.add_argument("--experiment", "-e", required=True, help="Experiment ID")
    sync_parser.add_argument("--direction", "-d", choices=["upload", "download"],
                             default="upload", help="Sync direction")

    # List command
    list_parser = subparsers.add_parser("list", help="List all experiments")

    # Setup EC2 command
    ec2_parser = subparsers.add_parser("setup-ec2", help="Setup EC2 instance")
    ec2_parser.add_argument("--experiment", "-e", required=True, help="Experiment ID")
    ec2_parser.add_argument("--key-name", help="SSH key pair name")
    ec2_parser.add_argument("--dry-run", action="store_true", help="Dry run")

    # Show command
    show_parser = subparsers.add_parser("show", help="Show experiment details")
    show_parser.add_argument("--experiment", "-e", required=True, help="Experiment ID")

    args = parser.parse_args()

    if args.command == "create":
        create_experiment(args)
    elif args.command == "record":
        record_run(args)
    elif args.command == "sync":
        sync_experiment(args)
    elif args.command == "list":
        list_experiments(args)
    elif args.command == "setup-ec2":
        setup_ec2(args)
    elif args.command == "show":
        show_experiment(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
