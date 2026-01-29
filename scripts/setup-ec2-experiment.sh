#!/bin/bash
#
# STED Experiment EC2 Setup Script
# =================================
# This script automates the setup and deployment of STED temperature experiments on EC2.
#
# Prerequisites:
#   - AWS CLI configured with appropriate credentials
#   - Docker installed locally (for building/pushing images)
#   - SSH key pair created in AWS
#
# Usage:
#   # Create new EC2 instance and run experiment
#   ./scripts/setup-ec2-experiment.sh --create --model us.anthropic.claude-sonnet-4-20250514-v1:0
#
#   # Deploy code updates to existing instance
#   ./scripts/setup-ec2-experiment.sh --deploy --instance-id i-xxxxx
#
#   # Run experiment on existing instance
#   ./scripts/setup-ec2-experiment.sh --run --instance-id i-xxxxx --model us.anthropic.claude-sonnet-4-20250514-v1:0
#
#   # Full setup: create, deploy, and run
#   ./scripts/setup-ec2-experiment.sh --full --model us.anthropic.claude-sonnet-4-20250514-v1:0

set -e

# =============================================================================
# Configuration
# =============================================================================

# EC2 Configuration
AMI_ID="${AMI_ID:-ami-0c02fb55956c7d316}"  # Amazon Linux 2023 (us-east-1)
INSTANCE_TYPE="${INSTANCE_TYPE:-t3.large}"
KEY_NAME="${KEY_NAME:-sted-experiment-key}"
SECURITY_GROUP="${SECURITY_GROUP:-sted-experiment-sg}"
IAM_ROLE="${IAM_ROLE:-sted-experiment-role}"
DISK_SIZE="${DISK_SIZE:-200}"
REGION="${AWS_DEFAULT_REGION:-us-east-1}"

# Container Configuration
ECR_REGISTRY="${ECR_REGISTRY:-}"  # Set to your ECR registry URL if using ECR
DOCKER_IMAGE="${DOCKER_IMAGE:-sted-experiment:latest}"
S3_BUCKET="${S3_BUCKET:-sted-experiment-data}"

# Experiment Configuration
TEMPERATURES="${TEMPERATURES:-0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0}"
SAMPLES="${SAMPLES:-100}"
RUNS="${RUNS:-10}"
MAX_TOKENS="${MAX_TOKENS:-3000}"
MODE="${MODE:-structured}"  # structured or tool-calling
DATA_DIR="${DATA_DIR:-/home/ec2-user/sharegpt_data}"

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# =============================================================================
# Helper Functions
# =============================================================================

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

error() {
    echo "[ERROR] $1" >&2
    exit 1
}

wait_for_instance() {
    local instance_id=$1
    log "Waiting for instance $instance_id to be running..."
    aws ec2 wait instance-running --instance-ids "$instance_id" --region "$REGION"

    log "Waiting for instance status checks..."
    aws ec2 wait instance-status-ok --instance-ids "$instance_id" --region "$REGION"

    log "Instance $instance_id is ready"
}

get_instance_ip() {
    local instance_id=$1
    aws ec2 describe-instances \
        --instance-ids "$instance_id" \
        --region "$REGION" \
        --query 'Reservations[0].Instances[0].PublicIpAddress' \
        --output text
}

run_ssm_command() {
    local instance_id=$1
    local command=$2
    local timeout=${3:-300}

    local cmd_id=$(aws ssm send-command \
        --instance-ids "$instance_id" \
        --document-name "AWS-RunShellScript" \
        --parameters "commands=[\"$command\"]" \
        --region "$REGION" \
        --output text \
        --query 'Command.CommandId')

    sleep 5

    # Wait for command to complete
    local status="InProgress"
    local elapsed=0
    while [[ "$status" == "InProgress" || "$status" == "Pending" ]] && [[ $elapsed -lt $timeout ]]; do
        sleep 5
        elapsed=$((elapsed + 5))
        status=$(aws ssm get-command-invocation \
            --command-id "$cmd_id" \
            --instance-id "$instance_id" \
            --region "$REGION" \
            --query 'Status' \
            --output text 2>/dev/null || echo "InProgress")
    done

    # Get output
    aws ssm get-command-invocation \
        --command-id "$cmd_id" \
        --instance-id "$instance_id" \
        --region "$REGION" \
        --query 'StandardOutputContent' \
        --output text 2>/dev/null
}

# =============================================================================
# Step 1: Create EC2 Instance
# =============================================================================

create_instance() {
    local name_tag="${1:-sted-experiment}"

    log "Creating EC2 instance: $name_tag"
    log "  Instance Type: $INSTANCE_TYPE"
    log "  Disk Size: ${DISK_SIZE}GB"
    log "  Region: $REGION"

    # Create instance with 200GB disk
    local instance_id=$(aws ec2 run-instances \
        --image-id "$AMI_ID" \
        --instance-type "$INSTANCE_TYPE" \
        --key-name "$KEY_NAME" \
        --security-groups "$SECURITY_GROUP" \
        --iam-instance-profile "Name=$IAM_ROLE" \
        --block-device-mappings "[{\"DeviceName\":\"/dev/xvda\",\"Ebs\":{\"VolumeSize\":$DISK_SIZE,\"VolumeType\":\"gp3\",\"DeleteOnTermination\":true}}]" \
        --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$name_tag}]" \
        --region "$REGION" \
        --query 'Instances[0].InstanceId' \
        --output text)

    log "Created instance: $instance_id"

    wait_for_instance "$instance_id"

    local ip=$(get_instance_ip "$instance_id")
    log "Instance IP: $ip"

    echo "$instance_id"
}

# =============================================================================
# Step 2: Setup Instance (Install Docker, Download Data)
# =============================================================================

setup_instance() {
    local instance_id=$1

    log "Setting up instance $instance_id..."

    # Install Docker and dependencies
    log "Installing Docker and dependencies..."
    run_ssm_command "$instance_id" "sudo yum update -y && sudo yum install -y docker git python3-pip && sudo systemctl start docker && sudo systemctl enable docker && sudo usermod -aG docker ec2-user"

    # Create symlink for python
    log "Creating python symlink..."
    run_ssm_command "$instance_id" "sudo ln -sf /usr/bin/python3 /usr/bin/python 2>/dev/null || true"

    # Install Python dependencies
    log "Installing Python dependencies..."
    run_ssm_command "$instance_id" "pip3 install langchain-text-splitters boto3 openai tqdm --user"

    # Create directories
    log "Creating directories..."
    run_ssm_command "$instance_id" "mkdir -p /home/ec2-user/sharegpt_data/combined /home/ec2-user/toucan_data /home/ec2-user/sharegpt_results /home/ec2-user/sted"

    # Download data from S3
    log "Downloading experiment data from S3..."
    run_ssm_command "$instance_id" "aws s3 cp s3://$S3_BUCKET/sharegpt_data/ /home/ec2-user/sharegpt_data/ --recursive 2>/dev/null || echo 'ShareGPT data not found in S3'"
    run_ssm_command "$instance_id" "aws s3 cp s3://$S3_BUCKET/toucan_data/ /home/ec2-user/toucan_data/ --recursive 2>/dev/null || echo 'Toucan data not found in S3'"

    log "Instance setup complete"
}

# =============================================================================
# Step 3: Deploy Code to Instance
# =============================================================================

deploy_code() {
    local instance_id=$1

    log "Deploying code to instance $instance_id..."

    # Create tarball of the project (excluding large files)
    local tarball="/tmp/sted-code-$(date +%Y%m%d%H%M%S).tar.gz"
    log "Creating code tarball..."
    tar -czf "$tarball" \
        -C "$PROJECT_ROOT" \
        --exclude='*.pyc' \
        --exclude='__pycache__' \
        --exclude='.git' \
        --exclude='results' \
        --exclude='*.tar.gz' \
        --exclude='node_modules' \
        --exclude='.venv' \
        --exclude='venv' \
        .

    # Upload to S3
    log "Uploading code to S3..."
    aws s3 cp "$tarball" "s3://$S3_BUCKET/code/sted-code.tar.gz" --region "$REGION"

    # Download and extract on instance
    log "Downloading and extracting code on instance..."
    run_ssm_command "$instance_id" "aws s3 cp s3://$S3_BUCKET/code/sted-code.tar.gz /home/ec2-user/sted-code.tar.gz && cd /home/ec2-user/sted && tar -xzf /home/ec2-user/sted-code.tar.gz"

    # Cleanup
    rm -f "$tarball"

    log "Code deployment complete"
}

# =============================================================================
# Step 4: Run Experiment
# =============================================================================

run_experiment() {
    local instance_id=$1
    local model_id=$2

    if [[ -z "$model_id" ]]; then
        error "Model ID is required. Use --model <model-id>"
    fi

    log "Starting experiment on instance $instance_id"
    log "  Model: $model_id"
    log "  Mode: $MODE"
    log "  Temperatures: $TEMPERATURES"
    log "  Samples: $SAMPLES"
    log "  Runs: $RUNS"

    # Build the experiment command based on mode
    local experiment_cmd
    if [[ "$MODE" == "tool-calling" ]]; then
        experiment_cmd="export PYTHONPATH=/home/ec2-user/sted && cd /home/ec2-user/sted && nohup python3 scripts/eval/run_temperature_experiment.py --mode tool-calling --dataset-path /home/ec2-user/toucan_data/toucan_tool_calls_1006.json --output-dir /home/ec2-user/results --model-id $model_id --sample-limit $SAMPLES --temperatures $TEMPERATURES --run-num $RUNS --max-tokens $MAX_TOKENS --max-workers 10 > /home/ec2-user/experiment.log 2>&1 &"
    else
        experiment_cmd="export PYTHONPATH=/home/ec2-user/sted && cd /home/ec2-user/sted && nohup python3 scripts/eval/run_temperature_experiment.py --mode structured --data-dir $DATA_DIR --output-dir /home/ec2-user/sharegpt_results --model-id $model_id --sample-limit $SAMPLES --temperatures $TEMPERATURES --run-num $RUNS --max-tokens $MAX_TOKENS --include-schema > /home/ec2-user/experiment.log 2>&1 &"
    fi

    # For OpenRouter models, set OPENAI_API_KEY
    if [[ "$model_id" == *"/"* ]]; then
        if [[ -z "$OPENAI_API_KEY" ]]; then
            error "OPENAI_API_KEY environment variable required for OpenRouter models"
        fi
        experiment_cmd="export OPENAI_API_KEY='$OPENAI_API_KEY' && $experiment_cmd"
    fi

    # Kill any existing experiment
    run_ssm_command "$instance_id" "pkill -f python || true"

    # Start experiment
    log "Starting experiment..."
    aws ssm send-command \
        --instance-ids "$instance_id" \
        --document-name "AWS-RunShellScript" \
        --parameters "commands=[\"$experiment_cmd\"]" \
        --region "$REGION" \
        --output text \
        --query 'Command.CommandId'

    log "Experiment started. Monitor with:"
    log "  aws ssm send-command --instance-ids $instance_id --document-name AWS-RunShellScript --parameters 'commands=[\"tail -50 /home/ec2-user/experiment.log\"]'"

    log "Or check status with:"
    log "  ./scripts/setup-ec2-experiment.sh --status --instance-id $instance_id"
}

# =============================================================================
# Status Check
# =============================================================================

check_status() {
    local instance_id=$1

    log "Checking experiment status on $instance_id..."

    echo ""
    echo "=== Instance Info ==="
    aws ec2 describe-instances \
        --instance-ids "$instance_id" \
        --region "$REGION" \
        --query 'Reservations[0].Instances[0].[InstanceId,State.Name,InstanceType,PublicIpAddress]' \
        --output table

    echo ""
    echo "=== Running Processes ==="
    run_ssm_command "$instance_id" "ps aux | grep python | grep -v grep | head -5 || echo 'No Python processes running'"

    echo ""
    echo "=== Latest Log Output ==="
    run_ssm_command "$instance_id" "tail -30 /home/ec2-user/experiment.log 2>/dev/null || tail -30 /home/ec2-user/sharegpt_experiment.log 2>/dev/null || echo 'No log file found'"

    echo ""
    echo "=== Disk Usage ==="
    run_ssm_command "$instance_id" "df -h / | tail -1"
}

# =============================================================================
# Main
# =============================================================================

show_help() {
    cat << EOF
STED Experiment EC2 Setup Script

Usage: $0 [OPTIONS]

Actions:
    --create              Create a new EC2 instance
    --setup               Setup instance (install deps, download data)
    --deploy              Deploy local code to instance
    --run                 Run experiment on instance
    --full                Full setup: create, setup, deploy, and run
    --status              Check experiment status

Options:
    --instance-id ID      EC2 instance ID (required for deploy/run/status)
    --model MODEL_ID      Model ID for experiment (required for run)
    --name NAME           Instance name tag (default: sted-experiment)
    --mode MODE           Experiment mode: structured or tool-calling (default: structured)
    --samples N           Number of samples (default: 100)
    --runs N              Runs per temperature (default: 10)
    --help                Show this help message

Environment Variables:
    AWS_DEFAULT_REGION    AWS region (default: us-east-1)
    OPENAI_API_KEY        Required for OpenRouter models (grok, gemini, etc.)
    S3_BUCKET             S3 bucket for data/code (default: sted-experiment-data)

Examples:
    # Create new instance and run Claude Sonnet 4 experiment
    $0 --full --model us.anthropic.claude-sonnet-4-20250514-v1:0 --name sted-exp-sonnet4

    # Deploy code updates to existing instance
    $0 --deploy --instance-id i-0123456789abcdef0

    # Run experiment on existing instance
    $0 --run --instance-id i-0123456789abcdef0 --model us.meta.llama3-3-70b-instruct-v1:0

    # Check status
    $0 --status --instance-id i-0123456789abcdef0

Available Bedrock Models:
    us.anthropic.claude-opus-4-5-20251101-v1:0     (Claude Opus 4.5)
    us.anthropic.claude-sonnet-4-5-20250929-v1:0   (Claude Sonnet 4.5)
    us.anthropic.claude-sonnet-4-20250514-v1:0     (Claude Sonnet 4)
    us.anthropic.claude-opus-4-20250514-v1:0       (Claude Opus 4)
    us.anthropic.claude-3-7-sonnet-20250219-v1:0   (Claude 3.7 Sonnet)
    us.anthropic.claude-3-5-sonnet-20241022-v2:0   (Claude 3.5 Sonnet)
    us.anthropic.claude-3-5-haiku-20241022-v1:0    (Claude 3.5 Haiku)
    us.anthropic.claude-haiku-4-5-20251001-v1:0    (Claude Haiku 4.5)
    us.amazon.nova-pro-v1:0                        (Nova Pro)
    us.amazon.nova-2-lite-v1:0                     (Nova 2 Lite)
    us.meta.llama3-3-70b-instruct-v1:0             (Llama 3.3 70B)

Available OpenRouter Models (requires OPENAI_API_KEY):
    x-ai/grok-4.1-fast                             (Grok 4.1 Fast)
    google/gemini-2.5-flash-lite                   (Gemini 2.5 Flash Lite)
    openai/gpt-4.1-mini                            (GPT 4.1 Mini)
EOF
}

# Parse arguments
ACTION=""
INSTANCE_ID=""
MODEL_ID=""
INSTANCE_NAME="sted-experiment"

while [[ $# -gt 0 ]]; do
    case $1 in
        --create)
            ACTION="create"
            shift
            ;;
        --setup)
            ACTION="setup"
            shift
            ;;
        --deploy)
            ACTION="deploy"
            shift
            ;;
        --run)
            ACTION="run"
            shift
            ;;
        --full)
            ACTION="full"
            shift
            ;;
        --status)
            ACTION="status"
            shift
            ;;
        --instance-id)
            INSTANCE_ID="$2"
            shift 2
            ;;
        --model)
            MODEL_ID="$2"
            shift 2
            ;;
        --name)
            INSTANCE_NAME="$2"
            shift 2
            ;;
        --mode)
            MODE="$2"
            shift 2
            ;;
        --samples)
            SAMPLES="$2"
            shift 2
            ;;
        --runs)
            RUNS="$2"
            shift 2
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            error "Unknown option: $1"
            ;;
    esac
done

# Execute action
case $ACTION in
    create)
        create_instance "$INSTANCE_NAME"
        ;;
    setup)
        [[ -z "$INSTANCE_ID" ]] && error "--instance-id required for setup"
        setup_instance "$INSTANCE_ID"
        ;;
    deploy)
        [[ -z "$INSTANCE_ID" ]] && error "--instance-id required for deploy"
        deploy_code "$INSTANCE_ID"
        ;;
    run)
        [[ -z "$INSTANCE_ID" ]] && error "--instance-id required for run"
        [[ -z "$MODEL_ID" ]] && error "--model required for run"
        run_experiment "$INSTANCE_ID" "$MODEL_ID"
        ;;
    full)
        [[ -z "$MODEL_ID" ]] && error "--model required for full setup"
        INSTANCE_ID=$(create_instance "$INSTANCE_NAME")
        setup_instance "$INSTANCE_ID"
        deploy_code "$INSTANCE_ID"
        run_experiment "$INSTANCE_ID" "$MODEL_ID"
        ;;
    status)
        [[ -z "$INSTANCE_ID" ]] && error "--instance-id required for status"
        check_status "$INSTANCE_ID"
        ;;
    *)
        show_help
        exit 1
        ;;
esac
