# STED Experiment EC2 Setup Guide

This guide explains how to set up and run STED temperature experiments on AWS EC2 instances.

## Prerequisites

### 1. AWS CLI Configuration
```bash
# Install AWS CLI
pip install awscli

# Configure credentials
aws configure
# Enter: AWS Access Key ID, Secret Access Key, Region (us-east-1), Output format (json)
```

### 2. Create IAM Role for EC2
Create an IAM role with the following permissions:
- `AmazonBedrockFullAccess` - For Bedrock model access
- `AmazonSSMManagedInstanceCore` - For SSM command execution
- `AmazonS3ReadOnlyAccess` - For downloading experiment data

```bash
# Create role via AWS Console or CLI
aws iam create-role --role-name sted-experiment-role \
    --assume-role-policy-document '{
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Principal": {"Service": "ec2.amazonaws.com"},
            "Action": "sts:AssumeRole"
        }]
    }'

# Attach policies
aws iam attach-role-policy --role-name sted-experiment-role \
    --policy-arn arn:aws:iam::aws:policy/AmazonBedrockFullAccess
aws iam attach-role-policy --role-name sted-experiment-role \
    --policy-arn arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore
aws iam attach-role-policy --role-name sted-experiment-role \
    --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess

# Create instance profile
aws iam create-instance-profile --instance-profile-name sted-experiment-role
aws iam add-role-to-instance-profile --instance-profile-name sted-experiment-role --role-name sted-experiment-role
```

### 3. Create Security Group
```bash
aws ec2 create-security-group \
    --group-name sted-experiment-sg \
    --description "Security group for STED experiments"

# Allow SSH (optional, for debugging)
aws ec2 authorize-security-group-ingress \
    --group-name sted-experiment-sg \
    --protocol tcp --port 22 --cidr 0.0.0.0/0
```

### 4. Create SSH Key Pair (optional)
```bash
aws ec2 create-key-pair --key-name sted-experiment-key \
    --query 'KeyMaterial' --output text > ~/.ssh/sted-experiment-key.pem
chmod 400 ~/.ssh/sted-experiment-key.pem
```

### 5. Upload Experiment Data to S3
```bash
# Create S3 bucket
aws s3 mb s3://sted-experiment-data

# Upload ShareGPT data
aws s3 cp sharegpt_data/ s3://sted-experiment-data/sharegpt_data/ --recursive

# Upload Toucan data (for tool-calling experiments)
aws s3 cp toucan_data/ s3://sted-experiment-data/toucan_data/ --recursive
```

---

## Quick Start

### Option 1: Using the Setup Script (Recommended)

```bash
# Full setup: create EC2, install deps, deploy code, run experiment
./scripts/setup-ec2-experiment.sh --full \
    --model us.anthropic.claude-sonnet-4-20250514-v1:0 \
    --name sted-exp-sonnet4

# Check status
./scripts/setup-ec2-experiment.sh --status --instance-id i-xxxxx
```

### Option 2: Manual Step-by-Step

#### Step 1: Create EC2 Instance

```bash
# Launch instance with 200GB disk
aws ec2 run-instances \
    --image-id ami-0c02fb55956c7d316 \
    --instance-type t3.large \
    --key-name sted-experiment-key \
    --security-groups sted-experiment-sg \
    --iam-instance-profile Name=sted-experiment-role \
    --block-device-mappings '[{"DeviceName":"/dev/xvda","Ebs":{"VolumeSize":200,"VolumeType":"gp3"}}]' \
    --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=sted-experiment}]' \
    --query 'Instances[0].InstanceId' \
    --output text
```

#### Step 2: Setup Instance via SSM

```bash
INSTANCE_ID=i-xxxxx  # Replace with your instance ID

# Install dependencies
aws ssm send-command \
    --instance-ids "$INSTANCE_ID" \
    --document-name "AWS-RunShellScript" \
    --parameters 'commands=[
        "sudo yum update -y",
        "sudo yum install -y docker git python3-pip",
        "sudo systemctl start docker",
        "sudo systemctl enable docker",
        "sudo usermod -aG docker ec2-user",
        "sudo ln -sf /usr/bin/python3 /usr/bin/python",
        "pip3 install langchain-text-splitters boto3 openai tqdm --user"
    ]'

# Create directories
aws ssm send-command \
    --instance-ids "$INSTANCE_ID" \
    --document-name "AWS-RunShellScript" \
    --parameters 'commands=[
        "mkdir -p /home/ec2-user/sharegpt_data/combined",
        "mkdir -p /home/ec2-user/toucan_data",
        "mkdir -p /home/ec2-user/sharegpt_results",
        "mkdir -p /home/ec2-user/sted"
    ]'

# Download data from S3
aws ssm send-command \
    --instance-ids "$INSTANCE_ID" \
    --document-name "AWS-RunShellScript" \
    --parameters 'commands=[
        "aws s3 cp s3://sted-experiment-data/sharegpt_data/ /home/ec2-user/sharegpt_data/ --recursive",
        "aws s3 cp s3://sted-experiment-data/toucan_data/ /home/ec2-user/toucan_data/ --recursive"
    ]'
```

#### Step 3: Deploy Code

```bash
# Create tarball locally
tar -czf /tmp/sted-code.tar.gz \
    --exclude='*.pyc' \
    --exclude='__pycache__' \
    --exclude='.git' \
    --exclude='results' \
    -C /path/to/sted-internal .

# Upload to S3
aws s3 cp /tmp/sted-code.tar.gz s3://sted-experiment-data/code/sted-code.tar.gz

# Download and extract on instance
aws ssm send-command \
    --instance-ids "$INSTANCE_ID" \
    --document-name "AWS-RunShellScript" \
    --parameters 'commands=[
        "aws s3 cp s3://sted-experiment-data/code/sted-code.tar.gz /home/ec2-user/",
        "cd /home/ec2-user/sted && tar -xzf /home/ec2-user/sted-code.tar.gz"
    ]'
```

#### Step 4: Run Experiment

```bash
# For ShareGPT (structured output) experiments
aws ssm send-command \
    --instance-ids "$INSTANCE_ID" \
    --document-name "AWS-RunShellScript" \
    --parameters 'commands=[
        "export PYTHONPATH=/home/ec2-user/sted && cd /home/ec2-user/sted && nohup python3 scripts/eval/run_temperature_experiment.py --mode structured --data-dir /home/ec2-user/sharegpt_data --output-dir /home/ec2-user/sharegpt_results --model-id us.anthropic.claude-sonnet-4-20250514-v1:0 --sample-limit 100 --temperatures 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 --run-num 10 --max-tokens 3000 --include-schema > /home/ec2-user/experiment.log 2>&1 &"
    ]'

# For Toucan (tool-calling) experiments
aws ssm send-command \
    --instance-ids "$INSTANCE_ID" \
    --document-name "AWS-RunShellScript" \
    --parameters 'commands=[
        "export PYTHONPATH=/home/ec2-user/sted && cd /home/ec2-user/sted && nohup python3 scripts/eval/run_temperature_experiment.py --mode tool-calling --dataset-path /home/ec2-user/toucan_data/toucan_tool_calls_1006.json --output-dir /home/ec2-user/results --model-id us.anthropic.claude-sonnet-4-20250514-v1:0 --sample-limit 1006 --temperatures 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 --run-num 10 --max-tokens 2048 --max-workers 10 > /home/ec2-user/experiment.log 2>&1 &"
    ]'
```

---

## Monitoring Experiments

### Check Experiment Progress
```bash
aws ssm send-command \
    --instance-ids "$INSTANCE_ID" \
    --document-name "AWS-RunShellScript" \
    --parameters 'commands=["tail -50 /home/ec2-user/experiment.log"]'
```

### Check Running Processes
```bash
aws ssm send-command \
    --instance-ids "$INSTANCE_ID" \
    --document-name "AWS-RunShellScript" \
    --parameters 'commands=["ps aux | grep python | grep -v grep"]'
```

### Check Disk Usage
```bash
aws ssm send-command \
    --instance-ids "$INSTANCE_ID" \
    --document-name "AWS-RunShellScript" \
    --parameters 'commands=["df -h /"]'
```

---

## Available Models

### Bedrock Models (AWS Credentials)
| Model ID | Display Name | Workers |
|----------|--------------|---------|
| `us.anthropic.claude-opus-4-5-20251101-v1:0` | Claude Opus 4.5 | 12 |
| `us.anthropic.claude-sonnet-4-5-20250929-v1:0` | Claude Sonnet 4.5 | 20 |
| `us.anthropic.claude-sonnet-4-20250514-v1:0` | Claude Sonnet 4 | 20 |
| `us.anthropic.claude-opus-4-20250514-v1:0` | Claude Opus 4 | 20 |
| `us.anthropic.claude-3-7-sonnet-20250219-v1:0` | Claude 3.7 Sonnet | 25 |
| `us.anthropic.claude-3-5-sonnet-20241022-v2:0` | Claude 3.5 Sonnet | 10 |
| `us.anthropic.claude-3-5-haiku-20241022-v1:0` | Claude 3.5 Haiku | 40 |
| `us.anthropic.claude-haiku-4-5-20251001-v1:0` | Claude Haiku 4.5 | 12 |
| `us.amazon.nova-pro-v1:0` | Nova Pro | 50 |
| `us.amazon.nova-2-lite-v1:0` | Nova 2 Lite | 50 |
| `us.meta.llama3-3-70b-instruct-v1:0` | Llama 3.3 70B | 50 |

### OpenRouter Models (Requires OPENAI_API_KEY)
| Model ID | Display Name |
|----------|--------------|
| `x-ai/grok-4.1-fast` | Grok 4.1 Fast |
| `google/gemini-2.5-flash-lite` | Gemini 2.5 Flash Lite |
| `openai/gpt-4.1-mini` | GPT 4.1 Mini |

For OpenRouter models, set the API key before running:
```bash
export OPENAI_API_KEY="your-openrouter-api-key"
```

---

## Troubleshooting

### Common Issues

1. **"No module named 'langchain_text_splitters'"**
   ```bash
   aws ssm send-command --instance-ids "$INSTANCE_ID" \
       --parameters 'commands=["pip3 install langchain-text-splitters --user"]'
   ```

2. **AWS Credential Errors**
   - Ensure the IAM role is attached to the instance
   - Check the role has `AmazonBedrockFullAccess` policy

3. **"python: command not found"**
   ```bash
   aws ssm send-command --instance-ids "$INSTANCE_ID" \
       --parameters 'commands=["sudo ln -sf /usr/bin/python3 /usr/bin/python"]'
   ```

4. **Disk Full**
   - Check disk usage: `df -h /`
   - Clean old results: `rm -rf /home/ec2-user/results/old_*`

5. **Unicode Decode Errors**
   - Update to latest `run_temperature_experiment.py` which handles encoding errors

---

## Collecting Results

### Download Results from Instance
```bash
# Compress results
aws ssm send-command --instance-ids "$INSTANCE_ID" \
    --parameters 'commands=["cd /home/ec2-user && tar -czf results.tar.gz sharegpt_results/"]'

# Upload to S3
aws ssm send-command --instance-ids "$INSTANCE_ID" \
    --parameters 'commands=["aws s3 cp /home/ec2-user/results.tar.gz s3://sted-experiment-data/results/results-$(date +%Y%m%d).tar.gz"]'

# Download locally
aws s3 cp s3://sted-experiment-data/results/results-YYYYMMDD.tar.gz ./
```

---

## Cost Management

### Stop Instance When Not In Use
```bash
aws ec2 stop-instances --instance-ids "$INSTANCE_ID"
```

### Start Instance
```bash
aws ec2 start-instances --instance-ids "$INSTANCE_ID"
aws ec2 wait instance-running --instance-ids "$INSTANCE_ID"
```

### Terminate Instance (Deletes Everything)
```bash
aws ec2 terminate-instances --instance-ids "$INSTANCE_ID"
```

### Estimated Costs
- `t3.large`: ~$0.08/hour
- `t3.xlarge`: ~$0.16/hour
- 200GB gp3 storage: ~$16/month
- Bedrock API calls: varies by model
