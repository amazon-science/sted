#!/bin/bash
# EC2 Setup Script for Multi-LLM Benchmark
# Run this on a fresh EC2 instance (Amazon Linux 2 or Ubuntu)

set -e

echo "=========================================="
echo "Setting up Multi-LLM Benchmark Environment"
echo "=========================================="

# Update system
echo "Updating system packages..."
sudo yum update -y 2>/dev/null || sudo apt-get update -y

# Install Python 3.11+ and git
echo "Installing Python and dependencies..."
sudo yum install -y python3.11 python3.11-pip git 2>/dev/null || \
sudo apt-get install -y python3.11 python3-pip git

# Create working directory
mkdir -p ~/sted_benchmark
cd ~/sted_benchmark

# Clone the repository (or copy files)
echo "Setting up project..."
if [ -d "sted-internal" ]; then
    echo "Project already exists, pulling latest..."
    cd sted-internal && git pull && cd ..
else
    echo "Please copy the project files to ~/sted_benchmark/sted-internal"
fi

# Create virtual environment
echo "Creating virtual environment..."
python3.11 -m venv venv || python3 -m venv venv
source venv/bin/activate

# Install dependencies
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install boto3 openai numpy scipy tqdm python-dotenv sentence-transformers

# Install STED package
if [ -d "sted-internal" ]; then
    cd sted-internal
    pip install -e .
    cd ..
fi

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Copy your .env file with AWS credentials to ~/sted_benchmark/sted-internal/"
echo "2. Run: source ~/sted_benchmark/venv/bin/activate"
echo "3. Run: cd ~/sted_benchmark/sted-internal"
echo "4. Run: ./research/experiments/multi_llm_benchmark/run_background.sh"
