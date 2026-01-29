# STED Temperature Experiment Docker Image
# Each EC2 instance runs ONE model - specify via MODEL_ID environment variable
#
# Usage:
#   # Build image
#   docker build -t sted-experiment .
#
#   # Tool-calling mode (Toucan dataset - included in image)
#   docker run -e MODEL_ID=us.amazon.nova-2-lite-v1:0 sted-experiment
#
#   # Structured output mode (ShareGPT dataset - mount at runtime)
#   docker run -e MODEL_ID=us.amazon.nova-2-lite-v1:0 -e MODE=structured \
#       -v /path/to/sharegpt_data:/app/sharegpt_data sted-experiment
#
# Available models are defined in sted/model_config.py (MODEL_REGISTRY)

FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app
ENV AWS_DEFAULT_REGION=us-east-1
ENV OPENAI_BASE_URL=https://openrouter.ai/api/v1

# Experiment configuration (override at runtime)
ENV MODEL_ID=""
ENV MODE="tool-calling"
ENV TEMPERATURES="0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0"
ENV SAMPLES=""
ENV RUNS=10
ENV MAX_WORKERS=""
ENV MAX_TOKENS=""
ENV INCLUDE_SCHEMA="true"

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy project files
COPY pyproject.toml README.md ./
COPY sted/ ./sted/
COPY scripts/ ./scripts/
COPY toucan_data/ ./toucan_data/

# Create sharegpt_data directory (data can be mounted at runtime)
RUN mkdir -p ./sharegpt_data

# Copy ShareGPT data if available (will be empty if not present locally)
# Use .dockerignore to exclude if needed, or mount at runtime
COPY sharegpt_dat[a]/ ./sharegpt_data/

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e .

# Create directories for results and logs
RUN mkdir -p /app/results /app/logs /app/sharegpt_data

# Copy entrypoint script
COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

# Entrypoint
ENTRYPOINT ["/docker-entrypoint.sh"]
