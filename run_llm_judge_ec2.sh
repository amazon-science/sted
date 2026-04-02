#!/bin/bash
# Script to run LLM-as-judge experiments on EC2
# Run this script locally to execute experiments on the remote EC2

EC2_IP="${EC2_IP:?Set EC2_IP environment variable}"
INSTANCE_ID="${EC2_INSTANCE_ID:?Set EC2_INSTANCE_ID environment variable}"
REGION="us-east-1"

echo "=== Step 1: Sync files to EC2 ==="
rsync -avz --progress \
  ./scripts/data/human_validation/best_match_selection/ \
  ubuntu@${EC2_IP}:/home/ubuntu/sted-internal/scripts/data/human_validation/best_match_selection/

echo ""
echo "=== Step 2: Run T=0.7 experiment (343 samples × 5 runs) ==="
ssh ubuntu@${EC2_IP} "cd /home/ubuntu/sted-internal && \
  nohup bash -c 'PYTHONPATH=\$PWD python scripts/data/human_validation/best_match_selection/run_llm_judge_baseline.py \
    --dataset scripts/data/human_validation/best_match_selection/data/toucan_validation_dataset.json \
    --output scripts/data/human_validation/best_match_selection/data/llm_judge_consistency_t07_full.json \
    --n-runs 5 --temperature 0.7 --workers 10 --no-resume \
    2>&1 | tee llm_judge_t07.log' > /dev/null 2>&1 &"

echo "T=0.7 experiment started in background on EC2"
echo ""
echo "=== Step 3: Run T=0.0 experiment (343 samples × 5 runs) ==="
ssh ubuntu@${EC2_IP} "cd /home/ubuntu/sted-internal && \
  nohup bash -c 'PYTHONPATH=\$PWD python scripts/data/human_validation/best_match_selection/run_llm_judge_baseline.py \
    --dataset scripts/data/human_validation/best_match_selection/data/toucan_validation_dataset.json \
    --output scripts/data/human_validation/best_match_selection/data/llm_judge_consistency_t00_full.json \
    --n-runs 5 --temperature 0.0 --workers 1 --no-parallel-runs --no-resume \
    2>&1 | tee llm_judge_t00.log' > /dev/null 2>&1 &"

echo "T=0.0 experiment started in background on EC2"
echo ""
echo "=== Monitor progress ==="
echo "SSH into EC2 and check logs:"
echo "  ssh ubuntu@${EC2_IP}"
echo "  tail -f /home/ubuntu/sted-internal/llm_judge_t07.log"
echo "  tail -f /home/ubuntu/sted-internal/llm_judge_t00.log"
echo ""
echo "=== Download results when complete ==="
echo "  scp ubuntu@${EC2_IP}:/home/ubuntu/sted-internal/scripts/data/human_validation/best_match_selection/data/llm_judge_consistency_t07_full.json ."
echo "  scp ubuntu@${EC2_IP}:/home/ubuntu/sted-internal/scripts/data/human_validation/best_match_selection/data/llm_judge_consistency_t00_full.json ."
