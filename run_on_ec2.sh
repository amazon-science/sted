#!/bin/bash
# Run LLM-as-judge experiments on EC2

set -e

INSTANCE_ID="${EC2_INSTANCE_ID:?Set EC2_INSTANCE_ID environment variable}"
REGION="us-east-1"

echo "=== Getting EC2 IP ==="
EC2_IP=$(aws ec2 describe-instances --instance-ids $INSTANCE_ID --region $REGION --query "Reservations[0].Instances[0].PublicIpAddress" --output text)
echo "EC2 IP: $EC2_IP"

echo ""
echo "=== Git pull on EC2 ==="
ssh ubuntu@$EC2_IP "cd /home/ubuntu/sted-internal && git pull"

echo ""
echo "=== Starting T=0.7 experiment (343 samples × 5 runs) ==="
ssh ubuntu@$EC2_IP "cd /home/ubuntu/sted-internal && nohup bash -c 'PYTHONPATH=\$PWD python scripts/data/human_validation/best_match_selection/run_llm_judge_baseline.py --dataset scripts/data/human_validation/best_match_selection/data/toucan_validation_dataset.json --output scripts/data/human_validation/best_match_selection/data/llm_judge_consistency_t07_full.json --n-runs 5 --temperature 0.7 --workers 10 --no-resume 2>&1 | tee llm_judge_t07.log' > /dev/null 2>&1 &"
echo "T=0.7 started"

echo ""
echo "=== Starting T=0.0 experiment (343 samples × 5 runs) ==="
ssh ubuntu@$EC2_IP "cd /home/ubuntu/sted-internal && nohup bash -c 'PYTHONPATH=\$PWD python scripts/data/human_validation/best_match_selection/run_llm_judge_baseline.py --dataset scripts/data/human_validation/best_match_selection/data/toucan_validation_dataset.json --output scripts/data/human_validation/best_match_selection/data/llm_judge_consistency_t00_full.json --n-runs 5 --temperature 0.0 --workers 1 --no-parallel-runs --no-resume 2>&1 | tee llm_judge_t00.log' > /dev/null 2>&1 &"
echo "T=0.0 started"

echo ""
echo "=== Verifying experiments running ==="
sleep 2
ssh ubuntu@$EC2_IP "ps aux | grep run_llm_judge | grep -v grep"

echo ""
echo "=== DONE ==="
echo "Monitor progress: ssh ubuntu@$EC2_IP 'tail -f /home/ubuntu/sted-internal/llm_judge_t07.log'"
