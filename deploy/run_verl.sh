#!/bin/bash
# Launch veRL training on AutoDL
#
# Usage:
#   bash deploy/run_verl.sh                    # defaults
#   MODEL=Qwen/Qwen2.5-7B-Instruct bash deploy/run_verl.sh
#
# Environment variables:
#   MODEL         - model name (default: Qwen/Qwen2.5-3B-Instruct)
#   NUM_GPUS      - GPU count (default: auto-detect)
#   NUM_PROBLEMS  - training problems (default: 200)
#   BATCH_SIZE    - batch size (default: 8)
#   NUM_EPOCHS    - training epochs (default: 3)
#   WANDB_API_KEY - enables W&B logging

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/root/rllm}"
cd "$PROJECT_DIR"

# Activate venv if it exists
if [ -f "/root/venv_rllm/bin/activate" ]; then
    source /root/venv_rllm/bin/activate
fi

# Auto-detect GPUs
NUM_GPUS=${NUM_GPUS:-$(nvidia-smi -L 2>/dev/null | wc -l)}
MODEL=${MODEL:-"Qwen/Qwen2.5-3B-Instruct"}
NUM_PROBLEMS=${NUM_PROBLEMS:-200}
BATCH_SIZE=${BATCH_SIZE:-8}
NUM_EPOCHS=${NUM_EPOCHS:-3}
OUTPUT_DIR=${OUTPUT_DIR:-"./outputs/verl_$(date +%Y%m%d_%H%M%S)"}

echo "=== veRL Agent RL Training ==="
echo "  Model:      $MODEL"
echo "  GPUs:       $NUM_GPUS"
echo "  Problems:   $NUM_PROBLEMS"
echo "  Batch size: $BATCH_SIZE"
echo "  Epochs:     $NUM_EPOCHS"
echo "  Output:     $OUTPUT_DIR"
echo ""

# W&B args
WANDB_ARGS=""
if [ -n "${WANDB_API_KEY:-}" ]; then
    WANDB_ARGS="--wandb-project rllm_verl --wandb-run verl_$(date +%m%d_%H%M)"
    echo "  W&B: enabled"
fi

# Launch
python -m rllm_verl.train \
    --model "$MODEL" \
    --num-gpus "$NUM_GPUS" \
    --num-problems "$NUM_PROBLEMS" \
    --batch-size "$BATCH_SIZE" \
    --num-epochs "$NUM_EPOCHS" \
    --output-dir "$OUTPUT_DIR" \
    $WANDB_ARGS \
    "$@"

echo ""
echo "Training complete. Output: $OUTPUT_DIR"
