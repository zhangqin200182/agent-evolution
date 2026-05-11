#!/bin/bash
# Launch TRL-based training on AutoDL (single GPU, simpler setup)
# Use this as a fallback when veRL is not needed or for quick experiments.
#
# Usage:
#   bash deploy/run_trl.sh
#   bash deploy/run_trl.sh "200 problems, 3 epochs, qwen-3b"

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/root/rllm}"
cd "$PROJECT_DIR"

if [ -f "/root/venv_rllm/bin/activate" ]; then
    source /root/venv_rllm/bin/activate
fi

echo "=== TRL Agent RL Training (single GPU) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
echo ""

# Pass all arguments to the training script
if [ $# -gt 0 ]; then
    python -m rllm_train.train "$@"
else
    python -m rllm_train.train
fi
