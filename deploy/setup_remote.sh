#!/bin/bash
# One-time setup on the NPU server for rllm remote training.
# Run from the dev machine (requires SSH access to the server).
#
# Usage:
#   bash deploy/setup_remote.sh
#
# Environment variables:
#   SSH_HOST     - server IP (default: 192.168.9.142)
#   SSH_USER     - server user (default: root)
#   CONTAINER    - container name (default: agent5.0.0_qjy)

set -euo pipefail

SSH_HOST="${SSH_HOST:-192.168.9.142}"
SSH_USER="${SSH_USER:-root}"
SSH_PORT="${SSH_PORT:-22}"
CONTAINER="${CONTAINER:-agent5.0.0_qjy}"

ssh_cmd() {
    ssh -o StrictHostKeyChecking=no -p "$SSH_PORT" "$SSH_USER@$SSH_HOST" "$@"
}

docker_exec() {
    ssh_cmd "docker exec $CONTAINER bash -c '$1'"
}

echo "=== NPU Server Setup for rllm Remote Training ==="
echo "  Server:    $SSH_USER@$SSH_HOST:$SSH_PORT"
echo "  Container: $CONTAINER"
echo ""

# 1. Verify SSH connectivity
echo "[1/5] Checking SSH connectivity..."
if ssh_cmd "echo ok" > /dev/null 2>&1; then
    echo "  SSH: OK"
else
    echo "  SSH: FAILED - cannot connect to $SSH_USER@$SSH_HOST"
    echo "  Verify: ssh key is configured, server is reachable"
    exit 1
fi

# 2. Verify container is running
echo "[2/5] Checking container..."
if ssh_cmd "docker ps --filter name=$CONTAINER --format '{{.Names}}'" | grep -q "$CONTAINER"; then
    echo "  Container '$CONTAINER': running"
else
    echo "  Container '$CONTAINER': NOT RUNNING"
    echo "  Available containers:"
    ssh_cmd "docker ps -a --format '{{.Names}}  {{.Status}}'" || true
    echo ""
    echo "  Start with: docker start $CONTAINER"
    exit 1
fi

# 3. Verify AgentSDK installation in container
echo "[3/6] Checking AgentSDK installation..."
AGENTSDK_DIR="/home/work/AgentSDK/aura"
docker_exec "test -f $AGENTSDK_DIR/run_start_in_local.sh && echo '  run_start_in_local.sh: OK' || echo '  run_start_in_local.sh: MISSING'"
docker_exec "test -d $AGENTSDK_DIR/configs && echo '  configs/: OK' || echo '  configs/: MISSING'"
docker_exec "test -f $AGENTSDK_DIR/aura/start.py && echo '  start.py: OK' || echo '  start.py: MISSING'"

# 4. Check Python packages in container
echo "[4/6] Checking Python packages..."
docker_exec "python3 -c 'import yaml; print(\"pyyaml: OK\")' 2>/dev/null || pip install pyyaml -q"

docker_exec "python3 -c 'import torch; print(\"torch: OK\")' 2>/dev/null" || echo "  WARNING: torch not found in container"

# 5. Create output directories
echo "[5/6] Creating output directories..."
docker_exec "mkdir -p $AGENTSDK_DIR/outputs"
docker_exec "mkdir -p $AGENTSDK_DIR/configs"  # ensure configs dir exists for uploaded configs
echo "  Output dir: $AGENTSDK_DIR/outputs"
echo "  Configs dir: $AGENTSDK_DIR/configs"

# 6. Verify model and data paths
echo "[6/6] Verifying model and data paths..."

MODEL_PATH="/opt/DPC/models/l00619320/code/AGENTIC_RL_WS/AgenticRL_Binary_Files/models/Qwen2.5-7B-Instruct"
TRAIN_DATA="/opt/DPC/models/l00619320/code/VERL_NPU_WS/data/gsm8k/train.parquet"
VAL_DATA="/opt/DPC/models/l00619320/code/VERL_NPU_WS/data/gsm8k/test.parquet"

docker_exec "test -f $MODEL_PATH/config.json && echo '  Model: OK ($MODEL_PATH)' || echo '  Model: MISSING ($MODEL_PATH)'"
docker_exec "test -f $TRAIN_DATA && echo '  Train data: OK ($TRAIN_DATA)' || echo '  Train data: MISSING ($TRAIN_DATA)'"
docker_exec "test -f $VAL_DATA && echo '  Val data: OK ($VAL_DATA)' || echo '  Val data: MISSING ($VAL_DATA)'"

echo ""
echo "=== Setup complete ==="
echo ""
echo "Next steps:"
echo "  python -m rllm_remote.train --dry-run   # preview config"
echo "  python -m rllm_remote.train --check     # verify connectivity"
echo "  bash deploy/run_remote.sh               # launch training"
