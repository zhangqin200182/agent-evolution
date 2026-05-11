#!/bin/bash
# Launch remote NPU training via rllm_remote.
#
# Usage:
#   bash deploy/run_remote.sh                           # defaults
#   bash deploy/run_remote.sh --lr 5e-7 --epochs 200    # custom params
#   bash deploy/run_remote.sh path/to/config.json       # from JSON config
#
# Environment variables:
#   SSH_HOST        - server IP (default: <server-ip>)
#   SSH_USER        - server user (default: root)
#   SSH_KEY         - SSH key path (default: ~/.ssh/id_rsa)
#   CONTAINER       - container name (default: agent5.0.0_qjy)
#   MODEL_PATH      - model path on server
#   TRAIN_DATA      - training data path on server
#   VAL_DATA        - validation data path on server
#   LR              - learning rate (default: 1e-6)
#   BATCH_SIZE      - batch size (default: 32)
#   MINI_BATCH_SIZE - mini batch size (default: 8)
#   EPOCHS          - training epochs (default: 100)
#   NUM_SAMPLES     - samples per prompt (default: 8)
#   TP_SIZE         - tensor parallel size (default: 4)
#   NUM_NPUS        - NPU count (default: 8)
#   KL_COEF         - KL coefficient (default: 0.001)
#   TEMPERATURE     - temperature (default: 1.0)
#   RUN_ID          - custom run ID
#   WANDB_API_KEY   - enables W&B logging
#   DRY_RUN         - set to 1 for dry-run

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

echo "=== Remote NPU Agent RL Training ==="
echo "  Server:    ${SSH_HOST:-<server-ip>}"
echo "  Container: ${CONTAINER:-agent5.0.0_qjy}"
echo ""

# Build CLI args
ARGS=()

[ -n "${SSH_HOST:-}" ]     && ARGS+=(--ssh-host "$SSH_HOST")
[ -n "${SSH_USER:-}" ]     && ARGS+=(--ssh-user "$SSH_USER")
[ -n "${SSH_KEY:-}" ]      && ARGS+=(--ssh-key "$SSH_KEY")
[ -n "${SSH_PASSWORD:-}" ]  && ARGS+=(--ssh-password "$SSH_PASSWORD")
[ -n "${CONTAINER:-}" ]    && ARGS+=(--container "$CONTAINER")
[ -n "${MODEL_PATH:-}" ]   && ARGS+=(--model-path "$MODEL_PATH")
[ -n "${TRAIN_DATA:-}" ]   && ARGS+=(--train-data "$TRAIN_DATA")
[ -n "${VAL_DATA:-}" ]     && ARGS+=(--val-data "$VAL_DATA")
[ -n "${LR:-}" ]           && ARGS+=(--lr "$LR")
[ -n "${BATCH_SIZE:-}" ]   && ARGS+=(--batch-size "$BATCH_SIZE")
[ -n "${MINI_BATCH_SIZE:-}" ] && ARGS+=(--mini-batch-size "$MINI_BATCH_SIZE")
[ -n "${EPOCHS:-}" ]       && ARGS+=(--epochs "$EPOCHS")
[ -n "${NUM_SAMPLES:-}" ]  && ARGS+=(--num-samples "$NUM_SAMPLES")
[ -n "${TP_SIZE:-}" ]      && ARGS+=(--tp-size "$TP_SIZE")
[ -n "${NUM_NPUS:-}" ]     && ARGS+=(--num-npus "$NUM_NPUS")
[ -n "${KL_COEF:-}" ]      && ARGS+=(--kl-coef "$KL_COEF")
[ -n "${TEMPERATURE:-}" ]  && ARGS+=(--temperature "$TEMPERATURE")
[ -n "${RUN_ID:-}" ]       && ARGS+=(--run-id "$RUN_ID")

# W&B
WANDB_ARGS=""
if [ -n "${WANDB_API_KEY:-}" ]; then
    WANDB_ARGS="--wandb-project rllm_remote --wandb-run remote_$(date +%m%d_%H%M)"
fi

if [ "${DRY_RUN:-0}" = "1" ]; then
    python -m rllm_remote.train --dry-run "${ARGS[@]}" $WANDB_ARGS "$@"
else
    python -m rllm_remote.train "${ARGS[@]}" $WANDB_ARGS "$@"
fi
