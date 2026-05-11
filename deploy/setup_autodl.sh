#!/bin/bash
# AutoDL environment setup script for rllm agent RL training
# Tested on: AutoDL A100/4090 instances with Ubuntu 22.04 + CUDA 12.x
#
# Usage:
#   bash deploy/setup_autodl.sh
#
# This script:
# 1. Installs system dependencies
# 2. Sets up Python environment
# 3. Installs PyTorch + veRL + project dependencies
# 4. Downloads model weights
# 5. Runs a quick sanity check

set -euo pipefail

echo "=== AutoDL Setup for rllm Agent RL ==="
echo "GPU info:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "  (no GPU detected)"
echo ""

# --- Configuration ---
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-3B-Instruct}"
VENV_DIR="${VENV_DIR:-/root/venv_rllm}"
PROJECT_DIR="${PROJECT_DIR:-/root/rllm}"
PYTHON_VERSION="python3.10"

# --- System packages ---
echo "[1/5] Installing system dependencies..."
apt-get update -qq && apt-get install -y -qq git wget curl tmux htop > /dev/null 2>&1
echo "  Done."

# --- Python venv ---
echo "[2/5] Setting up Python environment..."
if [ ! -d "$VENV_DIR" ]; then
    $PYTHON_VERSION -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
pip install --upgrade pip setuptools wheel -q

# --- Install dependencies ---
echo "[3/5] Installing Python packages..."
pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -q transformers accelerate datasets tokenizers sentencepiece
pip install -q trl peft bitsandbytes
pip install -q vllm
pip install -q pandas pyarrow
pip install -q wandb
pip install -q verl

# Install project in editable mode
cd "$PROJECT_DIR"
if [ -f "pyproject.toml" ]; then
    pip install -e . -q
elif [ -f "setup.py" ]; then
    pip install -e . -q
else
    # Add project to PYTHONPATH
    export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
fi

echo "  Done."

# --- Download model ---
echo "[4/5] Downloading model: $MODEL_NAME"
python -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
print(f'  Downloading tokenizer...')
AutoTokenizer.from_pretrained('$MODEL_NAME', trust_remote_code=True)
print(f'  Downloading model weights...')
AutoModelForCausalLM.from_pretrained('$MODEL_NAME', trust_remote_code=True, torch_dtype='auto')
print(f'  Model cached.')
"

# --- Sanity check ---
echo "[5/5] Running sanity check..."
python -c "
import torch
from rllm_common.base import BaseAgent, BaseEnv
from rllm_common.math_env import MathCalcEnv, generate_math_problems
from rllm_common.tools import CalculateTool, FinishTool
from rllm_verl.config import VerlTrainConfig
from rllm_verl.reward import reward_function

print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
print(f'  Imports OK')
"

echo ""
echo "=== Setup complete ==="
echo "Activate with: source $VENV_DIR/bin/activate"
echo "Run training:  python -m rllm_verl.train --num-gpus \$(nvidia-smi -L | wc -l)"
