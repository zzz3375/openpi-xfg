#!/usr/bin/env bash
set -euo pipefail

# ── 选择数据存放盘符：C 盘或 D 盘 ──
# 用法：
#   DATA_DRIVE=d bash train-libero.sh    # D 盘 (推荐，WSL2 避免 vhdx I/O)
#   DATA_DRIVE=c bash train-libero.sh    # C 盘
#   bash train-libero.sh                 # 默认 C 盘
DATA_DRIVE="${DATA_DRIVE:-c}"
DATA_ROOT="/mnt/${DATA_DRIVE}/Users/13694"

export OPENPI_DATA_HOME="$DATA_ROOT/openpi"
export HF_HOME="$DATA_ROOT/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_LEROBOT_HOME="$HF_HOME/lerobot"
export TMPDIR="$DATA_ROOT/tmp"

mkdir -p "$OPENPI_DATA_HOME" "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" "$HF_LEROBOT_HOME" "$TMPDIR"

# ── 清理残留 lock ──
find "$HF_DATASETS_CACHE" -name "*.lock" -delete 2>/dev/null || true

CONFIG="pi0_libero_low_mem_finetune"
EXP_NAME="pi0_libero_low_mem_finetune"

# ── Step 1: 计算归一化统计量 ──
echo "=== Computing normalization stats ==="
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 WANDB_MODE=offline \
    uv run scripts/compute_norm_stats.py --config-name="$CONFIG"

# ── Step 2: 启动训练 ──
echo "=== Starting training ==="
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 WANDB_MODE=disabled \
    uv run scripts/train.py "$CONFIG" \
    --exp-name="$EXP_NAME" --overwrite --num-workers=0
