#!/usr/bin/env bash
set -euo pipefail

# ── 自动选择数据存放盘符：优先 D 盘，否则使用 C 盘 ──
# 用法：
#   bash train-libero.sh                 # D 盘存在则用 D，否则用 C
if [ -d /mnt/d ]; then
    DATA_DRIVE="d"
else
    DATA_DRIVE="c"
fi
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

# ── 拉取 lerobot fork 最新 commit ──
echo "=== Upgrading lerobot to latest commit ==="
uv lock --upgrade-package lerobot
uv sync --no-install-project

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
