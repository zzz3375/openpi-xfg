#!/usr/bin/env bash
set -euo pipefail

# ── 将所有缓存指向 D 盘，避免 C 盘 (WSL2 vhdx) 异常读盘 ──
export HF_HOME=/mnt/c/Users/13694/hf_home
export HF_HUB_CACHE=/mnt/c/Users/13694/hf_home/hub
export HF_DATASETS_CACHE=/mnt/c/Users/13694/hf_home/datasets
export HF_LEROBOT_HOME=/mnt/c/Users/13694/hf_home/lerobot
export OPENPI_DATA_HOME=/mnt/c/Users/13694/openpi_data_home
export TMPDIR=/mnt/c/Users/13694/tmp
mkdir -p /mnt/c/Users/13694/tmp

# ── 清理可能残留的 lock 文件 (上次被中断运行留下的) ──
find "$HF_DATASETS_CACHE" -name "*.lock" -delete 2>/dev/null || true

CONFIG="pi0_libero_low_mem_finetune"
EXP_NAME="pi0_libero_low_mem_finetune"

# ── Step 1: 计算归一化统计量 (只需运行一次，重复运行会跳过已有结果) ──
echo "=== Computing normalization stats ==="
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 WANDB_MODE=offline \
    uv run scripts/compute_norm_stats.py --config-name="$CONFIG"

# ── Step 2: 启动训练 ──
echo "=== Starting training ==="
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 WANDB_MODE=disabled \
    uv run scripts/train.py "$CONFIG" \
    --exp-name="$EXP_NAME" --overwrite --num-workers=0
