# ── 自动选择数据存放盘符：优先 D 盘，否则使用 C 盘 ──
# 用法：
#   bash train-libero.sh                 # D 盘存在则用 D，否则用 C
if mountpoint -q /mnt/d 2>/dev/null; then
    DATA_DRIVE="d"
else
    DATA_DRIVE="c"
fi
DATA_ROOT="/mnt/${DATA_DRIVE}/Users/13694"

export OPENPI_DATA_HOME="$DATA_ROOT/openpi_data_home"
export HF_HOME="$DATA_ROOT/hf_home"

# export HTTP_PROXY=http://127.0.0.1:7890
# export HTTPS_PROXY=https://127.0.0.1:7890


CONFIG="pi0_libero_low_mem_finetune"
EXP_NAME="pi0_libero_low_mem_finetune"

# ── Step 1: 计算归一化统计量 ──
# echo "=== Computing normalization stats ==="
# XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 WANDB_MODE=offline \
#     uv run scripts/compute_norm_stats.py --config-name="$CONFIG"

# ── Step 2: 启动训练 ──
echo "=== Starting training ==="
XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 WANDB_MODE=disabled \
    uv run scripts/train.py "$CONFIG" \
    --exp-name="$EXP_NAME" --resume --num-workers=0
