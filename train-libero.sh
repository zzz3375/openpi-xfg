# ── 自动选择数据存放盘符：优先 D 盘，否则使用 C 盘 ──
# 用法：
#   bash train-libero.sh                 # D 盘存在则用 D，否则用 C
if [ -d /mnt/d ]; then
    DATA_DRIVE="d"
else
    DATA_DRIVE="c"
fi
DATA_ROOT="/mnt/${DATA_DRIVE}/Users/13694"
WIN_TEMP="/mnt/c/Users/13694/AppData/Local/Temp"

# ── 所有缓存和临时文件都映射到 Windows，避免写满 WSL 虚拟磁盘 ──
export OPENPI_DATA_HOME="$DATA_ROOT/openpi_data_home"
export HF_HOME="$DATA_ROOT/hf_home"
# JAX/XLA 编译临时文件 → Windows Temp（重启后自动清理）
export TMPDIR="$WIN_TEMP/openpi_tmp"
mkdir -p "$TMPDIR"

CONFIG="pi05_libero_low_mem_finetune"
EXP_NAME="pi05_libero_low_mem_finetune"

# ── Step 1: 计算归一化统计量 ──
echo "=== Computing normalization stats ==="
# uv run scripts/compute_norm_stats.py --config-name="$CONFIG"

# ── Step 2: 启动训练 ──
echo "=== Starting training ==="
XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 \
    uv run scripts/train.py "$CONFIG" \
    --exp-name="$EXP_NAME" --num-workers=0 --resume \
    --checkpoint-base-dir "$DATA_ROOT/openpi_checkpoints" #--overwrite 
