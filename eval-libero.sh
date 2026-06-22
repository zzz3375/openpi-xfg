#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
# LIBERO 评估脚本（适用于 RTX 5090 / CUDA 12.8+ / WSL2）
# ─────────────────────────────────────────────────────────────
# 用法：
#   bash eval-libero.sh                          # 默认评估 libero_spatial
#   bash eval-libero.sh libero_10                # 评估 libero_10
#   bash eval-libero.sh libero_90 20             # 评估 libero_90，每任务 20 次 rollout
#   bash eval-libero.sh --docker                 # 使用 Docker 运行客户端（WSL 推荐）
#   bash eval-libero.sh --help                   # 显示帮助
# ─────────────────────────────────────────────────────────────

set -euo pipefail

# ── 默认值 ──
USE_DOCKER=false
TASK_SUITE="libero_spatial"
NUM_TRIALS=50

# ── 解析参数 ──
for arg in "$@"; do
    case "$arg" in
        --docker)
            USE_DOCKER=true
            ;;
        --help|-h)
            echo "用法: bash eval-libero.sh [TASK_SUITE] [NUM_TRIALS] [--docker]"
            echo ""
            echo "参数:"
            echo "  TASK_SUITE    任务套件 (默认 libero_spatial)"
            echo "                可选: libero_spatial, libero_object, libero_goal, libero_10, libero_90"
            echo "  NUM_TRIALS    每个任务 rollout 次数 (默认 50)"
            echo "  --docker      使用 Docker 运行 LIBERO 客户端 (WSL2 推荐)"
            echo ""
            echo "示例:"
            echo "  bash eval-libero.sh                         # 默认: libero_spatial, 50 trials"
            echo "  bash eval-libero.sh libero_10               # libero_10, 50 trials"
            echo "  bash eval-libero.sh libero_90 20 --docker   # libero_90, 20 trials, Docker"
            exit 0
            ;;
        --*)
            echo "未知选项: $arg (使用 --help 查看帮助)"
            exit 1
            ;;
        *)
            if [[ "$arg" =~ ^[0-9]+$ ]]; then
                NUM_TRIALS="$arg"
            else
                TASK_SUITE="$arg"
            fi
            ;;
    esac
done

# ── 自动选择数据存放盘符（与 train-libero.sh 保持一致） ──
if [ -d /mnt/d ]; then
    DATA_DRIVE="d"
else
    DATA_DRIVE="c"
fi
DATA_ROOT="/mnt/${DATA_DRIVE}/Users/13694"
WIN_TEMP="/mnt/c/Users/13694/AppData/Local/Temp"

export OPENPI_DATA_HOME="$DATA_ROOT/openpi_data_home"
export HF_HOME="$DATA_ROOT/hf_home"
export TMPDIR="$WIN_TEMP/openpi_tmp"
mkdir -p "$TMPDIR"

# ── 训练配置（必须与 train-libero.sh 一致） ──
CONFIG="pi05_libero_low_mem_finetune"
EXP_NAME="pi05_libero_low_mem_finetune"
CHECKPOINT_BASE="$DATA_ROOT/openpi_checkpoints"

# ── 查找最新 checkpoint ──
CHECKPOINT_DIR="$CHECKPOINT_BASE/$EXP_NAME/$EXP_NAME"
if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "错误: 未找到 checkpoint 目录: $CHECKPOINT_DIR"
    echo "请确认训练已完成，或修改脚本中的 CHECKPOINT_BASE。"
    exit 1
fi

LATEST_STEP=$(ls -d "$CHECKPOINT_DIR"/*/ 2>/dev/null | sort -t '/' -k1 -n | tail -1 | xargs basename 2>/dev/null || echo "")
if [ -z "$LATEST_STEP" ]; then
    echo "错误: $CHECKPOINT_DIR 中没有 checkpoint step 目录"
    exit 1
fi
CHECKPOINT_PATH="$CHECKPOINT_DIR/$LATEST_STEP"

echo "============================================"
echo " LIBERO 评估"
echo "============================================"
echo " 配置:      $CONFIG"
echo " 任务套件:  $TASK_SUITE"
echo " 每任务次数: $NUM_TRIALS"
echo " Checkpoint: $CHECKPOINT_PATH"
echo " 模式:      $([ $USE_DOCKER = true ] && echo 'Docker' || echo '本地')"
echo "============================================"
echo ""

# ── 清理函数 ──
cleanup() {
    echo ""
    echo "正在停止模型服务器 (PID: ${SERVER_PID:-unknown})..."
    if [ -n "${SERVER_PID:-}" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    echo "已退出。"
}
trap cleanup EXIT INT TERM

# ═══════════════════════════════════════════════════════════════
# 启动模型服务器（后台运行，日志写入临时文件）
# ═══════════════════════════════════════════════════════════════
echo ">>> 启动模型服务器..."
SERVER_LOG=$(mktemp /tmp/openpi_server.XXXXXX.log)
XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 WANDB_MODE=disabled \
    uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config="$CONFIG" \
    --policy.dir="$CHECKPOINT_PATH" \
    > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!

# 监听日志，等待 "server listening" 出现（不改动端口，不干扰 websocket）
echo -n ">>> 等待服务器就绪"
for i in $(seq 1 120); do
    if grep -q "server listening" "$SERVER_LOG" 2>/dev/null; then
        echo ""
        echo ">>> 服务器已就绪 (PID: $SERVER_PID)"
        break
    fi
    if [ $i -eq 120 ]; then
        echo ""
        echo "错误: 服务器启动超时 (120s)。最近日志:"
        tail -20 "$SERVER_LOG"
        exit 1
    fi
    echo -n "."
    sleep 1
done

# 启动后清理日志文件（评估期间不再需要）
rm -f "$SERVER_LOG"

# ═══════════════════════════════════════════════════════════════
# 运行 LIBERO 评估客户端
# ═══════════════════════════════════════════════════════════════
if $USE_DOCKER; then
    echo ">>> Docker 模式: 仅启动 LIBERO 仿真容器（模型服务器在 WSL 本地运行）"

    # 检查 Docker 权限
    if ! docker ps > /dev/null 2>&1; then
        echo ""
        echo "错误: 没有 Docker 权限。请执行以下操作之一："
        echo "  1) 将当前用户加入 docker 组: sudo usermod -aG docker \$USER && newgrp docker"
        echo "  2) 或使用 sudo 运行此脚本: sudo bash eval-libero.sh --docker"
        echo ""
        exit 1
    fi

    export CLIENT_ARGS="--args.task-suite-name=$TASK_SUITE --args.num-trials-per-task=$NUM_TRIALS"

    # 首次运行需构建镜像
    if ! docker image inspect libero > /dev/null 2>&1; then
        echo ">>> 首次运行，构建 Docker 镜像（约 3-5 分钟）..."
        docker compose -f examples/libero/compose.yml build runtime
    fi

    # 只启动 runtime 服务（通过 --no-deps 跳过 compose 中的 openpi_server）
    # runtime 容器使用 network_mode: host，可直接访问 localhost:8000
    docker compose -f examples/libero/compose.yml up \
        --no-deps \
        --abort-on-container-exit \
        --remove-orphans \
        runtime
else
    echo ">>> 本地模式: 激活 LIBERO 虚拟环境..."

    LIBERO_VENV="examples/libero/.venv"
    if [ ! -f "$LIBERO_VENV/bin/activate" ]; then
        echo ""
        echo "错误: LIBERO 虚拟环境未创建: $LIBERO_VENV"
        echo ""
        echo "请执行以下命令初始化环境:"
        echo "  uv venv --python 3.8 $LIBERO_VENV"
        echo "  source $LIBERO_VENV/bin/activate"
        echo "  uv pip sync examples/libero/requirements.txt third_party/libero/requirements.txt \\"
        echo "    --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy=unsafe-best-match"
        echo "  uv pip install -e packages/openpi-client"
        echo "  uv pip install -e third_party/libero"
        exit 1
    fi

    source "$LIBERO_VENV/bin/activate"
    export PYTHONPATH="$PYTHONPATH:$PWD/third_party/libero"

    echo ">>> 运行评估..."
    python examples/libero/main.py \
        --args.task-suite-name="$TASK_SUITE" \
        --args.num-trials-per-task="$NUM_TRIALS"
fi
