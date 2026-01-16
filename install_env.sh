sudo apt-get update && apt-get install -y --no-install-recommends \
    git \
    clang \
    curl \
    ca-certificates \
    build-essential \
    wget \
    ffmpeg \
    libavutil-dev \
    openssh-server 

GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .