sudo apt update
sudo apt install -y --no-install-recommends \
    git \
    clang \
    curl \
    ca-certificates \
    build-essential \
    wget \
    libavutil-dev \
    openssh-server g++-11 libnsl-dev libstdc++-11-dev libtirpc-dev

conda install -c conda-forge ffmpeg==7.1.1 gcc=12 -y
pip install uv

GIT_LFS_SKIP_SMUDGE=1 CC=/usr/bin/cc uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .