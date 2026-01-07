# Use NVIDIA CUDA base image with Ubuntu 22.04 and Python 3.11
FROM nvidia/cuda:12.3.1-devel-ubuntu22.04

# Avoid interactive prompts during package install
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    ca-certificates \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv (ultra-fast Python package installer)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:/root/.local/bin:$PATH"

# Set up working directory
WORKDIR /workspace

# Clone repo + submodules (shallow to reduce size, but full if needed)
RUN git clone --recurse-submodules https://github.com/Physical-Intelligence/openpi.git .
# If you prefer SSH (for private forks), replace with:
# RUN git clone --recurse-submodules git@github.com:Physical-Intelligence/openpi.git .

# Install Python dependencies with uv (skip LFS during install to avoid large pulls)
RUN GIT_LFS_SKIP_SMUDGE=1 uv sync
RUN GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .

# Default cache directory (override with OPENPI_DATA_HOME at runtime)
ENV OPENPI_DATA_HOME=/workspace/.openpi_cache
RUN mkdir -p $OPENPI_DATA_HOME

# Expose policy server port (used by serve_policy.py)
EXPOSE 8000

# Health check: verify Python + basic import
HEALTHCHECK --interval=5m --timeout=3s \
  CMD python -c "import openpi; print('✅ openpi imported')" || exit 1

# Default command: interactive shell (override for training/inference)
CMD ["bash"]