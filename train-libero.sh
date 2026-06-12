export HF_HOME=/mnt/c/Users/13694/hf_home
export OPENPI_DATA_HOME=/mnt/c/Users/13694/openpi_data_home
# XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 WANDB_MODE=offline uv run scripts/compute_norm_stats.py --config-name=pi0_libero_low_mem_finetune
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 WANDB_MODE=offline &&\ 
    uv run scripts/train.py pi0_libero_low_mem_finetune &&\
    --exp-name=pi0_libero_low_mem_finetune --assets-base-dir=/mnt/c/Users/13694/openpi_data_home/assets --overwrite