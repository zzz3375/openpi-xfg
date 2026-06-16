export OPENPI_DATA_HOME=/root/private_data/robot_ws/openpi_cache
uv run scripts/compute_norm_stats.py --config-name pi05_xfg_full
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 WANDB_MODE=offline uv run scripts/train.py pi05_xfg_full --exp-name=pi05_xfg_full --overwrite