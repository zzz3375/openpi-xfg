"""Monkey-patch LeRobotDataset to fix extremely slow timestamp extraction on WSL2.

The original code:
    timestamps = torch.stack(list(self.hf_dataset["timestamp"])).numpy()
    episode_indices = torch.stack(list(self.hf_dataset["episode_index"])).numpy()

This iterates through ALL 273k+ rows one-by-one with transform applied,
which is extremely slow on WSL2 DrvFs (minutes instead of milliseconds).

The fix uses Arrow's native columnar extraction instead.
"""
import logging
import torch
import numpy as np
import lerobot.common.datasets.lerobot_dataset as lds

_original_init = lds.LeRobotDataset.__init__
logger = logging.getLogger(__name__)


def _fast_init(self, *args, **kwargs):
    _original_init(self, *args, **kwargs)
    # After the original __init__ completes, we need to replace the
    # already-computed (likely hanging) timestamp/index arrays with fast ones.
    # Actually, we can't do it after because it hangs during __init__.
    # We need to patch BEFORE.


# Better approach: patch at the import level, replacing the slow path.
# We modify the module BEFORE LeRobotDataset is used.

def _patched_dataset_init(self, *args, **kwargs):
    """Replacement for LeRobotDataset.__init__ that uses fast column extraction."""
    # We need to call the original but intercept the timestamp extraction.
    # Strategy: temporarily disable the transform, do the slow part, then re-enable.
    # But the slow part IS the timestamp extraction.
    # Better: call original up to the point of timestamp extraction, then use fast path.

    import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
    from pathlib import Path

    # Replicate the beginning of __init__
    repo_id = args[0] if args else kwargs.get("repo_id")
    root = kwargs.get("root", None)
    image_transforms = kwargs.get("image_transforms", None)
    delta_timestamps = kwargs.get("delta_timestamps", None)
    episodes = kwargs.get("episodes", None)
    tolerance_s = kwargs.get("tolerance_s", 1e-4)
    revision = kwargs.get("revision", None)
    force_cache_sync = kwargs.get("force_cache_sync", False)
    download_videos = kwargs.get("download_videos", True)
    video_backend = kwargs.get("video_backend", None)

    # Match positional args: repo_id, root, image_transforms, delta_timestamps, ...
    pos_args = list(args)
    if len(pos_args) > 0:
        repo_id = pos_args[0]
    if len(pos_args) > 1:
        root = pos_args[1]
    if len(pos_args) > 2:
        image_transforms = pos_args[2]
    if len(pos_args) > 3:
        delta_timestamps = pos_args[3]

    self.repo_id = repo_id
    self.root = Path(root) if root else lerobot_dataset.HF_LEROBOT_HOME / repo_id
    self.image_transforms = image_transforms
    self.delta_timestamps = delta_timestamps
    self.episodes = episodes
    self.tolerance_s = tolerance_s
    self.revision = revision if revision else lerobot_dataset.CODEBASE_VERSION
    self.video_backend = video_backend if video_backend else lerobot_dataset.get_safe_default_codec()
    self.delta_indices = None
    self.image_writer = None
    self.episode_buffer = None

    self.root.mkdir(exist_ok=True, parents=True)

    # Load metadata
    self.meta = lerobot_dataset.LeRobotDatasetMetadata(
        self.repo_id, self.root, self.revision, force_cache_sync=force_cache_sync
    )
    if self.episodes is not None and self.meta._version >= lerobot_dataset.packaging.version.parse("v2.1"):
        episodes_stats = [self.meta.episodes_stats[ep_idx] for ep_idx in self.episodes]
        self.stats = lerobot_dataset.aggregate_stats(episodes_stats)

    # Load actual data
    try:
        if force_cache_sync:
            raise FileNotFoundError
        assert all((self.root / fpath).is_file() for fpath in self.get_episodes_file_paths())
        self.hf_dataset = self.load_hf_dataset()
    except (AssertionError, FileNotFoundError, NotADirectoryError):
        self.revision = lerobot_dataset.get_safe_version(self.repo_id, self.revision)
        self.download_episodes(download_videos)
        self.hf_dataset = self.load_hf_dataset()

    self.episode_data_index = lerobot_dataset.get_episode_data_index(self.meta.episodes, self.episodes)

    # ===== FAST PATH: use Arrow column extraction instead of per-row iteration =====
    timestamps = self.hf_dataset.data.table.column("timestamp").to_numpy()
    episode_indices = self.hf_dataset.data.table.column("episode_index").to_numpy()
    ep_data_index_np = {k: t.numpy() for k, t in self.episode_data_index.items()}
    lerobot_dataset.check_timestamps_sync(
        timestamps, episode_indices, ep_data_index_np, self.fps, self.tolerance_s
    )

    # Setup delta_indices
    if self.delta_timestamps is not None:
        lerobot_dataset.check_delta_timestamps(self.delta_timestamps, self.fps, self.tolerance_s)
        self.delta_indices = lerobot_dataset.get_delta_indices(self.delta_timestamps, self.fps)


def apply_patch():
    """Apply the monkey-patch to LeRobotDataset."""
    lds.LeRobotDataset.__init__ = _patched_dataset_init
    logger.info("Patched LeRobotDataset.__init__ with fast timestamp extraction")
