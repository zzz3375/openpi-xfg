"""
Convert a LeRobot dataset on disk into a Libero-style RLDS episode structure.

Input (LeRobot):
  - A local LeRobot dataset directory (e.g., under $HF_LEROBOT_HOME/<repo_id>).

Output (Libero-style, RLDS-like):
  output_dir/
    metadata.json
    episodes/
      episode_000000.npz
      episode_000001.npz
      ...

Each episode .npz stores:
  - observation_image: (T, H, W, C) uint8
  - observation_wrist_image: (T, H, W, C) uint8
  - observation_state: (T, 8) float32
  - action: (T, 7) float32
  - language_instruction: (T,) bytes (per-step, Libero RLDS convention)

This mirrors the Libero RLDS step keys used in `convert_libero_data_to_lerobot.py`,
but does not write TFDS metadata or TFRecord shards. Use this as a concrete,
portable Libero-style export that you can adapt into TFDS if needed.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import tyro


@dataclass(frozen=True)
class Args:
    input_dir: str
    output_dir: str


def _lerobot_frame_to_libero_step(frame: dict) -> dict:
    task = frame.get("task", "")
    if isinstance(task, bytes):
        language_instruction = task
    else:
        language_instruction = str(task).encode()

    return {
        "observation": {
            "image": frame["image"],
            "wrist_image": frame["wrist_image"],
            "state": frame["state"],
        },
        "action": frame["actions"],
        "language_instruction": language_instruction,
    }


def _iter_lerobot_episodes(input_dir: Path):
    try:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as exc:
        raise ImportError(
            "LeRobot is not installed. Install it before running this script "
            "(e.g., `uv pip install lerobot`)."
        ) from exc

    dataset = LeRobotDataset(str(input_dir))

    current_episode = None
    frames: list[dict] = []
    for frame in dataset:
        episode_id = frame.get("episode_index", frame.get("episode_id"))
        if episode_id is None:
            raise KeyError(
                "Expected LeRobot frames to include `episode_index` or `episode_id` for grouping."
            )

        if current_episode is None:
            current_episode = episode_id
        if episode_id != current_episode:
            yield current_episode, frames
            frames = []
            current_episode = episode_id

        frames.append(frame)

    if frames:
        yield current_episode, frames


def _write_episode_npz(output_dir: Path, episode_index: int, steps: list[dict]) -> None:
    episode_path = output_dir / "episodes" / f"episode_{episode_index:06d}.npz"
    observation_image = np.stack([step["observation"]["image"] for step in steps], axis=0)
    observation_wrist_image = np.stack([step["observation"]["wrist_image"] for step in steps], axis=0)
    observation_state = np.stack([step["observation"]["state"] for step in steps], axis=0)
    action = np.stack([step["action"] for step in steps], axis=0)
    language_instruction = np.asarray([step["language_instruction"] for step in steps], dtype=object)

    np.savez_compressed(
        episode_path,
        observation_image=observation_image,
        observation_wrist_image=observation_wrist_image,
        observation_state=observation_state,
        action=action,
        language_instruction=language_instruction,
    )


def main(args: Args) -> None:
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "episodes").mkdir(parents=True, exist_ok=True)

    episode_count = 0
    step_count = 0
    for episode_id, frames in _iter_lerobot_episodes(input_dir):
        steps = [_lerobot_frame_to_libero_step(frame) for frame in frames]
        _write_episode_npz(output_dir, int(episode_id), steps)
        episode_count += 1
        step_count += len(steps)

    metadata = {
        "input_dir": str(input_dir),
        "episodes": episode_count,
        "steps": step_count,
        "format": "libero_rlds_npz_v1",
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main(tyro.cli(Args))
