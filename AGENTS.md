# Repository Guidelines

## Project Structure & Module Organization
- `src/openpi/`: core library code (models, policies, training, serving, shared utilities).
- `packages/openpi-client/`: Python client/runtime package used by external apps.
- `scripts/`: training, serving, and dataset utilities.
- `examples/`: runnable demos and robot-specific workflows (ALOHA, DROID, LIBERO, UR5).
- `docs/`: focused guides (Docker, remote inference, normalization stats).
- `third_party/`: vendored dependencies and assets; avoid modifying unless necessary.

## Build, Test, and Development Commands
- `uv sync` + `uv pip install -e .`: install dependencies and editable package.
- `uv run scripts/compute_norm_stats.py --config-name <config>`: precompute normalization stats.
- `uv run scripts/train.py <config> --exp-name=<name>`: launch training.
- `uv run scripts/serve_policy.py policy:checkpoint --policy.config=<cfg> --policy.dir=<ckpt>`: serve a policy over websocket.
- `pytest`: run unit tests across `src`, `scripts`, and `packages`.
- `ruff format .` / `ruff check .`: format and lint (run via `pre-commit` when possible).

## Coding Style & Naming Conventions
- Python, 4-space indentation, line length 120 (Ruff config).
- Naming: `snake_case` for functions/files, `PascalCase` for classes, `UPPER_SNAKE_CASE` for constants.
- Keep imports one per line; Ruff enforces import sorting.

## Testing Guidelines
- Framework: `pytest` (see `pyproject.toml` for config).
- Tests live alongside code and use `*_test.py` naming (e.g., `src/openpi/models/pi0_test.py`).
- Mark long/manual tests with `@pytest.mark.manual` as needed.

## Commit & Pull Request Guidelines
- Commit messages appear short, sentence-style, and imperative (e.g., “use EGL for headless GPU rendering”).
- PRs should include a clear title/description and run `pre-commit`, `ruff`, and tests.
- For new robot/environment support, consider opening an issue or discussion first.

## Configuration & Data Notes
- Checkpoints download to `~/.cache/openpi` by default; override with `OPENPI_DATA_HOME`.
- Many examples assume GPUs and Linux (tested on Ubuntu 22.04).
