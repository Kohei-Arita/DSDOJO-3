# Repository Guidelines

## Project Structure & Module Organization
- `experiments/exp0001/` holds canonical baseline assets: `training.py`, notebooks, model artifacts, and persistent CV logs under `logs/`.
- `configs/` centralizes reusable YAML settings (data sources, CV strategy, LightGBM defaults, feature bundles).
- `scripts/` contains CLI utilities; prefer extending existing entry points rather than duplicating notebook logic.
- `data/` stores large raw and interim CSVs. Treat it as read-only within version control and avoid adding derived artifacts here.
- `docs/` captures workflow playbooks; update alongside major process or tooling changes.

## Build, Test, and Development Commands
- `docker-compose up -d` launches the reproducible dev stack with Jupyter and dependencies pre-installed.
- `jupyter lab` (inside the virtualenv) opens notebooks for exploratory work when Docker is not used.
- `python experiments/exp0001/training.py` runs the end-to-end LightGBM pipeline and records CV metrics to `experiments/exp0001/logs/`.
- `python -m scripts.run_local_experiment --config experiments/exp0001/config.yaml --output-dir experiments/exp0001/artifacts` executes the scripted workflow, emitting `metrics.json`, OOF predictions, and a submission file.

## Coding Style & Naming Conventions
- Follow PEP 8: 4-space indentation, snake_case for modules/functions, UpperCamelCase for classes.
- Keep notebook-derived scripts ASCII-only; add concise comments only where intent is non-obvious.
- Persist experiment artifacts with explicit prefixes (e.g., `host_baseline_###_*.csv`) to mirror README references.
- Prefer `Path` objects over raw strings for filesystem work and seed all stochastic processes with `SEED` constants.

## Testing Guidelines
- Place automated tests under `tests/` using `pytest` naming (`test_*.py`).
- Mock heavy I/O; sample rows from `data/` rather than loading full files in unit tests.
- Run `pytest` (or targeted `pytest tests/test_x.py`) before proposing changes. For notebook exports, sanity-check with `python -m compileall experiments/exp0001/training.py`.

## Commit & Pull Request Guidelines
- Use concise, imperative commit messages (`Add LightGBM logging hook`). Current history favors short English summaries with optional qualifiers.
- Bundle related code, docs, and config updates in a single commit to maintain reproducibility.
- PRs should describe the experiment or fix, list affected artifacts, link to metrics (e.g., `logs/host_baseline_###_metrics.json`), and include screenshots when UI outputs change.
- Confirm CV deltas and submission impacts in the PR description; stale experiments should be rebased or superseded before merge.

## Experiment Tracking & Logs
- After each training run, capture `experiments/exp0001/logs/host_baseline_###_metrics.json` in your notes and update the README experiments table when results are promotion-worthy.
- Archive large artifacts in the `experiments/expXXXX/` subtree only; avoid polluting the repository root or `data/`.

##Rule
- 日本語で回答すること
- 不明点がある場合は確認を求めること。むやみにタスクを進めないこと