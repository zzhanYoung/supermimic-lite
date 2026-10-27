# SuperMimic Lite

Standalone SuperMimic motion tracking for Unitree G1, using official [mjlab](https://github.com/mujocolab/mjlab) as an external dependency. This repository owns the learner, motion commands, task configuration, evaluation and data tools. It requires no changes to mjlab.

The package structure follows `supermimic_reference/supermimic_mjlab`; the learner and training semantics remain those of the existing **lite** implementation. Only SuperMimic training and inference are exposed. Plotting still supports historical PPO TensorBoard logs.

## Install

Use Python 3.11–3.13, Git and [uv](https://docs.astral.sh/uv/). The default Python version is 3.13. Linux with an NVIDIA GPU is recommended for training.

```bash
uv sync --locked --extra cu128 --group dev
uv run --no-sync supermimic --help
```

For CPU development, replace `--extra cu128` with `--extra cpu`. Use `--no-sync` for subsequent commands to preserve the selected Torch variant. Initial installation requires internet access.

`pyproject.toml` pins official mjlab to commit `b517e0c489139e7fcee95702cfb2b01931264985`, matching lite's APIs and RSL-RL 5.5.0. `uv.lock` locks dependencies. There is no submodule, fork, namespace injection or `PYTHONPATH` setup. mjlab and its robot assets are installed in the virtual environment.

## Connect assets

Checkpoints, replay shards, motions and logs are **not bundled** in Git or the wheel. No public Hugging Face asset repository has been configured yet. Use an existing compatible run and motion dataset.

```bash
cp assets.example.toml .supermimic.local.toml
# Edit the asset paths.
uv run --no-sync supermimic doctor
uv run --no-sync supermimic data check
```

Paths in the ignored `.supermimic.local.toml` are relative to that file. `--asset-config /path/to/assets.toml` selects another configuration; explicit command flags override its defaults. `protected_roots` prevent writing new output into preserved input directories.

In the migration workspace, this configuration already points to `../../supermimic-lite`. Original assets stay there. New training writes under this repository's `checkpoints/`; evaluation and plots write under `logs/`.

## Play and evaluate

```bash
# Native viewer; requires a display.
uv run --no-sync supermimic play

# Remote/headless playback.
uv run --no-sync supermimic play --headless --steps 100 --device cuda:0

# Full-motion evaluation at 50 Hz.
uv run --no-sync supermimic eval --num-envs 40 --device cuda:0
```

Use `--motion <NPZ-stem>` to select a playback clip. Both commands accept `--checkpoint`, `--motion-folder` and `--run-dir`. The run directory contains `params/env.yaml` and `params/agent.yaml`; by default it is the checkpoint parent, unless configured for the selected checkpoint.

Inference loads the actor from the main checkpoint and needs no replay shard. Legacy lite checkpoint names are translated per load, without rewriting source files. New saves use format version 2 with data-only configuration signatures. Load only trusted PyTorch training checkpoints.

Evaluation preserves lite's frame-zero start, full reference horizon, failure thresholds, terminal-state scoring and normalized return. Each invocation creates a new output directory with per-motion metrics and figures.

## Train or resume

```bash
uv run --no-sync supermimic train --max-iterations 320000
uv run --no-sync supermimic train --resume --max-iterations 20000
uv run --no-sync supermimic train --gpu-ids '[0,1]' --max-iterations 320000
uv run --no-sync supermimic train --help-full
```

`--max-iterations` is the **total target iteration**: resuming iteration 5000 to 20000 performs 15000 additional iterations. Resume requires the main checkpoint, every matching `model_N.rankR.replay.pt` shard and saved configuration. Dataset, task signature, rank count and replay dimensions must match. Missing replay is an error. The saved iteration and learning-rate schedule continue.

`--output-dir` selects a new or empty destination. Resume always writes a new run; it does not overwrite the source. Advanced options use `--env.*` and `--agent.*` flags exposed by `--help-full`. Collections use quoted Python syntax, e.g. `--agent.critic.hidden-dims '(256,128)'`; booleans use explicit `True`/`False` values.

## Data, plots and development

```bash
uv run --no-sync supermimic data csv --help
uv run --no-sync supermimic data qpos --help
uv run --no-sync supermimic data play-npz --help
uv run --no-sync supermimic data play-bvh --help
uv run --no-sync supermimic plot --source /path/to/run --output-dir logs/plots
uv run --no-sync supermimic data inventory
make check
make test-fast
make test
uv build
```

Tests include synthetic-motion simulation, training/save/resume and two-rank CPU synchronization. Set `MUJOCO_GL=disable` for headless use. See [architecture](docs/architecture.md), [asset migration](docs/assets.md), [validation](docs/validation.md) and [NOTICE](NOTICE.md). License: Apache-2.0.
