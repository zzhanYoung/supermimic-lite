"""Train SuperMimic, optionally resuming a preserved run into a new directory."""

from __future__ import annotations

import argparse
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal, cast

import mjlab
import tyro
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.utils.gpu import select_gpus
from mjlab.utils.os import dump_yaml
from mjlab.utils.torch import configure_torch_backends

from supermimic.assets import DEFAULT_ASSETS, LocalAssets, require_path
from supermimic.cli.evaluate import read_snapshot
from supermimic.compat.checkpoints import load_checkpoint
from supermimic.envs import SuperMimicEnv
from supermimic.evaluation.runtime import restore_config
from supermimic.learning.config import SuperMimicRunnerCfg, runner_config
from supermimic.learning.runner import SuperMimicRunner
from supermimic.mdp.return_sampler import ReturnMotionCommandCfg
from supermimic.tasks import load_env_cfg
from supermimic.tasks.tracking_env_cfg import TrackingEnvCfg


@dataclass
class TrainConfig:
  env: TrackingEnvCfg
  agent: SuperMimicRunnerCfg
  gpu_ids: list[int] | Literal["all"] | None = field(default_factory=lambda: [0])
  checkpoint: str | None = None
  enable_nan_guard: bool = False


def run_train(cfg: TrainConfig, output: Path) -> None:
  rank = int(os.environ.get("RANK", "0"))
  local_rank = int(os.environ.get("LOCAL_RANK", "0"))
  device = f"cuda:{local_rank}" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
  if device != "cpu":
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(local_rank)
  configure_torch_backends()
  cfg.env.seed = cfg.agent.seed = cfg.agent.seed + rank
  cfg.env.sim.nan_guard.enabled = cfg.enable_nan_guard
  env = SuperMimicEnv(cfg.env, device=device)
  runner = None
  try:
    wrapper = RslRlVecEnvWrapper(env, clip_actions=cfg.agent.clip_actions)
    runner = SuperMimicRunner(wrapper, asdict(cfg.agent), str(output), device=device)
    if cfg.checkpoint:
      runner.load(cfg.checkpoint)
    remaining = cfg.agent.max_iterations - runner.current_learning_iteration
    if remaining <= 0:
      raise ValueError("--max-iterations must exceed the checkpoint iteration.")
    if rank == 0:
      dump_yaml(output / "params/env.yaml", asdict(cfg.env))
      dump_yaml(output / "params/agent.yaml", asdict(cfg.agent))
    runner.add_git_repo_to_log(__file__)
    runner.learn(remaining)
  finally:
    if runner is not None:
      writer = getattr(runner.logger, "writer", None)
      if writer is not None:
        writer.close()
    env.close()


def parse_config(
  argv: list[str] | None, assets: LocalAssets
) -> tuple[TrainConfig, Path]:
  parser = argparse.ArgumentParser(
    description=__doc__,
    epilog="Advanced options: --agent.* / --env.* / --gpu-ids; --help-full lists them.",
  )
  parser.add_argument("--resume", action="store_true")
  parser.add_argument("--checkpoint", type=Path)
  parser.add_argument("--run-dir", type=Path)
  parser.add_argument("--motion-folder", type=Path)
  parser.add_argument("--output-dir", type=Path)
  parser.add_argument(
    "--max-iterations",
    type=int,
    help=f"Total learner iteration target (default: {SuperMimicRunnerCfg().max_iterations}).",
  )
  parser.add_argument("--device", help="cpu or cuda:N; use --gpu-ids for multiple GPUs")
  parser.add_argument("--help-full", action="store_true")
  options, advanced = parser.parse_known_args(argv)
  cfg = TrainConfig(load_env_cfg(), SuperMimicRunnerCfg())
  checkpoint = None
  if options.help_full:
    tyro.cli(TrainConfig, default=cfg, args=["--help"], config=mjlab.TYRO_FLAGS)
  if options.checkpoint is not None and not options.resume:
    raise ValueError("Use --resume with --checkpoint to continue training.")
  if options.resume:
    checkpoint = require_path(options.checkpoint or assets.checkpoint, "--checkpoint")
    run_dir = assets.checkpoint_run_dir(checkpoint, options.run_dir)
    cfg.env = restore_config(cfg.env, read_snapshot(run_dir / "params/env.yaml"))
    saved = load_checkpoint(checkpoint)
    cfg.agent = runner_config(saved["config"])
    cfg.agent.max_iterations = SuperMimicRunnerCfg().max_iterations
    cfg.checkpoint = str(checkpoint)
  motion = cast(ReturnMotionCommandCfg, cfg.env.commands["motion"])
  motion.motion_folder = str(
    options.motion_folder or assets.motion_folder or motion.motion_folder
  )
  if options.max_iterations is not None:
    cfg.agent.max_iterations = options.max_iterations
  if options.device:
    if options.device == "cpu":
      cfg.gpu_ids = None
    elif options.device.startswith("cuda:"):
      cfg.gpu_ids = [int(options.device.split(":", 1)[1])]
    else:
      raise ValueError("--device must be cpu or cuda:N")
  cfg = tyro.cli(TrainConfig, default=cfg, args=advanced, config=mjlab.TYRO_FLAGS)
  cfg.agent.validate()
  motion = cast(ReturnMotionCommandCfg, cfg.env.commands["motion"])
  motion_dir = require_path(motion.motion_folder, "--motion-folder", directory=True)
  motion.motion_folder = str(motion_dir)
  motion.resolve_motion_files()
  if checkpoint:
    saved = load_checkpoint(checkpoint)
    if cfg.agent.max_iterations <= saved["iteration"]:
      raise ValueError("--max-iterations must exceed the checkpoint iteration.")
    for rank in range(saved["world_size"]):
      require_path(
        checkpoint.with_name(f"{checkpoint.stem}.rank{rank}.replay.pt"),
        "matching replay shard (required for resume, not inference)",
      )
  stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
  name = stamp + ("_" + cfg.agent.run_name if cfg.agent.run_name else "")
  output = options.output_dir or Path("checkpoints") / cfg.agent.experiment_name / name
  output = assets.protect_output(
    output,
    checkpoint,
    motion_dir,
    assets.checkpoint_run_dir(checkpoint, options.run_dir)
    if checkpoint
    else options.run_dir,
  )
  if output.exists() and (not output.is_dir() or any(output.iterdir())):
    raise ValueError(f"Output must be a new or empty directory: {output}")
  return cfg, output


def main(argv: list[str] | None = None, *, assets: LocalAssets = DEFAULT_ASSETS) -> int:
  cfg, output = parse_config(argv, assets)
  selected, count = select_gpus(cfg.gpu_ids)
  if count > 2:
    raise ValueError("SuperMimic supports one or two training ranks.")
  if cfg.checkpoint:
    expected = load_checkpoint(cfg.checkpoint)["world_size"]
    if expected != max(1, count):
      raise ValueError(f"Resume requires {expected} rank(s), got {max(1, count)}.")
  os.environ["CUDA_VISIBLE_DEVICES"] = (
    "" if selected is None else ",".join(map(str, selected))
  )
  output.mkdir(parents=True, exist_ok=True)
  print(f"Training output: {output}", flush=True)
  if count <= 1:
    run_train(cfg, output)
  else:
    import torchrunx

    logging.basicConfig(level=logging.INFO)
    os.environ["TORCHRUNX_LOG_DIR"] = str(output / "torchrunx")
    torchrunx.Launcher(
      hostnames=["localhost"],
      workers_per_host=count,
      backend=None,
      agent_timeout=900,
      copy_env_vars=torchrunx.DEFAULT_ENV_VARS_FOR_COPY + ("MUJOCO*",),
    ).run(run_train, cfg, output)
  return 0
