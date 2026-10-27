"""Play a SuperMimic actor without loading replay, optimizers, or a training runner."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, cast

import torch
from mjlab.rl import RslRlVecEnvWrapper
from tensordict import TensorDict

from supermimic.assets import DEFAULT_ASSETS, LocalAssets, require_path
from supermimic.cli.evaluate import read_snapshot, scan_motions
from supermimic.envs import SuperMimicEnv
from supermimic.evaluation.runtime import environment_config, load_policy
from supermimic.tasks import TASK_ID


class PlaybackEnv(RslRlVecEnvWrapper):
  """Reset completed lanes after the terminal transition has been produced."""

  def step(self, actions):
    obs, reward, done, info = super().step(actions)
    if bool(done.any()):
      reset_obs, _ = self.unwrapped.reset(env_ids=done.nonzero().flatten())
      obs = TensorDict(cast(Any, reset_obs), batch_size=[self.num_envs])
    return obs, reward, done, info


def main(argv: list[str] | None = None, *, assets: LocalAssets = DEFAULT_ASSETS) -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--checkpoint", type=Path, default=assets.checkpoint)
  parser.add_argument("--motion-folder", type=Path, default=assets.motion_folder)
  parser.add_argument("--run-dir", type=Path)
  parser.add_argument("--motion", help="NPZ stem; default: first motion in the folder")
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--headless", action="store_true")
  parser.add_argument("--steps", type=int, default=0, help="0: run until viewer closes")
  args = parser.parse_args(argv)
  if args.steps < 0 or (args.headless and args.steps == 0):
    parser.error("Headless playback requires a positive --steps limit.")
  checkpoint = require_path(args.checkpoint, "--checkpoint")
  directory = require_path(args.motion_folder, "--motion-folder", directory=True)
  motions = scan_motions(directory)
  if args.motion:
    motions = [m for m in motions if m["name"] == args.motion]
    if not motions:
      raise ValueError(f"Motion not found: {args.motion}")
  motions = motions[:1]
  run_dir = assets.checkpoint_run_dir(checkpoint, args.run_dir)
  method: dict = {
    "task_id": TASK_ID,
    "env_config": read_snapshot(run_dir / "params/env.yaml"),
    "agent_config": read_snapshot(run_dir / "params/agent.yaml"),
  }
  cfg = environment_config(motions, method, num_envs=1, seed=args.seed)
  env = SuperMimicEnv(cfg, device=args.device)
  try:
    policy = load_policy(env, method, checkpoint, "supermimic")
    wrapped = PlaybackEnv(env, clip_actions=method["agent_config"].get("clip_actions"))
    wrapped.reset()
    with torch.inference_mode():
      if args.headless:
        observation = wrapped.get_observations()
        for _ in range(args.steps):
          observation, _, _, _ = wrapped.step(policy(observation))
      else:
        from mjlab.viewer.native.viewer import NativeMujocoViewer

        NativeMujocoViewer(wrapped, policy).run(num_steps=args.steps or None)
    print(f"Playback complete: {motions[0]['name']}.")
  finally:
    env.close()
  return 0
