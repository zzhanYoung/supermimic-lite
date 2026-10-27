"""Full-motion G1 evaluation runtime for the generic eval command."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, fields, is_dataclass, replace
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Protocol, cast

import torch
from mjlab.envs import ManagerBasedRlEnv as BaseEnv
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.utils.torch import configure_torch_backends

from supermimic.cli.evaluate import METRICS, TASKS, json_data
from supermimic.compat.checkpoints import (
  canonical_config,
  canonical_symbol,
  load_checkpoint,
)
from supermimic.envs import SuperMimicEnv as ManagerBasedRlEnv
from supermimic.learning.adapter import TrackingAdapter
from supermimic.learning.config import runner_config
from supermimic.learning.models import TrackingActor
from supermimic.mdp import metrics
from supermimic.mdp.commands import MotionCommand, MotionCommandCfgBase
from supermimic.tasks import load_env_cfg


def _compute_step_metrics(command):
  return {name: getattr(metrics, "compute_" + name)(command) for name in METRICS}


def restore_config(template: Any, saved: Any) -> Any:
  """Restore known config types without evaluating YAML's Python tags."""
  if is_dataclass(template) and not isinstance(template, type):
    known = {f.name: f for f in fields(template)}
    if not isinstance(saved, dict) or set(saved) - set(known):
      raise ValueError("Saved configuration has unsupported fields.")
    return replace(
      template,
      **{
        name: restore_config(getattr(template, name), value)
        for name, value in saved.items()
        if known[name].init
      },
    )
  if isinstance(template, Enum):
    if saved["python_type"] != f"{type(template).__module__}.{type(template).__name__}":
      raise ValueError("Configuration enum type changed.")
    return type(template)(*saved["args"])
  if isinstance(template, slice):
    if saved.get("python_type") != "builtins.slice":
      raise ValueError("Expected a slice configuration.")
    return slice(*saved["args"])
  if callable(template):
    expected = canonical_config(template)
    # PyYAML records __name__, including the terrain's default spec lambda;
    # checkpoint signatures use __qualname__. Keep the known template callable.
    yaml_name = canonical_symbol(f"{template.__module__}.{template.__name__}")
    if canonical_config(saved) not in (expected, yaml_name):
      raise ValueError(f"Configuration function changed: {saved} != {expected}")
    return template
  if isinstance(template, dict):
    if not isinstance(saved, dict):
      return deepcopy(saved)  # e.g. scalar versus per-joint action scale.
    result = {}
    for key, value in saved.items():
      # Snapshot JSON normalizes integer keys to strings. Restore typed keys
      # (e.g. COM randomization axes 0/1/2) from the known configuration.
      typed_key = (
        key
        if key in template
        else next((item for item in template if str(item) == key), key)
      )
      if typed_key not in template:
        if any(is_dataclass(v) for v in template.values()):
          raise ValueError(f"Unsupported saved configuration entry: {key}")
        result[typed_key] = deepcopy(value)
      else:
        result[typed_key] = restore_config(template[typed_key], value)
    return result
  if isinstance(template, (tuple, list)):
    if any(is_dataclass(v) for v in template):
      if len(template) != len(saved):
        raise ValueError("Configuration sequence layout changed.")
      values = [restore_config(a, b) for a, b in zip(template, saved, strict=True)]
    else:
      values = deepcopy(saved)
    return tuple(values) if isinstance(template, tuple) else values
  return deepcopy(saved)


class EvaluationMotionCommand(MotionCommand):
  """Fixed lane assignments, exact reset frame, no wraparound or learning."""

  def _sample_reference_frames(self, env_ids: torch.Tensor) -> None:
    self.time_steps[env_ids] = 0

  def _update_command(self, env_ids: torch.Tensor | None = None) -> None:
    if env_ids is None:
      self.time_steps.copy_(torch.minimum(self.time_steps + 1, self.motion_lengths - 1))
    self.update_relative_body_poses()

  def assign(self, motion_ids: list[int]) -> None:
    self.motion_ids.copy_(torch.tensor(motion_ids, device=self.device))
    self.motion_lengths.copy_(self.motion.time_step_totals[self.motion_ids])


@dataclass(kw_only=True)
class EvaluationMotionCfg(MotionCommandCfgBase):
  motion_files: tuple[str, ...] = ()

  def resolve_motion_files(self) -> tuple[str, ...]:
    return self.motion_files

  def build(self, env: BaseEnv) -> EvaluationMotionCommand:
    return EvaluationMotionCommand(self, env)


def full_motion_end(env: ManagerBasedRlEnv) -> torch.Tensor:
  command = cast(EvaluationMotionCommand, env.command_manager.get_term("motion"))
  return env.episode_length_buf >= command.motion_lengths - 1


def environment_config(motions: list[dict], method: dict, *, num_envs: int, seed: int):
  cfg = load_env_cfg(method["task_id"], play=False)
  saved = method["env_config"]
  # Restore the policy's sensor history and action mapping. Shared scoring and
  # physical conditions are applied only after its input contract is restored.
  cfg.observations = restore_config(cfg.observations, saved["observations"])
  cfg.actions = restore_config(cfg.actions, saved["actions"])
  robot = cfg.scene.entities["robot"]
  saved_robot = saved["scene"]["entities"]["robot"]
  cfg.scene.entities["robot"] = restore_config(robot, saved_robot)
  expected_dt = saved["sim"]["mujoco"]["timestep"] * saved["decimation"]
  if abs(expected_dt - 0.02) > 1e-9:
    raise ValueError("Checkpoint control frequency is not 50 Hz.")
  canonical = load_env_cfg(TASKS["supermimic"], play=False)
  cfg.rewards = deepcopy(canonical.rewards)
  cfg.terminations = deepcopy(canonical.terminations)
  cfg.terminations.pop("time_out", None)
  cfg.terminations["motion_end"] = TerminationTermCfg(
    func=full_motion_end, time_out=True
  )
  cfg.scene.entities["robot"].collisions = deepcopy(
    canonical.scene.entities["robot"].collisions
  )
  cfg.sim = deepcopy(canonical.sim)
  cfg.decimation = canonical.decimation
  cfg.refresh_kinematics_before_rewards = True
  cfg.scale_rewards_by_dt = True
  cfg.auto_reset = False
  cfg.seed = seed
  cfg.scene.num_envs = num_envs
  cfg.events = {}
  cfg.curriculum = {}
  for group in cfg.observations.values():
    group.enable_corruption = False
  original = cfg.commands["motion"]
  saved_motion = saved["commands"]["motion"]
  for field in ("body_names", "anchor_body_name", "entity_name"):
    if json_data(getattr(original, field)) != saved_motion[field]:
      raise ValueError(f"Motion input layout changed: {field}")
  command = EvaluationMotionCfg(
    **{
      f.name: deepcopy(getattr(original, f.name)) for f in fields(MotionCommandCfgBase)
    }
  )
  command.motion_files = tuple(m["path"] for m in motions)
  command.pose_range = {}
  command.velocity_range = {}
  command.joint_position_range = (0.0, 0.0)
  command.sampling_mode = "start"
  command.debug_vis = False
  command.resampling_time_range = (1e9, 1e9)
  cfg.commands = {"motion": command}
  cfg.episode_length_s = max(m["frames"] for m in motions) * 0.02 + 1
  return cfg


def load_supermimic_actor(checkpoint: dict, adapter: Any, device: str):
  actor = TrackingActor(runner_config(checkpoint["config"]), adapter)
  indices = actor.proprio_indices.clone()
  actor.load_state_dict(checkpoint["algorithm"]["actor"], strict=True)
  if not torch.equal(actor.proprio_indices, indices):
    raise ValueError("SuperMimic proprioceptive feature order changed.")
  return actor.to(device).eval()


def load_policy(env: ManagerBasedRlEnv, method: dict, checkpoint: Path, algorithm: str):
  device = env.device
  if algorithm != "supermimic":
    raise ValueError("Only SuperMimic policies are supported.")
  saved = load_checkpoint(checkpoint)
  adapter = TrackingAdapter(env)
  signature = saved["signature"]
  saved_observations = canonical_config(signature["observations"])
  if saved_observations != canonical_config(
    method["env_config"]["observations"]["actor"]
  ):
    raise ValueError("Checkpoint and saved actor observation configuration differ.")
  for key in ("activation", "clip_actions"):
    if saved["config"][key] != method["agent_config"][key]:
      raise ValueError(f"Checkpoint and saved agent configuration differ: {key}")
  for key in ("hidden_dims", "latent_dim"):
    if json_data(saved["config"]["imagination"][key]) != json_data(
      method["agent_config"]["imagination"][key]
    ):
      raise ValueError(f"Checkpoint actor architecture changed: {key}")
  # Inference may use held-out motions; the training dataset is not a constraint.
  if (
    signature.get("actor_input")
    != "proprio_orientation_history.990_future_reference.vibed.730.v1"
  ):
    raise ValueError("Unsupported SuperMimic actor input contract.")
  # The inference path deliberately does not construct SuperMimicRunner or read
  # its replay shards, world model, critic, optimizers or sampler state.
  model = load_supermimic_actor(saved, adapter, device)

  def policy(obs):
    return model(
      adapter.unpack_history(obs["actor"]),
      adapter.future_reference(adapter.command.motion_ids, adapter.command.time_steps),
    )

  return policy


class SimulationBackend:
  def __init__(self, env: ManagerBasedRlEnv, clip_actions: float | None):
    self.env = env
    self.command = cast(EvaluationMotionCommand, env.command_manager.get_term("motion"))
    self.num_envs, self.device = env.num_envs, env.device
    self.clip_actions = clip_actions

  def reset_batch(self, motion_ids: list[int]):
    self.command.assign(motion_ids)
    obs, _ = self.env.reset(seed=self.env.cfg.seed)
    if bool(self.command.time_steps.any()):
      raise ValueError("Evaluation reset skipped frame zero.")
    return obs

  def reset_done(self, done: torch.Tensor):
    obs, _ = self.env.reset(env_ids=done.nonzero().flatten())
    return obs

  def step(self, actions: torch.Tensor) -> Transition:
    command = self.command
    ref = SimpleNamespace(num_envs=self.num_envs, device=self.device, cfg=command.cfg)
    for name in (
      "body_pos_w",
      "joint_pos",
      "joint_vel",
      "body_lin_vel_w",
      "body_ang_vel_w",
    ):
      setattr(ref, name, getattr(command, name).clone())
    if self.clip_actions is not None:
      actions = actions.clamp(-self.clip_actions, self.clip_actions)
    obs, reward, terminated, truncated, _ = self.env.step(actions)
    for name in (
      "body_pos_w",
      "joint_pos",
      "joint_vel",
      "body_lin_vel_w",
      "body_ang_vel_w",
    ):
      setattr(ref, f"robot_{name}", getattr(command, f"robot_{name}"))
    manager = self.env.termination_manager
    return Transition(
      obs,
      reward,
      terminated.clone(),
      truncated.clone(),
      {
        "height_failure": manager.get_term("anchor_pos").clone(),
        "tilt_failure": manager.get_term("anchor_ori").clone(),
      },
      _compute_step_metrics(cast(MotionCommand, ref)),
      manager.get_term("nan_detection").clone(),
    )


def evaluate_folder(
  motions: list[dict],
  method: dict,
  checkpoint: Path,
  algorithm: str,
  *,
  num_envs: int,
  seed: int,
  device: str,
  on_motion,
):
  configure_torch_backends()
  env = ManagerBasedRlEnv(
    environment_config(motions, method, num_envs=num_envs, seed=seed), device=device
  )
  try:
    policy = load_policy(env, method, checkpoint, algorithm)
    backend = SimulationBackend(env, method["agent_config"].get("clip_actions"))
    return evaluate(backend, policy, motions, on_motion)
  finally:
    env.close()


@dataclass
class Transition:
  observation: Any
  reward: torch.Tensor
  terminated: torch.Tensor
  truncated: torch.Tensor
  failures: dict[str, torch.Tensor]
  errors: dict[str, torch.Tensor]
  invalid_state: torch.Tensor


class Backend(Protocol):
  num_envs: int
  device: str

  def reset_batch(self, motion_ids: list[int]) -> Any: ...
  def step(self, actions: torch.Tensor) -> Transition: ...
  def reset_done(self, done: torch.Tensor) -> Any: ...


@torch.no_grad()
def evaluate(
  backend: Backend,
  policy: Callable,
  motions: list[dict],
  on_motion: Callable[[dict], None] | None = None,
) -> list[dict]:
  rows = []
  width = backend.num_envs
  if width < 1:
    raise ValueError("Evaluation needs at least one environment.")
  for start in range(0, len(motions), width):
    count = min(width, len(motions) - start)
    ids = [min(start + i, len(motions) - 1) for i in range(width)]
    obs = backend.reset_batch(ids)
    expected = torch.tensor(
      [motions[i]["frames"] - 1 for i in ids], device=backend.device
    )
    if (expected < 1).any():
      raise ValueError("Motions need at least two frames.")
    active = torch.arange(width, device=backend.device) < count
    steps = torch.zeros_like(expected)
    rewards = torch.zeros(width, dtype=torch.float64, device=backend.device)
    errors = {key: torch.zeros_like(rewards) for key in METRICS}
    while bool(active.any()):
      actions = policy(obs)
      if actions.shape != (width, 29) or not torch.isfinite(actions[active]).all():
        raise ValueError("Invalid/nonfinite policy actions.")
      actions = torch.where(active[:, None], actions, 0.0)
      result = backend.step(actions)
      if bool(result.invalid_state[active].any()):
        raise ValueError("Nonfinite simulator state (nan_detection).")
      for value in (result.reward, *result.errors.values()):
        if not torch.isfinite(value[active]).all():
          raise ValueError("Nonfinite active reward or tracking error.")
      if set(result.errors) != set(METRICS):
        raise ValueError("Incomplete tracking metrics.")
      steps += active.long()
      rewards += torch.where(active, result.reward, 0.0).double()
      for key in METRICS:
        errors[key] += torch.where(active, result.errors[key], 0.0).double()
      # A horizon cutoff is valid only when the full reference was executed.
      early_cutoff = active & result.truncated & ~result.terminated & (steps < expected)
      if bool(early_cutoff.any()):
        raise ValueError("Unexpected timeout before full motion completion.")
      finished = active & (result.terminated | (steps >= expected))
      for lane in finished.nonzero().flatten().tolist():
        failed = bool(result.terminated[lane])
        reasons = [key for key, value in result.failures.items() if bool(value[lane])]
        if failed and not reasons:
          reasons = ["physical_failure"]
        if reasons and not failed:
          raise ValueError("Failure threshold crossed without termination.")
        length, completed = int(expected[lane]), int(steps[lane])
        total = float(rewards[lane])
        row = {
          "motion": motions[ids[lane]]["name"],
          "status": "complete",
          "steps": completed,
          "expected_steps": length,
          "completion": completed / length,
          "success": not failed,
          "return": total,
          "normalized_return": total / (0.1 * length),
          "failure_reason": "+".join(reasons) or None,
          "first_failure_step": completed if failed else None,
          "mean_errors": {
            key: float(value[lane]) / completed for key, value in errors.items()
          },
        }
        rows.append(row)
        if on_motion is not None:
          on_motion(row)
      active &= ~finished
      obs = result.observation
      done = result.terminated | result.truncated
      # Manual reset happens AFTER scoring the true terminal state. Completed
      # lanes may keep simulating, but the active mask can never become true again.
      if bool(active.any()) and bool(done.any()):
        obs = backend.reset_done(done)
  order = {m["name"]: i for i, m in enumerate(motions)}
  return sorted(rows, key=lambda row: order[row["motion"]])
