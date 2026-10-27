"""CPU/mock tests of full-motion execution; no physical environment is built."""

from __future__ import annotations

from dataclasses import asdict
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock

import pytest
import torch
import yaml

from supermimic.cli.evaluate import (
  METRICS,
  TASKS,
  SnapshotLoader,
  json_data,
  summarize,
)
from supermimic.evaluation.runtime import (
  EvaluationMotionCfg,
  EvaluationMotionCommand,
  SimulationBackend,
  Transition,
  environment_config,
  evaluate,
  full_motion_end,
  load_policy,
  load_supermimic_actor,
)
from supermimic.learning.config import SuperMimicRunnerCfg
from supermimic.learning.models import TrackingActor
from supermimic.tasks import load_env_cfg


def motions():
  return [
    {"name": f"motion{i}", "frames": n, "path": f"motion{i}.npz"}
    for i, n in enumerate([3, 5, 4, 6, 2])
  ]


class FakeBackend:
  def __init__(self, width, *, invalid=None):
    self.num_envs, self.device = width, "cpu"
    self.invalid = invalid
    self.resets = 0

  def reset_batch(self, motion_ids):
    self.ids = torch.tensor(motion_ids)
    self.expected = torch.tensor([motions()[i]["frames"] - 1 for i in motion_ids])
    self.steps = torch.zeros(self.num_envs, dtype=torch.long)
    self.pending = torch.zeros(self.num_envs, dtype=torch.bool)
    return self.steps.clone()

  def step(self, actions):
    assert not self.pending.any(), "Terminal state must be read, then reset."
    self.steps += 1
    terminated = ((self.ids == 1) & (self.steps == 2)) | (
      (self.ids == 2) & (self.steps == self.expected)
    )
    truncated = self.steps >= self.expected
    self.pending = terminated | truncated
    reward = torch.full((self.num_envs,), 0.08, dtype=torch.float64)
    errors = {key: self.steps.double() for key in METRICS}
    invalid = torch.zeros(self.num_envs, dtype=torch.bool)
    if self.invalid == "reward":
      reward[0] = float("nan")
    elif self.invalid == "metric":
      errors["mpkpe"][0] = float("inf")
    elif self.invalid == "state":
      invalid[0] = True
    elif self.invalid == "timeout":
      truncated[0] = True
    return Transition(
      self.steps.clone(),
      reward,
      terminated,
      truncated,
      {"height_failure": terminated},
      errors,
      invalid,
    )

  def reset_done(self, done):
    self.steps[done] = 0
    self.pending[done] = False
    self.resets += int(done.sum())
    return self.steps.clone()


@pytest.mark.parametrize("width", [1, 2, 4, 5])
def test_full_motion_scoring_batching_failure_and_terminal_errors(width):
  backend = FakeBackend(width)
  rows = evaluate(backend, lambda _: torch.zeros(width, 29), motions())
  assert [r["steps"] for r in rows] == [2, 2, 3, 5, 1]
  assert [r["success"] for r in rows] == [True, False, False, True, True]
  assert rows[1]["normalized_return"] == pytest.approx(0.4)
  assert rows[1]["first_failure_step"] == 2
  assert rows[1]["mean_errors"]["mpkpe"] == 1.5  # Includes terminal step.
  assert rows[2]["completion"] == 1.0 and not rows[2]["success"]
  summary = summarize(rows, motions())
  assert summary["successes"] == 3
  assert summary["normalized_return"] == pytest.approx(0.72)
  assert summary["normalized_return_framew"] == pytest.approx(13 * 0.08 / (15 * 0.1))


@pytest.mark.parametrize("invalid", ["reward", "metric", "state", "timeout", "action"])
def test_bad_transitions_are_evaluation_errors(invalid):
  backend = FakeBackend(2, invalid=invalid)

  def policy(_):
    return torch.full((2, 29), float("nan") if invalid == "action" else 0.0)

  with pytest.raises(ValueError, match="[Nn]onfinite|timeout"):
    evaluate(backend, policy, motions())


def test_deterministic_command_keeps_reset_frame_and_clamps_at_end():
  command = object.__new__(EvaluationMotionCommand)
  command.time_steps = torch.tensor([0, 0])
  command.motion_lengths = torch.tensor([2, 4])
  command.update_relative_body_poses = Mock()
  command._update_command(torch.tensor([0, 1]))
  assert command.time_steps.tolist() == [0, 0]
  for _ in range(5):
    command._update_command()
  assert command.time_steps.tolist() == [1, 3]
  command._sample_reference_frames(torch.tensor([0]))
  assert command.time_steps.tolist() == [0, 3]
  env: Any = SimpleNamespace(
    command_manager=SimpleNamespace(get_term=lambda _: command),
    episode_length_buf=torch.tensor([1, 2]),
  )
  assert full_motion_end(cast(Any, env)).tolist() == [True, False]


def snapshot(value):
  return json_data(yaml.load(yaml.dump(asdict(value)), Loader=SnapshotLoader))


@pytest.mark.parametrize("algorithm", ["supermimic"])
def test_eval_configuration_restores_inputs_and_shares_scoring_without_simulation(
  algorithm,
):
  original = load_env_cfg(TASKS[algorithm], play=False)
  saved = snapshot(original)
  saved["actions"]["joint_pos"]["scale"] = 0.25
  saved["observations"]["actor"]["history_length"] = 7
  method = {"task_id": TASKS[algorithm], "env_config": saved}
  cfg = environment_config(motions(), method, num_envs=2, seed=42)
  assert cast(Any, cfg.actions["joint_pos"]).scale == 0.25
  assert cfg.observations["actor"].history_length == 7
  assert not cfg.observations["actor"].enable_corruption
  assert not cfg.events and not cfg.curriculum and not cfg.auto_reset
  assert cfg.refresh_kinematics_before_rewards and cfg.scale_rewards_by_dt
  assert "time_out" not in cfg.terminations and "ee_body_pos" not in cfg.terminations
  assert cfg.terminations["anchor_pos"].params["threshold"] == 0.25
  assert cfg.terminations["anchor_ori"].params["threshold"] == 0.8
  assert cfg.rewards["motion_global_root_height"].weight == 0.5
  assert cfg.rewards["joint_limit"].weight == -2
  assert "self_collisions" not in cfg.rewards
  assert sum(max(0, term.weight) for term in cfg.rewards.values()) == 5
  collision = cfg.scene.entities["robot"].collisions[0]
  assert collision.contype == 0 and collision.conaffinity == 1
  assert isinstance(collision.friction, dict)
  assert list(collision.friction.values()) == [(0.6,)]
  command = cfg.commands["motion"]
  assert isinstance(command, EvaluationMotionCfg)
  assert command.pose_range == {} and command.velocity_range == {}
  assert command.joint_position_range == (0, 0)
  assert command.resolve_motion_files() == tuple(m["path"] for m in motions())
  assert snapshot(original) != saved  # Original registered configuration untouched.


def test_step_metrics_use_pre_step_reference_and_pre_reset_robot(monkeypatch):
  command = SimpleNamespace(cfg=object())
  names = ("body_pos_w", "joint_pos", "joint_vel", "body_lin_vel_w", "body_ang_vel_w")
  for name in names:
    setattr(command, name, torch.tensor([[2.0]]))
    setattr(command, f"robot_{name}", torch.tensor([[3.0]]))

  def step(_):
    for name in names:
      getattr(command, name).fill_(100)  # Reference advanced by command.compute.
      getattr(command, f"robot_{name}").fill_(5)  # True terminal state.
    return {}, torch.tensor([0.1]), torch.tensor([True]), torch.tensor([False]), {}

  env: Any = SimpleNamespace(
    num_envs=1,
    device="cpu",
    step=step,
    command_manager=SimpleNamespace(get_term=lambda _: command),
    termination_manager=SimpleNamespace(
      get_term=lambda name: torch.tensor([name == "anchor_pos"])
    ),
  )
  monkeypatch.setattr(
    "supermimic.evaluation.runtime._compute_step_metrics",
    lambda ref: {k: (ref.robot_joint_pos - ref.joint_pos).squeeze(-1) for k in METRICS},
  )
  result = SimulationBackend(cast(Any, env), None).step(torch.zeros(1, 29))
  assert result.errors["mpjpe"].item() == 3  # 5 - 2, not 5 - 100.
  assert result.terminated.item()


def test_supermimic_loads_actor_and_normalizers_only(tmp_path, monkeypatch):
  cfg = SuperMimicRunnerCfg()
  cfg.imagination.hidden_dims = (8,)
  adapter: Any = SimpleNamespace(
    proprio_features=list(range(99)), future_reference_dim=730, history_length=10
  )
  actor = TrackingActor(cfg, cast(Any, adapter))
  actor.normalizer.mean.fill_(0.25)
  actor.goal_normalizer.mean.fill_(0.5)
  actor.normalizer.count.fill_(12)
  saved = {"config": asdict(cfg), "algorithm": {"actor": actor.state_dict()}}
  loaded = load_supermimic_actor(saved, adapter, "cpu")
  assert not loaded.training
  for name, value in actor.state_dict().items():
    torch.testing.assert_close(value, loaded.state_dict()[name])
  # No world-model/critic/optimizer fields, and no replay file exists.
  path = tmp_path / "model_2.pt"
  saved.update(
    iteration=2,
    signature={
      "dataset": "synthetic",
      "observations": {"history_length": 10},
      "actor_input": "proprio_orientation_history.990_future_reference.vibed.730.v1",
    },
  )
  torch.save(saved, path)
  adapter.command = SimpleNamespace(
    motion=SimpleNamespace(dataset_fingerprint="held-out-dataset"),
    motion_ids=torch.zeros(1, dtype=torch.long),
    time_steps=torch.zeros(1, dtype=torch.long),
  )
  adapter.unpack_history = lambda obs: obs
  adapter.future_reference = lambda *_: torch.zeros(1, 730)
  monkeypatch.setattr(
    "supermimic.evaluation.runtime.TrackingAdapter", lambda _: adapter
  )
  original_load = torch.load
  reads = []

  def tracked_load(path, **kwargs):
    reads.append(str(path))
    return original_load(path, **kwargs)

  monkeypatch.setattr(torch, "load", tracked_load)
  method = {
    "agent_config": json_data(asdict(cfg)),
    "env_config": {"observations": {"actor": {"history_length": 10}}},
  }
  policy = load_policy(
    cast(Any, SimpleNamespace(device="cpu")),
    method,
    path,
    "supermimic",
  )
  with torch.no_grad():
    output = policy({"actor": torch.zeros(1, 10, 157)})
  assert output.shape == (1, 29) and reads == [str(path)]
  method["env_config"]["observations"]["actor"]["history_length"] = 9
  with pytest.raises(ValueError, match="observation configuration differ"):
    load_policy(
      cast(Any, SimpleNamespace(device="cpu")),
      method,
      path,
      "supermimic",
    )
