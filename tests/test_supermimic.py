"""Algorithm contracts independent of simulator throughput or GPU availability."""

from copy import deepcopy
from dataclasses import asdict

import pytest
import torch
from torch import nn

from supermimic.learning.algorithm import (
  frozen,
  imagined_objective,
  lambda_returns,
  optimizer_step,
)
from supermimic.learning.config import AdamCfg, SuperMimicRunnerCfg
from supermimic.learning.models import RunningNorm
from supermimic.learning.replay import ReplayBuffer
from supermimic.mdp.return_sampler import ReturnSampler, ReturnSamplerCfg
from supermimic.tasks import load_env_cfg, load_rl_cfg

TASK = "Mjlab-Tracking-Flat-Unitree-G1-SuperMimic"


@pytest.mark.parametrize("play", [False, True])
def test_supermimic_task_preserves_general_tracker_contract(play: bool) -> None:
  from supermimic.tasks.g1.env_cfgs import unitree_g1_flat_tracking_env_cfg

  original = unitree_g1_flat_tracking_env_cfg(multi_motion=not play, play=play)
  if play:
    from supermimic.tasks.tracking_env_cfg import (
      configure_multi_motion_tracking_profile,
    )

    original = configure_multi_motion_tracking_profile(original)
  cfg = load_env_cfg(TASK, play=play)
  for name in ("observations", "actions"):
    assert {key: asdict(value) for key, value in getattr(cfg, name).items()} == {
      key: asdict(value) for key, value in getattr(original, name).items()
    }
  expected_rewards = {key: asdict(value) for key, value in original.rewards.items()}
  expected_rewards.pop("self_collisions")
  expected_rewards["joint_limit"]["weight"] = -2.0
  expected_rewards["motion_global_root_height"]["weight"] = 0.5
  assert {key: asdict(value) for key, value in cfg.rewards.items()} == expected_rewards
  assert "push_robot" not in cfg.events
  if not play:
    assert "push_robot" in original.events
    assert not cfg.auto_reset
    assert cfg.scene.num_envs == 4096
  agent = load_rl_cfg(TASK)
  assert isinstance(agent, SuperMimicRunnerCfg)
  agent.validate()
  assert agent.max_iterations == 320_000
  assert agent.world_model.window_length == 8
  assert agent.imagination.horizon == 32
  assert agent.imagination.optimizer.betas == (0.9, 0.95)
  assert agent.num_steps_per_env == 0
  assert agent.update_to_data_ratio == 0.4
  assert cfg.refresh_kinematics_before_rewards
  assert not original.refresh_kinematics_before_rewards


def test_return_sampler_mixture_tails_and_rehearsal() -> None:
  sampler = ReturnSampler(torch.tensor([52, 101]), 0.02, 0.092, ReturnSamplerCfg())
  assert sampler.starts.tolist() == [0, 50, 0, 50]
  assert sampler.ends.tolist() == [50, 51, 50, 100]
  assert sampler.g_max == pytest.approx(9.2)
  torch.testing.assert_close(sampler.probabilities(), torch.full((4,), 0.25))
  sampler.returns[:] = torch.tensor([0.0, 9.2, 9.2, 9.2])
  sampler.observe(torch.tensor([0]), torch.tensor([0.0]))
  sampler.commit()
  assert sampler.probabilities()[0] > sampler.probabilities()[1]
  sampler.returns.fill_(9.2)
  probabilities = sampler.probabilities()
  assert probabilities[0] > probabilities[1]  # Previously hard bin is rehearsed.
  assert torch.all(probabilities >= 0.5 / 4)
  motions, frames = sampler.sample(2000)
  assert torch.all(frames < sampler.lengths[motions] - 1)
  assert torch.any((motions == 0) & (frames == 50))
  sampler.observe(torch.tensor([0, 0]), torch.tensor([1.0, 3.0]))
  sampler.commit()
  assert sampler.returns[0] == pytest.approx(0.9 * 9.2 + 0.1 * 2.0)


def test_return_sampler_resume_reproduces_sampling() -> None:
  first = ReturnSampler(torch.tensor([500, 51]), 0.02, 0.092, ReturnSamplerCfg(), 42)
  first.observe(torch.tensor([0, 1]), torch.tensor([1.0, 9.0]))
  first.commit()
  first.observe(torch.tensor([1]), torch.tensor([7.0]))
  first.sample(9)
  state = deepcopy(first.state_dict())
  second = ReturnSampler(torch.tensor([500, 51]), 0.02, 0.092, ReturnSamplerCfg(), 1)
  second.load_state_dict(state)
  first.commit()
  second.commit()
  torch.testing.assert_close(first.returns, second.returns)
  torch.testing.assert_close(first.history, second.history)
  for a, b in zip(first.sample(30), second.sample(30), strict=True):
    assert torch.equal(a, b)


def _append(replay: ReplayBuffer, episode: int, age: int, done: bool = False) -> None:
  replay.append(
    {
      "state": torch.tensor([[episode * 10.0 + age]]),
      "obs": torch.tensor([[episode * 10.0 + age]]),
      "episode": torch.tensor([episode]),
      "age": torch.tensor([age]),
      "terminated": torch.tensor([done]),
      "truncated": torch.tensor([False]),
    }
  )


def test_replay_windows_keep_terminal_and_mask_reset(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  replay = ReplayBuffer(10, 1, "cpu")
  _append(replay, 0, 0)
  _append(replay, 0, 1, True)
  _append(replay, 1, 0)
  monkeypatch.setattr(replay, "_draw", lambda _: (torch.tensor([0]), torch.tensor([0])))
  batch = replay.windows(1, 8)
  assert batch["mask"][:, 0].tolist() == [
    True,
    True,
    False,
    False,
    False,
    False,
    False,
    False,
  ]
  assert batch["terminated"][1, 0]
  assert torch.isfinite(batch["state"]).all()


def test_replay_history_reset_backfill_fifo_and_resume() -> None:
  replay = ReplayBuffer(10, 1, "cpu")
  for episode in range(6):
    for age in range(3):
      _append(replay, episode, age, age == 2)
  assert replay.size == 10
  seeds = replay.starts(50, 4)
  for history, episode, age in zip(
    seeds["history"], seeds["episode"], seeds["age"], strict=True
  ):
    expected = episode * 10 + (age - torch.arange(3, -1, -1)).clamp_min(0)
    torch.testing.assert_close(history[:, 0], expected.float())
  restored = ReplayBuffer(10, 1, "cpu")
  restored.load_state_dict(replay.state_dict())
  assert restored.total_steps == replay.total_steps
  torch.manual_seed(32)
  first = replay.starts(20, 4)
  torch.manual_seed(32)
  second = restored.starts(20, 4)
  for key in first:
    assert torch.equal(first[key], second[key])


def test_lambda_return_failure_and_timeout_bootstraps() -> None:
  reward = torch.tensor([[1.0], [2.0]])
  value = torch.tensor([[10.0], [20.0], [30.0]], requires_grad=True)
  failure = lambda_returns(reward, torch.tensor([[1.0], [0.0]]), value, 0.9, 1.0)
  torch.testing.assert_close(failure[:, 0], torch.tensor([2.8, 2.0]))
  timeout = torch.tensor([[True], [False]])
  target = lambda_returns(reward, torch.ones_like(reward), value, 0.9, 0.95, timeout)
  assert target[0] == 19
  assert not target.requires_grad
  objective = imagined_objective(reward, torch.ones_like(reward), value, 0.9, timeout)
  assert objective == 19


def test_bptt_matches_finite_difference_through_frozen_model_and_critic() -> None:
  dynamics, critic = (
    nn.Linear(1, 1, bias=False).double(),
    nn.Linear(1, 1, bias=False).double(),
  )
  with torch.no_grad():
    dynamics.weight.fill_(0.8)
    critic.weight.fill_(1.5)
  theta = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)
  saved = [p.clone() for net in (dynamics, critic) for p in net.parameters()]

  def rollout(parameter: torch.Tensor) -> torch.Tensor:
    state = torch.ones(1, 1, dtype=torch.float64)
    rewards, continuation, values = [], [], [critic(state).squeeze(-1)]
    for _ in range(3):
      action = parameter * state
      state = dynamics(state) + action
      rewards.append(-(state.square() + action.square()).squeeze(-1))
      continuation.append(torch.sigmoid(state.squeeze(-1)))
      values.append(critic(state).squeeze(-1))
    return imagined_objective(
      torch.stack(rewards),
      torch.stack(continuation),
      torch.stack(values),
      0.99,
      torch.zeros(3, 1, dtype=torch.bool),
    )

  with frozen(dynamics, critic):
    rollout(theta).backward()
    eps = 1e-6
    numerical = (rollout(theta.detach() + eps) - rollout(theta.detach() - eps)) / (
      2 * eps
    )
  torch.testing.assert_close(theta.grad, numerical, atol=1e-7, rtol=1e-7)
  for before, after in zip(
    saved, [p for net in (dynamics, critic) for p in net.parameters()], strict=True
  ):
    assert torch.equal(before, after)
    assert after.grad is None


def test_actor_gradient_clips_global_norm_before_elements() -> None:
  module = nn.Linear(2, 1, bias=False)
  optimizer = torch.optim.Adam(module.parameters(), lr=2e-4)
  optimizer_step(
    module(torch.tensor([[10.0, 0.2]])).sum(), module, optimizer, AdamCfg(), 0.1
  )
  torch.testing.assert_close(
    module.weight.grad, torch.tensor([[0.1, 0.2 / (100.04**0.5)]])
  )


def test_normalizer_forward_is_fixed_and_differentiable() -> None:
  normalizer = RunningNorm(2)
  normalizer.update(torch.tensor([[1.0, 3.0], [3.0, 7.0]]))
  count = normalizer.count.clone()
  inputs = torch.tensor([[3.0, 7.0]], requires_grad=True)
  normalizer(inputs).sum().backward()
  torch.testing.assert_close(inputs.grad, torch.tensor([[1.0, 0.5]]))
  assert normalizer.count == count
