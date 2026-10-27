"""A successful short clip must not become a hard negative for reset sampling."""

from copy import deepcopy
from types import SimpleNamespace
from typing import cast

import pytest
import torch
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.utils.os import get_checkpoint_path

from supermimic.learning.adapter import TrackingAdapter
from supermimic.learning.config import SuperMimicRunnerCfg
from supermimic.learning.runner import SuperMimicRunner
from supermimic.mdp.return_sampler import (
  ReturnSampler,
  ReturnSamplerCfg,
  episode_returns,
)


def test_successful_clip_tail_is_not_harder_than_a_long_episode():
  rewards = torch.full((500, 3), 0.1)
  lengths = torch.tensor([1, 50, 500])
  # Padding can contain stale rewards, and must have no effect on scores.
  rewards[1:, 0] = float("nan")
  scores = episode_returns(rewards, lengths, torch.zeros(3, dtype=torch.bool), 0.99)
  torch.testing.assert_close(scores[0], torch.full((3,), 10.0))
  assert scores[1:, 0].count_nonzero() == 0
  sampler = ReturnSampler(
    torch.tensor([51, 51, 51]), 0.02, 0.1, ReturnSamplerCfg(ema_alpha=1.0)
  )
  sampler.observe(torch.arange(3), scores[0])
  sampler.commit()
  torch.testing.assert_close(sampler.probabilities(), torch.full((3,), 1 / 3))

  # An actual fall still has a zero-valued tail and remains a hard negative.
  failure = episode_returns(rewards, lengths, torch.ones(3, dtype=torch.bool), 0.99)
  assert failure[0, 0] == pytest.approx(0.1)
  assert failure[0, 1] == pytest.approx(10 * (1 - 0.99**50))
  sampler.observe(torch.arange(3), failure[0])
  sampler.commit()
  assert sampler.probabilities()[0] > sampler.probabilities()[1]


def test_episode_suffix_scores_match_discounted_returns():
  rewards = torch.tensor([[0.1, 0.1], [0.4, 0.4], [-0.2, -0.2]])
  gamma = 0.9
  scores = episode_returns(
    rewards, torch.tensor([3, 3]), torch.tensor([False, True]), gamma
  )
  for start in range(3):
    raw = sum(gamma ** (t - start) * rewards[t, 0] for t in range(start, 3))
    torch.testing.assert_close(scores[start, 1], raw)
    torch.testing.assert_close(scores[start, 0], raw / (1 - gamma ** (3 - start)))


def test_runner_scores_each_episode_once_at_its_reset_bin():
  sampler = ReturnSampler(
    torch.tensor([101, 101]), 0.02, 0.1, ReturnSamplerCfg(ema_alpha=1.0)
  )
  runner = SuperMimicRunner.__new__(SuperMimicRunner)
  runner.cfg = SuperMimicRunnerCfg()
  runner.env = cast(RslRlVecEnvWrapper, SimpleNamespace(num_envs=3))
  runner.device = "cpu"
  runner.adapter = cast(
    TrackingAdapter, SimpleNamespace(command=SimpleNamespace(return_sampler=sampler))
  )
  runner.episode_discounted_returns = torch.zeros(3)
  runner.episode_start_bins = torch.zeros(3, dtype=torch.long)
  runner._record_returns(
    torch.tensor([0, 0, 0]),
    torch.tensor([0, 0, 2]),
    torch.tensor([0.1, 0.05, 0.02]),
    torch.tensor([False, False, True]),
    torch.tensor([False, False, True]),
  )
  # The first two episodes cross into bin 1; the third environment resets.
  # Neither crossing nor resetting should contaminate another bin's score.
  runner._record_returns(
    torch.tensor([1, 1, 0]),
    torch.tensor([1, 1, 3]),
    torch.tensor([0.1, 0.05, 0.04]),
    torch.zeros(3, dtype=torch.bool),
    torch.zeros(3, dtype=torch.bool),
  )
  runner._record_returns(
    torch.tensor([2, 2, 1]),
    torch.tensor([1, 1, 3]),
    torch.tensor([0.1, 0.05, 0.04]),
    torch.ones(3, dtype=torch.bool),
    torch.tensor([False, True, False]),
  )
  torch.testing.assert_close(sampler.pending_count, torch.tensor([2.0, 0.0, 1.0, 1.0]))
  sampler.commit()
  failed_return = 0.05 * (1 + 0.99 + 0.99**2)
  torch.testing.assert_close(
    sampler.returns, torch.tensor([(10 + failed_return) / 2, 0.0, 0.02, 4.0])
  )


@pytest.mark.parametrize("length", [1, 50, 500])
def test_streaming_scores_match_discounted_episode_returns(length):
  sampler = ReturnSampler(
    torch.tensor([501, 501]), 0.02, 0.1, ReturnSamplerCfg(ema_alpha=1.0)
  )
  runner = SuperMimicRunner.__new__(SuperMimicRunner)
  runner.cfg = SuperMimicRunnerCfg()
  runner.adapter = cast(
    TrackingAdapter, SimpleNamespace(command=SimpleNamespace(return_sampler=sampler))
  )
  runner.episode_discounted_returns = torch.zeros(2)
  runner.episode_start_bins = torch.zeros(2, dtype=torch.long)
  rewards = torch.linspace(0.03, 0.1, length)[:, None].expand(-1, 2)
  failed = torch.tensor([False, True])
  for t in range(length):
    bins = sampler.bin_ids(torch.arange(2), torch.full((2,), t))
    runner._record_returns(
      torch.full((2,), t), bins, rewards[t], torch.full((2,), t == length - 1), failed
    )
  sampler.commit()
  expected = episode_returns(rewards, torch.full((2,), length), failed, 0.99)[0]
  torch.testing.assert_close(sampler.returns[sampler.offsets], expected)
  assert sampler.returns.count_nonzero() == 2


@pytest.mark.parametrize("ratio", [1.0, 20.0, None])
@pytest.mark.parametrize("hard_bins", [1, 25])
def test_probability_cap_preserves_mass_and_uniform_floor(ratio, hard_bins):
  sampler = ReturnSampler(
    torch.full((1000,), 51),
    0.02,
    0.1,
    ReturnSamplerCfg(max_probability_ratio=ratio),
  )
  sampler.returns.fill_(10)
  sampler.returns[:hard_bins] = -100
  sampler.history.zero_()
  sampler.history[:hard_bins] = 1 / hard_bins
  probabilities = sampler.probabilities()
  assert probabilities.sum() == pytest.approx(1)
  assert probabilities.min() >= sampler.cfg.uniform_share / len(probabilities)
  if ratio is None:
    assert probabilities[0] == pytest.approx(0.5 / 1000 + 0.5 / hard_bins)
  else:
    assert probabilities.max() <= ratio / len(probabilities) + 1e-7
    if ratio == 1:
      torch.testing.assert_close(probabilities, torch.full((1000,), 0.001))
    else:
      assert probabilities[0] > probabilities[-1]


def test_history_forgets_old_difficulty_only_when_new_episodes_arrive():
  sampler = ReturnSampler(
    torch.tensor([101]), 0.02, 0.1, ReturnSamplerCfg(ema_alpha=1.0)
  )
  sampler.observe(torch.tensor([0, 1]), torch.tensor([0.0, 10.0]))
  sampler.commit()
  hard_history = sampler.history.clone()
  assert hard_history[0] > hard_history[1]
  sampler.commit()
  torch.testing.assert_close(sampler.history, hard_history)
  for _ in range(100):
    sampler.observe(torch.tensor([0, 1]), torch.tensor([10.0, 10.0]))
    sampler.commit()
  torch.testing.assert_close(
    sampler.history, 0.5 + (hard_history - 0.5) * (1 - sampler.cfg.history_alpha) ** 100
  )
  assert sampler.history.sum() == pytest.approx(1)


@pytest.mark.parametrize("max_steps", [1, 50, 500])
def test_expected_length_matches_enumerated_reset_distribution(max_steps):
  sampler = ReturnSampler(
    torch.tensor([2, 52, 101, 2001]), 0.02, 0.1, ReturnSamplerCfg()
  )
  sampler.returns.copy_(torch.linspace(0, 10, len(sampler.returns)))
  expected = 0.0
  for probability, motion, start, end in zip(
    sampler.probabilities(),
    sampler.motion_ids,
    sampler.starts,
    sampler.ends,
    strict=True,
  ):
    available = (
      sampler.lengths[motion] - torch.arange(int(start), int(end))
    ).clamp_max(max_steps)
    expected += probability * available.float().mean()
  torch.testing.assert_close(sampler.expected_episode_length(max_steps), expected)


@pytest.mark.parametrize("estimator", [None, "timeout_discount_normalized.v1"])
def test_legacy_sampler_statistics_reset_without_resetting_rng(estimator):
  sampler = ReturnSampler(torch.tensor([101]), 0.02, 0.1, ReturnSamplerCfg(), seed=42)
  sampler.sample(20)
  legacy = deepcopy(sampler.state_dict())
  legacy.pop("return_estimator")
  if estimator is not None:
    legacy["return_estimator"] = estimator
  legacy["returns"].fill_(3)
  legacy["history"].copy_(torch.tensor([100.0, 1.0]))
  legacy["pending_count"].fill_(12)
  restored = ReturnSampler(torch.tensor([101]), 0.02, 0.1, ReturnSamplerCfg(), seed=0)
  with pytest.warns(UserWarning, match="Resetting legacy sampler"):
    restored.load_state_dict(legacy)
  assert restored.returns.count_nonzero() == 0
  assert restored.pending_count.count_nonzero() == 0
  torch.testing.assert_close(restored.probabilities(), torch.tensor([0.5, 0.5]))
  assert torch.equal(restored.generator.get_state(), legacy["rng"])
  bad = {**restored.state_dict(), "return_estimator": "unknown"}
  with pytest.raises(ValueError, match="Unsupported sampler"):
    restored.load_state_dict(bad)


def test_auto_resume_selects_model_instead_of_replay_shard(tmp_path):
  run = tmp_path / "run"
  run.mkdir()
  for name in (
    "model_2500.pt",
    "model_5000.pt",
    "model_2500.rank0.replay.pt",
    "model_5000.rank0.replay.pt",
  ):
    (run / name).touch()
  checkpoint = get_checkpoint_path(
    tmp_path, checkpoint=SuperMimicRunnerCfg().load_checkpoint
  )
  assert checkpoint.name == "model_5000.pt"
