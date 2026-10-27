"""SuperMimic Appendix D.1: real-return-based reset sampling."""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch
import torch.distributed

from .commands import MultiMotionCommand, MultiMotionCommandCfg

# PyTorch exposes distributed symbols conditionally in its type annotations.
dist: Any = torch.distributed

RETURN_ESTIMATOR = "episode_start_timeout_normalized.v2"

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


@dataclass
class ReturnSamplerCfg:
  gamma: float = 0.99
  ema_alpha: float = 0.1
  temperature: float = 25.0
  uniform_share: float = 0.5
  difficulty_share: float = 0.4
  history_share: float = 0.1
  history_alpha: float = 0.01
  # Maximum bin probability relative to uniform; None disables the cap.
  max_probability_ratio: float | None = 20.0


def episode_returns(
  rewards: torch.Tensor,
  lengths: torch.Tensor,
  terminated: torch.Tensor,
  gamma: float,
) -> torch.Tensor:
  """Return-to-go scores without treating a successful truncation as failure.

  Rewards are [time, episode]. At a failure the unobserved tail is zero. At a
  time limit or motion end, divide by the observed discount mass relative to
  an infinite horizon: constant reward r then scores r / (1 - gamma), even
  when only one frame remains. This estimates tracking quality from the
  observed suffix without bootstrapping from the learned critic.
  """
  steps = torch.arange(len(rewards), device=rewards.device)[:, None]
  valid = steps < lengths[None]
  discounts = gamma**steps
  weighted = torch.where(valid, rewards, 0.0) * discounts
  returns = weighted.flip(0).cumsum(0).flip(0) / discounts
  remaining = (lengths[None] - steps).clamp_min(1)
  discount_mass = -torch.expm1(remaining * math.log(gamma))
  returns = torch.where(terminated[None], returns, returns / discount_mass)
  return torch.where(valid, returns, 0.0)


class ReturnSampler:
  def __init__(
    self,
    lengths: torch.Tensor,
    step_dt: float,
    max_reward: float,
    cfg: ReturnSamplerCfg,
    seed: int = 0,
  ) -> None:
    if torch.any(lengths < 2):
      raise ValueError("Return sampling requires motions with at least two frames.")
    if not 0 < cfg.gamma < 1 or not 0 < cfg.ema_alpha <= 1:
      raise ValueError("Invalid sampler discount or EMA coefficient.")
    if not 0 < cfg.history_alpha <= 1:
      raise ValueError("Sampler history EMA coefficient must be in (0, 1].")
    if cfg.max_probability_ratio is not None and (
      not math.isfinite(cfg.max_probability_ratio) or cfg.max_probability_ratio < 1
    ):
      raise ValueError("Sampler probability ratio must be finite and at least one.")
    shares = (cfg.uniform_share, cfg.difficulty_share, cfg.history_share)
    if min(shares) < 0 or abs(sum(shares) - 1) > 1e-6:
      raise ValueError("Sampler mixture weights must sum to one.")
    if max_reward <= 0 or cfg.temperature <= 0:
      raise ValueError("Sampler reward upper bound and temperature must be positive.")
    self.cfg = cfg
    self.lengths = lengths.clone()
    self.steps_per_bin = round(1 / step_dt)
    if self.steps_per_bin < 1 or abs(self.steps_per_bin * step_dt - 1) > 1e-6:
      raise ValueError("A one-second sampling bin must contain whole control steps.")
    # The final frame is a timeout state, not an action-bearing reset point.
    counts = (lengths - 2) // self.steps_per_bin + 1
    self.offsets = torch.cat((counts.new_zeros(1), counts.cumsum(0)[:-1]))
    self.motion_ids = torch.repeat_interleave(
      torch.arange(len(lengths), device=lengths.device), counts
    )
    indexes = torch.arange(len(self.motion_ids), device=lengths.device)
    self.starts = (indexes - self.offsets[self.motion_ids]) * self.steps_per_bin
    self.ends = torch.minimum(
      self.starts + self.steps_per_bin, lengths[self.motion_ids] - 1
    )
    self.g_max = max_reward / (1 - cfg.gamma)
    self.returns = torch.zeros(len(indexes), device=lengths.device)
    self.history = torch.full_like(self.returns, 1 / len(indexes))
    self.pending_sum = torch.zeros_like(self.returns)
    self.pending_count = torch.zeros_like(self.returns)
    self.generator = torch.Generator(device=lengths.device).manual_seed(seed)

  def bin_ids(self, motion_ids: torch.Tensor, frames: torch.Tensor) -> torch.Tensor:
    frames = torch.minimum(frames, self.lengths[motion_ids] - 2)
    return self.offsets[motion_ids] + frames // self.steps_per_bin

  def probabilities(self) -> torch.Tensor:
    difficulty = torch.softmax(-self.cfg.temperature * self.returns / self.g_max, 0)
    probabilities = (
      self.cfg.uniform_share / len(self.returns)
      + self.cfg.difficulty_share * difficulty
      + self.cfg.history_share * self.history / self.history.sum()
    )
    count = len(probabilities)
    ratio = self.cfg.max_probability_ratio
    if ratio is not None and ratio < count:
      # Redistribute clipped mass uniformly. Reserve room for redistribution:
      # threshold + (1 - threshold) / count == ratio / count. This guarantees
      # the final cap without iterative clipping or a device synchronization.
      threshold = (ratio - 1) / (count - 1)
      clipped = probabilities.clamp_max(threshold)
      probabilities = clipped + (1 - clipped.sum()).clamp_min(0) / count
    return probabilities

  def expected_episode_length(self, max_steps: int) -> torch.Tensor:
    """Mean available steps under current reset probabilities, without failures.

    The environment evaluates the last reference frame before timing out, so
    a reset at frame f has min(max_steps, length - f) available transitions.
    Average the capped integer sequence analytically within each reset bin.
    """
    count = self.ends - self.starts
    low = self.lengths[self.motion_ids] - self.ends + 1
    high = self.lengths[self.motion_ids] - self.starts
    uncapped = (high.clamp_max(max_steps) - low + 1).clamp_min(0)
    total = (low + high.clamp_max(max_steps)) * uncapped / 2
    total += (count - uncapped) * max_steps
    return (self.probabilities() * total / count).sum()

  def sample(self, count: int) -> tuple[torch.Tensor, torch.Tensor]:
    bins = torch.multinomial(
      self.probabilities(), count, replacement=True, generator=self.generator
    )
    offsets = torch.rand(count, device=bins.device, generator=self.generator)
    frames = (
      self.starts[bins] + (offsets * (self.ends[bins] - self.starts[bins])).long()
    )
    return self.motion_ids[bins], frames

  def observe(self, bins: torch.Tensor, returns: torch.Tensor) -> None:
    valid = torch.isfinite(returns)
    self.pending_sum.scatter_add_(0, bins[valid], returns[valid])
    self.pending_count.scatter_add_(0, bins[valid], torch.ones_like(returns[valid]))

  def commit(self) -> None:
    if dist.is_initialized():
      dist.all_reduce(self.pending_sum)
      dist.all_reduce(self.pending_count)
    visited = self.pending_count > 0
    means = self.pending_sum / self.pending_count.clamp_min(1)
    self.returns[visited] = torch.lerp(
      self.returns[visited], means[visited], self.cfg.ema_alpha
    )
    # Forget obsolete difficulty while retaining recent rehearsal. No finished
    # episodes on any rank means no update, even if collect() was called again.
    difficulty = torch.softmax(-self.cfg.temperature * self.returns / self.g_max, 0)
    self.history.lerp_(difficulty, self.cfg.history_alpha * visited.any())
    self.pending_sum.zero_()
    self.pending_count.zero_()

  def state_dict(self) -> dict:
    return {
      "return_estimator": RETURN_ESTIMATOR,
      "lengths": self.lengths,
      "returns": self.returns,
      "history": self.history,
      "pending_sum": self.pending_sum,
      "pending_count": self.pending_count,
      "rng": self.generator.get_state(),
    }

  def load_state_dict(self, state: dict) -> None:
    if not torch.equal(self.lengths.cpu(), state["lengths"].cpu()):
      raise ValueError("Sampler motion lengths changed.")
    estimator = state.get("return_estimator")
    if estimator not in (None, "timeout_discount_normalized.v1", RETURN_ESTIMATOR):
      raise ValueError(f"Unsupported sampler return estimator: {estimator}.")
    if estimator != RETURN_ESTIMATOR:
      # Along-trajectory scores include accumulated tracking error and do not
      # estimate the difficulty of resetting to a bin's reference state.
      self.returns.zero_()
      self.history.fill_(1 / len(self.history))
      self.pending_sum.zero_()
      self.pending_count.zero_()
      self.generator.set_state(state["rng"].cpu())
      warnings.warn(
        "Resetting legacy sampler return/history statistics after switching to "
        "episode-start scoring; model, optimizer, replay and sampler RNG are "
        "retained.",
        stacklevel=2,
      )
      return
    for name in ("returns", "history", "pending_sum", "pending_count"):
      value = state[name].to(self.returns.device)
      if value.shape != self.returns.shape or not torch.isfinite(value).all():
        raise ValueError(f"Invalid sampler {name}.")
      getattr(self, name).copy_(value)
    self.generator.set_state(state["rng"].cpu())


class ReturnMotionCommand(MultiMotionCommand):
  def __init__(self, cfg: ReturnMotionCommandCfg, env: ManagerBasedRlEnv):
    if cfg.adaptive_sampler is not None or cfg.sampling_mode != "uniform":
      raise ValueError("Return sampling cannot be combined with other samplers.")
    super().__init__(cfg, env)
    reward_bound = sum(max(0.0, term.weight) for term in env.cfg.rewards.values())
    if env.cfg.scale_rewards_by_dt:
      reward_bound *= env.step_dt
    self.return_sampler = ReturnSampler(
      self.motion.time_step_totals,
      env.step_dt,
      reward_bound,
      cfg.return_sampler,
      int(env.cfg.seed or 0),
    )

  def _sample_reference_frames(self, env_ids: torch.Tensor) -> None:
    motion_ids, frames = self.return_sampler.sample(len(env_ids))
    self.motion_ids[env_ids] = motion_ids
    self.motion_lengths[env_ids] = self.motion.time_step_totals[motion_ids]
    self.time_steps[env_ids] = frames
    probabilities = self.return_sampler.probabilities()
    self.metrics["sampling_entropy"][env_ids] = -(
      probabilities * probabilities.log()
    ).sum() / math.log(max(len(probabilities), 2))
    self.metrics["sampling_top1_prob"][env_ids] = probabilities.max()
    self.metrics["sampling_top1_bin"][env_ids] = probabilities.argmax().float() / max(
      len(probabilities) - 1, 1
    )

  def _update_command(self, env_ids: torch.Tensor | None = None) -> None:
    # Preserve terminal state until the collector has recorded it. Reset-time
    # compute(dt=0) must also preserve the exact frame drawn by the sampler.
    if env_ids is None:
      self.time_steps.copy_(torch.minimum(self.time_steps + 1, self.motion_lengths - 1))
    self.update_relative_body_poses()


@dataclass(kw_only=True)
class ReturnMotionCommandCfg(MultiMotionCommandCfg):
  return_sampler: ReturnSamplerCfg = field(default_factory=ReturnSamplerCfg)

  def build(self, env: ManagerBasedRlEnv) -> ReturnMotionCommand:
    return ReturnMotionCommand(self, env)
