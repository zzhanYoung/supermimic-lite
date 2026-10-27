"""Deterministic dynamics, task heads and goal-conditioned actor/critic."""

from __future__ import annotations

from typing import Any

import torch
import torch.distributed
from torch import nn

from .adapter import TrackingAdapter
from .config import SuperMimicRunnerCfg
from .dynamics import (
  integrate_delta,
  normalize_context,
  state_delta,
)
from .heads import AnalyticHeads
from .kinematics import LinkState

dist: Any = torch.distributed


def mlp(
  inputs: int,
  hidden: tuple[int, ...],
  outputs: int,
  activation: str,
  *,
  layer_norm: bool = False,
) -> nn.Sequential:
  activations = {"elu": nn.ELU, "silu": nn.SiLU, "tanh": nn.Tanh}
  layers: list[nn.Module] = []
  for width in hidden:
    layers.append(nn.Linear(inputs, width))
    if layer_norm:
      layers.append(nn.LayerNorm(width))
    layers.append(activations[activation]())
    inputs = width
  layers.append(nn.Linear(inputs, outputs))
  return nn.Sequential(*layers)


class RunningNorm(nn.Module):
  """Parallel Welford moments; forward never changes statistics or detaches inputs."""

  mean: torch.Tensor
  variance: torch.Tensor
  count: torch.Tensor

  def __init__(
    self, size: int, *, eps: float | None = None, max_count: float | None = None
  ):
    super().__init__()
    self.eps = eps
    self.max_count = max_count
    self.register_buffer("mean", torch.zeros(size))
    self.register_buffer("variance", torch.ones(size))
    self.register_buffer("count", torch.zeros(()))

  @property
  def std(self) -> torch.Tensor:
    if self.eps is not None:
      return (self.variance + self.eps**2).sqrt()
    return self.variance.clamp_min(1e-4).sqrt()

  @torch.no_grad()
  def update(self, values: torch.Tensor) -> None:
    if self.max_count is not None and self.count >= self.max_count:
      return
    values = values.detach().reshape(-1, self.mean.numel())
    values = values[torch.isfinite(values).all(-1)]
    stats = torch.cat(
      (values.sum(0), values.square().sum(0), values.new_tensor([len(values)]))
    )
    if dist.is_initialized():
      dist.all_reduce(stats)
    n = stats[-1]
    if n == 0:
      return
    batch_mean = stats[: self.mean.numel()] / n
    batch_var = (stats[self.mean.numel() : -1] / n - batch_mean.square()).clamp_min(0)
    total = self.count + n
    delta = batch_mean - self.mean
    variance = (
      self.variance * self.count
      + batch_var * n
      + delta.square() * self.count * n / total
    ) / total
    self.mean.add_(delta * n / total)
    self.variance.copy_(variance)
    self.count.copy_(total)

  def forward(self, values: torch.Tensor) -> torch.Tensor:
    return (values - self.mean) / self.std

  def denormalize(self, values: torch.Tensor) -> torch.Tensor:
    return values * self.std + self.mean


class TrackingActor(nn.Module):
  """Decode future references with proprioceptive and relative-orientation history."""

  proprio_indices: torch.Tensor

  def __init__(self, cfg: SuperMimicRunnerCfg, adapter: TrackingAdapter):
    super().__init__()
    self.register_buffer("proprio_indices", torch.tensor(adapter.proprio_features))
    self.normalizer = RunningNorm(len(adapter.proprio_features))
    self.goal_normalizer = RunningNorm(adapter.future_reference_dim)
    self.encoder = mlp(
      adapter.future_reference_dim,
      cfg.imagination.hidden_dims,
      cfg.imagination.latent_dim,
      cfg.activation,
    )
    self.decoder = mlp(
      len(adapter.proprio_features) * adapter.history_length
      + cfg.imagination.latent_dim,
      cfg.imagination.hidden_dims,
      29,
      cfg.activation,
    )

  def forward(self, history: torch.Tensor, future: torch.Tensor) -> torch.Tensor:
    proprio = self.normalizer(history.index_select(-1, self.proprio_indices)).flatten(1)
    latent = self.encoder(self.goal_normalizer(future)).tanh()
    return self.decoder(torch.cat((latent, proprio), -1))


class TrackingCritic(nn.Module):
  def __init__(self, cfg: SuperMimicRunnerCfg):
    super().__init__()
    self.normalizer = RunningNorm(TrackingAdapter.critic_input_dim)
    self.network = mlp(
      TrackingAdapter.critic_input_dim, cfg.critic.hidden_dims, 1, cfg.activation
    )

  def forward(self, observation: torch.Tensor) -> torch.Tensor:
    return self.network(self.normalizer(observation)).squeeze(-1)


class WorldModel(nn.Module):
  """Raw physical-state residuals with independently scaled geometric loss."""

  error_floor: torch.Tensor

  def __init__(self, cfg: SuperMimicRunnerCfg, adapter: TrackingAdapter):
    super().__init__()
    self.adapter = adapter
    self.in_norm = RunningNorm(133, eps=1e-2, max_count=2e8)
    self.out_norm = RunningNorm(71, eps=1e-4, max_count=2e8)
    self.loss_norm = RunningNorm(70, eps=1e-4, max_count=2e8)
    self.register_buffer(
      "error_floor",
      torch.cat(
        [
          torch.full((size,), scale)
          for size, scale in zip(
            (3, 3, 3, 3, 29, 29),
            (3e-3, 5e-3, 2e-2, 5e-2, 3e-3, 5e-2),
            strict=True,
          )
        ]
      ),
    )
    self.heads = AnalyticHeads(adapter)
    wm = cfg.world_model
    self.dynamics = mlp(
      133,
      wm.hidden_dims,
      71,
      "elu",
      layer_norm=True,
    )
    # Zero output predicts the mean raw increment once statistics are fitted.
    final = self.dynamics[-1]
    assert isinstance(final, nn.Linear)
    nn.init.zeros_(final.weight)
    nn.init.zeros_(final.bias)

  @staticmethod
  def inputs(
    state: torch.Tensor, action: torch.Tensor, context: torch.Tensor
  ) -> torch.Tensor:
    return torch.cat((state[..., :71], action, normalize_context(context)), -1)

  @staticmethod
  def targets(state: torch.Tensor, next_state: torch.Tensor) -> torch.Tensor:
    return next_state[..., :71] - state[..., :71]

  @staticmethod
  def state_error(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return state_delta(target, predicted)

  def step(
    self, state: torch.Tensor, action: torch.Tensor, context: torch.Tensor
  ) -> torch.Tensor:
    inputs = self.in_norm(self.inputs(state, action, context))
    delta = self.out_norm.denormalize(self.dynamics(inputs))
    return integrate_delta(state, delta, action)

  def observe(
    self,
    state: torch.Tensor,
    goal: torch.Tensor,
    context: torch.Tensor,
    links: LinkState,
  ) -> torch.Tensor:
    return self.heads.observation(state, goal, context, links)

  def signals(
    self,
    state: torch.Tensor,
    action: torch.Tensor,
    next_state: torch.Tensor,
    goal: torch.Tensor,
    before: LinkState,
    after: LinkState,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    terms = self.heads.reward_terms(state, action, next_state, goal, before, after)
    return torch.stack(list(terms.values())).sum(0), self.heads.soft_termination(
      next_state, goal
    )
