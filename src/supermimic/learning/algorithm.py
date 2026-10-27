"""Equations 1--3: autoregressive model fit and short-horizon analytic gradients."""

from __future__ import annotations

import math
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import torch
import torch.distributed
from torch import nn

from .adapter import TrackingAdapter
from .config import AdamCfg, SuperMimicRunnerCfg
from .kinematics import LinkState
from .models import TrackingActor, TrackingCritic, WorldModel
from .replay import ReplayBuffer

dist: Any = torch.distributed


@contextmanager
def frozen(*modules: nn.Module) -> Iterator[None]:
  parameters = [parameter for module in modules for parameter in module.parameters()]
  previous = [p.requires_grad for p in parameters]
  try:
    for parameter in parameters:
      parameter.requires_grad_(False)
      parameter.grad = None
    yield
  finally:
    for parameter, requires_grad in zip(parameters, previous, strict=True):
      parameter.requires_grad_(requires_grad)


def lambda_returns(
  rewards: torch.Tensor,
  continuation: torch.Tensor,
  values: torch.Tensor,
  gamma: float,
  lam: float,
  truncated: torch.Tensor | None = None,
) -> torch.Tensor:
  """Fixed Eq. 3 targets; time limits bootstrap without crossing a reset."""
  target = values[-1]
  targets = []
  for t in range(len(rewards) - 1, -1, -1):
    if truncated is not None:
      target = torch.where(truncated[t], values[t + 1], target)
    target = rewards[t] + gamma * continuation[t] * (
      (1 - lam) * values[t + 1] + lam * target
    )
    targets.append(target)
  return torch.stack(targets[::-1]).detach()


def imagined_objective(
  rewards: torch.Tensor,
  continuation: torch.Tensor,
  values: torch.Tensor,
  gamma: float,
  truncated: torch.Tensor,
) -> torch.Tensor:
  """Eq. 2 with an early value bootstrap for known episode time limits."""
  survival = torch.ones_like(rewards[0])
  objective = torch.zeros_like(survival)
  for t in range(len(rewards)):
    objective = objective + gamma**t * survival * rewards[t]
    next_survival = survival * continuation[t]
    objective = (
      objective + gamma ** (t + 1) * next_survival * truncated[t] * values[t + 1]
    )
    survival = next_survival * (~truncated[t])
  return (objective + gamma ** len(rewards) * survival * values[-1]).mean()


def optimizer_step(
  loss: torch.Tensor,
  module: nn.Module,
  optimizer: torch.optim.Adam,
  config: AdamCfg,
  elementwise_clip: float | None = None,
) -> float:
  optimizer.zero_grad(set_to_none=True)
  loss.backward()
  parameters = [p for p in module.parameters() if p.grad is not None]
  if dist.is_initialized():
    # All ranks construct the same graph, including masked terminal samples.
    for parameter in parameters:
      assert parameter.grad is not None
      dist.all_reduce(parameter.grad)
      parameter.grad.div_(dist.get_world_size())
  norm = nn.utils.clip_grad_norm_(
    parameters, config.max_grad_norm, error_if_nonfinite=True
  )
  if elementwise_clip is not None:
    nn.utils.clip_grad_value_(parameters, elementwise_clip)
  optimizer.step()
  return float(norm)


class SuperMimic:
  def __init__(self, cfg: SuperMimicRunnerCfg, adapter: TrackingAdapter, device: str):
    self.cfg = cfg
    self.adapter = adapter
    self.world_model = WorldModel(cfg, adapter).to(device)
    self.actor = TrackingActor(cfg, adapter).to(device)
    self.critic = TrackingCritic(cfg).to(device)
    self.modules = (self.world_model, self.actor, self.critic)
    self.optimizer_configs = (
      cfg.world_model.optimizer,
      cfg.imagination.optimizer,
      cfg.critic.optimizer,
    )
    self.world_size = dist.get_world_size() if dist.is_initialized() else 1
    self.optimizers = [
      torch.optim.Adam(
        module.parameters(),
        lr=opt.learning_rate * math.sqrt(self.world_size),
        betas=opt.betas,
        eps=opt.eps,
        weight_decay=opt.weight_decay,
      )
      for module, opt in zip(self.modules, self.optimizer_configs, strict=True)
    ]
    self.schedule_iteration = 0
    if dist.is_initialized():
      for module in self.modules:
        for tensor in module.state_dict().values():
          dist.broadcast(tensor, src=0)
    # Compile tensor-only helpers, leaving modules and checkpoint keys intact.
    # Default mode avoids CUDA graph output reuse across the H-step BPTT graph.
    self.compile_enabled = (
      torch.device(device).type == "cuda" if cfg.compile is None else cfg.compile
    )
    self.world_model_loss = self._world_model_loss
    self.imagination_step = self._imagination_step
    if self.compile_enabled:
      self.world_model_loss = torch.compile(
        self._world_model_loss, fullgraph=True, dynamic=False
      )
      self.imagination_step = torch.compile(
        self._imagination_step, fullgraph=True, dynamic=False
      )

  def set_iteration(self, iteration: int) -> None:
    self.schedule_iteration = iteration
    progress = min(iteration / self.cfg.decay_iterations, 1)
    factor = (
      self.cfg.final_lr_fraction
      + (1 - self.cfg.final_lr_fraction) * (1 + math.cos(math.pi * progress)) / 2
    )
    for optimizer, cfg in zip(self.optimizers, self.optimizer_configs, strict=True):
      for group in optimizer.param_groups:
        group["lr"] = cfg.learning_rate * math.sqrt(self.world_size) * factor

  @torch.no_grad()
  def update_normalizers(self, data: dict[str, torch.Tensor]) -> None:
    wm = self.world_model
    wm.in_norm.update(wm.inputs(data["state"], data["action"], data["context"]))
    wm.out_norm.update(wm.targets(data["state"], data["next_state"]))
    wm.loss_norm.update(wm.state_error(data["next_state"], data["state"]))
    self.actor.normalizer.update(
      data["obs"].index_select(-1, self.actor.proprio_indices)
    )
    # Bound FK temporary memory independently of collection/environment count.
    for start in range(0, len(data["state"]), 4096):
      end = start + 4096
      state, context = data["state"][start:end], data["context"][start:end]
      motion, frame = data["motion"][start:end], data["frame"][start:end]
      goal = self.adapter.reference(motion, frame)
      future = self.adapter.future_reference(motion, frame)
      self.actor.goal_normalizer.update(future)
      features = wm.heads.critic_input(
        state, goal, context, future, wm.heads.kinematics(state)
      )
      self.critic.normalizer.update(features)

  def _world_model_loss(
    self,
    state: torch.Tensor,
    actions: torch.Tensor,
    contexts: torch.Tensor,
    targets: torch.Tensor,
    masks: torch.Tensor,
  ) -> torch.Tensor:
    """Autoregressive k-step state loss with reset masks and full BPTT."""
    wm = self.world_model
    loss = state.new_zeros(())
    for t in range(actions.shape[0]):
      state = wm.step(state, actions[t], contexts[t])
      error = wm.state_error(state, targets[t]) / torch.maximum(
        wm.loss_norm.std, wm.error_floor
      )
      loss = loss + (error.square().mean(-1) * masks[t].float()).mean()
    return loss

  def fit_world_model(self, replay: ReplayBuffer) -> dict[str, float]:
    wm = self.world_model
    metrics = {"state": 0.0, "grad_norm": 0.0}
    for _ in range(self.cfg.world_model.batches_per_update):
      batch = replay.windows(
        self.cfg.world_model.batch_size, self.cfg.world_model.window_length
      )
      loss = self.world_model_loss(
        batch["state"][0],
        batch["action"],
        batch["context"],
        batch["next_state"],
        batch["mask"],
      )
      grad_norm = optimizer_step(
        loss, wm, self.optimizers[0], self.cfg.world_model.optimizer
      )
      metrics["state"] += float(loss.detach())
      metrics["grad_norm"] += grad_norm
    return {
      f"world_model/{key}": value / self.cfg.world_model.batches_per_update
      for key, value in metrics.items()
    }

  def _imagination_step(
    self,
    state: torch.Tensor,
    history: torch.Tensor,
    context: torch.Tensor,
    goal: torch.Tensor,
    future: torch.Tensor,
    links: LinkState,
    next_goal: torch.Tensor,
    next_future: torch.Tensor,
    noise: torch.Tensor,
  ) -> tuple[
    torch.Tensor,
    torch.Tensor,
    LinkState,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
  ]:
    """Fuse one differentiable actor/dynamics/FK/task/value step."""
    heads = self.world_model.heads
    action = self.actor(history, future)
    if self.cfg.clip_actions is not None:
      action = action.clamp(-self.cfg.clip_actions, self.cfg.clip_actions)
    next_state = self.world_model.step(state, action, context)
    next_links = heads.kinematics(next_state)
    # Rewards use the pre-action reference; observations use the next frame.
    reward, termination = self.world_model.signals(
      state, action, next_state, goal, links, next_links
    )
    actor_obs = self.world_model.observe(next_state, next_goal, context, next_links)
    next_history = torch.cat((history[:, 1:], (actor_obs + noise)[:, None]), 1)
    critic_obs = heads.critic_input(
      next_state, next_goal, context, next_future, next_links
    )
    value = self.critic(critic_obs)
    return next_state, next_history, next_links, reward, termination, critic_obs, value

  def improve_policy(self, replay: ReplayBuffer) -> dict[str, float]:
    cfg = self.cfg
    start_time = time.perf_counter()
    seed = replay.starts(cfg.imagination.batch_size)
    state, history, context = seed["state"], seed["history"], seed["context"]
    frames = seed["frame"]
    observations, rewards, continuations, values, truncated = [], [], [], [], []
    active = torch.ones_like(frames, dtype=torch.bool)
    masks = []
    with frozen(self.world_model, self.critic):
      heads = self.world_model.heads
      goal = self.adapter.reference(seed["motion"], frames)
      links = heads.kinematics(state)
      future = self.adapter.future_reference(seed["motion"], frames)
      critic_obs = heads.critic_input(state, goal, context, future, links)
      observations.append(critic_obs.detach())
      values.append(self.critic(critic_obs))
      for t in range(cfg.imagination.horizon):
        masks.append(active)
        # Real termination evaluates the reference frame before command advance.
        timeout = (seed["age"] + t + 1 >= self.adapter.env.max_episode_length) | (
          frames >= self.adapter.command.motion.time_step_totals[seed["motion"]] - 1
        )
        frames = torch.minimum(
          frames + 1, self.adapter.command.motion.time_step_totals[seed["motion"]] - 1
        )
        next_goal = self.adapter.reference(seed["motion"], frames)
        next_future = self.adapter.future_reference(seed["motion"], frames)
        # Sample outside the compiled graph to keep eager/compiled RNG identical.
        noise = self.adapter.noise_low + torch.rand_like(history[:, -1]) * (
          self.adapter.noise_high - self.adapter.noise_low
        )
        state, history, links, reward, termination, critic_obs, value = (
          self.imagination_step(
            state, history, context, goal, future, links, next_goal, next_future, noise
          )
        )
        goal, future = next_goal, next_future
        observations.append(critic_obs.detach())
        values.append(value)
        rewards.append(reward)
        continuations.append(1 - termination)
        truncated.append(timeout)
        active = active & ~timeout
      reward_tensor, continuation = torch.stack(rewards), torch.stack(continuations)
      value_tensor, timeouts = torch.stack(values), torch.stack(truncated)
      objective = imagined_objective(
        reward_tensor, continuation, value_tensor, cfg.gamma, timeouts
      )
      targets = lambda_returns(
        reward_tensor.detach(),
        continuation.detach(),
        value_tensor.detach(),
        cfg.gamma,
        cfg.lam,
        timeouts,
      )
      actor_norm = optimizer_step(
        -objective,
        self.actor,
        self.optimizers[1],
        cfg.imagination.optimizer,
        cfg.imagination.elementwise_grad_clip,
      )
    imagination_time = time.perf_counter() - start_time
    start_time = time.perf_counter()
    valid = torch.stack(masks).flatten()
    critic_inputs = torch.stack(observations[:-1]).flatten(0, 1)[valid]
    critic_targets = targets.flatten()[valid]
    critic_loss = critic_norm = 0.0
    for _ in range(cfg.critic.epochs):
      indices = torch.randperm(len(critic_inputs), device=state.device)
      # Default horizon/batch always exceeds four, including the smoke preset.
      for minibatch in indices.tensor_split(cfg.critic.minibatches):
        if not len(minibatch):
          raise ValueError("Imagined batch is too small for critic minibatches.")
        loss = (
          (self.critic(critic_inputs[minibatch]) - critic_targets[minibatch])
          .square()
          .mean()
        )
        critic_norm += optimizer_step(
          loss, self.critic, self.optimizers[2], cfg.critic.optimizer
        )
        critic_loss += float(loss.detach())
    updates = cfg.critic.epochs * cfg.critic.minibatches
    return {
      "actor/objective": float(objective.detach()),
      "actor/grad_norm": actor_norm,
      "critic/loss": critic_loss / updates,
      "critic/grad_norm": critic_norm / updates,
      "time/imagination_bptt": imagination_time,
      "time/critic": time.perf_counter() - start_time,
    }

  def state_dict(self) -> dict:
    return {
      "world_model": self.world_model.state_dict(),
      "actor": self.actor.state_dict(),
      "critic": self.critic.state_dict(),
      "optimizers": [opt.state_dict() for opt in self.optimizers],
      "schedule_iteration": self.schedule_iteration,
    }

  def load_state_dict(self, state: dict) -> None:
    for name in ("world_model", "actor", "critic"):
      getattr(self, name).load_state_dict(state[name])
    for opt, saved in zip(self.optimizers, state["optimizers"], strict=True):
      opt.load_state_dict(saved)
    self.set_iteration(state["schedule_iteration"])
