"""Simulator collection, model fitting and imagined actor--critic training."""

from __future__ import annotations

import math
import os
import random
import time
from collections import deque
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed
from mjlab.rl.vecenv_wrapper import RslRlVecEnvWrapper
from rsl_rl.utils.logger import Logger
from tensordict import TensorDict

from supermimic.compat.checkpoints import FORMAT_VERSION, load_checkpoint
from supermimic.envs import SuperMimicEnv
from supermimic.mdp.return_sampler import (
  ReturnMotionCommand,
  ReturnSamplerCfg,
)

from .adapter import TrackingAdapter
from .algorithm import SuperMimic
from .config import runner_config
from .replay import ReplayBuffer

dist: Any = torch.distributed


class SuperMimicRunner:
  def __init__(
    self,
    env: RslRlVecEnvWrapper,
    train_cfg: dict,
    log_dir: str | None = None,
    device: str = "cpu",
    registry_name: str | None = None,
  ) -> None:
    del registry_name
    self.cfg = runner_config(train_cfg)
    self.env, self.device, self.log_dir = env, device, log_dir
    if env.cfg.auto_reset:
      raise ValueError("SuperMimic collection requires auto_reset=False.")
    if "push_robot" in env.cfg.events:
      raise ValueError("The deterministic SuperMimic model requires pushes disabled.")
    self.rank = int(os.environ.get("RANK", "0"))
    self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if self.world_size not in (1, 2):
      raise ValueError("The paper presets support one or two training ranks.")
    if self.world_size > 1 and not dist.is_initialized():
      if torch.device(device).type == "cuda":
        torch.cuda.set_device(torch.device(device))
      dist.init_process_group(backend="nccl" if "cuda" in device else "gloo")
    command = env.unwrapped.command_manager.get_term("motion")
    if not isinstance(command, ReturnMotionCommand):
      raise ValueError("SuperMimic requires ReturnMotionCommand.")
    if command.return_sampler.cfg.gamma != self.cfg.gamma:
      raise ValueError("Sampler and learner discounts must agree.")
    if not isinstance(env.unwrapped, SuperMimicEnv):
      raise ValueError("SuperMimic requires its standalone environment subclass.")
    self.adapter = TrackingAdapter(env.unwrapped)
    self.signature = self.adapter.signature()
    if dist.is_initialized():
      signatures: list = [None] * self.world_size
      dist.all_gather_object(signatures, self.signature)
      if any(signature != self.signature for signature in signatures):
        raise ValueError("Distributed task or dataset signatures differ.")
    self.alg = SuperMimic(self.cfg, self.adapter, device)
    self.replay = ReplayBuffer(self.cfg.replay_steps, env.num_envs, device)
    logger_cfg = {**asdict(self.cfg), "algorithm": {}}
    self.logger = Logger(
      log_dir,
      logger_cfg,
      env.cfg,
      env.num_envs,
      self.world_size > 1,
      self.world_size,
      self.rank,
      device,
    )
    self.current_learning_iteration = 0
    self.real_steps = 0
    self.collection_remainder = 0.0
    self.initial_collection_done = False
    self.optimizer_updates = {"world_model": 0, "actor": 0, "critic": 0}
    self.episode_ids = torch.zeros(env.num_envs, dtype=torch.long, device=device)
    self.episode_discounted_returns = torch.zeros(env.num_envs, device=device)
    self.episode_start_bins = torch.zeros(env.num_envs, dtype=torch.long, device=device)
    self.observations = env.get_observations()
    self.last_metrics: dict[str, float] = {}
    self.episode_metrics: dict[str, deque[float]] = {
      name: deque(maxlen=100)
      for name in ("failure_fraction", "completion_fraction", "available_length")
    }

  def add_git_repo_to_log(self, path: str) -> None:
    self.logger.git_status_repos.append(path)

  @torch.no_grad()
  def _record_returns(
    self,
    age: torch.Tensor,
    bins: torch.Tensor,
    rewards: torch.Tensor,
    done: torch.Tensor,
    terminated: torch.Tensor,
  ) -> None:
    starting = age == 0
    self.episode_start_bins.copy_(torch.where(starting, bins, self.episode_start_bins))
    self.episode_discounted_returns.copy_(
      torch.where(starting, 0.0, self.episode_discounted_returns)
      + self.cfg.gamma**age * rewards
    )
    finished = done.nonzero().flatten()
    if not len(finished):
      return
    lengths = age[finished] + 1
    returns = self.episode_discounted_returns[finished]
    discount_mass = -torch.expm1(lengths * math.log(self.cfg.gamma))
    scores = torch.where(terminated[finished], returns, returns / discount_mass)
    # Each episode contributes one observation to its reference reset bin.
    self.adapter.command.return_sampler.observe(
      self.episode_start_bins[finished], scores
    )

  @torch.no_grad()
  def collect(self, num_steps: int) -> dict[str, torch.Tensor]:
    if num_steps < 1:
      raise ValueError("Collection needs at least one simulator step.")
    collected: dict[str, list[torch.Tensor]] = {}
    raw = self.env.unwrapped
    command = self.adapter.command
    for _ in range(num_steps):
      actor_observation = self.observations["actor"]
      assert isinstance(actor_observation, torch.Tensor)
      history = self.adapter.unpack_history(actor_observation)
      frames, motions = command.time_steps.clone(), command.motion_ids.clone()
      future = self.adapter.future_reference(motions, frames)
      action = self.alg.actor(history, future) + self.cfg.action_noise * torch.randn(
        self.env.num_envs, 29, device=self.device
      )
      if self.cfg.clip_actions is not None:
        action = action.clamp(-self.cfg.clip_actions, self.cfg.clip_actions)
      state = self.adapter.state().clone()
      age = raw.episode_length_buf.clone()
      context = self.adapter.read_context()
      observations, reward, done, extras = self.env.step(action)
      transition = {
        "state": state,
        "next_state": self.adapter.state().clone(),
        "action": action,
        "context": context,
        "obs": history[:, -1].clone(),
        "reward": reward.clone(),
        "terminated": raw.reset_terminated.clone(),
        "truncated": raw.reset_time_outs.clone(),
        "motion": motions,
        "frame": frames,
        "next_frame": command.time_steps.clone(),
        "episode": self.episode_ids.clone(),
        "age": age,
      }
      if not all(torch.isfinite(value).all() for value in transition.values()):
        raise FloatingPointError(
          "Non-finite simulator transition; replay was not updated."
        )
      self.replay.append(transition)
      for key, value in transition.items():
        collected.setdefault(key, []).append(value)
      self._record_returns(
        age,
        command.return_sampler.bin_ids(motions, frames),
        reward,
        done.bool(),
        transition["terminated"],
      )
      finished = done.nonzero().flatten()
      if len(finished):
        available = (
          command.motion.time_step_totals[motions[finished]]
          - (frames[finished] - age[finished])
        ).clamp_max(raw.max_episode_length)
        episode_values = {
          "failure_fraction": transition["terminated"][finished].float(),
          "completion_fraction": (age[finished] + 1) / available,
          "available_length": available.float(),
        }
        for name, value in episode_values.items():
          self.episode_metrics[name].extend(value.cpu().tolist())
        # Every terminal label and next state above was captured before reset.
        reset_obs, reset_extras = raw.reset(env_ids=finished)
        extras = {
          **extras,
          "log": {**extras.get("log", {}), **reset_extras.get("log", {})},
        }
        observations = TensorDict(reset_obs, batch_size=[self.env.num_envs])
        self.episode_ids[finished] += 1
      self.logger.process_env_step(transition["reward"], done, extras)
      self.observations = observations
      self.real_steps += self.env.num_envs * self.world_size
    command.return_sampler.commit()
    return {key: torch.cat(values) for key, values in collected.items()}

  def next_collection_steps(self) -> int:
    """Carry fractional simulator steps across learner rounds and checkpoints."""
    self.collection_remainder += 1 / self.cfg.update_to_data_ratio
    steps = math.floor(self.collection_remainder + 1e-12)
    self.collection_remainder = max(0.0, self.collection_remainder - steps)
    return steps

  def learn(
    self, num_learning_iterations: int, init_at_random_ep_len: bool = False
  ) -> None:
    del init_at_random_ep_len
    self.logger.init_logging_writer()
    start = self.current_learning_iteration
    stop = start + num_learning_iterations
    print(
      "[SuperMimic] UTD = complete learner updates / simulator vector steps: "
      f"{self.cfg.update_to_data_ratio:g}; "
      f"{self.cfg.world_model.batches_per_update} WM / 1 actor / "
      f"{self.cfg.critic.epochs * self.cfg.critic.minibatches} critic "
      "optimizer steps per update."
    )
    if not self.initial_collection_done:
      self.alg.update_normalizers(self.collect(self.cfg.initial_collection_steps))
      self.initial_collection_done = True
    for iteration in range(start, stop):
      self.alg.set_iteration(iteration)
      tick = time.perf_counter()
      collection_steps = self.next_collection_steps()
      collected = self.collect(collection_steps)
      self.alg.update_normalizers(collected)
      collection_time = time.perf_counter() - tick
      tick = time.perf_counter()
      metrics = self.alg.fit_world_model(self.replay)
      self.optimizer_updates["world_model"] += self.cfg.world_model.batches_per_update
      metrics["time/world_model"] = time.perf_counter() - tick
      tick = time.perf_counter()
      metrics.update(self.alg.improve_policy(self.replay))
      self.optimizer_updates["actor"] += 1
      self.optimizer_updates["critic"] += (
        self.cfg.critic.epochs * self.cfg.critic.minibatches
      )
      learning_time = metrics["time/world_model"] + time.perf_counter() - tick
      probabilities = self.adapter.command.return_sampler.probabilities()
      metrics.update(
        {
          "time/collection": collection_time,
          "sampling/entropy": float(-(probabilities * probabilities.log()).sum()),
          "sampling/max_probability": float(probabilities.max()),
          "sampling/max_uniform_ratio": float(probabilities.max() * len(probabilities)),
          "sampling/effective_bin_fraction": float(
            1 / (len(probabilities) * probabilities.square().sum())
          ),
          "sampling/mean_return": float(
            self.adapter.command.return_sampler.returns.mean()
          ),
          "sampling/expected_episode_length": float(
            self.adapter.command.return_sampler.expected_episode_length(
              self.env.unwrapped.max_episode_length
            )
          ),
          **{
            f"episode/{name}": sum(values) / len(values)
            for name, values in self.episode_metrics.items()
            if values
          },
          "training/utd": self.cfg.update_to_data_ratio,
          "training/real_steps": float(self.real_steps),
          "training/vector_steps": float(
            self.real_steps // (self.env.num_envs * self.world_size)
          ),
          **{
            f"training/{name}_updates": float(count)
            for name, count in self.optimizer_updates.items()
          },
        }
      )
      self.last_metrics = metrics
      self.current_learning_iteration = iteration + 1
      # RSL's logger assumes fixed collection lengths. Supply this round's
      # length and include prefill/resumed steps in its cumulative total.
      self.logger.cfg["num_steps_per_env"] = collection_steps
      self.logger.tot_timesteps = (
        self.real_steps - collection_steps * self.env.num_envs * self.world_size
      )
      self.logger.log(
        iteration,
        start,
        stop,
        collection_time,
        learning_time,
        metrics,
        self.alg.optimizers[1].param_groups[0]["lr"],
        torch.zeros(29, device=self.device),
        None,
      )
      if (
        self.log_dir is not None
        and self.current_learning_iteration % self.cfg.save_interval == 0
      ):
        self.save(
          str(Path(self.log_dir) / f"model_{self.current_learning_iteration}.pt")
        )
    if self.log_dir is not None:
      self.save(str(Path(self.log_dir) / f"model_{self.current_learning_iteration}.pt"))
    flush = getattr(self.logger.writer, "flush", None)
    if callable(flush):
      flush()

  def save(self, path: str, infos: dict | None = None) -> None:
    """Collective checkpoint: all ranks write their own replay/RNG shard."""
    checkpoint = Path(path)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    shard = checkpoint.with_name(f"{checkpoint.stem}.rank{self.rank}.replay.pt")
    local = {
      "replay": self.replay.state_dict(),
      "sampler": self.adapter.command.return_sampler.state_dict(),
      "episode_ids": self.episode_ids,
      "torch_rng": torch.get_rng_state(),
      "numpy_rng": np.random.get_state(),
      "python_rng": random.getstate(),
      "cuda_rng": torch.cuda.get_rng_state(self.device)
      if "cuda" in self.device
      else None,
    }
    temporary = shard.with_suffix(".tmp")
    torch.save(local, temporary)
    temporary.replace(shard)
    if dist.is_initialized():
      dist.barrier()
    if self.rank == 0:
      sampler = self.adapter.command.return_sampler
      # A compact, directly inspectable distribution log alongside checkpoints;
      # restoration uses the complete sampler state in each replay shard.
      np.savez_compressed(
        checkpoint.with_suffix(".sampling.npz"),
        motion_ids=sampler.motion_ids.cpu().numpy(),
        start_frames=sampler.starts.cpu().numpy(),
        end_frames=sampler.ends.cpu().numpy(),
        probabilities=sampler.probabilities().cpu().numpy(),
        returns=sampler.returns.cpu().numpy(),
        historical_weights=sampler.history.cpu().numpy(),
      )
      state = {
        "format_version": FORMAT_VERSION,
        "algorithm": self.alg.state_dict(),
        "config": asdict(self.cfg),
        "signature": self.signature,
        "world_size": self.world_size,
        "iteration": self.current_learning_iteration,
        "real_steps": self.real_steps,
        "collection_remainder": self.collection_remainder,
        "initial_collection_done": self.initial_collection_done,
        "optimizer_updates": self.optimizer_updates,
        "common_step_counter": self.env.unwrapped.common_step_counter,
        "infos": infos or {},
      }
      temporary = checkpoint.with_suffix(".tmp")
      torch.save(state, temporary)
      temporary.replace(checkpoint)
      if self.cfg.upload_model:
        self.logger.save_model(str(checkpoint), self.current_learning_iteration)
    if dist.is_initialized():
      dist.barrier()

  def load(self, path: str, **kwargs) -> dict:
    del kwargs
    checkpoint = Path(path)
    state = load_checkpoint(checkpoint, map_location=self.device)
    # Fill omitted sampler defaults; all explicit task settings remain strict.
    signature = {
      **state["signature"],
      "sampler": asdict(ReturnSamplerCfg(**state["signature"]["sampler"])),
    }
    if signature != self.signature or state["world_size"] != self.world_size:
      raise ValueError(
        "Checkpoint dataset, task configuration (including actor inputs) or "
        "world size changed; start a new training run."
      )
    for key in (
      "world_model",
      "imagination",
      "critic",
      "gamma",
      "lam",
      "num_steps_per_env",
      "initial_collection_steps",
      "update_to_data_ratio",
      "replay_steps",
      "action_noise",
      "activation",
      "decay_iterations",
      "final_lr_fraction",
      "clip_actions",
    ):
      if state["config"][key] != asdict(self.cfg)[key]:
        raise ValueError(f"Checkpoint training configuration changed: {key}.")
    shard = checkpoint.with_name(f"{checkpoint.stem}.rank{self.rank}.replay.pt")
    local = torch.load(shard, map_location=self.device, weights_only=False)
    self.alg.load_state_dict(state["algorithm"])
    self.replay.load_state_dict(local["replay"])
    sampler = self.adapter.command.return_sampler
    sampler.load_state_dict(local["sampler"])
    self.current_learning_iteration = int(state["iteration"])
    self.real_steps = int(state["real_steps"])
    self.collection_remainder = float(state["collection_remainder"])
    self.initial_collection_done = bool(state["initial_collection_done"])
    self.optimizer_updates = dict(state["optimizer_updates"])
    self.env.unwrapped.common_step_counter = int(state["common_step_counter"])
    self.episode_ids.copy_(local["episode_ids"] + 1)
    torch.set_rng_state(local["torch_rng"].cpu())
    np.random.set_state(local["numpy_rng"])
    random.setstate(local["python_rng"])
    if local["cuda_rng"] is not None:
      torch.cuda.set_rng_state(local["cuda_rng"].cpu(), self.device)
    self.episode_discounted_returns.zero_()
    self.episode_start_bins.zero_()
    for values in self.episode_metrics.values():
      values.clear()
    # Physical simulation is restarted; retained replay is still valid dynamics
    # data. New episode IDs prevent a window from bridging the restart.
    self.observations, _ = self.env.reset()
    return state["infos"]
