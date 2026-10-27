"""Per-environment FIFO with episode-safe windows and observation histories."""

from __future__ import annotations

import torch


class ReplayBuffer:
  def __init__(self, capacity: int, num_envs: int, device: str | torch.device):
    self.capacity = capacity
    self.num_envs = num_envs
    self.device = torch.device(device)
    self.total_steps = 0
    self.data: dict[str, torch.Tensor] = {}

  @property
  def size(self) -> int:
    return min(self.total_steps, self.capacity)

  @torch.no_grad()
  def append(self, transition: dict[str, torch.Tensor]) -> None:
    if not self.data:
      self.data = {
        key: torch.empty(
          (self.capacity, *value.shape), dtype=value.dtype, device=self.device
        )
        for key, value in transition.items()
      }
    if self.data.keys() != transition.keys():
      raise ValueError("Replay transition fields changed.")
    slot = self.total_steps % self.capacity
    for key, value in transition.items():
      if value.shape != self.data[key].shape[1:]:
        raise ValueError(f"Replay shape changed for {key}.")
      self.data[key][slot].copy_(value.detach())
    self.total_steps += 1

  def _draw(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    if not self.size:
      raise RuntimeError("Cannot sample empty replay.")
    times = torch.randint(
      max(0, self.total_steps - self.capacity),
      self.total_steps,
      (batch_size,),
      device=self.device,
    )
    envs = torch.randint(self.num_envs, (batch_size,), device=self.device)
    return times, envs

  def windows(self, batch_size: int, length: int) -> dict[str, torch.Tensor]:
    times, envs = self._draw(batch_size)
    times = times[None] + torch.arange(length, device=self.device)[:, None]
    slots = times.remainder(self.capacity)
    result = {key: value[slots, envs] for key, value in self.data.items()}
    same_episode = result["episode"] == result["episode"][:1]
    valid = (times < self.total_steps) & same_episode
    valid &= (
      result["age"]
      == result["age"][:1] + torch.arange(length, device=self.device)[:, None]
    )
    done = result["terminated"] | result["truncated"]
    # Include the terminal transition, but never train on its reset successor.
    preceding_alive = torch.cat((torch.ones_like(done[:1]), ~done[:-1]))
    result["mask"] = (valid & preceding_alive).long().cumprod(0).bool()
    # Padded slots may be uninitialized before the ring fills. Replace their
    # contents before model evaluation; multiplying a NaN loss by zero is unsafe.
    for key in self.data:
      mask = result["mask"].reshape(
        *result["mask"].shape, *((1,) * (result[key].ndim - 2))
      )
      result[key] = torch.where(mask, result[key], result[key][:1])
    return result

  def starts(
    self, batch_size: int, history_length: int = 10
  ) -> dict[str, torch.Tensor]:
    # Avoid starts whose true observation history has already been overwritten.
    oldest = max(0, self.total_steps - self.capacity)
    times = torch.arange(oldest, self.total_steps, device=self.device)
    ages = self.data["age"][times.remainder(self.capacity)]
    available = times[:, None] - oldest
    needed = ages.clamp_max(history_length - 1)
    valid = available >= needed
    candidates = valid.nonzero()
    if not len(candidates):
      raise RuntimeError("Replay has no seeds with complete observation histories.")
    chosen = candidates[
      torch.randint(len(candidates), (batch_size,), device=self.device)
    ]
    times, envs = times[chosen[:, 0]], chosen[:, 1]
    slots = times.remainder(self.capacity)
    result = {key: value[slots, envs] for key, value in self.data.items()}
    # Oldest-to-newest, with reset's first noisy frame backfilled exactly as in
    # ObservationManager. No new noise is drawn for these historical frames.
    offsets = torch.arange(history_length - 1, -1, -1, device=self.device)
    history_times = times[:, None] - torch.minimum(offsets, result["age"][:, None])
    hs = history_times.remainder(self.capacity)
    if not torch.all(
      self.data["episode"][hs, envs[:, None]] == result["episode"][:, None]
    ):
      raise RuntimeError("Replay observation history crosses an episode boundary.")
    result["history"] = self.data["obs"][hs, envs[:, None]]
    return result

  def state_dict(self) -> dict:
    return {
      "capacity": self.capacity,
      "num_envs": self.num_envs,
      "total_steps": self.total_steps,
      "data": {key: value[: self.size].cpu() for key, value in self.data.items()},
    }

  def load_state_dict(self, state: dict) -> None:
    if state["capacity"] != self.capacity or state["num_envs"] != self.num_envs:
      raise ValueError("Replay capacity or number of environments changed.")
    self.total_steps = int(state["total_steps"])
    self.data = {}
    for key, value in state["data"].items():
      if value.shape[:2] != (self.size, self.num_envs):
        raise ValueError(f"Invalid replay shape for {key}.")
      full = torch.empty(
        (self.capacity, *value.shape[1:]), dtype=value.dtype, device=self.device
      )
      full[: self.size].copy_(value)
      self.data[key] = full
