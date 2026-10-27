"""Pure-Torch clip sampling from tracking quality and rehearsal age.

The sampler owns a stable table of virtual clips.  Outcome statistics are
queued separately from committed curriculum state so callers can all-reduce
the sufficient statistics before updating every distributed replica.
"""

from __future__ import annotations

import copy
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, NamedTuple

import torch
import torch.distributed as dist

__all__ = ["GmtAdaptiveMotionSampler", "GmtAdaptiveSamplerCfg"]

_STATE_SCHEMA = "mjlab.gmt_adaptive_sampler.v3"
_distributed_api: Any = dist
_PENDING_KEYS = (
  "sample_count",
  "success_count",
  "failure_count",
  "tracking_error_sum",
  "tracking_error_count",
)


@dataclass(kw_only=True)
class GmtAdaptiveSamplerCfg:
  """Configuration for quality-based sampling of fixed episode-length clips."""

  partition_seed: int | None = None
  """Base sampling seed (legacy name); fixed clip boundaries do not use RNG."""
  failure_weight: float = 0.5
  """Weight of recent failure rate; the remainder weights normalized error."""
  tracking_error_scale: float = 0.10
  """Local mean key-body error in meters corresponding to a score of 0.5."""
  ema_half_life_episodes: float = 32.0
  """Number of valid episode outcomes that halves the previous EMA weight."""
  uniform_floor: float = 0.20
  staleness_ratio: float = 0.10
  """Probability budget for clips not sampled recently within their group."""
  sync_interval_steps: int = 240

  def __post_init__(self) -> None:
    if self.partition_seed is not None and (
      isinstance(self.partition_seed, bool)
      or not isinstance(self.partition_seed, int)
      or self.partition_seed < 0
    ):
      raise ValueError("partition_seed must be a non-negative integer or None.")
    positive = ("tracking_error_scale", "ema_half_life_episodes")
    for name in positive:
      value = getattr(self, name)
      if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    for name in ("failure_weight", "uniform_floor", "staleness_ratio"):
      value = getattr(self, name)
      if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1].")
    if self.uniform_floor + self.staleness_ratio > 1.0:
      raise ValueError("uniform_floor + staleness_ratio must not exceed 1.")
    if (
      isinstance(self.sync_interval_steps, bool)
      or not isinstance(self.sync_interval_steps, int)
      or self.sync_interval_steps <= 0
    ):
      raise ValueError("sync_interval_steps must be a positive integer.")


class _GmtSamplerSample(NamedTuple):
  clip_ids: torch.Tensor
  motion_ids: torch.Tensor
  group_ids: torch.Tensor
  start_steps: torch.Tensor
  end_steps: torch.Tensor
  probabilities: torch.Tensor


class GmtAdaptiveMotionSampler:
  """Sample stable clips using recent quality, uniform coverage, and rehearsal."""

  def __init__(
    self,
    cfg: GmtAdaptiveSamplerCfg,
    motion_lengths: torch.Tensor,
    *,
    step_dt: float,
    clip_length_steps: int,
    motion_group_ids: torch.Tensor | None = None,
    device: str | torch.device | None = None,
    seed: int = 0,
    sampling_seed: int | None = None,
    dataset_fingerprint: str | None = None,
    motion_names: tuple[str, ...] | None = None,
  ) -> None:
    self.cfg = copy.deepcopy(cfg)
    if not math.isfinite(step_dt) or step_dt <= 0.0:
      raise ValueError("step_dt must be finite and positive.")
    self.step_dt = float(step_dt)
    if (
      isinstance(clip_length_steps, bool)
      or not isinstance(clip_length_steps, int)
      or clip_length_steps <= 0
    ):
      raise ValueError("clip_length_steps must be a positive integer.")
    self.clip_length_steps = clip_length_steps
    self.device = torch.device(device or motion_lengths.device)
    self._validate_seed(seed, "seed")
    if sampling_seed is None:
      sampling_seed = seed
    self._validate_seed(sampling_seed, "sampling_seed")
    self.seed = seed
    self.sampling_seed = sampling_seed

    self.motion_lengths = self._validate_index_tensor(
      motion_lengths, "motion_lengths", positive=True
    ).to(self.device)
    self.num_motions = int(self.motion_lengths.numel())
    if motion_group_ids is None:
      motion_group_ids = torch.zeros_like(self.motion_lengths)
    self.motion_group_ids = self._validate_index_tensor(
      motion_group_ids, "motion_group_ids", positive=False
    ).to(self.device)
    if self.motion_group_ids.shape != self.motion_lengths.shape:
      raise ValueError("motion_group_ids must have the same shape as motion_lengths.")
    if torch.any(self.motion_group_ids < 0):
      raise ValueError("motion_group_ids must be non-negative.")

    if dataset_fingerprint is not None and not dataset_fingerprint.strip():
      raise ValueError("dataset_fingerprint must be non-empty when provided.")
    self.dataset_fingerprint = dataset_fingerprint
    if motion_names is not None:
      if len(motion_names) != self.num_motions:
        raise ValueError("motion_names must contain one name per motion.")
      if any(not name for name in motion_names):
        raise ValueError("motion_names must be non-empty.")
      if len(set(motion_names)) != len(motion_names):
        raise ValueError("motion_names must be unique.")
    self.motion_names = motion_names

    self._sampling_generator = torch.Generator(device=self.device)
    self._sampling_generator.manual_seed(sampling_seed)
    self._distributed_sampling_rng_states: tuple[torch.Tensor, ...] | None = None
    self._build_clip_table()

    self.success_rate_ema = torch.full(
      (self.num_clips,),
      0.5,
      dtype=torch.float32,
      device=self.device,
    )
    self.tracking_error_ema = torch.full_like(
      self.success_rate_ema, self.cfg.tracking_error_scale
    )
    self.rehearsal_age = torch.zeros_like(self.success_rate_ema)
    self.sample_counts = torch.zeros_like(self.success_rate_ema)
    self.success_counts = torch.zeros_like(self.success_rate_ema)
    self.failure_counts = torch.zeros_like(self.success_rate_ema)
    self.tracking_error_sums = torch.zeros_like(self.success_rate_ema)
    self.tracking_error_counts = torch.zeros_like(self.success_rate_ema)
    self._pending = {
      key: torch.zeros_like(self.success_rate_ema) for key in _PENDING_KEYS
    }
    self._cached_probabilities = torch.empty_like(self.success_rate_ema)
    self._state_revision = 0
    self._last_sync_step = 0
    self._refresh_probabilities()

  @staticmethod
  def _validate_seed(seed: int, name: str) -> None:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
      raise ValueError(f"{name} must be a non-negative integer.")

  @staticmethod
  def _validate_index_tensor(
    value: torch.Tensor, name: str, *, positive: bool
  ) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
      raise TypeError(f"{name} must be a torch.Tensor.")
    if value.ndim != 1 or value.numel() == 0:
      raise ValueError(f"{name} must be a non-empty one-dimensional tensor.")
    integer_dtypes = {
      torch.uint8,
      torch.int8,
      torch.int16,
      torch.int32,
      torch.int64,
    }
    if value.dtype not in integer_dtypes:
      raise TypeError(f"{name} must have an integer dtype.")
    result = value.to(dtype=torch.long)
    if positive and torch.any(result <= 0):
      raise ValueError(f"{name} must contain only positive values.")
    return result.clone()

  def _build_clip_table(self) -> None:
    """Partition each source motion into fixed intervals, keeping every tail."""
    starts: list[int] = []
    ends: list[int] = []
    motions: list[int] = []
    for motion_id, raw_length in enumerate(self.motion_lengths.cpu().tolist()):
      length = int(raw_length)
      for start in range(0, length, self.clip_length_steps):
        starts.append(start)
        ends.append(min(start + self.clip_length_steps, length))
        motions.append(motion_id)

    self.clip_start_steps = torch.tensor(starts, dtype=torch.long, device=self.device)
    self.clip_end_steps = torch.tensor(ends, dtype=torch.long, device=self.device)
    self.clip_motion_ids = torch.tensor(motions, dtype=torch.long, device=self.device)
    self.num_clips = int(self.clip_motion_ids.numel())
    self.clip_group_ids = self.motion_group_ids[self.clip_motion_ids]
    self.group_ids, self._clip_group_indexes = torch.unique(
      self.clip_group_ids, sorted=True, return_inverse=True
    )
    self.num_groups = int(self.group_ids.numel())
    self._group_clip_counts = torch.bincount(
      self._clip_group_indexes, minlength=self.num_groups
    )

  @property
  def state_revision(self) -> int:
    """Local cache generation, advanced after committed changes or restoration."""
    return self._state_revision

  @property
  def cached_probabilities(self) -> torch.Tensor:
    """Cached proposal tensor, exposed for in-place distributed broadcast."""
    return self._cached_probabilities

  @property
  def sampling_rng_state(self) -> torch.Tensor:
    """Return a clone of this rank's current sampling RNG state."""
    return self._sampling_generator.get_state().clone()

  @property
  def distributed_sampling_rng_states(self) -> tuple[torch.Tensor, ...] | None:
    """Return the last gathered per-rank sampling states, if any."""
    if self._distributed_sampling_rng_states is None:
      return None
    return tuple(state.clone() for state in self._distributed_sampling_rng_states)

  def set_distributed_sampling_rng_states(self, states: Sequence[torch.Tensor]) -> None:
    """Store sampling RNG states gathered from every distributed rank."""
    if not states:
      raise ValueError("Distributed sampling RNG states must not be empty.")
    validated: list[torch.Tensor] = []
    for rank, state in enumerate(states):
      if not isinstance(state, torch.Tensor):
        raise TypeError(f"Sampling RNG state for rank {rank} must be a tensor.")
      generator = torch.Generator(device=self.device)
      try:
        generator.set_state(state.cpu())
      except RuntimeError as exc:
        raise ValueError(f"Sampling RNG state for rank {rank} is invalid.") from exc
      validated.append(state.cpu().clone())
    if _distributed_api.is_available() and _distributed_api.is_initialized():
      world_size = _distributed_api.get_world_size()
      if len(validated) != world_size:
        raise ValueError(
          "Distributed sampling RNG state count does not match the process-group "
          "world size."
        )
    self._distributed_sampling_rng_states = tuple(validated)

  @property
  def coverage(self) -> float:
    counts = self.sample_counts + self._pending["sample_count"]
    return float((counts > 0).float().mean().item())

  @property
  def total_sample_count(self) -> float:
    return float((self.sample_counts + self._pending["sample_count"]).sum().item())

  @property
  def total_success_count(self) -> float:
    return float((self.success_counts + self._pending["success_count"]).sum().item())

  @property
  def total_failure_count(self) -> float:
    return float((self.failure_counts + self._pending["failure_count"]).sum().item())

  @property
  def group_sample_counts(self) -> torch.Tensor:
    result = torch.zeros(self.num_groups, dtype=torch.float32, device=self.device)
    counts = self.sample_counts + self._pending["sample_count"]
    return result.scatter_add(0, self._clip_group_indexes, counts)

  @property
  def group_sample_shares(self) -> torch.Tensor:
    counts = self.group_sample_counts
    total = counts.sum()
    return torch.where(total > 0, counts / total, torch.zeros_like(counts))

  @property
  def success_rates(self) -> torch.Tensor:
    successes = self.success_counts + self._pending["success_count"]
    failures = self.failure_counts + self._pending["failure_count"]
    total = successes + failures
    return torch.where(total > 0, successes / total, torch.zeros_like(total))

  def scores(self) -> torch.Tensor:
    normalized_error = self.tracking_error_ema / (
      self.tracking_error_ema + self.cfg.tracking_error_scale
    )
    return (
      self.cfg.failure_weight * (1.0 - self.success_rate_ema)
      + (1.0 - self.cfg.failure_weight) * normalized_error
    )

  def base_probabilities(self) -> torch.Tensor:
    counts = self._group_clip_counts[self._clip_group_indexes].to(torch.float32)
    return 1.0 / (self.num_groups * counts)

  def probabilities(self) -> torch.Tensor:
    return self._cached_probabilities.clone()

  def _group_conditional(self, scores: torch.Tensor) -> torch.Tensor:
    group_score_sums = torch.zeros(
      self.num_groups, dtype=torch.float32, device=self.device
    ).scatter_add(0, self._clip_group_indexes, scores)
    denominators = group_score_sums[self._clip_group_indexes]
    uniform_conditional = 1.0 / self._group_clip_counts[self._clip_group_indexes].to(
      torch.float32
    )
    return torch.where(
      denominators > 0,
      scores / denominators.clamp_min(torch.finfo(torch.float32).tiny),
      uniform_conditional,
    )

  def _refresh_probabilities(self) -> None:
    adaptive = self._group_conditional(self.scores()) / self.num_groups
    rehearsal = self._group_conditional(self.rehearsal_age) / self.num_groups
    base = self.base_probabilities()
    adaptive_ratio = 1.0 - self.cfg.uniform_floor - self.cfg.staleness_ratio
    self._cached_probabilities.copy_(
      adaptive_ratio * adaptive
      + self.cfg.uniform_floor * base
      + self.cfg.staleness_ratio * rehearsal
    )
    self._state_revision += 1

  def set_cached_probabilities(self, probabilities: torch.Tensor) -> None:
    """Validate and copy a proposal broadcast from another rank."""
    value = probabilities.to(device=self.device, dtype=torch.float32)
    if value.shape != (self.num_clips,):
      raise ValueError("Cached probabilities have an incompatible shape.")
    if not torch.isfinite(value).all() or torch.any(value < 0):
      raise ValueError("Cached probabilities must be finite and non-negative.")
    value64 = value.to(torch.float64)
    if not torch.isclose(value64.sum(), value64.new_tensor(1.0), atol=1e-6):
      raise ValueError("Cached probabilities must sum to one.")
    group_mass = torch.zeros(
      self.num_groups, dtype=torch.float64, device=self.device
    ).scatter_add(0, self._clip_group_indexes, value64)
    expected = torch.full_like(group_mass, 1.0 / self.num_groups)
    if not torch.allclose(group_mass, expected, atol=1e-6, rtol=1e-6):
      raise ValueError("Cached probabilities must assign equal mass to every group.")
    self._cached_probabilities.copy_(value)
    self._state_revision += 1

  def sample(self, num_samples: int) -> _GmtSamplerSample:
    """Draw virtual clips and queue their per-clip sample counts."""
    if isinstance(num_samples, bool) or not isinstance(num_samples, int):
      raise TypeError("num_samples must be an integer.")
    if num_samples < 0:
      raise ValueError("num_samples must be non-negative.")
    if num_samples == 0:
      clip_ids = torch.empty(0, dtype=torch.long, device=self.device)
    else:
      clip_ids = torch.multinomial(
        self._cached_probabilities,
        num_samples,
        replacement=True,
        generator=self._sampling_generator,
      )
      self._pending["sample_count"].scatter_add_(
        0, clip_ids, torch.ones(num_samples, device=self.device)
      )
      self._distributed_sampling_rng_states = None
    return _GmtSamplerSample(
      clip_ids=clip_ids,
      motion_ids=self.clip_motion_ids[clip_ids],
      group_ids=self.clip_group_ids[clip_ids],
      start_steps=self.clip_start_steps[clip_ids],
      end_steps=self.clip_end_steps[clip_ids],
      probabilities=self._cached_probabilities[clip_ids],
    )

  def observe(
    self,
    clip_ids: torch.Tensor,
    successes: torch.Tensor,
    mean_key_body_errors: torch.Tensor,
    *,
    error_valid: torch.Tensor | None = None,
  ) -> None:
    """Queue completed-episode outcomes without changing probabilities."""
    if not isinstance(clip_ids, torch.Tensor):
      raise TypeError("clip_ids must be a torch.Tensor.")
    if clip_ids.ndim != 1 or clip_ids.dtype != torch.long:
      raise ValueError("clip_ids must be a one-dimensional torch.long tensor.")
    if not isinstance(successes, torch.Tensor) or successes.shape != clip_ids.shape:
      raise ValueError("successes must have the same shape as clip_ids.")
    if successes.dtype != torch.bool:
      raise TypeError("successes must have dtype torch.bool.")
    if (
      not isinstance(mean_key_body_errors, torch.Tensor)
      or mean_key_body_errors.shape != clip_ids.shape
    ):
      raise ValueError("mean_key_body_errors must have the same shape as clip_ids.")
    if error_valid is None:
      error_valid = torch.ones_like(successes)
    if (
      not isinstance(error_valid, torch.Tensor)
      or error_valid.shape != clip_ids.shape
      or error_valid.dtype != torch.bool
    ):
      raise ValueError("error_valid must be a bool tensor shaped like clip_ids.")
    clip_ids = clip_ids.to(self.device)
    successes = successes.to(self.device)
    error_valid = error_valid.to(self.device)
    errors = mean_key_body_errors.to(device=self.device, dtype=torch.float32)
    errors = torch.where(error_valid, errors, 0.0)
    if clip_ids.numel() == 0:
      return
    if torch.any(clip_ids < 0) or torch.any(clip_ids >= self.num_clips):
      raise IndexError("clip_ids contains an out-of-range clip ID.")
    if not torch.isfinite(errors).all() or torch.any(errors < 0):
      raise ValueError("Valid mean_key_body_errors must be finite and non-negative.")
    self._pending["success_count"].scatter_add_(
      0, clip_ids, successes.to(torch.float32)
    )
    self._pending["failure_count"].scatter_add_(
      0, clip_ids, (~successes).to(torch.float32)
    )
    self._pending["tracking_error_sum"].scatter_add_(0, clip_ids, errors)
    self._pending["tracking_error_count"].scatter_add_(
      0, clip_ids, error_valid.to(torch.float32)
    )

  def pending_statistics(self) -> dict[str, torch.Tensor]:
    """Return cloneable sufficient statistics ready for distributed reduction."""
    return {key: value.clone() for key, value in self._pending.items()}

  def clear_pending_statistics(self) -> None:
    for value in self._pending.values():
      value.zero_()

  def commit_pending_statistics(
    self,
    statistics: Mapping[str, torch.Tensor] | None = None,
    *,
    clear: bool = True,
    step: int | None = None,
  ) -> None:
    """Commit local or externally reduced sufficient statistics."""
    if step is not None and step < self._last_sync_step:
      raise ValueError("step cannot move backwards.")
    values = self._validated_statistics(
      self._pending if statistics is None else statistics
    )
    successes = values["success_count"]
    outcomes = successes + values["failure_count"]
    self._update_ema(self.success_rate_ema, successes, outcomes)
    self._update_ema(
      self.tracking_error_ema,
      values["tracking_error_sum"],
      values["tracking_error_count"],
    )

    group_draws = torch.zeros(
      self.num_groups, dtype=torch.float32, device=self.device
    ).scatter_add_(0, self._clip_group_indexes, values["sample_count"])
    group_rounds = group_draws / self._group_clip_counts
    self.rehearsal_age.add_(group_rounds[self._clip_group_indexes])
    self.rehearsal_age.masked_fill_(values["sample_count"] > 0, 0.0)
    self.sample_counts.add_(values["sample_count"])
    self.success_counts.add_(successes)
    self.failure_counts.add_(values["failure_count"])
    self.tracking_error_sums.add_(values["tracking_error_sum"])
    self.tracking_error_counts.add_(values["tracking_error_count"])
    if step is not None:
      self._last_sync_step = step
    self._refresh_probabilities()
    if clear:
      self.clear_pending_statistics()

  def _update_ema(
    self, ema: torch.Tensor, totals: torch.Tensor, counts: torch.Tensor
  ) -> None:
    # expm1 preserves small update coefficients for long half-lives.
    alpha = -torch.expm1(-math.log(2.0) * counts / self.cfg.ema_half_life_episodes)
    means = totals / counts.clamp_min(1.0)
    ema.copy_(torch.where(counts > 0, torch.lerp(ema, means, alpha), ema))

  def _validated_statistics(
    self, statistics: Mapping[str, torch.Tensor]
  ) -> dict[str, torch.Tensor]:
    if set(statistics) != set(_PENDING_KEYS):
      raise ValueError(f"Statistics must contain exactly {_PENDING_KEYS}.")
    result: dict[str, torch.Tensor] = {}
    for key in _PENDING_KEYS:
      value = statistics[key].to(device=self.device, dtype=torch.float32)
      if value.shape != (self.num_clips,):
        raise ValueError(f"Statistic {key!r} has an incompatible shape.")
      if not torch.isfinite(value).all() or torch.any(value < 0):
        raise ValueError(f"Statistic {key!r} must be finite and non-negative.")
      result[key] = value
    return result

  def should_sync(self, step: int) -> bool:
    if step < self._last_sync_step:
      raise ValueError("step cannot move backwards.")
    return step - self._last_sync_step >= self.cfg.sync_interval_steps

  def validate_dataset(
    self,
    *,
    dataset_fingerprint: str | None = None,
    motion_lengths: torch.Tensor | None = None,
    motion_group_ids: torch.Tensor | None = None,
    motion_names: tuple[str, ...] | None = None,
  ) -> None:
    """Validate caller metadata against the sampler's stable clip table."""
    if (
      dataset_fingerprint is not None
      and dataset_fingerprint != self.dataset_fingerprint
    ):
      raise ValueError("Dataset fingerprint does not match the sampler state.")
    if motion_lengths is not None and not torch.equal(
      motion_lengths.to(self.motion_lengths), self.motion_lengths
    ):
      raise ValueError("Motion lengths do not match the sampler state.")
    if motion_group_ids is not None and not torch.equal(
      motion_group_ids.to(self.motion_group_ids), self.motion_group_ids
    ):
      raise ValueError("Motion group IDs do not match the sampler state.")
    if (
      dataset_fingerprint is None
      and motion_names is not None
      and motion_names != self.motion_names
    ):
      raise ValueError("Motion names do not match the sampler state.")

  def state_dict(self, *, include_pending: bool = True) -> dict[str, Any]:
    """Return sampler state, optionally omitting rank-local pending statistics."""
    pending = {
      f"pending_{key}": value if include_pending else torch.zeros_like(value)
      for key, value in self._pending.items()
    }
    tensors = {
      "success_rate_ema": self.success_rate_ema,
      "tracking_error_ema": self.tracking_error_ema,
      "rehearsal_age": self.rehearsal_age,
      "sample_counts": self.sample_counts,
      "success_counts": self.success_counts,
      "failure_counts": self.failure_counts,
      "tracking_error_sums": self.tracking_error_sums,
      "tracking_error_counts": self.tracking_error_counts,
      "cached_probabilities": self._cached_probabilities,
      **pending,
    }
    return {
      "schema": _STATE_SCHEMA,
      "config": asdict(self.cfg),
      "step_dt": self.step_dt,
      "clip_length_steps": self.clip_length_steps,
      "seed": self.seed,
      "sampling_seed": self.sampling_seed,
      "dataset_fingerprint": self.dataset_fingerprint,
      "motion_names": self.motion_names,
      "motion_lengths": self.motion_lengths.clone(),
      "motion_group_ids": self.motion_group_ids.clone(),
      "clip_motion_ids": self.clip_motion_ids.clone(),
      "clip_start_steps": self.clip_start_steps.clone(),
      "clip_end_steps": self.clip_end_steps.clone(),
      "last_sync_step": self._last_sync_step,
      "sampling_rng_state": self._sampling_generator.get_state().clone(),
      "sampling_rng_states": (
        None
        if self._distributed_sampling_rng_states is None
        else tuple(state.clone() for state in self._distributed_sampling_rng_states)
      ),
      "sampling_rng_world_size": (
        1
        if self._distributed_sampling_rng_states is None
        else len(self._distributed_sampling_rng_states)
      ),
      **{name: value.clone() for name, value in tensors.items()},
    }

  def load_state_dict(
    self,
    state: Mapping[str, Any],
    *,
    strict_metadata: bool = True,
    restore_sampling_rng: bool = True,
    sampling_rng_rank: int | None = None,
    validate_only: bool = False,
  ) -> None:
    """Validate and optionally restore an exact state without partial mutation."""
    if state.get("schema") != _STATE_SCHEMA:
      raise ValueError(
        "Unsupported GMT adaptive sampler state schema: quality and rehearsal "
        "sampling require v3. Start a new training run; legacy policy weights "
        "remain usable for actor-only playback and evaluation."
      )
    saved_clip_length = state.get("clip_length_steps")
    if (
      isinstance(saved_clip_length, bool)
      or not isinstance(saved_clip_length, int)
      or saved_clip_length != self.clip_length_steps
    ):
      raise ValueError("Sampler clip_length_steps does not match checkpoint state.")
    if strict_metadata:
      if state.get("config") != asdict(self.cfg):
        raise ValueError("Sampler configuration does not match checkpoint state.")
      if state.get("step_dt") != self.step_dt:
        raise ValueError("Sampler step_dt does not match checkpoint state.")
      if state.get("seed") != self.seed:
        raise ValueError("Sampler base seed does not match checkpoint state.")
      if (
        restore_sampling_rng
        and sampling_rng_rank is None
        and state.get("sampling_seed") != self.sampling_seed
      ):
        raise ValueError("Sampler sampling seed does not match checkpoint state.")
      if state.get("dataset_fingerprint") != self.dataset_fingerprint:
        raise ValueError("Dataset fingerprint does not match checkpoint state.")
      if (
        self.dataset_fingerprint is None
        and state.get("motion_names") != self.motion_names
      ):
        raise ValueError("Motion names do not match checkpoint state.")
    metadata_tensors = (
      "motion_lengths",
      "motion_group_ids",
      "clip_motion_ids",
      "clip_start_steps",
      "clip_end_steps",
    )
    for name in metadata_tensors:
      current = getattr(self, name)
      value = state.get(name)
      if not isinstance(value, torch.Tensor) or not torch.equal(
        value.to(current), current
      ):
        raise ValueError(f"Checkpoint {name} does not match the sampler dataset.")

    tensor_names = (
      "success_rate_ema",
      "tracking_error_ema",
      "rehearsal_age",
      "sample_counts",
      "success_counts",
      "failure_counts",
      "tracking_error_sums",
      "tracking_error_counts",
      "cached_probabilities",
      *(f"pending_{key}" for key in _PENDING_KEYS),
    )
    loaded: dict[str, torch.Tensor] = {}
    for name in tensor_names:
      value = state.get(name)
      if not isinstance(value, torch.Tensor):
        raise ValueError(f"Checkpoint is missing tensor {name!r}.")
      value = value.to(device=self.device, dtype=torch.float32)
      if value.shape != (self.num_clips,):
        raise ValueError(f"Checkpoint tensor {name!r} has an incompatible shape.")
      if not torch.isfinite(value).all() or torch.any(value < 0):
        raise ValueError(f"Checkpoint tensor {name!r} is invalid.")
      loaded[name] = value
    if torch.any(loaded["success_rate_ema"] > 1.0):
      raise ValueError("Checkpoint success rate EMA must be in [0, 1].")

    cached = loaded["cached_probabilities"]
    cached64 = cached.to(torch.float64)
    if not torch.isclose(cached64.sum(), cached64.new_tensor(1.0), atol=1e-6):
      raise ValueError("Checkpoint cached probabilities do not sum to one.")
    cached_group_mass = torch.zeros(
      self.num_groups, dtype=torch.float64, device=self.device
    ).scatter_add(0, self._clip_group_indexes, cached64)
    expected_group_mass = torch.full_like(cached_group_mass, 1.0 / self.num_groups)
    if not torch.allclose(cached_group_mass, expected_group_mass, atol=1e-6, rtol=1e-6):
      raise ValueError(
        "Checkpoint cached probabilities do not assign equal group mass."
      )
    last_sync_step = state.get("last_sync_step")
    if not isinstance(last_sync_step, int) or last_sync_step < 0:
      raise ValueError("Checkpoint last_sync_step is invalid.")
    sampling_rng_state = state.get("sampling_rng_state")
    if not isinstance(sampling_rng_state, torch.Tensor):
      raise ValueError("Checkpoint RNG states are missing.")
    raw_sampling_rng_states = state.get("sampling_rng_states")
    sampling_rng_world_size = state.get("sampling_rng_world_size", 1)
    if not isinstance(sampling_rng_world_size, int) or sampling_rng_world_size <= 0:
      raise ValueError("Checkpoint sampling RNG world size is invalid.")
    distributed_rng_states: tuple[torch.Tensor, ...] | None = None
    if raw_sampling_rng_states is not None:
      if not isinstance(raw_sampling_rng_states, (tuple, list)):
        raise ValueError("Checkpoint distributed sampling RNG states are invalid.")
      if len(raw_sampling_rng_states) != sampling_rng_world_size:
        raise ValueError(
          "Checkpoint sampling RNG state count does not match its world size."
        )
      validated_states: list[torch.Tensor] = []
      for rank, rng_state in enumerate(raw_sampling_rng_states):
        if not isinstance(rng_state, torch.Tensor):
          raise ValueError(f"Checkpoint sampling RNG state {rank} is invalid.")
        validator = torch.Generator(device=self.device)
        try:
          validator.set_state(rng_state.cpu())
        except RuntimeError as exc:
          raise ValueError(f"Checkpoint sampling RNG state {rank} is invalid.") from exc
        validated_states.append(rng_state.cpu().clone())
      distributed_rng_states = tuple(validated_states)
    elif sampling_rng_world_size != 1:
      raise ValueError("Checkpoint is missing distributed sampling RNG states.")

    runtime_world_size = (
      int(_distributed_api.get_world_size())
      if _distributed_api.is_available() and _distributed_api.is_initialized()
      else int(os.environ.get("WORLD_SIZE", "1"))
    )
    if sampling_rng_world_size != runtime_world_size:
      raise ValueError(
        "Checkpoint sampling RNG world size does not match the runtime world size."
      )

    if sampling_rng_rank is not None:
      if (
        isinstance(sampling_rng_rank, bool)
        or not isinstance(sampling_rng_rank, int)
        or sampling_rng_rank < 0
      ):
        raise ValueError("sampling_rng_rank must be a non-negative integer.")
      if distributed_rng_states is None:
        if sampling_rng_rank != 0:
          raise ValueError(
            "A nonzero sampling rank requires distributed sampling RNG states."
          )
      elif sampling_rng_rank >= len(distributed_rng_states):
        raise ValueError("sampling_rng_rank is outside the checkpoint world size.")
    if (
      _distributed_api.is_available()
      and _distributed_api.is_initialized()
      and distributed_rng_states
    ):
      if len(distributed_rng_states) != _distributed_api.get_world_size():
        raise ValueError(
          "Checkpoint sampling RNG world size does not match the process group."
        )
      if (
        sampling_rng_rank is not None
        and sampling_rng_rank != _distributed_api.get_rank()
      ):
        raise ValueError("sampling_rng_rank does not match the process-group rank.")

    selected_sampling_rng_state = sampling_rng_state
    if sampling_rng_rank is not None and distributed_rng_states is not None:
      selected_sampling_rng_state = distributed_rng_states[sampling_rng_rank]
    sampling_generator = torch.Generator(device=self.device)
    try:
      if restore_sampling_rng:
        sampling_generator.set_state(selected_sampling_rng_state.cpu())
    except RuntimeError as exc:
      raise ValueError("Checkpoint RNG state is invalid.") from exc

    if validate_only:
      return

    self.success_rate_ema.copy_(loaded["success_rate_ema"])
    self.tracking_error_ema.copy_(loaded["tracking_error_ema"])
    self.rehearsal_age.copy_(loaded["rehearsal_age"])
    self.sample_counts.copy_(loaded["sample_counts"])
    self.success_counts.copy_(loaded["success_counts"])
    self.failure_counts.copy_(loaded["failure_counts"])
    self.tracking_error_sums.copy_(loaded["tracking_error_sums"])
    self.tracking_error_counts.copy_(loaded["tracking_error_counts"])
    self.set_cached_probabilities(cached)
    for key in _PENDING_KEYS:
      self._pending[key].copy_(loaded[f"pending_{key}"])
    self._last_sync_step = last_sync_step
    self._distributed_sampling_rng_states = distributed_rng_states
    if restore_sampling_rng:
      self._sampling_generator = sampling_generator
