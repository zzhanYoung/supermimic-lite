from __future__ import annotations

import copy
import hashlib
import math
import os
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import torch
import torch.distributed as dist
from mjlab.managers import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  quat_apply,
  quat_error_magnitude,
  quat_from_euler_xyz,
  quat_inv,
  quat_mul,
  sample_uniform,
  yaw_quat,
)
from mjlab.viewer.debug_visualizer import DebugVisualizer

from .gmt_sampler import GmtAdaptiveMotionSampler, GmtAdaptiveSamplerCfg

if TYPE_CHECKING:
  from collections.abc import Callable

  import viser
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv

_DESIRED_FRAME_COLORS = ((1.0, 0.5, 0.5), (0.5, 1.0, 0.5), (0.5, 0.5, 1.0))
_distributed_api = cast(Any, dist)

_MOTION_KEYS = (
  "joint_pos",
  "joint_vel",
  "body_pos_w",
  "body_quat_w",
  "body_lin_vel_w",
  "body_ang_vel_w",
)


def _build_motion_ids(
  num_envs: int, num_motions: int, device: str
) -> tuple[torch.Tensor, int]:
  if num_motions <= 0:
    raise ValueError("num_motions must be positive.")
  envs_per_motion = num_envs // num_motions
  if envs_per_motion == 0:
    motion_ids = torch.arange(num_envs, dtype=torch.long, device=device)
  else:
    counts = torch.full(
      (num_motions,), envs_per_motion, dtype=torch.long, device=device
    )
    counts[: num_envs % num_motions] += 1
    motion_ids = torch.arange(num_motions, dtype=torch.long, device=device)
    motion_ids = motion_ids.repeat_interleave(counts)
  return motion_ids, envs_per_motion


def _validate_motion_fps(
  motion_files: tuple[str, ...],
  motion_fps: tuple[float | None, ...],
  control_fps: float,
) -> None:
  """Validate optional motion-rate metadata against the control rate."""
  for motion_file, fps in zip(motion_files, motion_fps, strict=True):
    if fps is not None and not math.isclose(
      fps, control_fps, rel_tol=1e-6, abs_tol=1e-6
    ):
      raise ValueError(
        f"Motion file {motion_file} has fps {fps:g}, but the "
        f"environment control rate is {control_fps:g} Hz."
      )


def _motion_dataset_fingerprint(motion_files: tuple[str, ...]) -> str:
  """Hash ordered motion identities and file contents independent of location."""
  dataset_hasher = hashlib.sha256()
  for motion_index, motion_file in enumerate(motion_files):
    motion_path = Path(motion_file)
    identity = f"{motion_index}:{motion_path.name}".encode()
    dataset_hasher.update(len(identity).to_bytes(8, "big"))
    dataset_hasher.update(identity)

    content_hasher = hashlib.sha256()
    with motion_path.open("rb") as file_obj:
      while chunk := file_obj.read(1024 * 1024):
        content_hasher.update(chunk)
    dataset_hasher.update(content_hasher.digest())
  return dataset_hasher.hexdigest()


def _distributed_is_ready() -> bool:
  """Return whether PyTorch's optional distributed process group is ready."""
  is_initialized = getattr(_distributed_api, "is_initialized", None)
  return bool(
    _distributed_api.is_available() and is_initialized is not None and is_initialized()
  )


class MotionLoader:
  """Load one or more motions into packed tensors with per-motion offsets."""

  def __init__(
    self,
    motion_files: tuple[str, ...],
    body_indexes: torch.Tensor,
    device: str = "cpu",
  ) -> None:
    if not motion_files:
      raise ValueError("At least one motion file is required.")

    self._body_indexes = body_indexes
    self.motion_files = motion_files
    self.motion_names = tuple(
      f"{index}:{Path(motion_file).name}"
      for index, motion_file in enumerate(motion_files)
    )

    loaded: dict[str, list[torch.Tensor]] = {key: [] for key in _MOTION_KEYS}
    expected_shapes: dict[str, tuple[int, ...]] = {}
    expected_body_count: int | None = None
    time_step_totals: list[int] = []
    motion_fps: list[float | None] = []
    max_body_index = (
      int(body_indexes.max().item()) if body_indexes.numel() > 0 else None
    )

    for motion_file in motion_files:
      tensors, fps = self._load_motion_file(motion_file, device)
      body_count = tensors["body_pos_w"].shape[1]
      if expected_body_count is not None and body_count != expected_body_count:
        raise ValueError(
          f"Motion file {motion_file} has {body_count} bodies; "
          f"expected {expected_body_count}."
        )
      expected_body_count = body_count
      if max_body_index is not None and max_body_index >= body_count:
        raise ValueError(
          f"Motion file {motion_file} has {body_count} bodies, which cannot "
          f"provide requested body index {max_body_index}."
        )
      tensors["body_pos_w"] = tensors["body_pos_w"][:, body_indexes]
      tensors["body_quat_w"] = tensors["body_quat_w"][:, body_indexes]
      tensors["body_lin_vel_w"] = tensors["body_lin_vel_w"][:, body_indexes]
      tensors["body_ang_vel_w"] = tensors["body_ang_vel_w"][:, body_indexes]

      for key, tensor in tensors.items():
        shape = tuple(tensor.shape[1:])
        if key in expected_shapes and shape != expected_shapes[key]:
          raise ValueError(
            f"Motion file {motion_file} has incompatible {key} shape {shape}; "
            f"expected {expected_shapes[key]}."
          )
        expected_shapes.setdefault(key, shape)
        loaded[key].append(tensor)
      time_step_totals.append(tensors["joint_pos"].shape[0])
      motion_fps.append(fps)

    self.joint_pos = torch.cat(loaded["joint_pos"], dim=0)
    self.joint_vel = torch.cat(loaded["joint_vel"], dim=0)
    self.body_pos_w = torch.cat(loaded["body_pos_w"], dim=0)
    self.body_quat_w = torch.cat(loaded["body_quat_w"], dim=0)
    self.body_lin_vel_w = torch.cat(loaded["body_lin_vel_w"], dim=0)
    self.body_ang_vel_w = torch.cat(loaded["body_ang_vel_w"], dim=0)

    self.time_step_totals = torch.tensor(
      time_step_totals, dtype=torch.long, device=device
    )
    self.frame_offsets = torch.cat(
      [
        torch.zeros(1, dtype=torch.long, device=device),
        torch.cumsum(self.time_step_totals[:-1], dim=0),
      ]
    )
    self.num_motions = len(motion_files)
    self.fps = tuple(motion_fps)
    self._dataset_fingerprint: str | None = None

  @property
  def dataset_fingerprint(self) -> str:
    """Content fingerprint, computed only when adaptive state needs it."""
    if self._dataset_fingerprint is None:
      self._dataset_fingerprint = _motion_dataset_fingerprint(self.motion_files)
    return self._dataset_fingerprint

  @staticmethod
  def _load_motion_file(
    motion_file: str, device: str
  ) -> tuple[dict[str, torch.Tensor], float | None]:
    try:
      with np.load(motion_file, allow_pickle=False) as data:
        missing = [key for key in _MOTION_KEYS if key not in data]
        if missing:
          raise ValueError(
            f"Motion file {motion_file} is missing required arrays: {missing}."
          )
        arrays = {key: np.asarray(data[key]) for key in _MOTION_KEYS}
        fps_array = np.asarray(data["fps"]) if "fps" in data else None
    except (OSError, ValueError) as exc:
      if isinstance(exc, ValueError) and str(exc).startswith("Motion file"):
        raise
      raise ValueError(f"Failed to load motion file {motion_file}: {exc}") from exc

    fps: float | None = None
    if fps_array is not None:
      if fps_array.size != 1:
        raise ValueError(
          f"Motion file {motion_file} has fps shape {fps_array.shape}; "
          "expected a scalar."
        )
      try:
        fps = float(fps_array.reshape(-1)[0])
      except (TypeError, ValueError) as exc:
        raise ValueError(
          f"Motion file {motion_file} has non-numeric fps dtype {fps_array.dtype}."
        ) from exc
      if not math.isfinite(fps) or fps <= 0.0:
        raise ValueError(
          f"Motion file {motion_file} has invalid fps {fps!r}; "
          "expected a finite positive value."
        )

    joint_pos = arrays["joint_pos"]
    joint_vel = arrays["joint_vel"]
    if joint_pos.ndim != 2:
      raise ValueError(
        f"Motion file {motion_file} has joint_pos shape {joint_pos.shape}; "
        "expected [time, joints]."
      )
    if joint_vel.shape != joint_pos.shape:
      raise ValueError(
        f"Motion file {motion_file} has joint_vel shape {joint_vel.shape}; "
        f"expected {joint_pos.shape}."
      )
    if joint_pos.shape[0] == 0:
      raise ValueError(f"Motion file {motion_file} contains no frames.")

    time_steps = joint_pos.shape[0]
    expected_body_dims = {
      "body_pos_w": 3,
      "body_quat_w": 4,
      "body_lin_vel_w": 3,
      "body_ang_vel_w": 3,
    }
    body_count: int | None = None
    for key, final_dim in expected_body_dims.items():
      array = arrays[key]
      if array.ndim != 3 or array.shape[0] != time_steps or array.shape[2] != final_dim:
        raise ValueError(
          f"Motion file {motion_file} has {key} shape {array.shape}; expected "
          f"[{time_steps}, bodies, {final_dim}]."
        )
      if body_count is not None and array.shape[1] != body_count:
        raise ValueError(
          f"Motion file {motion_file} has inconsistent body count in {key}: "
          f"{array.shape[1]} instead of {body_count}."
        )
      body_count = array.shape[1]

    for key, array in arrays.items():
      try:
        finite = np.isfinite(array)
      except TypeError as exc:
        raise ValueError(
          f"Motion file {motion_file} has non-numeric {key} dtype {array.dtype}."
        ) from exc
      if not finite.all():
        first_index = tuple(int(i) for i in np.argwhere(~finite)[0])
        value = array[first_index]
        raise ValueError(
          f"Motion file {motion_file} contains non-finite value in {key} "
          f"at index {first_index}: {value!r}."
        )

    return (
      {
        key: torch.tensor(array, dtype=torch.float32, device=device)
        for key, array in arrays.items()
      },
      fps,
    )

  @property
  def time_step_total(self) -> int:
    if self.num_motions != 1:
      raise RuntimeError("time_step_total is only defined for a single motion.")
    return int(self.time_step_totals[0].item())


class MotionCommand(CommandTerm):
  cfg: MotionCommandCfgBase
  _env: ManagerBasedRlEnv

  def __init__(self, cfg: MotionCommandCfgBase, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)

    self.robot: Entity = env.scene[cfg.entity_name]
    self.robot_anchor_body_index = self.robot.body_names.index(
      self.cfg.anchor_body_name
    )
    self.motion_anchor_body_index = self.cfg.body_names.index(self.cfg.anchor_body_name)
    self.body_indexes = torch.tensor(
      self.robot.find_bodies(self.cfg.body_names, preserve_order=True)[0],
      dtype=torch.long,
      device=self.device,
    )

    self.motion = MotionLoader(
      self.cfg.resolve_motion_files(), self.body_indexes, device=self.device
    )
    self.motion_ids, self.envs_per_motion = _build_motion_ids(
      self.num_envs, self.motion.num_motions, self.device
    )
    # Single-motion and ungrouped multi-motion commands belong to one logical
    # cluster. MultiMotionCommand replaces this lookup when explicit motion
    # groups are configured.
    self.cluster_names = ("default",)
    self.motion_cluster_ids = torch.zeros(
      self.motion.num_motions, dtype=torch.long, device=self.device
    )
    self.motion_lengths = self.motion.time_step_totals[self.motion_ids]
    self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    self.body_pos_relative_w = torch.zeros(
      self.num_envs, len(cfg.body_names), 3, device=self.device
    )
    self.body_quat_relative_w = torch.zeros(
      self.num_envs, len(cfg.body_names), 4, device=self.device
    )
    self.body_quat_relative_w[:, :, 0] = 1.0

    steps_per_bin = 1 / env.step_dt
    self.bin_counts = (self.motion.time_step_totals / steps_per_bin).long() + 1
    self.bin_count = int(self.bin_counts.max().item())
    bin_indexes = torch.arange(self.bin_count, device=self.device)
    self._bin_mask = bin_indexes.unsqueeze(0) < self.bin_counts.unsqueeze(1)
    self.bin_failed_count = torch.zeros(
      self.motion.num_motions,
      self.bin_count,
      dtype=torch.float,
      device=self.device,
    )
    self._current_bin_failed = torch.zeros(
      self.motion.num_motions,
      self.bin_count,
      dtype=torch.float,
      device=self.device,
    )
    self.kernel = torch.tensor(
      [self.cfg.adaptive_lambda**i for i in range(self.cfg.adaptive_kernel_size)],
      device=self.device,
    )
    self.kernel = self.kernel / self.kernel.sum()

    self.metrics["error_anchor_pos"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_anchor_rot"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_anchor_lin_vel"] = torch.zeros(
      self.num_envs, device=self.device
    )
    self.metrics["error_anchor_ang_vel"] = torch.zeros(
      self.num_envs, device=self.device
    )
    self.metrics["error_body_pos"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_body_rot"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_joint_pos"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_joint_vel"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["sampling_entropy"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["sampling_top1_prob"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["sampling_top1_bin"] = torch.zeros(self.num_envs, device=self.device)

    self._ghost_model = None
    self._ghost_color = np.array(cfg.viz.ghost_color, dtype=np.float32)
    self._pending_forward = False

  @property
  def command(self) -> torch.Tensor:
    return torch.cat([self.joint_pos, self.joint_vel], dim=1)

  @property
  def frame_indices(self) -> torch.Tensor:
    return self.motion.frame_offsets[self.motion_ids] + self.time_steps

  @property
  def motion_end_steps(self) -> torch.Tensor:
    """Return the exclusive end frame for each environment's current motion."""
    return self.motion_lengths

  @property
  def cluster_ids(self) -> torch.Tensor:
    """Return the logical motion cluster assigned to every environment."""
    return self.motion_cluster_ids[self.motion_ids]

  @property
  def num_clusters(self) -> int:
    """Number of logical motion clusters available to the command."""
    return len(self.cluster_names)

  @property
  def joint_pos(self) -> torch.Tensor:
    return self.motion.joint_pos[self.frame_indices]

  @property
  def joint_vel(self) -> torch.Tensor:
    return self.motion.joint_vel[self.frame_indices]

  @property
  def body_pos_w(self) -> torch.Tensor:
    return (
      self.motion.body_pos_w[self.frame_indices]
      + self._env.scene.env_origins[:, None, :]
    )

  @property
  def body_quat_w(self) -> torch.Tensor:
    return self.motion.body_quat_w[self.frame_indices]

  @property
  def body_lin_vel_w(self) -> torch.Tensor:
    return self.motion.body_lin_vel_w[self.frame_indices]

  @property
  def body_ang_vel_w(self) -> torch.Tensor:
    return self.motion.body_ang_vel_w[self.frame_indices]

  @property
  def anchor_pos_w(self) -> torch.Tensor:
    return (
      self.motion.body_pos_w[self.frame_indices, self.motion_anchor_body_index]
      + self._env.scene.env_origins
    )

  @property
  def anchor_quat_w(self) -> torch.Tensor:
    return self.motion.body_quat_w[self.frame_indices, self.motion_anchor_body_index]

  @property
  def anchor_lin_vel_w(self) -> torch.Tensor:
    return self.motion.body_lin_vel_w[self.frame_indices, self.motion_anchor_body_index]

  @property
  def anchor_ang_vel_w(self) -> torch.Tensor:
    return self.motion.body_ang_vel_w[self.frame_indices, self.motion_anchor_body_index]

  @property
  def robot_joint_pos(self) -> torch.Tensor:
    return self.robot.data.joint_pos

  @property
  def robot_joint_vel(self) -> torch.Tensor:
    return self.robot.data.joint_vel

  @property
  def robot_body_pos_w(self) -> torch.Tensor:
    return self.robot.data.body_link_pos_w[:, self.body_indexes]

  @property
  def robot_body_quat_w(self) -> torch.Tensor:
    return self.robot.data.body_link_quat_w[:, self.body_indexes]

  @property
  def robot_body_lin_vel_w(self) -> torch.Tensor:
    return self.robot.data.body_link_lin_vel_w[:, self.body_indexes]

  @property
  def robot_body_ang_vel_w(self) -> torch.Tensor:
    return self.robot.data.body_link_ang_vel_w[:, self.body_indexes]

  @property
  def robot_anchor_pos_w(self) -> torch.Tensor:
    return self.robot.data.body_link_pos_w[:, self.robot_anchor_body_index]

  @property
  def robot_anchor_quat_w(self) -> torch.Tensor:
    return self.robot.data.body_link_quat_w[:, self.robot_anchor_body_index]

  @property
  def robot_anchor_lin_vel_w(self) -> torch.Tensor:
    return self.robot.data.body_link_lin_vel_w[:, self.robot_anchor_body_index]

  @property
  def robot_anchor_ang_vel_w(self) -> torch.Tensor:
    return self.robot.data.body_link_ang_vel_w[:, self.robot_anchor_body_index]

  def _update_metrics(self):
    self.metrics["error_anchor_pos"] = torch.norm(
      self.anchor_pos_w - self.robot_anchor_pos_w, dim=-1
    )
    self.metrics["error_anchor_rot"] = quat_error_magnitude(
      self.anchor_quat_w, self.robot_anchor_quat_w
    )
    self.metrics["error_anchor_lin_vel"] = torch.norm(
      self.anchor_lin_vel_w - self.robot_anchor_lin_vel_w, dim=-1
    )
    self.metrics["error_anchor_ang_vel"] = torch.norm(
      self.anchor_ang_vel_w - self.robot_anchor_ang_vel_w, dim=-1
    )

    self.metrics["error_body_pos"] = torch.norm(
      self.body_pos_relative_w - self.robot_body_pos_w, dim=-1
    ).mean(dim=-1)
    self.metrics["error_body_rot"] = quat_error_magnitude(
      self.body_quat_relative_w, self.robot_body_quat_w
    ).mean(dim=-1)

    self.metrics["error_body_lin_vel"] = torch.norm(
      self.body_lin_vel_w - self.robot_body_lin_vel_w, dim=-1
    ).mean(dim=-1)
    self.metrics["error_body_ang_vel"] = torch.norm(
      self.body_ang_vel_w - self.robot_body_ang_vel_w, dim=-1
    ).mean(dim=-1)

    self.metrics["error_joint_pos"] = torch.norm(
      self.joint_pos - self.robot_joint_pos, dim=-1
    )
    self.metrics["error_joint_vel"] = torch.norm(
      self.joint_vel - self.robot_joint_vel, dim=-1
    )

  def _adaptive_sampling_probabilities(self) -> torch.Tensor:
    uniform_mass = self.cfg.adaptive_uniform_ratio / self.bin_counts.float().unsqueeze(
      1
    )
    probabilities = torch.where(
      self._bin_mask, self.bin_failed_count + uniform_mass, 0.0
    )

    # Match the single-motion non-causal smoothing at each motion's right edge.
    last_valid = probabilities.gather(1, (self.bin_counts - 1).unsqueeze(1))
    probabilities = torch.where(self._bin_mask, probabilities, last_valid)
    probabilities = torch.nn.functional.pad(
      probabilities.unsqueeze(1),
      (0, self.cfg.adaptive_kernel_size - 1),
      mode="replicate",
    )
    probabilities = torch.nn.functional.conv1d(
      probabilities, self.kernel.view(1, 1, -1)
    ).squeeze(1)
    probabilities = torch.where(self._bin_mask, probabilities, 0.0)
    return probabilities / probabilities.sum(dim=1, keepdim=True)

  def _adaptive_sampling(self, env_ids: torch.Tensor):
    episode_failed = self._env.termination_manager.terminated[env_ids]
    if torch.any(episode_failed):
      failed_env_ids = env_ids[episode_failed]
      failed_motion_ids = self.motion_ids[failed_env_ids]
      failed_bin_counts = self.bin_counts[failed_motion_ids]
      failed_motion_lengths = self.motion_lengths[failed_env_ids]
      failed_bins = (
        self.time_steps[failed_env_ids] * failed_bin_counts
      ) // failed_motion_lengths
      failed_bins = torch.minimum(failed_bins, failed_bin_counts - 1)
      flat_failed_bins = failed_motion_ids * self.bin_count + failed_bins
      self._current_bin_failed += torch.bincount(
        flat_failed_bins,
        minlength=self.motion.num_motions * self.bin_count,
      ).view(self.motion.num_motions, self.bin_count)

    sampling_probabilities = self._adaptive_sampling_probabilities()
    env_motion_ids = self.motion_ids[env_ids]
    sampled_bins = torch.multinomial(
      sampling_probabilities[env_motion_ids], 1, replacement=True
    ).squeeze(1)
    sampled_bin_counts = self.bin_counts[env_motion_ids]
    sampled_motion_lengths = self.motion_lengths[env_ids]
    self.time_steps[env_ids] = (
      (sampled_bins + sample_uniform(0.0, 1.0, (len(env_ids),), device=self.device))
      / sampled_bin_counts
      * (sampled_motion_lengths - 1)
    ).long()

    # Update metrics.
    entropy = -(sampling_probabilities * (sampling_probabilities + 1e-12).log()).sum(
      dim=1
    )
    normalized_entropy = torch.where(
      self.bin_counts > 1,
      entropy / self.bin_counts.float().log(),
      torch.ones_like(entropy),
    )
    top_probability, top_bin = sampling_probabilities.max(dim=1)
    self.metrics["sampling_entropy"][env_ids] = normalized_entropy[env_motion_ids]
    self.metrics["sampling_top1_prob"][env_ids] = top_probability[env_motion_ids]
    self.metrics["sampling_top1_bin"][env_ids] = (
      top_bin[env_motion_ids].float() / sampled_bin_counts
    )

  def _uniform_sampling(self, env_ids: torch.Tensor):
    self.time_steps[env_ids] = (
      torch.rand(len(env_ids), device=self.device) * self.motion_lengths[env_ids]
    ).long()
    env_bin_counts = self.bin_counts[self.motion_ids[env_ids]]
    self.metrics["sampling_entropy"][env_ids] = 1.0
    self.metrics["sampling_top1_prob"][env_ids] = 1.0 / env_bin_counts
    self.metrics["sampling_top1_bin"][env_ids] = 0.5

  def _stratified_sampling(self, env_ids: torch.Tensor) -> None:
    """Assign deterministic, horizon-safe trajectory windows by environment."""
    if len(env_ids) == 0:
      return

    # reset() advances the command once via compute(dt=0), so leave one extra
    # reference frame after every requested evaluation window.
    available_start_counts = self.motion_lengths[env_ids] - self._env.max_episode_length
    if torch.any(available_start_counts <= 0):
      shortest_motion = int(self.motion_lengths[env_ids].min().item())
      raise ValueError(
        "Stratified evaluation requires episode length in steps to be shorter "
        f"than the motion ({self._env.max_episode_length} >= "
        f"{shortest_motion}). Reduce --episode-length-s."
      )
    latest_start = available_start_counts - 1
    candidate_counts = latest_start + 1
    stratum_centers = (env_ids.to(torch.float32) + 0.5) / self.num_envs
    sampled_steps = (stratum_centers * candidate_counts).long()
    self.time_steps[env_ids] = torch.minimum(sampled_steps, latest_start)

    env_bin_counts = self.bin_counts[self.motion_ids[env_ids]]
    self.metrics["sampling_entropy"][env_ids] = 1.0
    self.metrics["sampling_top1_prob"][env_ids] = 1.0 / env_bin_counts
    self.metrics["sampling_top1_bin"][env_ids] = 0.5

  def _write_reference_state_to_sim(
    self,
    env_ids: torch.Tensor,
    root_pos: torch.Tensor,
    root_ori: torch.Tensor,
    root_lin_vel: torch.Tensor,
    root_ang_vel: torch.Tensor,
    joint_pos: torch.Tensor,
    joint_vel: torch.Tensor,
  ) -> None:
    """Clip joint positions and write root + joint state to sim."""
    soft_limits = self.robot.data.soft_joint_pos_limits[env_ids]
    joint_pos = torch.clip(joint_pos, soft_limits[:, :, 0], soft_limits[:, :, 1])
    self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

    root_state = torch.cat([root_pos, root_ori, root_lin_vel, root_ang_vel], dim=-1)
    self.robot.write_root_state_to_sim(root_state, env_ids=env_ids)
    self.robot.reset(env_ids=env_ids)

  def _sample_reference_frames(self, env_ids: torch.Tensor) -> None:
    if self.cfg.sampling_mode == "start":
      self.time_steps[env_ids] = 0
    elif self.cfg.sampling_mode == "uniform":
      self._uniform_sampling(env_ids)
    elif self.cfg.sampling_mode == "stratified":
      self._stratified_sampling(env_ids)
    else:
      assert self.cfg.sampling_mode == "adaptive"
      self._adaptive_sampling(env_ids)

  def _resample_command(self, env_ids: torch.Tensor):
    self._sample_reference_frames(env_ids)

    root_pos = self.body_pos_w[env_ids, 0].clone()
    root_ori = self.body_quat_w[env_ids, 0].clone()
    root_lin_vel = self.body_lin_vel_w[env_ids, 0].clone()
    root_ang_vel = self.body_ang_vel_w[env_ids, 0].clone()

    range_list = [
      self.cfg.pose_range.get(key, (0.0, 0.0))
      for key in ["x", "y", "z", "roll", "pitch", "yaw"]
    ]
    ranges = torch.tensor(range_list, device=self.device)
    rand_samples = sample_uniform(
      ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device
    )
    root_pos += rand_samples[:, 0:3]
    orientations_delta = quat_from_euler_xyz(
      rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5]
    )
    root_ori = quat_mul(orientations_delta, root_ori)
    range_list = [
      self.cfg.velocity_range.get(key, (0.0, 0.0))
      for key in ["x", "y", "z", "roll", "pitch", "yaw"]
    ]
    ranges = torch.tensor(range_list, device=self.device)
    rand_samples = sample_uniform(
      ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device
    )
    root_lin_vel += rand_samples[:, :3]
    root_ang_vel += rand_samples[:, 3:]

    joint_pos = self.joint_pos[env_ids].clone()
    joint_vel = self.joint_vel[env_ids]

    joint_pos += sample_uniform(
      lower=self.cfg.joint_position_range[0],
      upper=self.cfg.joint_position_range[1],
      size=joint_pos.shape,
      device=joint_pos.device,  # type: ignore
    )

    self._write_reference_state_to_sim(
      env_ids,
      root_pos,
      root_ori,
      root_lin_vel,
      root_ang_vel,
      joint_pos,
      joint_vel,
    )
    self._pending_forward = True

  def update_relative_body_poses(self) -> None:
    """Recompute ``body_pos_relative_w`` and ``body_quat_relative_w``.

    Called after ``reset_to_frame`` so that termination checks that
    compare relative body positions see the correct state.
    """
    anchor_pos_w_repeat = self.anchor_pos_w[:, None, :].repeat(
      1, len(self.cfg.body_names), 1
    )
    anchor_quat_w_repeat = self.anchor_quat_w[:, None, :].repeat(
      1, len(self.cfg.body_names), 1
    )
    robot_anchor_pos_w_repeat = self.robot_anchor_pos_w[:, None, :].repeat(
      1, len(self.cfg.body_names), 1
    )
    robot_anchor_quat_w_repeat = self.robot_anchor_quat_w[:, None, :].repeat(
      1, len(self.cfg.body_names), 1
    )

    delta_pos_w = robot_anchor_pos_w_repeat
    delta_pos_w[..., 2] = anchor_pos_w_repeat[..., 2]
    delta_ori_w = yaw_quat(
      quat_mul(robot_anchor_quat_w_repeat, quat_inv(anchor_quat_w_repeat))
    )

    self.body_quat_relative_w = quat_mul(delta_ori_w, self.body_quat_w)
    self.body_pos_relative_w = delta_pos_w + quat_apply(
      delta_ori_w, self.body_pos_w - anchor_pos_w_repeat
    )

  def reset(self, env_ids: torch.Tensor | slice | None) -> dict[str, float]:
    extras = super().reset(env_ids)
    # Reset-path resamples are followed by the env's own forward(); only
    # compute-path resamples (wraparound or timer expiry) need our refresh.
    self._pending_forward = False
    return extras

  def _update_command(self, env_ids: torch.Tensor | None = None):
    if env_ids is None:
      self.time_steps += 1
    else:
      self.time_steps[env_ids] += 1
    wrap_ids = torch.where(self.time_steps >= self.motion_end_steps)[0]
    if wrap_ids.numel() > 0:
      self._resample_command(wrap_ids)

    # _resample_command writes qpos/qvel but does not refresh derived
    # quantities; forward() so update_relative_body_poses reads the
    # post-teleport robot anchor instead of the stale pre-resample pose.
    # Covers both the wraparound above and a timer-expiry resample that ran
    # in compute before this call.
    if self._pending_forward:
      self._pending_forward = False
      self._env.sim.forward()

    self.update_relative_body_poses()

    # Fold failure counts into the EMA only on the per-step update so
    # manual resets do not decay it faster.
    if (
      env_ids is None
      and self.cfg.sampling_mode == "adaptive"
      and not getattr(self, "adaptive_sampler", None)
    ):
      self.bin_failed_count = (
        self.cfg.adaptive_alpha * self._current_bin_failed
        + (1 - self.cfg.adaptive_alpha) * self.bin_failed_count
      )
      self._current_bin_failed.zero_()

  def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
    """Draw ghost robot or frames based on visualization mode."""
    env_indices = visualizer.get_env_indices(self.num_envs)
    if not env_indices:
      return

    if self.cfg.viz.mode == "ghost":
      if self._ghost_model is None:
        # Build a ghost model with only visual geoms visible. Collision geoms (nonzero
        # contype/conaffinity) get alpha=0 so the viewer's alpha filter excludes them.
        self._ghost_model = copy.deepcopy(self._env.sim.mj_model)
        for gi in range(self._ghost_model.ngeom):
          if (
            self._ghost_model.geom_contype[gi] != 0
            or self._ghost_model.geom_conaffinity[gi] != 0
          ):
            self._ghost_model.geom_rgba[gi, 3] = 0
          else:
            self._ghost_model.geom_rgba[gi] = self._ghost_color

      entity: Entity = self._env.scene[self.cfg.entity_name]
      indexing = entity.indexing
      free_joint_q_adr = indexing.free_joint_q_adr.cpu().numpy()
      joint_q_adr = indexing.joint_q_adr.cpu().numpy()

      for batch in env_indices:
        qpos = np.zeros(self._env.sim.mj_model.nq)
        qpos[free_joint_q_adr[0:3]] = self.body_pos_w[batch, 0].cpu().numpy()
        qpos[free_joint_q_adr[3:7]] = self.body_quat_w[batch, 0].cpu().numpy()
        qpos[joint_q_adr] = self.joint_pos[batch].cpu().numpy()

        visualizer.add_ghost_mesh(
          qpos,
          model=self._ghost_model,
          label=f"ghost_{batch}",
        )

    elif self.cfg.viz.mode == "frames":
      for batch in env_indices:
        desired_body_pos = self.body_pos_w[batch].cpu().numpy()
        desired_body_quat = self.body_quat_w[batch]
        desired_body_rotm = matrix_from_quat(desired_body_quat).cpu().numpy()

        current_body_pos = self.robot_body_pos_w[batch].cpu().numpy()
        current_body_quat = self.robot_body_quat_w[batch]
        current_body_rotm = matrix_from_quat(current_body_quat).cpu().numpy()

        for i, body_name in enumerate(self.cfg.body_names):
          visualizer.add_frame(
            position=desired_body_pos[i],
            rotation_matrix=desired_body_rotm[i],
            scale=0.08,
            label=f"desired_{body_name}_{batch}",
            axis_colors=_DESIRED_FRAME_COLORS,
          )
          visualizer.add_frame(
            position=current_body_pos[i],
            rotation_matrix=current_body_rotm[i],
            scale=0.12,
            label=f"current_{body_name}_{batch}",
          )

        desired_anchor_pos = self.anchor_pos_w[batch].cpu().numpy()
        desired_anchor_quat = self.anchor_quat_w[batch]
        desired_rotation_matrix = matrix_from_quat(desired_anchor_quat).cpu().numpy()
        visualizer.add_frame(
          position=desired_anchor_pos,
          rotation_matrix=desired_rotation_matrix,
          scale=0.1,
          label=f"desired_anchor_{batch}",
          axis_colors=_DESIRED_FRAME_COLORS,
        )

        current_anchor_pos = self.robot_anchor_pos_w[batch].cpu().numpy()
        current_anchor_quat = self.robot_anchor_quat_w[batch]
        current_rotation_matrix = matrix_from_quat(current_anchor_quat).cpu().numpy()
        visualizer.add_frame(
          position=current_anchor_pos,
          rotation_matrix=current_rotation_matrix,
          scale=0.15,
          label=f"current_anchor_{batch}",
        )

  def create_gui(
    self,
    name: str,
    server: viser.ViserServer,
    get_env_idx: Callable[[], int],
    on_change: Callable[[], None] | None = None,
    request_action: Callable[[str, Any], None] | None = None,
  ) -> None:
    """Create motion scrubber controls in the Viser viewer."""
    max_frame = int(self.motion.time_step_totals.max().item()) - 1

    with server.gui.add_folder(name.capitalize()):
      scrubber = server.gui.add_slider(
        "Frame",
        min=0,
        max=max_frame,
        step=1,
        initial_value=0,
      )

      @scrubber.on_update
      def _(_) -> None:
        idx = get_env_idx()
        self.time_steps[idx] = min(
          int(scrubber.value), int(self.motion_lengths[idx].item()) - 1
        )
        if on_change is not None:
          on_change()

      all_envs_cb = server.gui.add_checkbox("All envs", initial_value=True)
      start_btn = server.gui.add_button("Start Here")

      @start_btn.on_click
      def _(_) -> None:
        if request_action is not None:
          request_action(
            "CUSTOM",
            {"type": "gui_reset", "all_envs": all_envs_cb.value},
          )

    self._scrubber_handles = (scrubber, all_envs_cb, start_btn)
    self._set_scrubber_disabled(True)

  def _set_scrubber_disabled(self, disabled: bool) -> None:
    """Enable or disable the motion scrubber GUI controls."""
    for handle in self._scrubber_handles:
      handle.disabled = disabled

  def on_viewer_pause(self, paused: bool) -> None:
    if hasattr(self, "_scrubber_handles"):
      self._set_scrubber_disabled(not paused)

  def apply_gui_reset(self, env_ids: torch.Tensor) -> bool:
    if not hasattr(self, "_scrubber_handles"):
      return False
    frame = int(self._scrubber_handles[0].value)
    self.reset_to_frame(env_ids, frame)
    # reset_to_frame writes qpos/qvel; forward so update_relative_body_poses
    # reads the scrubbed pose instead of the stale pre-scrub kinematics.
    self._env.sim.forward()
    self.update_relative_body_poses()
    return True

  def reset_to_frame(self, env_ids: torch.Tensor, frame: int) -> None:
    """Reset to exact reference state at a specific frame.

    Like ``_resample_command`` but deterministic: no random
    perturbations to pose, velocity, or joint positions.
    """
    requested_frames = torch.full_like(self.time_steps[env_ids], frame)
    self.time_steps[env_ids] = torch.minimum(
      requested_frames, self.motion_lengths[env_ids] - 1
    ).clamp(min=0)
    self._write_reference_state_to_sim(
      env_ids,
      self.body_pos_w[env_ids, 0],
      self.body_quat_w[env_ids, 0],
      self.body_lin_vel_w[env_ids, 0],
      self.body_ang_vel_w[env_ids, 0],
      self.joint_pos[env_ids],
      self.joint_vel[env_ids],
    )


class MultiMotionCommand(MotionCommand):
  """Randomly select a motion and one of its frames on every reset."""

  def __init__(self, cfg: MultiMotionCommandCfg, env: ManagerBasedRlEnv):
    if cfg.adaptive_sampler is not None and cfg.sampling_mode == "adaptive":
      raise ValueError(
        "The GMT adaptive sampler cannot be combined with "
        "sampling_mode='adaptive'; use sampling_mode='uniform'."
      )
    super().__init__(cfg, env)

    _validate_motion_fps(
      self.motion.motion_files, self.motion.fps, control_fps=1.0 / env.step_dt
    )

    self.adaptive_sampler: GmtAdaptiveMotionSampler | None = None
    self.clip_ids = torch.full(
      (self.num_envs,), -1, dtype=torch.long, device=self.device
    )
    self.clip_start_steps = torch.zeros(
      self.num_envs, dtype=torch.long, device=self.device
    )
    self.clip_end_steps = self.motion_lengths.clone()
    self._adaptive_episode_max_key_body_error = torch.zeros(
      self.num_envs, device=self.device
    )
    self._adaptive_episode_error_sum = torch.zeros(self.num_envs, device=self.device)
    self._adaptive_episode_error_steps = torch.zeros(
      self.num_envs, dtype=torch.long, device=self.device
    )
    self._adaptive_assignment_steps = torch.full(
      (self.num_envs,), -1, dtype=torch.long, device=self.device
    )

    motion_groups = cfg.resolve_motion_groups()
    if not motion_groups:
      self._uses_motion_groups = False
      self._cluster_motion_offsets = torch.zeros(
        1, dtype=torch.long, device=self.device
      )
      self._cluster_motion_counts = torch.full(
        (1,), self.motion.num_motions, dtype=torch.long, device=self.device
      )
    else:
      self._uses_motion_groups = True
      self.cluster_names = tuple(name for name, _ in motion_groups)
      group_counts = [len(motion_files) for _, motion_files in motion_groups]
      self._cluster_motion_counts = torch.tensor(
        group_counts, dtype=torch.long, device=self.device
      )
      self._cluster_motion_offsets = torch.cat(
        [
          torch.zeros(1, dtype=torch.long, device=self.device),
          torch.cumsum(self._cluster_motion_counts[:-1], dim=0),
        ]
      )
      self.motion_cluster_ids = torch.repeat_interleave(
        torch.arange(self.num_clusters, dtype=torch.long, device=self.device),
        self._cluster_motion_counts,
      )

    if cfg.adaptive_sampler is not None:
      env_seed = getattr(env.cfg, "seed", None)
      rank = int(os.environ.get("RANK", "0"))
      sampler_seed = (
        int(env_seed or 0)
        if cfg.adaptive_sampler.partition_seed is None
        else cfg.adaptive_sampler.partition_seed
      )
      sampling_seed = sampler_seed + rank
      self.adaptive_sampler = GmtAdaptiveMotionSampler(
        cfg.adaptive_sampler,
        self.motion.time_step_totals,
        step_dt=env.step_dt,
        clip_length_steps=env.max_episode_length,
        motion_group_ids=self.motion_cluster_ids,
        device=self.device,
        seed=sampler_seed,
        sampling_seed=sampling_seed,
        dataset_fingerprint=self.motion.dataset_fingerprint,
        motion_names=self.motion.motion_names,
      )
      self._initialize_adaptive_metrics()
      self._adaptive_checkpoint_sync_steps: set[int] = set()
      self._adaptive_partition_validated = False

  @property
  def motion_end_steps(self) -> torch.Tensor:
    """Return source-motion or virtual-clip exclusive end frames."""
    if self.adaptive_sampler is None:
      return self.motion_lengths
    return self.clip_end_steps

  def record_adaptive_key_body_error(self, errors: torch.Tensor) -> None:
    """Accumulate body/time mean error and a diagnostic peak on valid frames."""
    if self.adaptive_sampler is None:
      raise RuntimeError("Adaptive error recording requires an active GMT sampler.")
    if errors.ndim != 2 or errors.shape[0] != self.num_envs or errors.shape[1] == 0:
      raise ValueError(
        "Adaptive key-body errors must have shape "
        f"({self.num_envs}, num_key_bodies), got {tuple(errors.shape)}."
      )
    valid = torch.isfinite(errors).all(dim=-1)
    finite_errors = torch.where(valid[:, None], errors, 0.0)
    self._adaptive_episode_error_sum.add_(finite_errors.mean(dim=-1))
    self._adaptive_episode_error_steps.add_(valid.long())
    self._adaptive_episode_max_key_body_error.copy_(
      torch.maximum(
        self._adaptive_episode_max_key_body_error, finite_errors.amax(dim=-1)
      )
    )
    self.metrics["adaptive_max_key_body_error"].copy_(
      self._adaptive_episode_max_key_body_error
    )
    self.metrics["adaptive_mean_key_body_error"].copy_(
      self._adaptive_episode_error_sum / self._adaptive_episode_error_steps.clamp_min(1)
    )

  def _initialize_adaptive_metrics(self) -> None:
    assert self.adaptive_sampler is not None
    metric_names = (
      "adaptive_sampling_probability",
      "adaptive_sampling_entropy",
      "adaptive_effective_clip_count",
      "adaptive_max_probability",
      "adaptive_clip_coverage",
      "adaptive_recent_success_rate",
      "adaptive_success_p10",
      "adaptive_success_p50",
      "adaptive_success_p90",
      "adaptive_tracking_error_ema",
      "adaptive_error_p10",
      "adaptive_error_p50",
      "adaptive_error_p90",
      "adaptive_difficulty",
      "adaptive_rehearsal_age",
      "adaptive_rehearsal_age_max",
      "adaptive_mean_key_body_error",
      "adaptive_max_key_body_error",
      "adaptive_success_rate",
      "adaptive_failure_rate",
      "adaptive_neutral_timeouts",
    )
    for metric_name in metric_names:
      self.metrics[metric_name] = torch.zeros(self.num_envs, device=self.device)
    for cluster_name in self.cluster_names:
      self.metrics[f"adaptive_group_share/{cluster_name}"] = torch.zeros(
        self.num_envs, device=self.device
      )

    self._adaptive_neutral_timeout_count = torch.zeros((), device=self.device)
    self._adaptive_pending_neutral_timeout_count = torch.zeros((), device=self.device)
    self._adaptive_snapshot_revision = -1
    self._adaptive_snapshot: dict[str, torch.Tensor] = {}
    self._update_adaptive_snapshot_metrics()

  def _update_metrics(self) -> None:
    super()._update_metrics()
    if self.adaptive_sampler is not None:
      self._update_adaptive_snapshot_metrics()

  def _update_adaptive_snapshot_metrics(self) -> None:
    """Refill resettable metrics without recomputing committed global summaries."""
    assert self.adaptive_sampler is not None
    if self._adaptive_snapshot_revision != self.adaptive_sampler.state_revision:
      self._rebuild_adaptive_snapshot()
    for metric_name, value in self._adaptive_snapshot.items():
      self.metrics[metric_name].fill_(value)
    active = self.clip_ids >= 0
    clip_ids = self.clip_ids.clamp_min(0)
    per_clip = {
      "adaptive_recent_success_rate": self.adaptive_sampler.success_rate_ema,
      "adaptive_tracking_error_ema": self.adaptive_sampler.tracking_error_ema,
      "adaptive_difficulty": self._adaptive_difficulty,
      "adaptive_rehearsal_age": self.adaptive_sampler.rehearsal_age,
    }
    for metric_name, values in per_clip.items():
      self.metrics[metric_name].copy_(torch.where(active, values[clip_ids], 0.0))

  def _rebuild_adaptive_snapshot(self) -> None:
    """Compute global summaries once after synchronization or state restoration."""
    assert self.adaptive_sampler is not None
    probabilities = self.adaptive_sampler.probabilities()
    quantiles = probabilities.new_tensor([0.1, 0.5, 0.9])
    success_quantiles = torch.quantile(
      self.adaptive_sampler.success_rate_ema, quantiles
    )
    error_quantiles = torch.quantile(
      self.adaptive_sampler.tracking_error_ema, quantiles
    )
    self._adaptive_difficulty = self.adaptive_sampler.scores()
    num_clips = probabilities.numel()

    entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum()
    normalized_entropy = (
      entropy / probabilities.new_tensor(float(num_clips)).log()
      if num_clips > 1
      else probabilities.new_tensor(1.0)
    )
    max_probability, top_clip = probabilities.max(dim=0)
    sample_counts = self.adaptive_sampler.sample_counts
    success_count = self.adaptive_sampler.success_counts.sum()
    failure_count = self.adaptive_sampler.failure_counts.sum()
    total_outcomes = (
      success_count + failure_count + self._adaptive_neutral_timeout_count
    )
    outcome_denominator = total_outcomes.clamp_min(1.0)

    snapshot = {
      "adaptive_sampling_entropy": normalized_entropy,
      "adaptive_effective_clip_count": entropy.exp(),
      "adaptive_max_probability": max_probability,
      "adaptive_clip_coverage": (sample_counts > 0).float().mean(),
      "adaptive_success_p10": success_quantiles[0],
      "adaptive_success_p50": success_quantiles[1],
      "adaptive_success_p90": success_quantiles[2],
      "adaptive_error_p10": error_quantiles[0],
      "adaptive_error_p50": error_quantiles[1],
      "adaptive_error_p90": error_quantiles[2],
      "adaptive_rehearsal_age_max": self.adaptive_sampler.rehearsal_age.max(),
      "adaptive_success_rate": success_count / outcome_denominator,
      "adaptive_failure_rate": failure_count / outcome_denominator,
      "adaptive_neutral_timeouts": (
        self._adaptive_neutral_timeout_count / outcome_denominator
      ),
    }
    snapshot["sampling_entropy"] = normalized_entropy
    snapshot["sampling_top1_prob"] = max_probability
    snapshot["sampling_top1_bin"] = top_clip.float() / max(num_clips - 1, 1)

    group_sample_counts = torch.zeros(
      self.num_clusters, dtype=torch.float32, device=self.device
    ).scatter_add(0, self.adaptive_sampler.clip_group_ids, sample_counts)
    total_samples = group_sample_counts.sum().clamp_min(1.0)
    group_shares = group_sample_counts / total_samples
    for group_id, cluster_name in enumerate(self.cluster_names):
      snapshot[f"adaptive_group_share/{cluster_name}"] = group_shares[group_id]
    self._adaptive_snapshot = snapshot
    self._adaptive_snapshot_revision = self.adaptive_sampler.state_revision

  def _observe_adaptive_outcomes(self, env_ids: torch.Tensor) -> None:
    assert self.adaptive_sampler is not None
    old_clip_ids = self.clip_ids[env_ids]
    active = old_clip_ids >= 0
    # Termination buffers are reset after commands. A second/manual reset in
    # the same environment step can therefore still expose the previous
    # episode's done flags; only assignments made before this step may consume
    # them.
    current_step = int(self._env.common_step_counter)
    active &= self._adaptive_assignment_steps[env_ids] < current_step
    if not torch.any(active):
      return

    active_env_ids = env_ids[active]
    old_clip_ids = old_clip_ids[active]
    termination_manager = self._env.termination_manager
    failures = termination_manager.terminated[active_env_ids]
    if "motion_end" in termination_manager.active_terms:
      completed = termination_manager.get_term("motion_end")[active_env_ids]
    else:
      completed = torch.zeros_like(failures)
    successes = completed & ~failures
    neutral_timeouts = (
      termination_manager.time_outs[active_env_ids] & ~completed & ~failures
    )
    observed = successes | failures
    if torch.any(observed):
      observed_env_ids = active_env_ids[observed]
      error_steps = self._adaptive_episode_error_steps[observed_env_ids]
      self.adaptive_sampler.observe(
        old_clip_ids[observed],
        successes[observed],
        self._adaptive_episode_error_sum[observed_env_ids] / error_steps.clamp_min(1),
        error_valid=error_steps > 0,
      )
    self._adaptive_pending_neutral_timeout_count += neutral_timeouts.sum()

  def _sample_adaptive_clips(self, env_ids: torch.Tensor) -> None:
    assert self.adaptive_sampler is not None
    batch = self.adaptive_sampler.sample(len(env_ids))
    self.clip_ids[env_ids] = batch.clip_ids
    self.motion_ids[env_ids] = batch.motion_ids
    self.motion_lengths[env_ids] = self.motion.time_step_totals[batch.motion_ids]
    self.clip_start_steps[env_ids] = batch.start_steps
    self.clip_end_steps[env_ids] = batch.end_steps
    self.time_steps[env_ids] = batch.start_steps
    self._adaptive_assignment_steps[env_ids] = int(self._env.common_step_counter)
    self._adaptive_episode_max_key_body_error[env_ids] = 0.0
    self._adaptive_episode_error_sum[env_ids] = 0.0
    self._adaptive_episode_error_steps[env_ids] = 0
    self.metrics["adaptive_max_key_body_error"][env_ids] = 0.0
    self.metrics["adaptive_mean_key_body_error"][env_ids] = 0.0

    self.metrics["adaptive_sampling_probability"][env_ids] = batch.probabilities
    self._update_adaptive_snapshot_metrics()

  def _synchronize_adaptive_statistics(self, step: int | None) -> bool:
    """Commit one globally consistent sampler refresh.

    Distributed training may construct environments before its process group is
    ready. In that case local observations remain queued until every rank can
    participate, preventing rank-specific distributions from being committed.
    """
    assert self.adaptive_sampler is not None
    distributed_expected = int(os.environ.get("WORLD_SIZE", "1")) > 1
    distributed_ready = _distributed_is_ready()
    if distributed_expected and not distributed_ready:
      return False
    if distributed_ready:
      self._validate_adaptive_partition_across_ranks()

    statistics = self.adaptive_sampler.pending_statistics()
    neutral_timeout_count = self._adaptive_pending_neutral_timeout_count.clone()

    if distributed_ready:
      reduce_op = _distributed_api.ReduceOp.SUM
      all_reduce = _distributed_api.all_reduce
      for value in statistics.values():
        all_reduce(value, op=reduce_op)
      all_reduce(neutral_timeout_count, op=reduce_op)

    self.adaptive_sampler.commit_pending_statistics(statistics, step=step)
    if distributed_ready:
      broadcast = _distributed_api.broadcast
      broadcast(self.adaptive_sampler.cached_probabilities, src=0)
    self._gather_adaptive_sampling_rng_states()
    self._adaptive_neutral_timeout_count += neutral_timeout_count
    self._adaptive_pending_neutral_timeout_count.zero_()
    self._update_adaptive_snapshot_metrics()
    return True

  def _adaptive_clip_table_digest(self) -> str:
    """Return a stable digest for cross-rank clip identity validation."""
    assert self.adaptive_sampler is not None
    digest = hashlib.sha256()
    if self.adaptive_sampler.dataset_fingerprint is not None:
      digest.update(self.adaptive_sampler.dataset_fingerprint.encode())
    for tensor in (
      self.adaptive_sampler.motion_group_ids,
      self.adaptive_sampler.clip_motion_ids,
      self.adaptive_sampler.clip_start_steps,
      self.adaptive_sampler.clip_end_steps,
    ):
      value = tensor.detach().to(device="cpu", dtype=torch.long).contiguous()
      digest.update(value.numpy().tobytes())
    return digest.hexdigest()

  def _validate_adaptive_partition_across_ranks(self) -> None:
    """Reject mismatched virtual clip identities before the first all-reduce."""
    if getattr(self, "_adaptive_partition_validated", False):
      return
    local_digest = self._adaptive_clip_table_digest()
    gathered: list[object] = [None] * _distributed_api.get_world_size()
    _distributed_api.all_gather_object(gathered, local_digest)
    if any(value != local_digest for value in gathered):
      raise RuntimeError(
        "GMT virtual clip tables differ across distributed ranks. Configure "
        "the same motions, groups, and episode length on every rank."
      )
    self._adaptive_partition_validated = True

  def synchronize_adaptive_sampler(self, step: int | None = None) -> bool:
    """Refresh adaptive state when its globally shared interval elapses."""
    if self.adaptive_sampler is None:
      return False
    if step is None:
      step = int(self._env.common_step_counter)
    if not self.adaptive_sampler.should_sync(step):
      return False
    return self._synchronize_adaptive_statistics(step)

  def flush_adaptive_sampler_for_checkpoint(self) -> bool:
    """Commit pending state on all ranks without shifting the periodic cadence."""
    if self.adaptive_sampler is None:
      return False
    return self._synchronize_adaptive_statistics(step=None)

  def configure_adaptive_checkpoint_sync_steps(self, steps: Iterable[int]) -> None:
    """Schedule all-rank sampler flushes immediately before checkpoints."""
    normalized: set[int] = set()
    for step in steps:
      if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError(
          "Adaptive checkpoint sync steps must be non-negative integers."
        )
      normalized.add(step)
    self._adaptive_checkpoint_sync_steps = normalized

  def _gather_adaptive_sampling_rng_states(self) -> None:
    """Cache each rank's sampling stream for an exact checkpoint resume."""
    assert self.adaptive_sampler is not None
    local_state = self.adaptive_sampler.sampling_rng_state
    distributed_ready = _distributed_is_ready()
    if not distributed_ready:
      self.adaptive_sampler.set_distributed_sampling_rng_states((local_state,))
      return

    get_world_size = _distributed_api.get_world_size
    all_gather_object = _distributed_api.all_gather_object
    gathered_objects: list[object] = [None] * get_world_size()
    all_gather_object(gathered_objects, local_state.cpu())
    gathered_states: list[torch.Tensor] = []
    for rank, gathered_state in enumerate(gathered_objects):
      if not isinstance(gathered_state, torch.Tensor):
        raise RuntimeError(
          f"Distributed sampling RNG state for rank {rank} is not a tensor."
        )
      gathered_states.append(gathered_state)
    self.adaptive_sampler.set_distributed_sampling_rng_states(gathered_states)

  def _update_command(self, env_ids: torch.Tensor | None = None) -> None:
    if self.adaptive_sampler is None:
      super()._update_command(env_ids)
      return

    # Adaptive clips end through the named ``motion_end`` termination. Do not
    # let the base command resample in-place at the boundary: auto_reset=False
    # must leave the completed clip intact until the caller resets, and clips
    # shorter than two frames must never trigger a second sample during
    # reset-time ``compute(dt=0)``.
    if env_ids is None:
      self.time_steps.copy_(
        torch.minimum(self.time_steps + 1, self.motion_end_steps - 1)
      )
    else:
      self.time_steps[env_ids] = torch.minimum(
        self.time_steps[env_ids] + 1, self.motion_end_steps[env_ids] - 1
      )
    self.update_relative_body_poses()

    if env_ids is not None:
      return

    step = int(self._env.common_step_counter)
    checkpoint_steps = getattr(self, "_adaptive_checkpoint_sync_steps", set())
    if step in checkpoint_steps:
      periodic_sync = self.adaptive_sampler.should_sync(step)
      sync_step = step if periodic_sync else None
      if self._synchronize_adaptive_statistics(sync_step):
        checkpoint_steps.discard(step)
    else:
      self.synchronize_adaptive_sampler(step)

  def _sample_reference_frames(self, env_ids: torch.Tensor) -> None:
    adaptive_sampler = getattr(self, "adaptive_sampler", None)
    if adaptive_sampler is not None:
      if len(env_ids) == 0:
        return
      self._observe_adaptive_outcomes(env_ids)
      self._sample_adaptive_clips(env_ids)
      return

    # Preserve the non-adaptive random stream and behavior unless explicit
    # motion groups are configured. getattr also supports lightweight command
    # fixtures that bypass __init__.
    if not getattr(self, "_uses_motion_groups", False):
      self.motion_ids[env_ids] = torch.randint(
        self.motion.num_motions, (len(env_ids),), device=self.device
      )
      self.motion_lengths[env_ids] = self.motion.time_step_totals[
        self.motion_ids[env_ids]
      ]
      self._uniform_sampling(env_ids)
      return

    sampled_cluster_ids = torch.randint(
      self.num_clusters, (len(env_ids),), device=self.device
    )
    sampled_cluster_counts = self._cluster_motion_counts[sampled_cluster_ids]
    sampled_group_offsets = (
      torch.rand(len(env_ids), device=self.device) * sampled_cluster_counts
    ).long()
    self.motion_ids[env_ids] = (
      self._cluster_motion_offsets[sampled_cluster_ids] + sampled_group_offsets
    )
    self.motion_lengths[env_ids] = self.motion.time_step_totals[
      self.motion_ids[env_ids]
    ]
    self._uniform_sampling(env_ids)


@dataclass(kw_only=True)
class MotionCommandCfgBase(CommandTermCfg):
  """Shared configuration for single- and multi-motion tracking commands."""

  anchor_body_name: str
  body_names: tuple[str, ...]
  entity_name: str
  pose_range: dict[str, tuple[float, float]] = field(default_factory=dict)
  velocity_range: dict[str, tuple[float, float]] = field(default_factory=dict)
  joint_position_range: tuple[float, float] = (-0.52, 0.52)
  adaptive_kernel_size: int = 1
  adaptive_lambda: float = 0.8
  adaptive_uniform_ratio: float = 0.1
  adaptive_alpha: float = 0.001
  sampling_mode: Literal["adaptive", "uniform", "start", "stratified"] = "adaptive"

  @dataclass
  class VizCfg:
    mode: Literal["ghost", "frames"] = "ghost"
    ghost_color: tuple[float, float, float, float] = (0.5, 0.7, 0.5, 0.5)

  viz: VizCfg = field(default_factory=VizCfg)

  def resolve_motion_files(self) -> tuple[str, ...]:
    raise NotImplementedError

  def build(self, env: ManagerBasedRlEnv) -> MotionCommand:
    return MotionCommand(self, env)


@dataclass(kw_only=True)
class MotionCommandCfg(MotionCommandCfgBase):
  """Configuration for tracking one NPZ motion file."""

  motion_file: str

  def resolve_motion_files(self) -> tuple[str, ...]:
    motion_path = Path(self.motion_file).expanduser()
    if not motion_path.is_file():
      raise FileNotFoundError(f"Motion file not found: {motion_path}")
    return (str(motion_path.resolve()),)


@dataclass(kw_only=True)
class MultiMotionCommandCfg(MotionCommandCfgBase):
  """Configuration for reset-time random sampling from multiple motions.

  ``motion_folder`` retains the default behavior of uniformly sampling all
  top-level NPZ files. ``motion_groups`` instead maps stable cluster names to
  explicit files and samples a cluster uniformly before sampling a motion
  uniformly within it.
  """

  motion_folder: str = ""
  motion_groups: dict[str, tuple[str, ...]] = field(default_factory=dict)
  adaptive_sampler: GmtAdaptiveSamplerCfg | None = None
  sampling_mode: Literal["adaptive", "uniform", "start", "stratified"] = "uniform"

  def resolve_motion_groups(
    self,
  ) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Resolve and validate explicitly configured motion groups."""
    if not self.motion_groups:
      return ()
    if self.motion_folder:
      raise ValueError(
        "motion_folder and motion_groups are mutually exclusive; configure "
        "only one motion source."
      )

    resolved_groups: list[tuple[str, tuple[str, ...]]] = []
    seen_files: set[str] = set()
    for cluster_name, motion_files in self.motion_groups.items():
      if not cluster_name:
        raise ValueError("Motion cluster names must be non-empty.")
      if not motion_files:
        raise ValueError(
          f"Motion cluster {cluster_name!r} must contain at least one file."
        )

      resolved_files: list[str] = []
      for motion_file in motion_files:
        motion_path = Path(motion_file).expanduser()
        if not motion_path.is_file():
          raise FileNotFoundError(f"Motion file not found: {motion_path}")
        resolved_path = str(motion_path.resolve())
        if resolved_path in seen_files:
          raise ValueError(
            f"Motion file appears in more than one cluster: {resolved_path}"
          )
        seen_files.add(resolved_path)
        resolved_files.append(resolved_path)
      resolved_groups.append((cluster_name, tuple(resolved_files)))
    return tuple(resolved_groups)

  def resolve_motion_files(self) -> tuple[str, ...]:
    motion_groups = self.resolve_motion_groups()
    if motion_groups:
      return tuple(
        motion_file
        for _, grouped_motion_files in motion_groups
        for motion_file in grouped_motion_files
      )

    motion_folder = Path(self.motion_folder).expanduser()
    if not motion_folder.is_dir():
      raise NotADirectoryError(f"Motion folder not found: {motion_folder}")
    motion_files = tuple(
      str(path.resolve())
      for path in sorted(motion_folder.glob("*.npz"), key=lambda path: path.name)
      if path.is_file()
    )
    if not motion_files:
      raise ValueError(f"Motion folder contains no .npz files: {motion_folder}")
    return motion_files

  def build(self, env: ManagerBasedRlEnv) -> MotionCommand:
    return MultiMotionCommand(self, env)
