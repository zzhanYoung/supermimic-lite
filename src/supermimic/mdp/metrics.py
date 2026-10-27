from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.utils.lab_api.math import quat_apply_inverse, yaw_quat

from .rewards import _get_body_index, _get_required_body_indexes

if TYPE_CHECKING:
  from supermimic.mdp.commands import MotionCommand


def compute_local_key_body_error(
  command: MotionCommand,
  body_names: tuple[str, ...],
  root_body_name: str = "pelvis",
) -> torch.Tensor:
  """Return the maximum key-body error in each trajectory's own heading frame."""
  return compute_local_key_body_errors(command, body_names, root_body_name).amax(-1)


def compute_local_key_body_errors(
  command: MotionCommand,
  body_names: tuple[str, ...],
  root_body_name: str = "pelvis",
) -> torch.Tensor:
  """Return per-key-body distances in each trajectory's own root heading frame."""
  root_index = _get_body_index(command, root_body_name)
  body_indexes = _get_required_body_indexes(command, body_names)
  num_bodies = len(body_indexes)
  reference_delta = (
    command.body_pos_w[:, body_indexes] - command.body_pos_w[:, root_index, None, :]
  )
  robot_delta = (
    command.robot_body_pos_w[:, body_indexes]
    - command.robot_body_pos_w[:, root_index, None, :]
  )
  reference_heading = yaw_quat(command.body_quat_w[:, root_index])[:, None, :]
  robot_heading = yaw_quat(command.robot_body_quat_w[:, root_index])[:, None, :]
  reference_local = quat_apply_inverse(
    reference_heading.expand(-1, num_bodies, -1), reference_delta
  )
  robot_local = quat_apply_inverse(
    robot_heading.expand(-1, num_bodies, -1), robot_delta
  )
  return torch.linalg.vector_norm(robot_local - reference_local, dim=-1)


def compute_mpkpe(command: MotionCommand) -> torch.Tensor:
  """Compute Mean Per-Keybody Position Error (MPKPE).

  MPKPE measures the average Euclidean distance between the reference and
  actual key body positions in the global world frame. It captures all
  tracking error, including global translation and heading drift.
  """
  pos_error = command.body_pos_w - command.robot_body_pos_w
  per_body_error = torch.norm(pos_error, dim=-1)  # (num_envs, num_bodies)
  return per_body_error.mean(dim=-1)  # (num_envs,)


def compute_mpjpe(command: MotionCommand) -> torch.Tensor:
  """Compute mean per-joint position error in radians."""
  return torch.abs(command.joint_pos - command.robot_joint_pos).mean(dim=-1)


def compute_mpjve(command: MotionCommand) -> torch.Tensor:
  """Compute mean per-joint velocity error in radians per second."""
  return torch.abs(command.joint_vel - command.robot_joint_vel).mean(dim=-1)


def compute_linear_velocity_error(command: MotionCommand) -> torch.Tensor:
  """Compute pelvis linear-velocity tracking error in the world frame."""
  pelvis_index = _get_pelvis_index(command)
  return torch.norm(
    command.body_lin_vel_w[:, pelvis_index]
    - command.robot_body_lin_vel_w[:, pelvis_index],
    dim=-1,
  )


def compute_yaw_velocity_error(command: MotionCommand) -> torch.Tensor:
  """Compute pelvis world-z angular-velocity tracking error."""
  pelvis_index = _get_pelvis_index(command)
  return torch.abs(
    command.body_ang_vel_w[:, pelvis_index, 2]
    - command.robot_body_ang_vel_w[:, pelvis_index, 2]
  )


def _get_pelvis_index(command: MotionCommand) -> int:
  """Return the pelvis index in the command's tracked body list."""
  try:
    return command.cfg.body_names.index("pelvis")
  except ValueError as exc:
    raise ValueError(
      "The pelvis body must be tracked to compute pelvis velocity errors. "
      f"Available bodies: {tuple(command.cfg.body_names)}."
    ) from exc
