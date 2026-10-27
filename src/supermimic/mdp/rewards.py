from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import (
  euler_xyz_from_quat,
  quat_apply_inverse,
  quat_error_magnitude,
  wrap_to_pi,
  yaw_quat,
)

from .commands import MotionCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def _get_body_indexes(
  command: MotionCommand, body_names: tuple[str, ...] | None
) -> list[int]:
  return [
    i
    for i, name in enumerate(command.cfg.body_names)
    if (body_names is None) or (name in body_names)
  ]


def _get_body_index(command: MotionCommand, body_name: str) -> int:
  """Resolve exactly one tracked body, failing clearly for invalid configs."""
  matches = [
    index
    for index, candidate in enumerate(command.cfg.body_names)
    if candidate == body_name
  ]
  if len(matches) != 1:
    raise ValueError(
      f"Expected body {body_name!r} exactly once in the motion command's "
      f"body_names, found {len(matches)}."
    )
  return matches[0]


def _get_required_body_indexes(
  command: MotionCommand, body_names: tuple[str, ...]
) -> list[int]:
  """Resolve an ordered, non-empty set of unique tracked bodies."""
  if not body_names:
    raise ValueError("body_names must contain at least one tracked body.")
  if len(set(body_names)) != len(body_names):
    raise ValueError("body_names must not contain duplicates.")
  return [_get_body_index(command, body_name) for body_name in body_names]


def _validate_std(name: str, std: float) -> None:
  if std <= 0.0:
    raise ValueError(f"{name} must be positive, got {std}.")


def motion_joint_position_error_exp(
  env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
  """Track all commanded joint positions with an exponential squared error."""
  _validate_std("std", std)
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  error = torch.sum(torch.square(command.joint_pos - command.robot_joint_pos), dim=-1)
  return torch.exp(-error / std**2)


def motion_joint_velocity_error_exp(
  env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
  """Track all commanded joint velocities with an exponential squared error."""
  _validate_std("std", std)
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  error = torch.sum(torch.square(command.joint_vel - command.robot_joint_vel), dim=-1)
  return torch.exp(-error / std**2)


def motion_root_pose_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  orientation_std: float,
  height_std: float,
  root_body_name: str = "pelvis",
) -> torch.Tensor:
  """Track root roll/pitch and world height while ignoring world XY and yaw."""
  _validate_std("orientation_std", orientation_std)
  _validate_std("height_std", height_std)
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  root_index = _get_body_index(command, root_body_name)

  reference_quat = command.body_quat_w[:, root_index]
  robot_quat = command.robot_body_quat_w[:, root_index]
  reference_roll, reference_pitch, _ = euler_xyz_from_quat(reference_quat)
  robot_roll, robot_pitch, _ = euler_xyz_from_quat(robot_quat)
  roll_error = wrap_to_pi(robot_roll - reference_roll)
  pitch_error = wrap_to_pi(robot_pitch - reference_pitch)
  orientation_error = torch.square(roll_error) + torch.square(pitch_error)

  height_error = torch.square(
    command.robot_body_pos_w[:, root_index, 2] - command.body_pos_w[:, root_index, 2]
  )
  return torch.exp(
    -orientation_error / orientation_std**2 - height_error / height_std**2
  )


def motion_root_velocity_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  linear_std: float,
  angular_std: float,
  root_body_name: str = "pelvis",
) -> torch.Tensor:
  """Track root twist in each trajectory's own heading-aligned local frame."""
  _validate_std("linear_std", linear_std)
  _validate_std("angular_std", angular_std)
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  root_index = _get_body_index(command, root_body_name)

  reference_heading = yaw_quat(command.body_quat_w[:, root_index])
  robot_heading = yaw_quat(command.robot_body_quat_w[:, root_index])
  reference_linear = quat_apply_inverse(
    reference_heading, command.body_lin_vel_w[:, root_index]
  )
  robot_linear = quat_apply_inverse(
    robot_heading, command.robot_body_lin_vel_w[:, root_index]
  )
  reference_angular = quat_apply_inverse(
    reference_heading, command.body_ang_vel_w[:, root_index]
  )
  robot_angular = quat_apply_inverse(
    robot_heading, command.robot_body_ang_vel_w[:, root_index]
  )

  linear_error = torch.sum(torch.square(robot_linear - reference_linear), dim=-1)
  angular_error = torch.sum(torch.square(robot_angular - reference_angular), dim=-1)
  return torch.exp(-linear_error / linear_std**2 - angular_error / angular_std**2)


def motion_local_key_body_position_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  body_names: tuple[str, ...],
  root_body_name: str = "pelvis",
) -> torch.Tensor:
  """Track key bodies relative to each trajectory's heading-aligned root frame."""
  _validate_std("std", std)
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
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

  body_error = torch.sum(torch.square(robot_local - reference_local), dim=-1)
  return torch.exp(-body_error.mean(dim=-1) / std**2)


def joint_velocity_l2_norm(
  env: ManagerBasedRlEnv, entity_name: str = "robot"
) -> torch.Tensor:
  """Return the Euclidean norm of all joint velocities."""
  return torch.linalg.vector_norm(env.scene[entity_name].data.joint_vel, dim=-1)


def joint_acceleration_l2_norm(
  env: ManagerBasedRlEnv, entity_name: str = "robot"
) -> torch.Tensor:
  """Return the Euclidean norm of all joint accelerations."""
  return torch.linalg.vector_norm(env.scene[entity_name].data.joint_acc, dim=-1)


def motion_global_anchor_position_error_exp(
  env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  error = torch.sum(
    torch.square(command.anchor_pos_w - command.robot_anchor_pos_w), dim=-1
  )
  return torch.exp(-error / std**2)


def motion_global_anchor_height_error_exp(
  env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
  """Track anchor world height, independent of horizontal drift and body tilt."""
  _validate_std("std", std)
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  error = torch.square(command.anchor_pos_w[:, 2] - command.robot_anchor_pos_w[:, 2])
  return torch.exp(-error / std**2)


def motion_global_anchor_orientation_error_exp(
  env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  error = quat_error_magnitude(command.anchor_quat_w, command.robot_anchor_quat_w) ** 2
  return torch.exp(-error / std**2)


def motion_relative_body_position_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  body_indexes = _get_body_indexes(command, body_names)
  error = torch.sum(
    torch.square(
      command.body_pos_relative_w[:, body_indexes]
      - command.robot_body_pos_w[:, body_indexes]
    ),
    dim=-1,
  )
  return torch.exp(-error.mean(-1) / std**2)


def motion_relative_body_orientation_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  body_indexes = _get_body_indexes(command, body_names)
  error = (
    quat_error_magnitude(
      command.body_quat_relative_w[:, body_indexes],
      command.robot_body_quat_w[:, body_indexes],
    )
    ** 2
  )
  return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_linear_velocity_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  body_indexes = _get_body_indexes(command, body_names)
  error = torch.sum(
    torch.square(
      command.body_lin_vel_w[:, body_indexes]
      - command.robot_body_lin_vel_w[:, body_indexes]
    ),
    dim=-1,
  )
  return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_angular_velocity_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  body_indexes = _get_body_indexes(command, body_names)
  error = torch.sum(
    torch.square(
      command.body_ang_vel_w[:, body_indexes]
      - command.robot_body_ang_vel_w[:, body_indexes]
    ),
    dim=-1,
  )
  return torch.exp(-error.mean(-1) / std**2)


def self_collision_cost(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  force_threshold: float = 10.0,
) -> torch.Tensor:
  """Penalize self-collisions.

  When the sensor provides force history (from ``history_length > 0``),
  counts substeps where any contact force exceeds *force_threshold*.
  Falls back to the instantaneous ``found`` count otherwise.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  data = sensor.data
  if data.force_history is not None:
    # force_history: [B, N, H, 3]
    force_mag = torch.norm(data.force_history, dim=-1)  # [B, N, H]
    hit = (force_mag > force_threshold).any(dim=1)  # [B, H]
    return hit.sum(dim=-1).float()  # [B]
  assert data.found is not None
  return data.found.squeeze(-1)
