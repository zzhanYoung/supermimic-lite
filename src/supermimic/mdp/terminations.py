from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
from mjlab.utils.lab_api.math import quat_apply_inverse

from .commands import MotionCommand
from .rewards import _get_body_index, _get_body_indexes

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.scene_entity_config import SceneEntityCfg


def motion_time_out(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Time out an episode after its current reference motion reaches the end."""
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  return command.time_steps >= command.motion_end_steps - 1


def bad_anchor_pos(
  env: ManagerBasedRlEnv, command_name: str, threshold: float
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  return (
    torch.norm(command.anchor_pos_w - command.robot_anchor_pos_w, dim=1) > threshold
  )


def bad_anchor_pos_z_only(
  env: ManagerBasedRlEnv, command_name: str, threshold: float
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  return (
    torch.abs(command.anchor_pos_w[:, -1] - command.robot_anchor_pos_w[:, -1])
    > threshold
  )


def bad_anchor_ori(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg, command_name: str, threshold: float
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]

  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  motion_projected_gravity_b = quat_apply_inverse(
    command.anchor_quat_w, asset.data.gravity_vec_w
  )

  robot_projected_gravity_b = quat_apply_inverse(
    command.robot_anchor_quat_w, asset.data.gravity_vec_w
  )

  return (
    motion_projected_gravity_b[:, 2] - robot_projected_gravity_b[:, 2]
  ).abs() > threshold


def bad_motion_root_height(
  env: ManagerBasedRlEnv,
  command_name: str,
  threshold: float,
  root_body_name: str = "pelvis",
) -> torch.Tensor:
  """Terminate on world-Z root error, independent of the tracking anchor."""
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  root_index = _get_body_index(command, root_body_name)
  return (
    command.body_pos_w[:, root_index, 2] - command.robot_body_pos_w[:, root_index, 2]
  ).abs() > threshold


def bad_motion_root_tilt(
  env: ManagerBasedRlEnv,
  command_name: str,
  threshold: float,
  root_body_name: str = "pelvis",
) -> torch.Tensor:
  """SuperMimic Appendix A: difference of root-frame gravity Z components."""
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  root_index = _get_body_index(command, root_body_name)
  gravity = command.robot.data.gravity_vec_w
  reference_gravity = quat_apply_inverse(command.body_quat_w[:, root_index], gravity)
  robot_gravity = quat_apply_inverse(command.robot_body_quat_w[:, root_index], gravity)
  return (reference_gravity[:, 2] - robot_gravity[:, 2]).abs() > threshold


def bad_motion_body_pos(
  env: ManagerBasedRlEnv,
  command_name: str,
  threshold: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))

  body_indexes = _get_body_indexes(command, body_names)
  error = torch.norm(
    command.body_pos_relative_w[:, body_indexes]
    - command.robot_body_pos_w[:, body_indexes],
    dim=-1,
  )
  return torch.any(error > threshold, dim=-1)


def bad_motion_body_pos_z_only(
  env: ManagerBasedRlEnv,
  command_name: str,
  threshold: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))

  body_indexes = _get_body_indexes(command, body_names)
  error = torch.abs(
    command.body_pos_relative_w[:, body_indexes, -1]
    - command.robot_body_pos_w[:, body_indexes, -1]
  )
  return torch.any(error > threshold, dim=-1)
