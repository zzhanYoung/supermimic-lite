from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  subtract_frame_transforms,
)

from .commands import MotionCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.scene_entity_config import SceneEntityCfg


def motion_anchor_pos_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))

  pos, _ = subtract_frame_transforms(
    command.robot_anchor_pos_w,
    command.robot_anchor_quat_w,
    command.anchor_pos_w,
    command.anchor_quat_w,
  )

  return pos.view(env.num_envs, -1)


def motion_anchor_ori_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))

  _, ori = subtract_frame_transforms(
    command.robot_anchor_pos_w,
    command.robot_anchor_quat_w,
    command.anchor_pos_w,
    command.anchor_quat_w,
  )
  mat = matrix_from_quat(ori)
  return mat[..., :2].reshape(mat.shape[0], -1)


def robot_body_pos_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))

  num_bodies = len(command.cfg.body_names)
  pos_b, _ = subtract_frame_transforms(
    command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
    command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
    command.robot_body_pos_w,
    command.robot_body_quat_w,
  )

  return pos_b.view(env.num_envs, -1)


def robot_body_ori_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))

  num_bodies = len(command.cfg.body_names)
  _, ori_b = subtract_frame_transforms(
    command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
    command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
    command.robot_body_pos_w,
    command.robot_body_quat_w,
  )
  mat = matrix_from_quat(ori_b)
  return mat[..., :2].reshape(mat.shape[0], -1)


def _expand_worlds(values: torch.Tensor, num_envs: int) -> torch.Tensor:
  """Broadcast an unexpanded simulator model field across environments."""
  if values.shape[0] == num_envs:
    return values
  if values.shape[0] != 1:
    raise ValueError(
      f"Simulator field has {values.shape[0]} worlds, expected 1 or {num_envs}."
    )
  return values.expand(num_envs, *values.shape[1:])


def privileged_dynamics(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  foot_asset_cfg: SceneEntityCfg,
  com_asset_cfg: SceneEntityCfg,
  wrench_asset_cfg: SceneEntityCfg,
  joint_friction_scale: float,
  foot_friction_scale: float,
  com_scale: tuple[float, float, float],
  force_scale: float,
  torque_scale: float,
) -> torch.Tensor:
  """Return scaled current simulator dynamics and external body wrenches.

  Features are ordered as actuator stiffness ratios, actuator damping ratios,
  joint friction, foot tangential friction, COM offset, then body force/torque.
  """
  if min(joint_friction_scale, foot_friction_scale, force_scale, torque_scale) <= 0:
    raise ValueError("Privileged dynamics scales must be positive.")
  if len(com_scale) != 3 or min(com_scale) <= 0:
    raise ValueError("COM scale must contain three positive values.")

  asset = env.scene[asset_cfg.name]
  ctrl_ids = asset.indexing.ctrl_ids[asset_cfg.actuator_ids]
  dof_ids = asset.indexing.joint_v_adr[asset_cfg.joint_ids]
  foot_geom_ids = asset.indexing.geom_ids[foot_asset_cfg.geom_ids]
  com_body_ids = asset.indexing.body_ids[com_asset_cfg.body_ids]

  gain = _expand_worlds(env.sim.model.actuator_gainprm[:, ctrl_ids, 0], env.num_envs)
  damping = _expand_worlds(env.sim.model.actuator_biasprm[:, ctrl_ids, 2], env.num_envs)
  default_gain = env.sim.get_default_field("actuator_gainprm")[ctrl_ids, 0]
  default_damping = env.sim.get_default_field("actuator_biasprm")[ctrl_ids, 2]
  if torch.any(default_gain == 0) or torch.any(default_damping == 0):
    raise ValueError("Privileged actuator ratios require non-zero default gains.")

  stiffness_ratio = gain / default_gain
  damping_ratio = damping / default_damping
  joint_friction = _expand_worlds(
    env.sim.model.dof_frictionloss[:, dof_ids], env.num_envs
  ) / float(joint_friction_scale)
  foot_friction = _expand_worlds(
    env.sim.model.geom_friction[:, foot_geom_ids, 0], env.num_envs
  ) / float(foot_friction_scale)

  current_com = _expand_worlds(env.sim.model.body_ipos[:, com_body_ids], env.num_envs)
  default_com = env.sim.get_default_field("body_ipos")[com_body_ids]
  com_offset = (current_com - default_com) / current_com.new_tensor(com_scale)

  wrench = asset.data.body_external_wrench[:, wrench_asset_cfg.body_ids].clone()
  wrench[..., :3] /= float(force_scale)
  wrench[..., 3:] /= float(torque_scale)

  return torch.cat(
    (
      stiffness_ratio,
      damping_ratio,
      joint_friction,
      foot_friction,
      com_offset.flatten(start_dim=1),
      wrench.flatten(start_dim=1),
    ),
    dim=-1,
  )
