"""Analytic task signals and Vibed-compatible privileged value features."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.utils.lab_api.math import matrix_from_quat, yaw_quat
from torch import nn

from .kinematics import (
  G1Kinematics,
  LinkState,
  angle_squared,
  conjugate,
  heading_matrix,
  multiply,
  rotate,
  rotation6d,
  transform,
)

if TYPE_CHECKING:
  from .adapter import TrackingAdapter


class AnalyticHeads(nn.Module):
  default_q: torch.Tensor
  default_qd: torch.Tensor
  limits: torch.Tensor

  def __init__(self, adapter: TrackingAdapter):
    super().__init__()
    env, robot, command = adapter.env, adapter.robot, adapter.command
    self.kinematics = G1Kinematics(env.sim.mj_model, robot)
    self.tracked = command.body_indexes.cpu().tolist()
    self.anchor = command.robot_anchor_body_index
    self.ref_anchor = command.motion_anchor_body_index
    self.ref_root = command.cfg.body_names.index("pelvis")
    self.body_count = len(self.tracked)
    if self.body_count != 14 or len(adapter.foot_ids) != 14:
      raise ValueError("SuperMimic critic requires 14 tracked bodies and foot geoms.")
    self.actor_terms = tuple(adapter.names["actor"])
    self.register_buffer("default_q", robot.data.default_joint_pos[0].clone())
    self.register_buffer("default_qd", robot.data.default_joint_vel[0].clone())
    self.register_buffer("limits", robot.data.soft_joint_pos_limits[0].clone())
    supported = {
      "motion_global_root_height": "motion_global_anchor_height_error_exp",
      "motion_global_root_ori": "motion_global_anchor_orientation_error_exp",
      "motion_body_pos": "motion_relative_body_position_error_exp",
      "motion_body_ori": "motion_relative_body_orientation_error_exp",
      "motion_body_lin_vel": "motion_global_body_linear_velocity_error_exp",
      "motion_body_ang_vel": "motion_global_body_angular_velocity_error_exp",
      "action_rate_l2": "action_rate_l2",
      "joint_limit": "joint_pos_limits",
    }
    if set(env.cfg.rewards) != set(supported):
      raise ValueError("Analytic heads require the SuperMimic tracking reward profile.")
    for name, function in supported.items():
      cfg = env.cfg.rewards[name]
      if cfg.func.__name__ != function or cfg.params.get("body_names") is not None:
        raise ValueError(f"Unsupported analytic reward configuration: {name}.")
    scale = env.step_dt if env.cfg.scale_rewards_by_dt else 1.0
    self.weights = {name: cfg.weight * scale for name, cfg in env.cfg.rewards.items()}
    self.stds = {
      name: float(cfg.params["std"])
      for name, cfg in env.cfg.rewards.items()
      if "std" in cfg.params
    }
    self.height_threshold = float(
      env.cfg.terminations["anchor_pos"].params["threshold"]
    )
    self.tilt_threshold = float(env.cfg.terminations["anchor_ori"].params["threshold"])

  def reference_parts(self, goal: torch.Tensor) -> tuple[torch.Tensor, ...]:
    n = self.body_count
    q, qd, p, rot, v, w = goal.split((29, 29, 3 * n, 4 * n, 3 * n, 3 * n), -1)
    return (
      q,
      qd,
      p.unflatten(-1, (n, 3)),
      rot.unflatten(-1, (n, 4)),
      v.unflatten(-1, (n, 3)),
      w.unflatten(-1, (n, 3)),
    )

  def observation(
    self,
    state: torch.Tensor,
    goal: torch.Tensor,
    context: torch.Tensor,
    links: LinkState,
  ) -> torch.Tensor:
    _, _, _, ref_q, _, _ = self.reference_parts(goal)
    imu_q = multiply(
      links.quaternion[..., self.kinematics.imu_body, :], self.kinematics.imu_rotation
    )
    values = {
      "command": goal[..., :58],
      "motion_anchor_ori_b": rotation6d(
        multiply(
          conjugate(links.quaternion[..., self.anchor, :]),
          ref_q[..., self.ref_anchor, :],
        )
      ),
      "base_ang_vel": rotate(
        conjugate(imu_q), links.angular_velocity[..., self.kinematics.imu_body, :]
      ),
      "projected_gravity": -matrix_from_quat(state[..., 3:7])[..., 2, :],
      "joint_pos": state[..., 7:36] + context[..., 1:30] - self.default_q,
      "joint_vel": state[..., 42:71] - self.default_qd,
      "actions": state[..., -29:],
    }
    return torch.cat([values[name] for name in self.actor_terms], -1)

  def reward_terms(
    self,
    state: torch.Tensor,
    action: torch.Tensor,
    next_state: torch.Tensor,
    goal: torch.Tensor,
    before: LinkState,
    after: LinkState,
  ) -> dict[str, torch.Tensor]:
    _, _, rp, rq, rv, rw = self.reference_parts(goal)
    # The command cached these targets before the action. Do not re-anchor them
    # at the next state: that would change the real environment's reward timing.
    anchor_p = before.position[..., self.anchor, :]
    reference_p = rp[..., self.ref_anchor, :]
    delta = yaw_quat(
      multiply(
        before.quaternion[..., self.anchor, :], conjugate(rq[..., self.ref_anchor, :])
      )
    )
    origin = torch.cat((anchor_p[..., :2], reference_p[..., 2:3]), -1)
    target_p = origin[..., None, :] + rotate(
      delta[..., None, :], rp - reference_p[..., None, :]
    )
    target_q = multiply(delta[..., None, :], rq)
    positions, quaternions = (
      after.position[..., self.tracked, :],
      after.quaternion[..., self.tracked, :],
    )
    errors = {
      "motion_global_root_height": (
        after.position[..., self.anchor, 2] - reference_p[..., 2]
      ).square(),
      "motion_global_root_ori": angle_squared(
        after.quaternion[..., self.anchor, :], rq[..., self.ref_anchor, :]
      ),
      "motion_body_pos": (target_p - positions).square().sum(-1).mean(-1),
      "motion_body_ori": angle_squared(target_q, quaternions).mean(-1),
      "motion_body_lin_vel": (rv - after.linear_velocity[..., self.tracked, :])
      .square()
      .sum(-1)
      .mean(-1),
      "motion_body_ang_vel": (rw - after.angular_velocity[..., self.tracked, :])
      .square()
      .sum(-1)
      .mean(-1),
    }
    terms = {
      name: torch.exp(-error / self.stds[name] ** 2) for name, error in errors.items()
    }
    terms["action_rate_l2"] = (action - state[..., -29:]).square().sum(-1)
    q = next_state[..., 7:36]
    terms["joint_limit"] = (
      (self.limits[:, 0] - q).relu() + (q - self.limits[:, 1]).relu()
    ).sum(-1)
    return {name: value * self.weights[name] for name, value in terms.items()}

  def termination_errors(
    self, state: torch.Tensor, goal: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor]:
    _, _, rp, rq, _, _ = self.reference_parts(goal)
    height = (state[..., 2] - rp[..., self.ref_root, 2]).abs()
    tilt = (
      matrix_from_quat(state[..., 3:7])[..., 2, 2]
      - matrix_from_quat(rq[..., self.ref_root, :])[..., 2, 2]
    ).abs()
    return height, tilt

  def hard_termination(self, state: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
    height, tilt = self.termination_errors(state, goal)
    return (height > self.height_threshold) | (tilt > self.tilt_threshold)

  def soft_termination(self, state: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
    height, tilt = self.termination_errors(state, goal)
    return 1 - (1 - ((height - self.height_threshold) / 0.02).sigmoid()) * (
      1 - ((tilt - self.tilt_threshold) / 0.04).sigmoid()
    )

  @staticmethod
  def normalize_context(context: torch.Tensor) -> torch.Tensor:
    return torch.cat(
      (
        (context[..., :1] - 0.75) / 0.45,
        context[..., 1:30] / 0.01,
        context[..., 30:33] / 0.05,
      ),
      -1,
    )

  def critic_input(
    self,
    state: torch.Tensor,
    goal: torch.Tensor,
    context: torch.Tensor,
    future: torch.Tensor,
    links: LinkState,
  ) -> torch.Tensor:
    ref_joints, _, rp, rq, _, _ = self.reference_parts(goal)
    p, q = state[..., :3], state[..., 3:7]
    r = matrix_from_quat(q)
    ht = heading_matrix(q).transpose(-1, -2)
    invariant = torch.cat(
      (
        p[..., 2:3],
        (ht @ r)[..., :, :2].flatten(-2),
        rotate(conjugate(q), state[..., 36:39]),
        state[..., 39:42],
        state[..., 7:36],
        state[..., 42:71],
      ),
      -1,
    )
    root_p, root_q = rp[..., self.ref_root, :], rq[..., self.ref_root, :]
    dr = heading_matrix(multiply(q, conjugate(root_q)))
    anchor = torch.cat((p[..., :2], root_p[..., 2:3]), -1)
    ref_bodies = anchor[..., None, :] + transform(
      dr[..., None, :, :], rp - root_p[..., None, :]
    )
    body_error = transform(
      ht[..., None, :, :], ref_bodies - links.position[..., self.tracked, :]
    ).flatten(-2)
    return torch.cat(
      (
        invariant,
        ref_joints - state[..., 7:36],
        body_error,
        transform(ht, root_p - p),
        rotation6d(multiply(conjugate(q), root_q)),
        self.normalize_context(context),
        future,
      ),
      -1,
    )
