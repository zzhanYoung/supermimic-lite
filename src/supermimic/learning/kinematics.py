"""Differentiable G1 link kinematics in the learner's MuJoCo state convention."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
import torch
from mjlab.entity import Entity
from mjlab.utils.lab_api.math import matrix_from_quat
from torch import nn
from torch.nn import functional as F


def conjugate(q: torch.Tensor) -> torch.Tensor:
  return torch.cat((q[..., :1], -q[..., 1:]), -1)


def multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
  a, b = torch.broadcast_tensors(a, b)
  aw, av, bw, bv = a[..., :1], a[..., 1:], b[..., :1], b[..., 1:]
  return torch.cat(
    (
      aw * bw - (av * bv).sum(-1, keepdim=True),
      aw * bv + bw * av + torch.linalg.cross(av, bv),
    ),
    -1,
  )


def rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
  xyz, v = torch.broadcast_tensors(q[..., 1:], v)
  t = 2 * torch.linalg.cross(xyz, v)
  return v + q[..., :1] * t + torch.linalg.cross(xyz, t)


def rotation6d(q: torch.Tensor) -> torch.Tensor:
  return matrix_from_quat(q)[..., :, :2].flatten(-2)


def angle_squared(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
  """Geodesic squared angle, with finite gradients at identical rotations."""
  q = F.normalize(multiply(a, conjugate(b)), dim=-1)
  angle = 2 * torch.atan2(torch.linalg.vector_norm(q[..., 1:], dim=-1), q[..., 0].abs())
  return angle.square()


def heading_matrix(q: torch.Tensor) -> torch.Tensor:
  """Vibed's projected-X heading, including its vertical-axis regularization."""
  r = matrix_from_quat(q)
  x, y = r[..., 0, 0], r[..., 1, 0]
  norm = (x.square() + y.square() + 1e-9).sqrt()
  c, s = x / norm, y / norm
  z, one = torch.zeros_like(c), torch.ones_like(c)
  return torch.stack((c, -s, z, s, c, z, z, z, one), -1).reshape(*q.shape[:-1], 3, 3)


def transform(r: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
  return (r @ v.unsqueeze(-1)).squeeze(-1)


@dataclass
class LinkState:
  position: torch.Tensor
  quaternion: torch.Tensor
  linear_velocity: torch.Tensor
  angular_velocity: torch.Tensor


class G1Kinematics(nn.Module):
  offsets: torch.Tensor
  rotations: torch.Tensor
  axes: torch.Tensor
  pivots: torch.Tensor
  joint_reference: torch.Tensor
  imu_rotation: torch.Tensor

  def __init__(self, model: mujoco.MjModel, robot: Entity):
    super().__init__()
    bodies = robot.indexing.body_ids.cpu().tolist()
    local = {body: i for i, body in enumerate(bodies)}
    joints = {
      int(adr): i for i, adr in enumerate(robot.indexing.joint_q_adr.cpu().tolist())
    }
    self.parents: list[int] = []
    self.joints: list[int] = []
    axes, pivots, refs = [], [], []
    for i, body in enumerate(bodies):
      n = int(model.body_jntnum[body])
      jid = int(model.body_jntadr[body])
      if i == 0:
        if n != 1 or model.jnt_type[jid] != mujoco.mjtJoint.mjJNT_FREE:
          raise ValueError("SuperMimic requires a free root followed by hinge links.")
        parent, joint = -1, -1
      else:
        parent = local[int(model.body_parentid[body])]
        if parent >= i or n > 1:
          raise ValueError("Expected a parent-first G1 tree with one joint per link.")
        joint = -1
        if n:
          if model.jnt_type[jid] != mujoco.mjtJoint.mjJNT_HINGE:
            raise ValueError("SuperMimic supports hinge joints below the free root.")
          joint = joints[int(model.jnt_qposadr[jid])]
      self.parents.append(parent)
      self.joints.append(joint)
      axes.append(model.jnt_axis[jid] if joint >= 0 else np.zeros(3))
      pivots.append(model.jnt_pos[jid] if joint >= 0 else np.zeros(3))
      refs.append(model.qpos0[model.jnt_qposadr[jid]] if joint >= 0 else 0.0)
    for name, value in (
      ("offsets", model.body_pos[bodies]),
      ("rotations", model.body_quat[bodies]),
      ("axes", np.array(axes)),
      ("pivots", np.array(pivots)),
      ("joint_reference", np.array(refs)),
    ):
      self.register_buffer(name, torch.tensor(value, dtype=torch.float32))
    sites, _ = robot.find_sites("imu_in_pelvis")
    if len(sites) != 1:
      raise ValueError("Expected one pelvis IMU site.")
    site = int(robot.indexing.site_ids[sites[0]])
    self.imu_body = local[int(model.site_bodyid[site])]
    self.register_buffer(
      "imu_rotation", torch.tensor(model.site_quat[site], dtype=torch.float32)
    )

  def forward(self, state: torch.Tensor) -> LinkState:
    p, q = state[..., :3], F.normalize(state[..., 3:7], dim=-1)
    positions, rotations = [p], [q]
    velocities = [state[..., 36:39]]
    angular = [rotate(q, state[..., 39:42])]
    for i in range(1, len(self.parents)):
      parent, joint = self.parents[i], self.joints[i]
      parent_q, parent_w = rotations[parent], angular[parent]
      offset = rotate(parent_q, self.offsets[i])
      origin = positions[parent] + offset
      base_q = multiply(parent_q, self.rotations[i])
      if joint >= 0:
        half = (state[..., 7 + joint] - self.joint_reference[i]) * 0.5
        turn = torch.cat(
          (half.cos()[..., None], half.sin()[..., None] * self.axes[i]), -1
        )
        child_q = multiply(base_q, turn)
        pivot = origin + rotate(base_q, self.pivots[i])
        arm = rotate(child_q, self.pivots[i])
        child_w = parent_w + rotate(base_q, self.axes[i]) * state[..., 42 + joint, None]
        child_p = pivot - arm
        child_v = (
          velocities[parent]
          + torch.linalg.cross(parent_w, pivot - positions[parent])
          - torch.linalg.cross(child_w, arm)
        )
      else:
        child_p, child_q, child_w = origin, base_q, parent_w
        child_v = velocities[parent] + torch.linalg.cross(parent_w, offset)
      positions.append(child_p)
      rotations.append(child_q)
      velocities.append(child_v)
      angular.append(child_w)
    return LinkState(
      torch.stack(positions, -2),
      torch.stack(rotations, -2),
      torch.stack(velocities, -2),
      torch.stack(angular, -2),
    )
