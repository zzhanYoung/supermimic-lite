"""Raw state integration and local geometric errors for world-model training."""

import torch
from torch.nn import functional as F

from .kinematics import (
  conjugate,
  heading_matrix,
  multiply,
  rotate,
  transform,
)


def rotvec_from_quat(quaternion: torch.Tensor) -> torch.Tensor:
  """Shortest relative rotation; q and -q give the same error away from pi."""
  q = F.normalize(quaternion, dim=-1)
  q = torch.where(q[..., :1] < 0, -q, q)
  vector = q[..., 1:]
  sine = torch.linalg.vector_norm(vector, dim=-1, keepdim=True)
  half_angle = torch.atan2(sine, q[..., :1])
  scale = torch.where(
    sine < 1e-6,
    2 / q[..., :1].clamp_min(1e-6),
    2 * half_angle / (sine + 1e-12),
  )
  return vector * scale


def normalize_context(context: torch.Tensor) -> torch.Tensor:
  return torch.cat(
    (
      (context[..., :1] - 0.75) / 0.45,
      context[..., 1:30] / 0.01,
      context[..., 30:33] / 0.05,
    ),
    -1,
  )


def state_delta(state: torch.Tensor, next_state: torch.Tensor) -> torch.Tensor:
  """70 local differences [dp_h, dphi_b, dv_b, dw_b, dq, dqd] at state."""
  q = F.normalize(state[..., 3:7], dim=-1)
  next_q = F.normalize(next_state[..., 3:7], dim=-1)
  # Compare both angular velocities in the current state's body frame.
  next_angular = rotate(multiply(conjugate(q), next_q), next_state[..., 39:42])
  return torch.cat(
    (
      transform(
        heading_matrix(q).transpose(-1, -2), next_state[..., :3] - state[..., :3]
      ),
      rotvec_from_quat(multiply(conjugate(q), next_q)),
      rotate(conjugate(q), next_state[..., 36:39] - state[..., 36:39]),
      next_angular - state[..., 39:42],
      next_state[..., 7:36] - state[..., 7:36],
      next_state[..., 42:71] - state[..., 42:71],
    ),
    -1,
  )


def integrate_delta(
  state: torch.Tensor, delta: torch.Tensor, action: torch.Tensor
) -> torch.Tensor:
  """Add 71 raw residuals, normalize orientation, and carry the current action."""
  physical = state[..., :71] + delta
  quaternion = F.normalize(physical[..., 3:7], dim=-1, eps=1e-8)
  return torch.cat((physical[..., :3], quaternion, physical[..., 7:], action), -1)
