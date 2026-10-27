"""Check raw state updates and geometric loss against matrix/SciPy calculations."""

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation
from test_supermimic_heads import analytic_adapter as analytic_adapter
from test_supermimic_updates import learner as learner
from test_tracking_observation_contract import standing_g1_motion as standing_g1_motion
from torch import nn

from supermimic.learning.dynamics import (
  integrate_delta,
  rotvec_from_quat,
  state_delta,
)
from supermimic.learning.models import RunningNorm


def random_state(batch: int = 6) -> torch.Tensor:
  rng = np.random.default_rng(18)
  state = rng.normal(size=(batch, 100))
  state[:, 3:7] = Rotation.random(batch, rng=rng).as_quat(scalar_first=True)
  return torch.from_numpy(state)


def matrix_state(state: torch.Tensor):
  s = state.detach().numpy()
  rotation = Rotation.from_quat(s[:, 3:7], scalar_first=True).as_matrix()
  heading = np.zeros_like(rotation)
  fx, fy = rotation[:, 0, 0], rotation[:, 1, 0]
  norm = np.sqrt(fx * fx + fy * fy + 1e-9)
  heading[:, 0, 0] = heading[:, 1, 1] = fx / norm
  heading[:, 1, 0] = fy / norm
  heading[:, 0, 1] = -fy / norm
  heading[:, 2, 2] = 1
  return s, rotation, heading


def test_raw_updates_and_geometric_errors_match_matrix_equations():
  state = random_state()
  s, rotation, heading = matrix_state(state)
  transpose = rotation.transpose(0, 2, 1)
  rng = np.random.default_rng(25)
  delta = rng.normal(size=(len(s), 71)) * 0.1
  action = torch.from_numpy(rng.normal(size=(len(s), 29)))
  updated = integrate_delta(state, torch.from_numpy(delta), action)
  result, next_rotation, _ = matrix_state(updated)
  expected = s[:, :71] + delta
  expected[:, 3:7] /= np.linalg.norm(expected[:, 3:7], axis=-1, keepdims=True)
  np.testing.assert_allclose(result[:, :71], expected, atol=1e-11)
  torch.testing.assert_close(updated[:, -29:], action)

  # Loss remains in the original 70D local geometry, independently of outputs.
  expected_error = np.concatenate(
    (
      np.einsum("bij,bj->bi", heading.transpose(0, 2, 1), delta[:, :3]),
      Rotation.from_matrix(transpose @ next_rotation).as_rotvec(),
      np.einsum("bij,bj->bi", transpose, delta[:, 36:39]),
      np.einsum("bij,bj->bi", transpose @ next_rotation, result[:, 39:42])
      - s[:, 39:42],
      delta[:, 7:36],
      delta[:, 42:71],
    ),
    axis=-1,
  )
  torch.testing.assert_close(
    state_delta(state, updated), torch.from_numpy(expected_error), atol=1e-8, rtol=1e-7
  )
  flipped = updated.clone()
  flipped[:, 3:7] *= -1
  torch.testing.assert_close(state_delta(state, flipped), state_delta(state, updated))
  torch.testing.assert_close(
    state_delta(state, state),
    torch.zeros_like(torch.from_numpy(expected_error)),
    atol=1e-12,
    rtol=0,
  )


@pytest.mark.parametrize("angle", [0.0, 1e-9, 0.7, np.pi - 1e-5, np.pi + 1e-5])
def test_rotation_error_values_and_gradients(angle):
  expected = Rotation.from_rotvec([angle, 0.0, 0.0])
  quaternion = torch.from_numpy(expected.as_quat(scalar_first=True)).requires_grad_()
  torch.testing.assert_close(
    rotvec_from_quat(quaternion),
    torch.from_numpy(expected.as_rotvec()),
    atol=1e-10,
    rtol=1e-10,
  )
  assert torch.autograd.gradcheck(rotvec_from_quat, (quaternion,))


def test_world_model_uses_raw_states_and_separate_increment_statistics(learner):
  alg, _ = learner
  wm = alg.world_model
  state = random_state()
  torch.manual_seed(35)
  action = torch.randn(6, 29, dtype=torch.float64)
  context = torch.randn(6, 33, dtype=torch.float64) * 0.01
  next_state = integrate_delta(
    state, torch.randn(6, 71, dtype=torch.float64) * 0.03, action
  )
  inputs, targets = wm.inputs(state, action, context), wm.targets(state, next_state)
  wm.in_norm.update(inputs)
  wm.out_norm.update(targets)
  wm.loss_norm.update(wm.state_error(next_state, state))
  torch.testing.assert_close(inputs[:, :71], state[:, :71])
  torch.testing.assert_close(inputs[:, 71:100], action)
  torch.testing.assert_close(targets, next_state[:, :71] - state[:, :71])
  old_output_norm = RunningNorm(70, eps=1e-4, max_count=2e8).double()
  old_output_norm.update(state_delta(state, next_state))
  for name, value in old_output_norm.state_dict().items():
    torch.testing.assert_close(wm.loss_norm.state_dict()[name], value)
  assert len(
    [layer for layer in wm.dynamics if isinstance(layer, nn.LayerNorm)]
  ) == len(alg.cfg.world_model.hidden_dims)
  assert isinstance(wm.dynamics[2], nn.ELU)
  assert wm.dynamics[-1].out_features == 71
  torch.testing.assert_close(inputs[:, 100:101], (context[:, :1] - 0.75) / 0.45)
  torch.testing.assert_close(inputs[:, 101:130], context[:, 1:30] / 0.01)
  torch.testing.assert_close(inputs[:, 130:133], context[:, 30:33] / 0.05)
  # Zero output adds the raw mean residual before normalizing the quaternion.
  nn.init.zeros_(wm.dynamics[-1].weight)
  nn.init.zeros_(wm.dynamics[-1].bias)
  expected = integrate_delta(state, targets.mean(0).expand(6, -1), action)
  torch.testing.assert_close(wm.step(state, action, context), expected)


def test_normalizer_eps_and_freeze():
  norm = RunningNorm(2, eps=1e-4, max_count=3).double()
  values = torch.tensor([[1.0, 0.0], [3.0, 0.0], [2.0, 0.0]], dtype=torch.float64)
  norm.update(values[:2])
  norm.update(values[2:])
  torch.testing.assert_close(norm.mean, values.mean(0))
  torch.testing.assert_close(norm.std, (values.var(0, unbiased=False) + 1e-8).sqrt())
  norm.update(values * 100)
  assert norm.count == 3
  torch.testing.assert_close(norm.mean, values.mean(0))
  torch.testing.assert_close(norm.denormalize(norm(values)), values)


def test_world_model_loss_uses_geometric_mean_error_and_reset_masks(learner):
  alg, replay = learner
  wm = alg.world_model
  batch = replay.windows(4, 2)
  with torch.no_grad():
    wm.loss_norm.variance.zero_()  # Exercise the per-quantity floors in the loss.
    wm.out_norm.variance.fill_(123)  # Output scales must not affect loss weighting.
    nn.init.zeros_(wm.dynamics[-1].weight)
    nn.init.zeros_(wm.dynamics[-1].bias)
  state = batch["state"][0]
  masks = torch.tensor([[True, True, False, True], [True, False, False, True]])
  targets = state.unsqueeze(0).expand(2, -1, -1).clone()
  targets[..., 2] += 3 * wm.error_floor[2]
  targets[..., 7] += 2 * wm.error_floor[12]
  targets[..., 3:7] *= -1  # Identical orientation must contribute zero error.
  # Identity predictions have errors of 3 and 2 nominal units, in exactly two
  # coordinates. Five valid samples contribute; the batch denominator stays 4.
  expected = state.new_tensor((3**2 + 2**2) / 70 * 5 / 4)
  actual = alg._world_model_loss(
    state, batch["action"], batch["context"], targets, masks
  )
  torch.testing.assert_close(actual, expected)
