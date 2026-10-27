"""Future-conditioned actor inputs in real-data normalization and imagination."""

from copy import deepcopy

import torch
from test_supermimic_heads import analytic_adapter as analytic_adapter
from test_supermimic_updates import learner as learner
from test_tracking_observation_contract import standing_g1_motion as standing_g1_motion


def test_actor_uses_future_reference_and_orientation_history(learner):
  alg, _ = learner
  torch.manual_seed(7)
  history = torch.randn(2, 10, 157, dtype=torch.float64, requires_grad=True)
  future = torch.randn(2, 730, dtype=torch.float64, requires_grad=True)
  action = alg.actor(history, future)
  assert alg.actor.normalizer.mean.numel() == 99
  assert isinstance(alg.actor.decoder[0], torch.nn.Linear)
  assert alg.actor.decoder[0].in_features == 10 * 99 + alg.cfg.imagination.latent_dim
  changed_future = future.detach().clone()
  changed_future[:, -73:] += 1
  assert not torch.allclose(action, alg.actor(history, changed_future))

  # Only command history is replaced by future references. Relative orientation
  # must still affect actions at every history step, including the current one.
  orientation = alg.adapter.slices["actor"]["motion_anchor_ori_b"]
  for step in range(10):
    changed_history = history.detach().clone()
    changed_history[:, step, orientation] += 1
    assert not torch.allclose(action, alg.actor(changed_history, future))
  command = alg.adapter.slices["actor"]["command"]
  changed_history = history.detach().clone()
  changed_history[..., command] += 10
  torch.testing.assert_close(action, alg.actor(changed_history, future), rtol=0, atol=0)
  action.square().sum().backward()
  assert future.grad is not None and torch.isfinite(future.grad).all()
  assert (future.grad.reshape(2, 10, 73).abs().sum(-1) > 0).all()
  assert history.grad is not None and torch.isfinite(history.grad).all()
  assert history.grad[..., command].count_nonzero() == 0
  assert (history.grad[..., orientation].abs().sum(-1) > 0).all()
  assert (history.grad[..., alg.actor.proprio_indices].abs().sum(-1) > 0).all()


def test_real_goal_statistics_and_imagined_frame_progression(learner, monkeypatch):
  alg, replay = learner
  library = alg.adapter.command.motion
  library.joint_pos += torch.arange(len(library.joint_pos))[:, None] * 0.01
  data = {key: value[: replay.size].flatten(0, 1) for key, value in replay.data.items()}
  alg.update_normalizers(data)
  goals = alg.adapter.future_reference(data["motion"], data["frame"])
  norm = alg.actor.goal_normalizer
  assert norm.count == replay.size * replay.num_envs
  torch.testing.assert_close(norm.mean, goals.mean(0))
  torch.testing.assert_close(norm.variance, goals.var(0, unbiased=False))
  torch.testing.assert_close(
    alg.actor.normalizer.mean, data["obs"][:, alg.actor.proprio_indices].mean(0)
  )
  assert alg.actor.normalizer.mean.numel() == 99
  saved = deepcopy(norm.state_dict())
  saved_history = deepcopy(alg.actor.normalizer.state_dict())

  # Use mixed starting frames, including two near the motion end.
  alg.cfg.imagination.horizon = 4
  seed = replay.starts(alg.cfg.imagination.batch_size)
  seed["frame"][:] = torch.tensor([0, 1, 98, 99])
  monkeypatch.setattr(replay, "starts", lambda _: seed)
  observed_goals = []

  def record_goal(_module, inputs):
    observed_goals.append(inputs[1].detach().clone())

  handle = alg.actor.register_forward_pre_hook(record_goal)
  try:
    alg.improve_policy(replay)
  finally:
    handle.remove()
  assert len(observed_goals) == alg.cfg.imagination.horizon
  for step, actual in enumerate(observed_goals):
    expected = alg.adapter.future_reference(seed["motion"], seed["frame"] + step)
    torch.testing.assert_close(actual, expected)
  for key, value in norm.state_dict().items():
    torch.testing.assert_close(value, saved[key], rtol=0, atol=0)
  for key, value in alg.actor.normalizer.state_dict().items():
    torch.testing.assert_close(value, saved_history[key], rtol=0, atol=0)
