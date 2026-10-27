"""Exercise the production rollout and optimizers without a physics simulator."""

from copy import deepcopy
from typing import cast

import pytest
import torch
from test_supermimic_heads import analytic_adapter as analytic_adapter
from test_tracking_observation_contract import standing_g1_motion as standing_g1_motion
from torch import nn

from supermimic.learning import algorithm
from supermimic.learning.algorithm import SuperMimic
from supermimic.learning.config import AdamCfg, SuperMimicRunnerCfg
from supermimic.learning.replay import ReplayBuffer


@pytest.fixture
def learner(analytic_adapter) -> tuple[SuperMimic, ReplayBuffer]:
  torch.manual_seed(17)
  cfg = SuperMimicRunnerCfg()
  cfg.world_model.hidden_dims = (16,)
  cfg.world_model.batch_size = 4
  cfg.world_model.batches_per_update = 1
  cfg.update_to_data_ratio = 0.2
  cfg.imagination.hidden_dims = (16,)
  cfg.imagination.batch_size = 4
  cfg.critic.hidden_dims = (16,)
  cfg.validate()

  adapter = analytic_adapter
  adapter.env.cfg.episode_length_s = 100 * adapter.env.step_dt
  library = adapter.command.motion
  for name in (
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
  ):
    value = getattr(library, name)
    setattr(library, name, value.repeat(20, *([1] * (value.ndim - 1))).double())
  library.time_step_totals[:] = 100
  alg = SuperMimic(cfg, adapter, "cpu")
  for module in alg.modules:
    module.double()
  # Exercise nonzero dynamics Jacobians, beyond the zero-residual initialization.
  final = alg.world_model.dynamics[-1]
  assert isinstance(final, nn.Linear)
  nn.init.normal_(final.weight, std=0.01)

  replay = ReplayBuffer(16, 2, "cpu")
  state = adapter.state().double()
  for t in range(12):
    action = torch.randn(2, 29, dtype=torch.float64) * 0.1
    next_state = state.clone()
    next_state[:, 7:36] += 0.02 * action
    next_state[:, -29:] = action
    replay.append(
      {
        "state": state,
        "next_state": next_state,
        "action": action,
        "context": adapter.read_context().double(),
        "obs": torch.randn(2, 157, dtype=torch.float64) * 0.1,
        "reward": -action.square().sum(-1),
        "terminated": torch.zeros(2, dtype=torch.bool),
        "truncated": torch.zeros(2, dtype=torch.bool),
        "motion": torch.zeros(2, dtype=torch.long),
        "frame": torch.full((2,), t, dtype=torch.long),
        "next_frame": torch.full((2,), t + 1, dtype=torch.long),
        "episode": torch.zeros(2, dtype=torch.long),
        "age": torch.full((2,), t, dtype=torch.long),
      }
    )
    state = next_state
  return alg, replay


def test_dynamics_exclude_previous_action_and_keep_current_action_gradients(
  learner: tuple[SuperMimic, ReplayBuffer],
) -> None:
  alg, replay = learner
  wm = alg.world_model
  first = wm.dynamics[0]
  assert isinstance(first, nn.Linear)
  assert first.in_features == 133
  state = replay.data["state"][0].clone().requires_grad_()
  action = replay.data["action"][0].clone().requires_grad_()
  context = replay.data["context"][0].clone().requires_grad_()
  changed = state.detach().clone()
  changed[..., -29:] += torch.randn_like(action) * 10
  # The fixture uses nonzero dynamics weights, so this also checks learned
  # rollouts rather than just the identity initialization.
  predicted = wm.step(state, action, context)
  torch.testing.assert_close(
    predicted, wm.step(changed, action, context), rtol=0, atol=0
  )
  torch.testing.assert_close(predicted[..., -29:], action, rtol=0, atol=0)
  physical = predicted[..., :71]
  state_grad, action_grad, context_grad = torch.autograd.grad(
    (physical * torch.randn_like(physical)).sum(), (state, action, context)
  )
  for gradient in (state_grad[..., :71], action_grad, context_grad):
    assert torch.isfinite(gradient).all()
    assert gradient.abs().sum() > 0
  assert torch.count_nonzero(state_grad[..., -29:]) == 0


@pytest.mark.parametrize("boundary", ["horizon", "episode", "motion"])
def test_production_actor_bptt_matches_finite_difference(
  learner: tuple[SuperMimic, ReplayBuffer],
  monkeypatch: pytest.MonkeyPatch,
  boundary: str,
) -> None:
  alg, replay = learner
  seed = replay.starts(alg.cfg.imagination.batch_size)
  if boundary == "episode":
    seed["age"][:] = torch.tensor([99, 98, 97, 96])
  elif boundary == "motion":
    seed["frame"][:] = torch.tensor([99, 98, 97, 96])
  monkeypatch.setattr(replay, "starts", lambda _: seed)
  parameters = list(alg.actor.parameters())
  direction = [torch.randn_like(p) for p in parameters]
  norm = torch.stack([d.square().sum() for d in direction]).sum().sqrt()
  direction = [d / norm for d in direction]
  gradients = []

  def backward_only(
    loss: torch.Tensor,
    module: nn.Module,
    optimizer: torch.optim.Adam,
    config: AdamCfg,
    elementwise_clip: float | None = None,
  ) -> float:
    del config, elementwise_clip
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    if module is alg.actor:
      assert all(p.grad is not None for p in parameters)
      gradients.append(
        torch.stack(
          [
            (cast(torch.Tensor, p.grad) * d).sum()
            for p, d in zip(parameters, direction, strict=True)
          ]
        )
        .sum()
        .item()
      )
    return 0.0

  # Suppress parameter steps while testing the exact production graph, including
  # history feedback, analytic termination and terminal critic bootstrap.
  monkeypatch.setattr(algorithm, "optimizer_step", backward_only)
  saved = [p.detach().clone() for p in parameters]
  objectives = []
  eps = 1e-5
  for offset in (0.0, eps, -eps):
    with torch.no_grad():
      for p, initial, d in zip(parameters, saved, direction, strict=True):
        p.copy_(initial + offset * d)
    torch.manual_seed(31)  # Identical observation noise in each difference sample.
    objectives.append(alg.improve_policy(replay)["actor/objective"])
  numerical = -(objectives[1] - objectives[2]) / (2 * eps)
  assert abs(numerical) > 1e-7
  assert gradients[0] == pytest.approx(numerical, rel=1e-5, abs=1e-8)


def test_production_updates_isolate_parameters_and_keep_rollout_graph(
  learner: tuple[SuperMimic, ReplayBuffer], monkeypatch: pytest.MonkeyPatch
) -> None:
  alg, replay = learner
  before = [deepcopy(module.state_dict()) for module in alg.modules]
  inputs, outputs = [], []
  original_step = alg.world_model.step

  def record_step(
    state: torch.Tensor, action: torch.Tensor, context: torch.Tensor
  ) -> torch.Tensor:
    inputs.append(state)
    result = original_step(state, action, context)
    result.retain_grad()
    outputs.append(result)
    return result

  with monkeypatch.context() as patch:
    patch.setattr(alg.world_model, "step", record_step)
    alg.fit_world_model(replay)
  assert len(outputs) == alg.cfg.world_model.window_length
  assert all(inputs[t] is outputs[t - 1] for t in range(1, len(inputs)))
  assert all(x.grad is not None and torch.isfinite(x.grad).all() for x in outputs)
  for module, initial in zip(alg.modules[1:], before[1:], strict=True):
    assert all(
      torch.equal(value, initial[k]) for k, value in module.state_dict().items()
    )
    assert all(p.grad is None for p in module.parameters())
  assert any(
    not torch.equal(value, before[0][key])
    for key, value in alg.world_model.state_dict().items()
  )

  before = [deepcopy(module.state_dict()) for module in alg.modules]
  calls = []
  actor_grads = []
  original_optimizer_step = algorithm.optimizer_step

  def record_update(
    loss: torch.Tensor,
    module: nn.Module,
    optimizer: torch.optim.Adam,
    config: AdamCfg,
    elementwise_clip: float | None = None,
  ) -> float:
    calls.append(module)
    result = original_optimizer_step(loss, module, optimizer, config, elementwise_clip)
    if module is alg.actor:
      actor_grads.extend(
        cast(torch.Tensor, p.grad).clone() for p in module.parameters()
      )
    else:
      for p, gradient in zip(alg.actor.parameters(), actor_grads, strict=True):
        torch.testing.assert_close(p.grad, gradient, rtol=0, atol=0)
    return result

  monkeypatch.setattr(algorithm, "optimizer_step", record_update)
  metrics = alg.improve_policy(replay)
  assert calls == [alg.actor] + [alg.critic] * (
    alg.cfg.critic.epochs * alg.cfg.critic.minibatches
  )
  assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())
  assert all(
    torch.equal(value, before[0][key])
    for key, value in alg.world_model.state_dict().items()
  )
  assert all(p.grad is None for p in alg.world_model.parameters())
  assert all(p.requires_grad for module in alg.modules for p in module.parameters())
  for module, initial in zip(alg.modules[1:], before[1:], strict=True):
    assert any(
      not torch.equal(value, initial[k]) for k, value in module.state_dict().items()
    )
