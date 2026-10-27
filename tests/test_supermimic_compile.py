"""Compile the production learning graphs and preserve their BPTT gradients."""

from copy import deepcopy
from functools import wraps

import torch
from test_supermimic_heads import analytic_adapter as analytic_adapter
from test_supermimic_updates import learner as learner
from test_tracking_observation_contract import standing_g1_motion as standing_g1_motion

from supermimic.learning.algorithm import (
  SuperMimic,
  frozen,
  imagined_objective,
)


def test_compiled_world_model_loss_and_autoregressive_gradients(learner):
  alg, replay = learner
  batch = replay.windows(4, 8)
  state = batch["state"][0].clone().requires_grad_()
  actions = batch["action"].clone().requires_grad_()
  masks = torch.arange(8)[:, None] < torch.tensor([8, 3, 0, 6])[None]
  parameters = tuple(alg.world_model.parameters())
  arguments = (state, actions, batch["context"], batch["next_state"], masks)
  compiled = torch.compile(
    alg._world_model_loss, backend="aot_eager", fullgraph=True, dynamic=False
  )
  for update_statistics in (False, True):
    if update_statistics:
      # Reuse the compiled callable after in-place running-statistic updates.
      wm = alg.world_model
      wm.in_norm.update(wm.inputs(batch["state"], batch["action"], batch["context"]))
      wm.out_norm.update(wm.targets(batch["state"], batch["next_state"]))
      wm.loss_norm.update(wm.state_error(batch["next_state"], batch["state"]))
    eager_loss = alg._world_model_loss(*arguments)
    eager_gradients = torch.autograd.grad(eager_loss, (*parameters, state, actions))
    compiled_loss = compiled(*arguments)
    compiled_gradients = torch.autograd.grad(
      compiled_loss, (*parameters, state, actions)
    )
    torch.testing.assert_close(compiled_loss, eager_loss, rtol=1e-10, atol=1e-12)
    for actual, expected in zip(compiled_gradients, eager_gradients, strict=True):
      assert torch.isfinite(actual).all()
      torch.testing.assert_close(actual, expected, rtol=1e-9, atol=1e-11)
    action_gradient = compiled_gradients[-1]
    assert (action_gradient[:, 0].abs().sum(-1) > 0).all()
    assert action_gradient[~masks].count_nonzero() == 0
    assert compiled_gradients[-2][0, :71].abs().sum() > 0


def test_compiled_imagination_preserves_multistep_actor_gradients(learner):
  alg, replay = learner
  horizon = 3
  seed = replay.starts(4)
  # A varying reference distinguishes the reward frame from the observation frame.
  library = alg.adapter.command.motion
  library.joint_pos += torch.arange(len(library.joint_pos))[:, None] * 0.01
  goals = [
    alg.adapter.reference(seed["motion"], seed["frame"] + t) for t in range(horizon + 1)
  ]
  futures = [
    alg.adapter.future_reference(seed["motion"], seed["frame"] + t)
    for t in range(horizon + 1)
  ]
  noises = [
    alg.adapter.add_noise(torch.zeros_like(seed["history"][:, -1]))
    for _ in range(horizon)
  ]
  compiled = torch.compile(
    alg._imagination_step, backend="aot_eager", fullgraph=True, dynamic=False
  )

  def rollout(step):
    state, history, context = seed["state"], seed["history"], seed["context"]
    heads = alg.world_model.heads
    links = heads.kinematics(state)
    critic_obs = heads.critic_input(state, goals[0], context, futures[0], links)
    values, rewards, continuation, outputs = [alg.critic(critic_obs)], [], [], []
    for t in range(horizon):
      state, history, links, reward, termination, critic_obs, value = step(
        state,
        history,
        context,
        goals[t],
        futures[t],
        links,
        goals[t + 1],
        futures[t + 1],
        noises[t],
      )
      assert state.requires_grad and history.requires_grad
      outputs.extend(
        (
          state,
          history,
          links.position,
          links.quaternion,
          links.linear_velocity,
          links.angular_velocity,
          reward,
          termination,
          critic_obs,
          value,
        )
      )
      values.append(value)
      rewards.append(reward)
      continuation.append(1 - termination)
    objective = imagined_objective(
      torch.stack(rewards),
      torch.stack(continuation),
      torch.stack(values),
      alg.cfg.gamma,
      torch.zeros(horizon, len(state), dtype=torch.bool),
    )
    gradients = torch.autograd.grad(objective, tuple(alg.actor.parameters()))
    return outputs, objective, gradients

  saved = [deepcopy(module.state_dict()) for module in alg.modules]
  with frozen(alg.world_model, alg.critic):
    eager_outputs, eager_objective, eager_gradients = rollout(alg._imagination_step)
    compiled_outputs, compiled_objective, compiled_gradients = rollout(compiled)
  for actual, expected in zip(compiled_outputs, eager_outputs, strict=True):
    torch.testing.assert_close(actual, expected, rtol=1e-9, atol=1e-11)
  torch.testing.assert_close(compiled_objective, eager_objective, rtol=1e-9, atol=1e-11)
  for actual, expected in zip(compiled_gradients, eager_gradients, strict=True):
    assert torch.isfinite(actual).all() and actual.abs().sum() > 0
    torch.testing.assert_close(actual, expected, rtol=1e-8, atol=1e-10)
  for module, initial in zip(alg.modules, saved, strict=True):
    assert all(p.grad is None and p.requires_grad for p in module.parameters())
    for key, value in module.state_dict().items():
      torch.testing.assert_close(value, initial[key], rtol=0, atol=0)


def test_compile_configuration_routes_updates_without_wrapping_modules(
  learner, monkeypatch
):
  eager, replay = learner
  assert not eager.compile_enabled  # None keeps the CPU test path eager.
  compiled_names, calls = [], []

  def compile_spy(function, *, fullgraph, dynamic):
    assert fullgraph is True and dynamic is False
    compiled_names.append(function.__name__)

    @wraps(function)
    def invoke(*args, **kwargs):
      calls.append(function.__name__)
      return function(*args, **kwargs)

    return invoke

  monkeypatch.setattr(torch, "compile", compile_spy)
  cfg = deepcopy(eager.cfg)
  cfg.compile = False
  disabled = SuperMimic(cfg, eager.adapter, "cpu")
  assert not disabled.compile_enabled and not compiled_names
  cfg = deepcopy(cfg)
  cfg.compile = True  # Explicit opt-in also works on CPU.
  cfg.imagination.horizon = 3
  cfg.critic.epochs = cfg.critic.minibatches = 1
  enabled = SuperMimic(cfg, eager.adapter, "cpu")
  for module in enabled.modules:
    module.double()
  enabled.load_state_dict(eager.state_dict())
  assert enabled.compile_enabled
  assert compiled_names == ["_world_model_loss", "_imagination_step"]
  enabled.fit_world_model(replay)
  enabled.improve_policy(replay)
  assert calls.count("_world_model_loss") == cfg.world_model.batches_per_update
  assert calls.count("_imagination_step") == cfg.imagination.horizon
  for module, original, optimizer in zip(
    enabled.modules, eager.modules, enabled.optimizers, strict=True
  ):
    assert module.state_dict().keys() == original.state_dict().keys()
    assert {id(p) for p in module.parameters()} == {
      id(p) for group in optimizer.param_groups for p in group["params"]
    }
