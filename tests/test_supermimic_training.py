"""Short real-simulator training and checkpoint integration tests."""

from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import cast

import numpy as np
import pytest
import torch
from mjlab.rl import RslRlVecEnvWrapper
from test_tracking_observation_contract import standing_g1_motion as standing_g1_motion

from supermimic.envs import SuperMimicEnv as ManagerBasedRlEnv
from supermimic.learning.config import SuperMimicRunnerCfg
from supermimic.learning.runner import SuperMimicRunner
from supermimic.mdp.return_sampler import ReturnMotionCommandCfg
from supermimic.tasks import load_env_cfg


@pytest.mark.slow
def test_supermimic_real_collection_update_and_resume(
  standing_g1_motion: Path,
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  torch.set_num_threads(2)
  cfg = load_env_cfg("Mjlab-Tracking-Flat-Unitree-G1-SuperMimic")
  cfg.scene.num_envs = 2
  cfg.seed = 13
  cast(ReturnMotionCommandCfg, cfg.commands["motion"]).motion_folder = str(
    standing_g1_motion.parent
  )
  # This is a unit integration preset; the documented acceptance smoke uses
  # full paper widths and horizons on the actual 40-clip LAFAN1 directory.
  agent = SuperMimicRunnerCfg(replay_steps=64)
  agent.world_model.hidden_dims = (32, 32)
  agent.imagination.hidden_dims = (32, 16)
  agent.critic.hidden_dims = (16,)
  agent.world_model.batch_size = 4
  agent.imagination.batch_size = 4
  env = ManagerBasedRlEnv(cfg, device="cpu")
  wrapped = RslRlVecEnvWrapper(env)
  try:
    runner = SuperMimicRunner(wrapped, asdict(agent), device="cpu")
    assert runner.adapter.command.return_sampler.g_max == pytest.approx(10.0)
    # Distinguish reference frames so collection checks catch stale/reset goals.
    library = runner.adapter.command.motion
    library.joint_pos += torch.arange(len(library.joint_pos))[:, None] * 0.005
    actor_obs = env.observation_manager.compute()["actor"]
    assert isinstance(actor_obs, torch.Tensor)
    before_history = actor_obs.clone()
    before_rng = torch.get_rng_state().clone()
    first_clean = runner.adapter.clean_observations()
    second_clean = runner.adapter.clean_observations()
    assert torch.equal(before_rng, torch.get_rng_state())
    after_history = env.observation_manager.compute()["actor"]
    assert isinstance(after_history, torch.Tensor)
    assert torch.equal(before_history, after_history)
    for first, second in zip(first_clean, second_clean, strict=True):
      assert torch.equal(first, second)
    clean_actor = first_clean[0]
    gravity_slice = runner.adapter.slices["actor"]["projected_gravity"]
    assert set(range(gravity_slice.start, gravity_slice.stop)).issubset(
      runner.adapter.proprio_features
    )
    for group, clean in zip(("actor", "critic"), first_clean, strict=True):
      torch.testing.assert_close(
        clean[:, runner.adapter.slices[group]["projected_gravity"]],
        env.scene["robot"].data.projected_gravity_b,
      )
    noisy = runner.adapter.add_noise(clean_actor)
    assert torch.all(noisy - clean_actor >= runner.adapter.noise_low - 1e-6)
    assert torch.all(noisy - clean_actor <= runner.adapter.noise_high + 1e-6)

    # Record the actual actor inputs and pre-reset physics to independently
    # verify replay's single-frame compression and terminal transition capture.
    histories: dict[tuple[int, int, int], torch.Tensor] = {}
    next_states: list[torch.Tensor] = []
    actor_goals: list[torch.Tensor] = []
    original_step = env.step

    def record_goal(_module, inputs):
      actor_goals.append(inputs[1].detach().clone())

    def record_step(action: torch.Tensor):
      command = runner.adapter.command
      torch.testing.assert_close(
        actor_goals[-1],
        runner.adapter.future_reference(command.motion_ids, command.time_steps),
      )
      observed = env.observation_manager.compute()["actor"]
      assert isinstance(observed, torch.Tensor)
      history = runner.adapter.unpack_history(observed)
      for index in range(env.num_envs):
        key = (
          index,
          int(runner.episode_ids[index]),
          int(env.episode_length_buf[index]),
        )
        histories[key] = history[index].clone()
      result = original_step(action)
      next_states.append(runner.adapter.state().clone())
      return result

    monkeypatch.setattr(env, "step", record_step)
    signature = runner.signature
    old_actor = [p.detach().clone() for p in runner.alg.actor.parameters()]
    handle = runner.alg.actor.register_forward_pre_hook(record_goal)
    try:
      runner.learn(1)
    finally:
      handle.remove()
    monkeypatch.setattr(env, "step", original_step)
    collected_steps = 32 + 2
    assert runner.replay.size == collected_steps
    assert runner.adapter.command.return_sampler.pending_count.sum() == 0
    assert runner.adapter.command.return_sampler.returns.abs().sum() > 0
    assert any(
      not torch.equal(old, new)
      for old, new in zip(old_actor, runner.alg.actor.parameters(), strict=True)
    )
    assert all(torch.isfinite(torch.tensor(x)) for x in runner.last_metrics.values())
    data = runner.replay.data
    assert torch.any(
      data["terminated"][:collected_steps] | data["truncated"][:collected_steps]
    )
    assert data["obs"].shape[-1] == 157
    assert data["context"].shape[-1] == 33
    wm = runner.alg.world_model
    inputs = wm.inputs(
      data["state"][:collected_steps],
      data["action"][:collected_steps],
      data["context"][:collected_steps],
    ).flatten(0, 1)
    increments = wm.targets(
      data["state"][:collected_steps], data["next_state"][:collected_steps]
    ).flatten(0, 1)
    for normalizer, values, size in (
      (wm.in_norm, inputs, 133),
      (wm.out_norm, increments, 71),
      (
        wm.loss_norm,
        wm.state_error(
          data["next_state"][:collected_steps], data["state"][:collected_steps]
        ).flatten(0, 1),
        70,
      ),
    ):
      assert normalizer.mean.numel() == size
      assert normalizer.count == collected_steps * env.num_envs
      torch.testing.assert_close(normalizer.mean, values.mean(0))
      torch.testing.assert_close(
        normalizer.variance, values.var(0, unbiased=False), atol=1e-6, rtol=1e-5
      )
    assert "next_critic" not in data
    assert "future" not in data
    assert runner.alg.critic.normalizer.mean.numel() == 914
    goal_norm = runner.alg.actor.goal_normalizer
    goals = runner.adapter.future_reference(
      data["motion"][:collected_steps].flatten(),
      data["frame"][:collected_steps].flatten(),
    )
    assert goal_norm.count == collected_steps * env.num_envs
    torch.testing.assert_close(goal_norm.mean, goals.mean(0))
    torch.testing.assert_close(goal_norm.variance, goals.var(0, unbiased=False))
    assert runner.optimizer_updates == {"world_model": 10, "actor": 1, "critic": 20}
    assert runner.collection_remainder == 0.5
    torch.testing.assert_close(
      data["next_state"][:collected_steps], torch.stack(next_states)
    )
    # Compare every collected state's reconstructed history, including resets.
    for index in range(collected_steps):
      for env_id in range(env.num_envs):
        episode, age = (
          int(data["episode"][index, env_id]),
          int(data["age"][index, env_id]),
        )
        slots = index - torch.arange(9, -1, -1).clamp_max(age)
        torch.testing.assert_close(
          data["obs"][slots, env_id], histories[env_id, episode, age]
        )
    command = runner.adapter.command
    motions = command.motion_ids.clone()
    command.time_steps.copy_(command.motion_lengths - 1)
    command._update_command()
    command._update_command()
    assert torch.equal(command.time_steps, command.motion_lengths - 1)
    assert torch.equal(command.motion_ids, motions)
    model = tmp_path / "model_1.pt"
    runner.save(str(model))
    wm_statistics = {
      name: deepcopy(getattr(wm, name).state_dict())
      for name in ("in_norm", "out_norm", "loss_norm")
    }
    for name in wm_statistics:
      getattr(wm, name).mean.fill_(123)
    with np.load(model.with_suffix(".sampling.npz")) as distribution:
      torch.testing.assert_close(
        torch.from_numpy(distribution["probabilities"]),
        command.return_sampler.probabilities().cpu(),
      )
    saved = [p.clone() for p in runner.alg.actor.parameters()]
    goal_statistics = deepcopy(goal_norm.state_dict())
    goal_norm.mean.fill_(123)
    runner.load(str(model))
    for name, statistics in wm_statistics.items():
      for key, value in statistics.items():
        torch.testing.assert_close(getattr(wm, name).state_dict()[key], value)
    assert runner.signature == signature
    assert runner.current_learning_iteration == 1
    assert runner.replay.size == collected_steps
    for key, value in goal_norm.state_dict().items():
      torch.testing.assert_close(value, goal_statistics[key])
    for before, after in zip(saved, runner.alg.actor.parameters(), strict=True):
      assert torch.equal(before, after)
    runner.learn(1)
    assert runner.current_learning_iteration == 2
    assert runner.replay.size == collected_steps + 3
    assert goal_norm.count == (collected_steps + 3) * env.num_envs
    assert runner.collection_remainder == 0
    assert runner.optimizer_updates == {"world_model": 20, "actor": 2, "critic": 40}
    assert all(torch.isfinite(torch.tensor(x)) for x in runner.last_metrics.values())
  finally:
    env.close()
