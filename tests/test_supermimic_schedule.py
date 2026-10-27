"""Count real optimizer calls and preserve fractional collection on resume."""

from collections import Counter
from dataclasses import asdict

import pytest
import torch
from mjlab.rl import RslRlVecEnvWrapper
from test_supermimic_heads import analytic_adapter as analytic_adapter
from test_tracking_observation_contract import standing_g1_motion as standing_g1_motion

from supermimic.learning import algorithm
from supermimic.learning.config import SuperMimicRunnerCfg
from supermimic.learning.runner import SuperMimicRunner


def test_updates_per_simulator_step_and_checkpoint(
  analytic_adapter, monkeypatch, tmp_path
):
  cfg = SuperMimicRunnerCfg(replay_steps=64)
  cfg.world_model.hidden_dims = (16,)
  cfg.world_model.batch_size = 4
  cfg.imagination.hidden_dims = (16,)
  cfg.imagination.batch_size = 4
  cfg.imagination.horizon = 2
  cfg.critic.hidden_dims = (16,)
  runner = SuperMimicRunner(
    RslRlVecEnvWrapper(analytic_adapter.env),
    asdict(cfg),
    log_dir=str(tmp_path / "logs"),
  )
  original = algorithm.optimizer_step
  counts = Counter()

  def count_update(loss, module, optimizer, config, elementwise_clip=None):
    name = ("world_model", "actor", "critic")[runner.alg.modules.index(module)]
    counts[name] += 1
    return original(loss, module, optimizer, config, elementwise_clip)

  monkeypatch.setattr(algorithm, "optimizer_step", count_update)
  runner.learn(9)
  assert runner.replay.total_steps == 32 + 22
  assert runner.collection_remainder == 0.5
  assert runner.logger.tot_timesteps == runner.real_steps
  checkpoint = tmp_path / "round9.pt"
  runner.save(str(checkpoint))
  runner.load(str(checkpoint))
  assert runner.initial_collection_done
  runner.learn(1)
  assert runner.replay.total_steps == 32 + 25
  assert runner.collection_remainder == 0
  assert counts == {"world_model": 100, "actor": 10, "critic": 200}
  assert runner.optimizer_updates == counts
  assert runner.real_steps == 57 * analytic_adapter.env.num_envs
  assert runner.logger.tot_timesteps == runner.real_steps
  # Reject mismatched actor inputs before loading model, optimizer or replay state.
  torch.save(
    {
      "signature": {
        **runner.signature,
        "actor_input": "proprio_history.930_future_reference.vibed.730.v1",
      },
      "world_size": runner.world_size,
    },
    tmp_path / "old_actor.pt",
  )
  with pytest.raises(ValueError, match="actor inputs.*start a new training run"):
    runner.load(str(tmp_path / "old_actor.pt"))

  # Preserve models/replay when migrating an otherwise compatible sampler.
  legacy = torch.load(checkpoint, weights_only=False)
  legacy["signature"]["sampler"].pop("history_alpha")
  legacy["signature"]["sampler"].pop("max_probability_ratio")
  legacy_model = tmp_path / "legacy.pt"
  torch.save(legacy, legacy_model)
  shard = torch.load(tmp_path / "round9.rank0.replay.pt", weights_only=False)
  shard["sampler"]["return_estimator"] = "timeout_discount_normalized.v1"
  shard["sampler"]["returns"].fill_(123)
  torch.save(shard, tmp_path / "legacy.rank0.replay.pt")
  with pytest.warns(UserWarning, match="Resetting legacy sampler"):
    runner.load(str(legacy_model))
  assert runner.current_learning_iteration == 9
  assert runner.replay.total_steps == 54
  assert runner.adapter.command.return_sampler.returns.count_nonzero() == 0
  assert runner.episode_discounted_returns.count_nonzero() == 0
  for name, value in runner.alg.actor.state_dict().items():
    torch.testing.assert_close(value, legacy["algorithm"]["actor"][name])

  # New explicit sampler settings remain part of the strict task signature.
  legacy["signature"]["sampler"]["max_probability_ratio"] = 5.0
  torch.save(legacy, legacy_model)
  with pytest.raises(ValueError, match="task configuration"):
    runner.load(str(legacy_model))


def test_fixed_collection_config_requires_migration():
  with pytest.raises(ValueError, match="obsolete"):
    SuperMimicRunnerCfg(num_steps_per_env=25).validate()
