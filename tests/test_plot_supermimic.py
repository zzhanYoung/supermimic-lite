"""Small offline checks for counts, derived metrics and snapshot refresh."""

from __future__ import annotations

import numpy as np
import pytest

from supermimic.cli import plot


def records(values):
  return [plot.Sample(i, value, 1000.0 + i) for i, value in enumerate(values)]


def training_run(algorithm="ppo", end=3):
  run = plot.Run(algorithm, algorithm, metadata={"algorithm": algorithm})
  run.samples = {
    "Train/mean_reward": records(list(range(end))),
    plot.TIME_TAGS[0]: records([1.0] * end),
    plot.TIME_TAGS[1]: records([3599.0] + [3.0] * (end - 1)),
    plot.FAILURE_TAGS[algorithm]: records([0.0] + [0.25] * (end - 1)),
  }
  if algorithm == "ppo":
    run.metadata["ppo_rollout"] = {
      "num_envs_per_rank": 8192,
      "num_steps_per_env": 24,
      "world_size": 8,
      "initial_real_steps": 0,
    }
    run.samples[plot.PPO_OUTCOMES[1]] = records([0.0] + [0.75] * (end - 1))
    run.samples[plot.PPO_OUTCOMES[2]] = records([0.0] * end)
  else:
    run.samples[plot.REAL_STEPS] = records([139264 + i * 12288 for i in range(end)])
    run.samples["Loss/world_model/state"] = records([1.0] * end)
  return run


def test_algorithm_counts_failure_mapping_and_time_share():
  ppo, sm = training_run(), training_run("supermimic")
  values, _ = plot.axis_values(ppo, "real-steps")
  assert values == {i: (i + 1) * 1572864 / 1e6 for i in range(3)}
  assert plot.axis_values(sm, "real-steps")[0][0] == 139264 / 1e6
  assert plot.axis_values(ppo, "train-time")[0] == {0: 60, 1: 3604 / 60, 2: 3608 / 60}
  curves, _ = plot.prepare_curves([ppo, sm], "iteration")
  np.testing.assert_allclose(curves["ppo"][plot.FAILURE].y, [np.nan, 25, 25])
  np.testing.assert_allclose(curves["supermimic"][plot.FAILURE].y, [0, 25, 25])
  np.testing.assert_allclose(
    curves["ppo"][plot.LEARNING_SHARE].y, [100 * 3599 / 3600, 75, 75]
  )
  metrics = plot.summarize([ppo, sm], curves, 2)
  failure = metrics[plot.FAILURE]["runs"]
  assert "cumulative" in failure["ppo"]["statistic_window"]
  assert "recent 100" in failure["supermimic"]["statistic_window"]
  assert metrics["Loss/critic/loss"]["runs"]["ppo"]["last_value"] is None
  ppo.metadata.pop("ppo_rollout")
  with pytest.raises(ValueError, match="ppo_rollout"):
    plot.axis_values(ppo, "real-steps")


def test_missing_records_duplicates_and_gaps_are_visible():
  run = training_run()
  run.samples["Train/mean_reward"].extend(
    [plot.Sample(1, 42, 2000), plot.Sample(1, 10, 1500)]
  )
  normalized, diagnostics = plot.normalize_runs([run])
  run = normalized[0]
  assert run.samples["Train/mean_reward"][1].value == 42
  assert diagnostics["ppo"]["Train/mean_reward"]["duplicates_removed"] == 2
  run.samples[plot.TIME_TAGS[1]].pop(1)
  values, warnings = plot.axis_values(run, "train-time")
  assert values == {0: 60}
  assert "iteration 1" in warnings[0]
  curves, _ = plot.prepare_curves([run], "iteration")
  assert np.isnan(curves["ppo"][plot.LEARNING_SHARE].y[1])
  run.samples[plot.TIME_TAGS[0]][2] = plot.Sample(2, 0, 1002)
  run.samples[plot.TIME_TAGS[1]][1] = plot.Sample(2, 0, 1002)
  assert np.isnan(plot.metric_records(run, plot.LEARNING_SHARE)[2].value)
  curve = plot.Curve(
    np.array([0, 1, 2, 4, 5]),
    np.array([0, 1, 2, 4, 5]),
    np.array([1, 3, np.nan, 8, 10]),
    np.arange(5),
  )
  np.testing.assert_allclose(plot.trailing_mean(curve, 2), [1, 2, np.nan, 8, 9])


def test_csv_snapshot_reuse_and_explicit_refresh(tmp_path, monkeypatch):
  source, output = tmp_path / "input.csv", tmp_path / "plots"
  run = training_run(end=2)
  plot.write_csv(source, [run])
  sources = [plot.Source(source, "PPO", run.metadata)]
  _, first = plot.load_snapshot(sources, output)
  plot.write_csv(source, [training_run(end=3)])
  with monkeypatch.context() as context:
    context.setattr(plot, "read_sources", lambda _: pytest.fail("cache read sources"))
    cached, second = plot.load_snapshot(sources, output)
  assert second == first
  assert len(cached[0].samples["Train/mean_reward"]) == 2
  refreshed, metadata = plot.load_snapshot(sources, output, refresh=True)
  assert len(refreshed[0].samples["Train/mean_reward"]) == 3
  assert refreshed[0].metadata == run.metadata
  assert "schema" not in metadata and "snapshot_sha256" not in metadata
