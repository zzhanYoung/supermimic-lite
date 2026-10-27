"""Generic motion-folder CLI and saved records, without physical simulation."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from supermimic.cli.evaluate import METRICS, parse_args, run_evaluation, scan_motions


def motion_file(path: Path, *, fps=50, frames=4, bodies=30, invalid=None):
  quaternions = np.zeros((frames, bodies, 4))
  quaternions[..., 0] = 1
  data: dict[str, Any] = {
    "fps": np.array([fps]),
    "joint_pos": np.zeros((frames, 29)),
    "joint_vel": np.zeros((frames, 29)),
    "body_pos_w": np.zeros((frames, bodies, 3)),
    "body_quat_w": quaternions,
    "body_lin_vel_w": np.zeros((frames, bodies, 3)),
    "body_ang_vel_w": np.zeros((frames, bodies, 3)),
  }
  if invalid == "nan":
    data["joint_pos"][0, 0] = np.nan
  elif invalid == "shape":
    data["joint_vel"] = np.zeros((frames, 28))
  elif invalid == "missing":
    del data["body_quat_w"]
  np.savez(path, **data)


@pytest.mark.parametrize("count", [1, 3, 41])
def test_folder_accepts_any_positive_motion_count(tmp_path, count):
  for index in reversed(range(count)):
    motion_file(tmp_path / f"clip_{index:03}.npz")
  motions = scan_motions(tmp_path)
  assert len(motions) == count
  assert [m["name"] for m in motions] == sorted(m["name"] for m in motions)
  assert all(m["frames"] == 4 for m in motions)


@pytest.mark.parametrize(
  "options",
  [
    {"fps": 30},
    {"frames": 1},
    {"invalid": "nan"},
    {"invalid": "shape"},
    {"invalid": "missing"},
  ],
)
def test_folder_rejects_incompatible_motion_data(tmp_path, options):
  motion_file(tmp_path / "bad.npz", **options)
  with pytest.raises(ValueError, match="Invalid motion.*bad.npz"):
    scan_motions(tmp_path)


def test_empty_folder_and_inconsistent_bodies(tmp_path):
  with pytest.raises(ValueError, match="No top-level NPZ"):
    scan_motions(tmp_path)
  motion_file(tmp_path / "a.npz")
  motion_file(tmp_path / "b.npz", bodies=29)
  with pytest.raises(ValueError, match="body layout"):
    scan_motions(tmp_path)


@pytest.fixture
def args(tmp_path):
  run = tmp_path / "training"
  (run / "params").mkdir(parents=True)
  (run / "params/env.yaml").write_text("{}\n")
  (run / "params/agent.yaml").write_text("{}\n")
  # No TensorBoard logs, replay files, or checkpoint naming convention required.
  (run / "chosen.pt").write_bytes(b"mock policy")
  folder = tmp_path / "held_out_motions"
  folder.mkdir()
  motion_file(folder / "a.npz")
  motion_file(folder / "b.npz", frames=5)
  return parse_args(
    [
      "--algorithm",
      "supermimic",
      "--checkpoint",
      str(run / "chosen.pt"),
      "--motion-folder",
      str(folder),
      "--output-dir",
      str(tmp_path / "results"),
      "--device",
      "cpu",
    ]
  )


def completed_row(motion, *, failed=False):
  expected = motion["frames"] - 1
  steps = 1 if failed else expected
  return {
    "motion": motion["name"],
    "status": "complete",
    "steps": steps,
    "expected_steps": expected,
    "completion": steps / expected,
    "success": not failed,
    "return": 0.08 * steps,
    "normalized_return": 0.8 * steps / expected,
    "failure_reason": "height_failure" if failed else None,
    "first_failure_step": steps if failed else None,
    "mean_errors": {key: 0.2 for key in METRICS},
  }


def fake_evaluator(motions, method, checkpoint, algorithm, **options):
  assert method["task_id"].endswith("SuperMimic")
  assert checkpoint.name == "chosen.pt" and algorithm == "supermimic"
  assert options["num_envs"] == 2 and options["device"] == "cpu"
  rows = []
  for index, motion in enumerate(motions):
    row = completed_row(motion, failed=index == 1)
    rows.append(row)
    options["on_motion"](row)
  return rows


def test_one_command_saves_summary_and_motion_records_without_training_logs(args):
  result = run_evaluation(args, evaluator=fake_evaluator)
  saved = json.loads((args.output_dir / "results.json").read_text())
  assert saved == result and saved["status"] == "complete"
  assert saved["summary"]["success_rate"] == 0.5
  assert saved["summary"]["mean_completion"] == 0.625
  with (args.output_dir / "summary.csv").open() as stream:
    summary = list(csv.DictReader(stream))
  assert len(summary) == 1 and summary[0]["completed_motions"] == "2"
  with (args.output_dir / "per_motion.csv").open() as stream:
    rows = list(csv.DictReader(stream))
  assert len(rows) == 2 and rows[1]["failure_reason"] == "height_failure"
  with pytest.raises(ValueError, match="not empty"):
    run_evaluation(args, evaluator=fake_evaluator)
  assert json.loads((args.output_dir / "results.json").read_text()) == saved


@pytest.mark.parametrize("error", [KeyboardInterrupt, RuntimeError])
def test_interruption_and_error_preserve_completed_motions(args, error):
  def interrupted(motions, _method, _checkpoint, _algorithm, **options):
    options["on_motion"](completed_row(motions[0]))
    raise error("stop")

  with pytest.raises(error):
    run_evaluation(args, evaluator=interrupted)
  saved = json.loads((args.output_dir / "results.json").read_text())
  assert saved["status"] == ("interrupted" if error is KeyboardInterrupt else "error")
  assert len(saved["per_motion"]) == 1 and saved["summary"] is None
  with (args.output_dir / "summary.csv").open() as stream:
    assert next(csv.DictReader(stream))["success_rate"] == ""


def test_default_results_under_logs_and_exported_checkpoint(
  args, tmp_path, monkeypatch
):
  monkeypatch.chdir(tmp_path)
  args.run_dir = args.checkpoint.parent
  exported = tmp_path / "chosen.pt"
  exported.write_bytes(args.checkpoint.read_bytes())
  args.checkpoint = exported
  args.output_dir = None
  result = run_evaluation(args, evaluator=fake_evaluator)
  destination = Path(result["output_dir"])
  assert destination.parent == tmp_path / "logs/eval"
  assert (destination / "summary.csv").is_file()


def test_invalid_parallelism_is_rejected():
  with pytest.raises(SystemExit) as caught:
    parse_args(
      [
        "--algorithm",
        "supermimic",
        "--checkpoint",
        "chosen.pt",
        "--motion-folder",
        "motions",
        "--num-envs",
        "0",
      ]
    )
  assert caught.value.code == 2
