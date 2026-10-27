"""Evaluate a G1 checkpoint on every NPZ motion in a folder."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from supermimic.assets import DEFAULT_ASSETS, LocalAssets, require_path

TASKS = {
  "supermimic": "Mjlab-Tracking-Flat-Unitree-G1-SuperMimic",
}
METRICS = {
  "mpkpe": "m (global key-body position)",
  "mpjpe": "rad (joint angle, not Cartesian joint position)",
  "mpjve": "rad/s",
  "linear_velocity_error": "m/s",
  "yaw_velocity_error": "rad/s",
}


def json_data(value: Any) -> Any:
  """Normalize tuples and integer mapping keys for stable JSON round trips."""
  return json.loads(json.dumps(value, allow_nan=False))


class SnapshotLoader(yaml.SafeLoader):
  """Represent Python tags as inert data; never import or execute YAML objects."""


def snapshot_sequence(loader: SnapshotLoader, node: yaml.Node) -> list:
  if not isinstance(node, yaml.SequenceNode):
    raise ValueError("Expected a configuration sequence.")
  return loader.construct_sequence(node)


SnapshotLoader.add_constructor("tag:yaml.org,2002:python/tuple", snapshot_sequence)
SnapshotLoader.add_multi_constructor(
  "tag:yaml.org,2002:python/name:", lambda _loader, name, _node: name
)
SnapshotLoader.add_multi_constructor(
  "tag:yaml.org,2002:python/object/apply:",
  lambda loader, name, node: {
    "python_type": name,
    "args": snapshot_sequence(loader, node),
  },
)


def read_snapshot(path: Path) -> dict:
  data = yaml.load(path.read_text(), Loader=SnapshotLoader)
  if not isinstance(data, dict):
    raise ValueError(f"Expected a configuration mapping: {path}")
  return json_data(data)


def validate_row(row: dict, motion: dict) -> None:
  expected = motion["frames"] - 1
  if row["status"] != "complete" or row["motion"] != motion["name"]:
    raise ValueError("Invalid or incomplete motion result.")
  if row["expected_steps"] != expected or not 1 <= row["steps"] <= expected:
    raise ValueError("Invalid motion length.")
  if set(row["mean_errors"]) != set(METRICS):
    raise ValueError("Unexpected tracking metrics.")
  numbers = [
    row["return"],
    row["normalized_return"],
    row["completion"],
    *row["mean_errors"].values(),
  ]
  if not all(math.isfinite(v) for v in numbers):
    raise ValueError("Nonfinite motion result.")
  if not math.isclose(row["normalized_return"], row["return"] / (0.1 * expected)):
    raise ValueError("Incorrect full-motion return denominator.")
  if not math.isclose(row["completion"], row["steps"] / expected):
    raise ValueError("Incorrect completion fraction.")
  failed = bool(row["failure_reason"])
  if row["success"] != (row["steps"] == expected and not failed):
    raise ValueError("Inconsistent success status.")
  if not row["success"] and not failed:
    raise ValueError("Unfinished motion cannot be a complete evaluation.")
  if failed and row["first_failure_step"] != row["steps"]:
    raise ValueError("Evaluation continued after failure.")
  if not failed and row["first_failure_step"] is not None:
    raise ValueError("Successful motion contains a failure step.")


def summarize(rows: list[dict], motions: list[dict]) -> dict:
  by_name = {m["name"]: m for m in motions}
  if len(rows) != len(motions) or {r["motion"] for r in rows} != set(by_name):
    raise ValueError("Cannot score incomplete or duplicate motion coverage.")
  for row in rows:
    validate_row(row, by_name[row["motion"]])
  return {
    "normalized_return": sum(r["normalized_return"] for r in rows) / len(rows),
    "normalized_return_framew": sum(r["return"] for r in rows)
    / (0.1 * sum(r["expected_steps"] for r in rows)),
    "successes": sum(r["success"] for r in rows),
    "success_rate": sum(r["success"] for r in rows) / len(rows),
    "n_failed": sum(not r["success"] for r in rows),
    "failed": [r["motion"] for r in rows if not r["success"]],
    "mean_completion": sum(r["completion"] for r in rows) / len(rows),
    **{key: sum(r["mean_errors"][key] for r in rows) / len(rows) for key in METRICS},
  }


def scan_motions(directory: Path) -> list[dict]:
  """Read any nonempty folder of compatible, finite G1 motion arrays."""
  if not directory.is_dir():
    raise ValueError(f"Motion folder does not exist: {directory}")
  files = sorted(directory.glob("*.npz"))
  if not files:
    raise ValueError(f"No top-level NPZ motions in {directory}")
  motions = []
  body_count = None
  for path in files:
    try:
      with np.load(path, allow_pickle=False) as data:
        fps = np.asarray(data["fps"]).reshape(-1)
        if fps.size != 1 or not np.isclose(fps[0], 50, rtol=0, atol=1e-6):
          raise ValueError("Motion must be 50 Hz.")
        joint_pos = data["joint_pos"]
        if joint_pos.ndim != 2 or joint_pos.shape[1] != 29 or len(joint_pos) < 2:
          raise ValueError("Motion needs at least two frames and 29 joints.")
        frames = len(joint_pos)
        positions = data["body_pos_w"]
        if positions.ndim != 3 or positions.shape[1] < 1:
          raise ValueError("Invalid body_pos_w shape.")
        bodies = positions.shape[1]
        if body_count is not None and bodies != body_count:
          raise ValueError("Motions must have the same body layout.")
        body_count = bodies
        for key, shape in {
          "joint_pos": (frames, 29),
          "joint_vel": (frames, 29),
          "body_pos_w": (frames, bodies, 3),
          "body_quat_w": (frames, bodies, 4),
          "body_lin_vel_w": (frames, bodies, 3),
          "body_ang_vel_w": (frames, bodies, 3),
        }.items():
          array = data[key]
          if array.shape != shape or not np.isfinite(array).all():
            raise ValueError(f"Invalid shape or nonfinite values in {key}.")
    except (KeyError, ValueError, TypeError) as error:
      raise ValueError(f"Invalid motion {path}: {error}") from error
    motions.append(
      {"name": path.stem, "path": str(path.resolve()), "frames": frames, "fps": 50.0}
    )
  return motions


def write_results(directory: Path, result: dict) -> None:
  """Save completed motions even if a later rollout is interrupted or fails."""
  temporary = directory / "results.json.tmp"
  temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
  temporary.replace(directory / "results.json")
  summary = {
    "checkpoint": result["checkpoint"],
    "algorithm": result["algorithm"],
    "status": result["status"],
    "motion_count": len(result["motions"]),
    "completed_motions": len(result["per_motion"]),
    "evaluation_wall_seconds": result["evaluation_wall_seconds"],
    **(result["summary"] or {}),
  }
  summary.pop("failed", None)
  summary_fields = [
    "checkpoint",
    "algorithm",
    "status",
    "motion_count",
    "completed_motions",
    "evaluation_wall_seconds",
    "normalized_return",
    "normalized_return_framew",
    "successes",
    "success_rate",
    "n_failed",
    "mean_completion",
    *METRICS,
  ]
  motion_fields = [
    "motion",
    "steps",
    "expected_steps",
    "completion",
    "success",
    "return",
    "normalized_return",
    "failure_reason",
    "first_failure_step",
    *METRICS,
  ]
  rows = [{**row, **row["mean_errors"]} for row in result["per_motion"]]
  for name, fields, records in (
    ("summary.csv", summary_fields, [summary]),
    ("per_motion.csv", motion_fields, rows),
  ):
    temporary = directory / f"{name}.tmp"
    with temporary.open("w", newline="") as stream:
      writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
      writer.writeheader()
      writer.writerows(records)
    temporary.replace(directory / name)


def parse_args(
  argv: list[str] | None = None, *, assets: LocalAssets = DEFAULT_ASSETS
) -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--algorithm", choices=tuple(TASKS), default="supermimic")
  parser.add_argument(
    "--checkpoint", "--checkpoint-file", type=Path, default=assets.checkpoint
  )
  parser.add_argument("--motion-folder", type=Path, default=assets.motion_folder)
  parser.add_argument(
    "--run-dir",
    type=Path,
    help="Training directory with params/env.yaml and agent.yaml; defaults to checkpoint parent.",
  )
  parser.add_argument(
    "--output-dir",
    type=Path,
    help="Empty output directory; default: logs/eval/<UTC timestamp>_<checkpoint>.",
  )
  parser.add_argument(
    "--num-envs", type=int, default=40, help="Maximum parallel motions."
  )
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--seed", type=int, default=42)
  args = parser.parse_args(argv)
  args.assets = assets
  if args.num_envs < 1:
    parser.error("--num-envs must be positive")
  return args


def run_evaluation(args: argparse.Namespace, *, evaluator=None) -> dict:
  checkpoint = require_path(args.checkpoint, "--checkpoint")
  args.motion_folder = require_path(
    args.motion_folder, "--motion-folder", directory=True
  )
  if not checkpoint.is_file():
    raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
  assets = getattr(args, "assets", DEFAULT_ASSETS)
  run_dir = assets.checkpoint_run_dir(checkpoint, args.run_dir)
  method = {
    "task_id": TASKS[args.algorithm],
    "env_config": read_snapshot(run_dir / "params/env.yaml"),
    "agent_config": read_snapshot(run_dir / "params/agent.yaml"),
  }
  motions = scan_motions(args.motion_folder.expanduser().resolve())
  timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
  directory = (
    (args.output_dir or Path("logs/eval") / f"{timestamp}_{checkpoint.stem}")
    .expanduser()
    .resolve()
  )
  directory = assets.protect_output(directory, checkpoint, args.motion_folder, run_dir)
  if directory.exists() and (not directory.is_dir() or any(directory.iterdir())):
    raise ValueError(
      f"Output directory is not empty; choose a new --output-dir: {directory}"
    )
  directory.mkdir(parents=True, exist_ok=True)
  result: dict[str, Any] = {
    "status": "running",
    "checkpoint": str(checkpoint),
    "algorithm": args.algorithm,
    "run_dir": str(run_dir),
    "motion_folder": str(args.motion_folder.expanduser().resolve()),
    "output_dir": str(directory),
    "num_envs": min(args.num_envs, len(motions)),
    "seed": args.seed,
    "device": args.device,
    "metric_units": METRICS,
    "protocol": {
      "control_dt": 0.02,
      "start_frame": 0,
      "expected_steps": "frames - 1",
      "normalized_return": "sum(rewards) / (5 * 0.02 * (frames - 1))",
      "aggregation": "equal weight per motion; errors include terminal steps",
      "reward": "local SuperMimic, including anchor-height reward",
      "reference_timing": "post-physics state versus pre-step reference",
      "height_failure_m": 0.25,
      "gravity_z_failure": 0.8,
      "foot_friction": 0.6,
      "self_collisions": False,
      "randomization": False,
      "reset_noise": False,
      "observation_noise": False,
      "action_noise": False,
    },
    "motions": motions,
    "per_motion": [],
    "summary": None,
    "evaluation_wall_seconds": 0.0,
  }
  print(f"Evaluating {len(motions)} motions; results: {directory}", flush=True)
  write_results(directory, result)
  started = time.perf_counter()

  def completed(row):
    result["per_motion"].append(row)
    result["evaluation_wall_seconds"] = time.perf_counter() - started
    write_results(directory, result)
    print(
      f"[{len(result['per_motion'])}/{len(motions)}] {row['motion']}: "
      f"success={row['success']}, steps={row['steps']}/{row['expected_steps']}, "
      f"return={row['normalized_return']:.4f}",
      flush=True,
    )

  try:
    if evaluator is None:
      from supermimic.evaluation.runtime import evaluate_folder

      evaluator = evaluate_folder
    rows = evaluator(
      motions,
      method,
      checkpoint,
      args.algorithm,
      num_envs=result["num_envs"],
      seed=args.seed,
      device=args.device,
      on_motion=completed,
    )
    summary = summarize(rows, motions)
    result.update(status="complete", per_motion=rows, summary=summary)
  except KeyboardInterrupt:
    result.update(status="interrupted", reason="User interrupted evaluation.")
    raise
  except Exception as error:
    result.update(status="error", reason=str(error))
    raise
  finally:
    result["evaluation_wall_seconds"] = time.perf_counter() - started
    write_results(directory, result)
  print(
    f"Success: {summary['successes']}/{len(motions)} "
    f"({summary['success_rate']:.1%}); mean completion: {summary['mean_completion']:.1%}",
    flush=True,
  )
  return result


def main(argv: list[str] | None = None, *, assets: LocalAssets = DEFAULT_ASSETS) -> int:
  args = parse_args(argv, assets=assets)
  try:
    run_evaluation(args)
  except KeyboardInterrupt:
    print("Interrupted; completed motion records were saved.")
    return 130
  except (OSError, ValueError, KeyError, RuntimeError) as error:
    print(f"Evaluation error: {error}")
    return 1
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
