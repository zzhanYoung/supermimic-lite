"""Plot PPO/SuperMimic efficiency and SuperMimic diagnostics from offline logs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import EngFormatter, FuncFormatter, MaxNLocator
from tensorboard.backend.event_processing.event_file_loader import LegacyEventFileLoader
from tensorboard.util import tensor_util

matplotlib.use("Agg")

DEFAULT_CONFIG = Path(__file__).with_name("train_comparison.json")
TIME_TAGS = ("Perf/collection_time", "Perf/learning_time")
REAL_STEPS = "Loss/training/real_steps"
FAILURE = "Derived/failure_percent"
LEARNING_SHARE = "Derived/learning_percent"
FAILURE_TAGS = {
  "ppo": "Metrics/motion/adaptive_failure_rate",
  "supermimic": "Loss/episode/failure_fraction",
}
FAILURE_WINDOWS = {
  "ppo": "cumulative episodes, synchronized sampler statistics",
  "supermimic": "recent 100 completed episodes on the logging rank",
}
PPO_OUTCOMES = (
  FAILURE_TAGS["ppo"],
  "Metrics/motion/adaptive_success_rate",
  "Metrics/motion/adaptive_neutral_timeouts",
)
AXES = {
  "iteration": "Learner iteration",
  "train-time": "Cumulative logged training time (min)",
  "real-steps": "Real environment steps (millions)",
}
COLORS = ("#D57835", "#2575B9", "#008B83", "#9856A5", "#C6475C", "#7F8540")


@dataclass(frozen=True)
class Panel:
  title: str
  ylabel: str
  tags: tuple[str, ...]
  caption: str
  log_positive: bool = False


COMPARISON_PANELS = (
  Panel(
    "Episode reward", "Mean episode return", ("Train/mean_reward",), "Train/mean_reward"
  ),
  Panel(
    "Episode length",
    "Control steps",
    ("Train/mean_episode_length",),
    "Train/mean_episode_length",
  ),
  Panel(
    "Failure fraction",
    "Failed episodes (%)",
    (FAILURE,),
    "PPO: Metrics/motion/adaptive_failure_rate\nSM: Loss/episode/failure_fraction",
  ),
  Panel(
    "Body position tracking error",
    "Mean body position error (m)",
    ("Metrics/motion/error_body_pos",),
    "Metrics/motion/error_body_pos",
  ),
)
SUPERMIMIC_PANELS = (
  Panel(
    "WM loss",
    "Normalized state loss",
    ("Loss/world_model/state",),
    "Loss/world_model/state",
    True,
  ),
  Panel(
    "Actor objective (higher is better)",
    "Imagined objective",
    ("Loss/actor/objective",),
    "Loss/actor/objective",
  ),
  Panel("Critic loss", "Value MSE", ("Loss/critic/loss",), "Loss/critic/loss", True),
  Panel(
    "Actor learning rate",
    "Learning rate",
    ("Loss/learning_rate",),
    "Loss/learning_rate (actor only; unsmoothed)",
  ),
  Panel(
    "Gradient norms",
    "Pre-clipping gradient norm",
    tuple(f"Loss/{part}/grad_norm" for part in ("world_model", "actor", "critic")),
    "Loss/{world_model,actor,critic}/grad_norm",
    True,
  ),
  Panel(
    "Learning time breakdown",
    "Component time / iteration (s)",
    tuple(
      f"Loss/time/{part}" for part in ("world_model", "imagination_bptt", "critic")
    ),
    "Loss/time/{world_model,imagination_bptt,critic}",
    True,
  ),
)
PANEL_GROUPS = {"comparison": COMPARISON_PANELS, "supermimic": SUPERMIMIC_PANELS}
METRIC_TAGS = tuple(
  tag for panels in PANEL_GROUPS.values() for panel in panels for tag in panel.tags
) + ("Perf/total_fps", LEARNING_SHARE)
CAPTURE_TAGS = tuple(
  dict.fromkeys(
    (
      *(tag for tag in METRIC_TAGS if not tag.startswith("Derived/")),
      *TIME_TAGS,
      REAL_STEPS,
      *FAILURE_TAGS.values(),
      *PPO_OUTCOMES,
      "Loss/value",
      "Loss/surrogate",
    )
  )
)
COMPONENT_LABELS = {
  "world_model": "WM",
  "actor": "Actor",
  "critic": "Critic",
  "imagination_bptt": "Imagination / BPTT",
}


@dataclass(frozen=True)
class Source:
  path: Path
  label: str | None = None
  metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Sample:
  step: int
  value: float
  wall_time: float


@dataclass
class Run:
  name: str
  label: str
  samples: dict[str, list[Sample]] = field(default_factory=dict)
  metadata: dict[str, Any] = field(default_factory=dict)

  @property
  def algorithm(self) -> str:
    return self.metadata.get("algorithm", "unknown")


@dataclass
class Curve:
  steps: np.ndarray
  x: np.ndarray
  y: np.ndarray
  wall: np.ndarray


def utc(timestamp: float) -> str:
  return datetime.fromtimestamp(timestamp, timezone.utc).isoformat()


def write_json(path: Path, value: Any) -> None:
  temporary = path.with_suffix(path.suffix + ".tmp")
  temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
  temporary.replace(path)


def csv_run_name(path: Path) -> str:
  summary = path.with_name("summary.json")
  if summary.exists():
    metadata = json.loads(summary.read_text())
    if isinstance(metadata.get("run"), str):
      return Path(metadata["run"]).name
  return path.parent.name if path.stem == "scalars" else path.stem


def read_csv(path: Path) -> list[Run]:
  runs: dict[str, Run] = {}
  with path.open(newline="") as stream:
    reader = csv.DictReader(stream)
    required = {"tag", "step", "value", "wall_time_utc"}
    if not required.issubset(reader.fieldnames or []):
      raise ValueError(f"{path}: CSV requires columns {sorted(required)}")
    default_name = csv_run_name(path)
    for line, row in enumerate(reader, 2):
      if row["tag"] not in CAPTURE_TAGS:
        continue
      try:
        timestamp = datetime.fromisoformat(row["wall_time_utc"])
        if timestamp.tzinfo is None:
          timestamp = timestamp.replace(tzinfo=timezone.utc)
        sample = Sample(int(row["step"]), float(row["value"]), timestamp.timestamp())
      except (ValueError, TypeError) as error:
        raise ValueError(f"{path}:{line}: invalid scalar record: {error}") from error
      name = row.get("run") or default_name
      run = runs.setdefault(name, Run(name, name))
      run.samples.setdefault(row["tag"], []).append(sample)
  provenance = path.with_name("provenance.json")
  if provenance.exists():
    metadata = json.loads(provenance.read_text())
    for run in runs.values():
      run.label = metadata.get("run_labels", {}).get(run.name, run.label)
      run.metadata = metadata.get("run_metadata", {}).get(run.name, {})
  return list(runs.values())


def read_tensorboard(path: Path) -> list[Run]:
  files = sorted(path.glob("events.out.tfevents.*"))
  if not files:
    raise ValueError(f"{path}: no TensorBoard event files")
  run = Run(path.name, path.name)
  for event_file in files:
    # Read both protobuf scalar encodings directly, without migrating every event.
    for event in LegacyEventFileLoader(str(event_file)).Load():
      for value in event.summary.value:
        if value.tag not in CAPTURE_TAGS:
          continue
        if value.HasField("tensor"):
          array = tensor_util.make_ndarray(value.tensor)
          if array.size != 1:
            raise ValueError(f"{event_file}: {value.tag} is not a scalar")
          scalar = float(array.reshape(-1)[0])
        elif value.HasField("simple_value"):
          scalar = float(value.simple_value)
        else:
          continue
        run.samples.setdefault(value.tag, []).append(
          Sample(int(event.step), scalar, float(event.wall_time))
        )
  return [run]


def read_sources(sources: list[Source]) -> list[Run]:
  runs: list[Run] = []
  names: set[str] = set()
  for source in sources:
    if source.path.is_dir():
      group = read_tensorboard(source.path)
    elif source.path.is_file() and source.path.suffix.lower() == ".csv":
      group = read_csv(source.path)
    else:
      raise ValueError(f"Source is not a TensorBoard directory or CSV: {source.path}")
    if not group or not any(r.samples.get(t) for r in group for t in METRIC_TAGS):
      raise ValueError(f"{source.path}: no supported training metrics recorded")
    for run in group:
      run.metadata.update(source.metadata)
      max_iteration = run.metadata.get("max_iteration")
      if max_iteration is not None:
        if type(max_iteration) is not int or max_iteration < 0:
          raise ValueError(f"{run.name}: max_iteration must be a nonnegative integer")
        run.samples = {
          tag: [sample for sample in samples if sample.step <= max_iteration]
          for tag, samples in run.samples.items()
        }
      if run.algorithm == "unknown":
        if any(
          tag.startswith(("Loss/world_model/", "Loss/actor/")) for tag in run.samples
        ):
          run.metadata["algorithm"] = "supermimic"
        elif any(
          tag in run.samples
          for tag in ("Loss/value", "Loss/surrogate", FAILURE_TAGS["ppo"])
        ):
          run.metadata["algorithm"] = "ppo"
      if run.algorithm not in FAILURE_TAGS:
        raise ValueError(
          f"{run.name}: cannot identify algorithm; set algorithm to ppo or "
          "supermimic in the source config"
        )
      if source.label:
        run.label = source.label if len(group) == 1 else f"{source.label} / {run.name}"
      if run.name in names:
        raise ValueError(f"Duplicate run name {run.name!r} across sources")
      names.add(run.name)
      runs.append(run)
  return runs


def write_csv(path: Path, runs: list[Run]) -> None:
  temporary = path.with_suffix(".csv.tmp")
  with temporary.open("w", newline="") as stream:
    writer = csv.writer(stream)
    writer.writerow(["run", "tag", "step", "value", "wall_time_utc"])
    for run in runs:
      for tag, samples in run.samples.items():
        for sample in samples:
          writer.writerow(
            [run.name, tag, sample.step, sample.value, utc(sample.wall_time)]
          )
  temporary.replace(path)


def normalize_runs(runs: list[Run]) -> tuple[list[Run], dict]:
  """Keep latest-wall-time records, but retain NaNs as explicit curve gaps."""
  result, diagnostics = [], {}
  for run in runs:
    normalized = Run(run.name, run.label, metadata=dict(run.metadata))
    counts = {}
    for tag, samples in run.samples.items():
      selected: dict[int, Sample] = {}
      for sample in samples:
        if sample.step < 0 or not math.isfinite(sample.wall_time):
          raise ValueError(f"{run.name}/{tag}: invalid step or wall time")
        prior = selected.get(sample.step)
        if prior is None or sample.wall_time >= prior.wall_time:
          selected[sample.step] = sample
      ordered = [selected[step] for step in sorted(selected)]
      normalized.samples[tag] = ordered
      counts[tag] = {
        "raw_records": len(samples),
        "duplicates_removed": len(samples) - len(ordered),
        "nonfinite_records": sum(not math.isfinite(s.value) for s in ordered),
      }
    diagnostics[run.name] = counts
    result.append(normalized)
  return result, diagnostics


def load_snapshot(
  sources: list[Source], output: Path, refresh: bool = False
) -> tuple[list[Run], dict]:
  snapshot, provenance = output / "scalars.csv", output / "provenance.json"
  if not sources:
    raise ValueError("At least one source is required")
  if any(source.path.resolve() == snapshot.resolve() for source in sources):
    raise ValueError("Output directory must differ from the input snapshot directory")
  specs = [
    {"path": str(s.path.resolve()), "label": s.label, **s.metadata} for s in sources
  ]
  metadata = json.loads(provenance.read_text()) if provenance.exists() else {}
  reusable = (
    not refresh
    and snapshot.exists()
    and set(CAPTURE_TAGS).issubset(metadata.get("capture_tags", []))
    and metadata.get("source_specs") == specs
  )
  if reusable:
    runs = read_csv(snapshot)
    for run in runs:
      run.label = metadata["run_labels"][run.name]
  else:
    runs = read_sources(sources)
    # Validate before replacing an existing snapshot.
    normalize_runs(runs)
    output.mkdir(parents=True, exist_ok=True)
    write_csv(snapshot, runs)
    files = []
    for source in sources:
      paths = (
        sorted(source.path.glob("events.out.tfevents.*"))
        if source.path.is_dir()
        else [source.path]
      )
      files.extend(
        {"path": str(p.resolve()), "size": p.stat().st_size, "mtime": p.stat().st_mtime}
        for p in paths
      )
    metadata = {
      "snapshot_created_utc": datetime.now(timezone.utc).isoformat(),
      "source_specs": specs,
      "source_files": files,
      "capture_tags": list(CAPTURE_TAGS),
      "run_labels": {r.name: r.label for r in runs},
      "run_metadata": {r.name: r.metadata for r in runs},
      "metric_definitions": {
        "failure_percent": {
          algorithm: {
            "tag": tag,
            "window": FAILURE_WINDOWS[algorithm],
            "formula": "100 * logged fraction",
          }
          for algorithm, tag in FAILURE_TAGS.items()
        },
        "learning_percent": "100 * learning_time / (collection_time + learning_time)",
        "ppo_real_steps": (
          "initial_real_steps + (iteration + 1) * "
          "num_envs_per_rank * num_steps_per_env * world_size"
        ),
        "supermimic_real_steps": REAL_STEPS
        + " (exact iteration join, includes prefill)",
        "train_time": "cumulative sum of Perf/collection_time + Perf/learning_time",
      },
    }
    write_json(provenance, metadata)
  normalized, diagnostics = normalize_runs(runs)
  return normalized, {**metadata, "diagnostics": diagnostics}


def metric_records(run: Run, tag: str) -> list[Sample]:
  """Resolve algorithm-specific fields and derive percentages by exact iteration."""
  if tag == FAILURE:
    records = run.samples.get(FAILURE_TAGS.get(run.algorithm, ""), [])
    outcomes = [{s.step: s.value for s in run.samples.get(t, [])} for t in PPO_OUTCOMES]
    result = []
    for sample in records:
      value = sample.value
      if run.algorithm == "ppo" and all(t.get(sample.step) == 0 for t in outcomes):
        value = math.nan  # No committed outcomes yet; this is not a zero failure rate.
      if not 0 <= value <= 1:
        value = math.nan
      result.append(Sample(sample.step, value * 100, sample.wall_time))
    return result
  if tag == LEARNING_SHARE:
    timings = [{s.step: s for s in run.samples.get(t, [])} for t in TIME_TAGS]
    result = []
    for step in sorted(timings[0].keys() | timings[1].keys()):
      pair = [t.get(step) for t in timings]
      values = [s.value if s else math.nan for s in pair]
      value = math.nan
      if all(math.isfinite(v) and v >= 0 for v in values) and sum(values) > 0:
        value = 100 * values[1] / sum(values)
      result.append(Sample(step, value, max(s.wall_time for s in pair if s)))
    return result
  return run.samples.get(tag, [])


def ppo_steps_per_iteration(run: Run) -> tuple[int, int]:
  rollout = run.metadata.get("ppo_rollout", {})
  keys = ("num_envs_per_rank", "num_steps_per_env", "world_size")
  if any(type(rollout.get(key)) is not int or rollout[key] <= 0 for key in keys):
    raise ValueError(
      f"{run.name}: real-steps requires a native {REAL_STEPS} counter or a "
      "source config with ppo_rollout: num_envs_per_rank, num_steps_per_env, "
      "world_size and initial_real_steps (the count before iteration 0)"
    )
  initial = rollout.get("initial_real_steps")
  if type(initial) is not int or initial < 0:
    raise ValueError(f"{run.name}: ppo_rollout.initial_real_steps must be nonnegative")
  return math.prod(rollout[key] for key in keys), initial


def axis_values(run: Run, axis: str) -> tuple[dict[int, float], list[str]]:
  steps = {s.step for tag in METRIC_TAGS for s in metric_records(run, tag)}
  if not steps:
    raise ValueError(f"{run.name}: no metric records")
  if axis == "iteration":
    return {step: float(step) for step in steps}, []
  if axis == "real-steps":
    samples = run.samples.get(REAL_STEPS, [])
    if not samples and run.algorithm == "ppo":
      per_iteration, initial = ppo_steps_per_iteration(run)
      return {step: (initial + (step + 1) * per_iteration) / 1e6 for step in steps}, []
    values = {s.step: s.value / 1e6 for s in samples if math.isfinite(s.value)}
    if any(v <= 0 for v in values.values()) or any(
      np.diff([values[step] for step in sorted(values)]) < 0
    ):
      raise ValueError(
        f"{run.name}: real steps decrease or are nonpositive on log axis"
      )
    if not steps.intersection(values):
      raise ValueError(f"{run.name}: real-steps axis requires {REAL_STEPS}")
    missing = len(steps.difference(values))
    warnings = [f"{run.name}: real-step coordinates missing at {missing} iterations"]
    return values, warnings if missing else []
  if axis != "train-time":
    raise ValueError(f"Unknown x-axis: {axis}")
  timings = [{s.step: s.value for s in run.samples.get(tag, [])} for tag in TIME_TAGS]
  values, total = {}, 0.0
  for step in range(max(steps) + 1):
    pair = [timing.get(step, math.nan) for timing in timings]
    if any(not math.isfinite(value) or value < 0 for value in pair):
      reason = f"{run.name}: train-time needs finite paired timings at iteration {step}"
      if not steps.intersection(values):
        raise ValueError(reason + "; cannot construct cumulative time from iteration 0")
      return values, [reason + "; time axis stops before this gap"]
    total += sum(pair)
    values[step] = total / 60
  return values, []


def prepare_curves(runs: list[Run], axis: str) -> tuple[dict, list[str]]:
  curves, warnings = {}, []
  for run in runs:
    coordinates, messages = axis_values(run, axis)
    warnings.extend(messages)
    curves[run.name] = {}
    for tag in METRIC_TAGS:
      records = metric_records(run, tag)
      curves[run.name][tag] = Curve(
        np.asarray([s.step for s in records], dtype=np.int64),
        np.asarray([coordinates.get(s.step, math.nan) for s in records]),
        np.asarray([s.value if math.isfinite(s.value) else math.nan for s in records]),
        np.asarray([s.wall_time for s in records]),
      )
  return curves, warnings


def trailing_mean(curve: Curve, window: int) -> np.ndarray:
  if window < 1:
    raise ValueError("Smoothing window must be positive")
  result = np.full(len(curve.y), np.nan)
  queue: deque[float] = deque()
  total = 0.0
  previous = -2
  for i, (step, x, value) in enumerate(zip(curve.steps, curve.x, curve.y, strict=True)):
    if step != previous + 1 or not np.isfinite([x, value]).all():
      queue.clear()
      total = 0.0
    previous = step
    if not np.isfinite([x, value]).all():
      continue
    queue.append(float(value))
    total += value
    if len(queue) > window:
      total -= queue.popleft()
    result[i] = total / len(queue)
  return result


def line_points(curve: Curve, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
  """Insert display-only separators at missing iterations; never invent samples."""
  gaps = np.flatnonzero(np.diff(curve.steps) != 1) + 1
  return np.insert(curve.x, gaps, np.nan), np.insert(y, gaps, np.nan)


def finite_or_none(value: float) -> float | None:
  return float(value) if math.isfinite(value) else None


def summarize(
  runs: list[Run], curves: dict, window: int, tags: tuple[str, ...] = METRIC_TAGS
) -> dict:
  panels = {}
  for tag in tags:
    ranges = []
    for run in runs:
      curve = curves[run.name][tag]
      valid = np.isfinite(curve.x) & np.isfinite(curve.y)
      if valid.any():
        ranges.append((float(curve.x[valid][0]), float(curve.x[valid][-1])))
    common = (max(r[0] for r in ranges), min(r[1] for r in ranges)) if ranges else None
    if len(ranges) < 2 or (common and common[0] > common[1]):
      common = None
    panel: dict[str, Any] = {"common_x_range": common, "runs": {}}
    for run in runs:
      curve = curves[run.name][tag]
      plotted = np.flatnonzero(np.isfinite(curve.x))
      entry: dict[str, Any] = {
        "status": "recorded" if len(curve.y) else "not_recorded",
        "records": len(curve.y),
        "unmapped_records": int((~np.isfinite(curve.x)).sum()),
        "nonfinite_records": int((~np.isfinite(curve.y)).sum()),
        "last_record_iteration": int(curve.steps[-1]) if len(curve.steps) else None,
        "last_event_utc": utc(float(curve.wall[-1])) if len(curve.wall) else None,
        "last_iteration": None,
        "last_x": None,
        "last_value": None,
        "last_window_mean": None,
        "common_last_window_mean": None,
        "common_last_iteration": None,
      }
      if tag == FAILURE:
        entry["source_tag"] = FAILURE_TAGS[run.algorithm]
        entry["statistic_window"] = FAILURE_WINDOWS[run.algorithm]
        entry["cross_algorithm_comparison"] = "Different windows; no ranking or delta"
      else:
        entry["source_tag"] = list(TIME_TAGS) if tag == LEARNING_SHARE else tag
      if len(plotted):
        last = int(plotted[-1])
        entry.update(
          last_iteration=int(curve.steps[last]),
          last_x=float(curve.x[last]),
          last_value=finite_or_none(curve.y[last]),
          last_window_mean=finite_or_none(trailing_mean(curve, window)[last]),
        )
      if common:
        mask = (curve.x >= common[0]) & (curve.x <= common[1])
        subset = Curve(
          curve.steps[mask], curve.x[mask], curve.y[mask], curve.wall[mask]
        )
        if mask.any():
          entry["common_last_window_mean"] = finite_or_none(
            trailing_mean(subset, window)[-1]
          )
          entry["common_last_iteration"] = int(subset.steps[-1])
      panel["runs"][run.name] = entry
    panels[tag] = panel
  return panels


def comparison_label(run: Run) -> str:
  details = []
  for axis in ("train-time", "real-steps"):
    try:
      values, warnings = axis_values(run, axis)
    except ValueError:
      continue
    if warnings:
      continue
    last = values[max(values)]
    if axis == "train-time":
      details.append(f"{last:,.1f} min")
    else:
      steps = f"{last / 1000:,.2f}B" if last >= 1000 else f"{last:,.1f}M"
      details.append(f"{steps} real steps")
  return run.label + ("\n" + " | ".join(details) if details else "")


def render_figure(runs: list[Run], curves: dict, summary: dict, title: str) -> Figure:
  panels = PANEL_GROUPS[summary["report"]]
  comparison = summary["report"] == "comparison"
  axis = summary["x_axis"]
  style = {
    "font.family": "DejaVu Sans",
    "font.size": 15 if comparison else 10,
    "text.color": "#1D3045",
    "axes.labelcolor": "#40536A",
    "axes.edgecolor": "#CAD5DF",
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#E6ECF2",
    "grid.linewidth": 0.7,
    "legend.frameon": False,
    "savefig.facecolor": "white",
  }
  coordinates = [
    c.x[np.isfinite(c.x)]
    for run in runs
    for c in curves[run.name].values()
    if np.isfinite(c.x).any()
  ]
  xmin = min(float(x.min()) for x in coordinates)
  xmax = max(float(x.max()) for x in coordinates)
  with plt.rc_context(style):  # pyright: ignore[reportArgumentType]
    fig, axes = plt.subplots(
      2, 2 if comparison else 3, figsize=(16, 11) if comparison else (19, 12)
    )
    fig.subplots_adjust(
      left=0.09 if comparison else 0.065,
      right=0.975 if comparison else 0.98,
      bottom=0.09 if comparison else 0.25,
      top=0.78,
      hspace=0.48 if comparison else 0.78,
      wspace=0.28 if comparison else 0.30,
    )
    fig.suptitle(
      title,
      x=0.5 if comparison else 0.065,
      y=0.965,
      ha="center" if comparison else "left",
      fontsize=28 if comparison else 24,
      weight="bold",
    )
    if not comparison:
      fig.text(
        0.065,
        0.92,
        f"{AXES[axis]}{' | logarithmic x-axis' if axis == 'real-steps' else ''}"
        f"  |  Snapshot {summary['snapshot_created_utc'][:19]} UTC",
        color="#53667B",
      )
    fig.legend(
      [
        Line2D([0], [0], color=summary["colors"][r.name], lw=3 if comparison else 2.5)
        for r in runs
      ],
      [comparison_label(r) if comparison else r.label for r in runs],
      loc="upper center" if comparison else "upper left",
      bbox_to_anchor=(0.5, 0.91) if comparison else (0.06, 0.895),
      ncol=min(3, len(runs)),
      fontsize=15 if comparison else 11,
    )
    for ax, panel in zip(axes.flat, panels, strict=True):
      ax.set_title(
        panel.title,
        loc="left",
        fontsize=19 if comparison else 12.5,
        pad=16 if comparison else 36,
      )
      if not comparison:
        ax.text(
          0,
          1.035,
          panel.caption,
          transform=ax.transAxes,
          fontsize=7.7,
          color="#66788A",
          linespacing=1.3,
        )
      positive, missing, excluded, endpoints = [], [], [], []
      for run in runs:
        color = summary["colors"][run.name]
        for component, tag in enumerate(panel.tags):
          curve = curves[run.name][tag]
          part = (
            COMPONENT_LABELS.get(
              tag.split("/")[2] if tag.startswith("Loss/time/") else tag.split("/")[1],
              "",
            )
            if len(panel.tags) > 1
            else ""
          )
          label = f"{run.label}: {part}" if part and len(runs) > 1 else part
          if tag == FAILURE:
            label = (
              "PPO: cumulative"
              if run.algorithm == "ppo"
              else "SuperMimic: recent 100 episodes"
            )
          if not len(curve.y):
            missing.append(label or run.label)
            continue
          valid = np.isfinite(curve.x) & np.isfinite(curve.y)
          positive.extend(curve.y[valid].tolist())
          if (~np.isfinite(curve.x)).any():
            excluded.append(
              f"{run.label}: {int((~np.isfinite(curve.x)).sum())} unmapped"
            )
          values = (
            curve.y
            if tag == "Loss/learning_rate"
            else trailing_mean(curve, summary["smooth_window"])
          )
          linestyle = ("-", "--", ":")[component % 3]
          if tag == FAILURE and run.algorithm == "ppo":
            linestyle = "--"
          if tag != "Loss/learning_rate":
            ax.plot(
              *line_points(curve, curve.y),
              color=color,
              lw=0.65,
              alpha=0.16,
              rasterized=True,
              linestyle=linestyle,
            )
          ax.plot(
            *line_points(curve, values),
            color=color,
            lw=2.6 if comparison else 1.8,
            linestyle=linestyle,
            label=label or "_nolegend_",
          )
          if valid.any() and not comparison:
            last = int(np.flatnonzero(valid)[-1])
            ax.plot(curve.x[last], values[last], "o", color=color, ms=3.5)
            endpoints.append((values[last], curve.x[last], color))
      if panel.log_positive and positive:
        if min(positive) > 0:
          ax.set_yscale("log")
        else:
          ax.text(
            0.02,
            0.97,
            "Linear scale: nonpositive values",
            transform=ax.transAxes,
            va="top",
            fontsize=8,
          )
      if panel.tags == ("Loss/learning_rate",):
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.5e}"))
      if panel.tags == ("Perf/total_fps",):
        ax.yaxis.set_major_formatter(EngFormatter())
      if panel.tags in ((FAILURE,), (LEARNING_SHARE,)):
        ax.set_ylim(-3, 103)
      else:
        ax.margins(y=0.16)
      for rank, (value, x, color) in enumerate(sorted(endpoints)):
        label = (
          f"{value:.5e}" if panel.tags == ("Loss/learning_rate",) else f"{value:.3g}"
        )
        offset = 9 + rank * 12 if rank >= len(endpoints) // 2 else -15 - rank * 12
        ax.annotate(
          label,
          (x, value),
          xytext=(-4, offset),
          textcoords="offset points",
          ha="right",
          color=color,
          fontsize=8,
          bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 1},
        )
      if missing or excluded:
        messages = (["Not recorded: " + "; ".join(missing)] if missing else []) + list(
          dict.fromkeys(excluded)
        )
        ax.text(
          0.02,
          0.03,
          "\n".join(messages),
          transform=ax.transAxes,
          fontsize=12 if comparison else 8,
          color="#66788A",
        )
      if (
        len(panel.tags) > 1 or panel.tags == (FAILURE,)
      ) and ax.get_legend_handles_labels()[0]:
        ax.legend(
          loc="lower right" if comparison else "best",
          bbox_to_anchor=(1, 0.14) if comparison else None,
          fontsize=12 if comparison else 8,
        )
      ax.set_xlabel(AXES[axis])
      ax.set_ylabel(panel.ylabel)
      ax.set_axisbelow(True)
      if axis == "real-steps":
        ax.set_xscale("log")
        ax.set_xlim(xmin / 1.2, xmax * 1.2)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
      else:
        ax.xaxis.set_major_locator(MaxNLocator(5, integer=axis == "iteration"))
        ax.set_xlim(0, max(xmax * 1.04, 1e-6))
        if axis == "iteration":
          ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
    if comparison:
      return fig
    footer = [
      f"Faint: logged values | Bold: trailing {summary['smooth_window']}-record mean"
      " | Actor LR unsmoothed | No smoothing across gaps",
      "Reward/length already average recent completed episodes. "
      "Endpoint labels show displayed values.",
    ]
    for run in runs:
      records = [s for tag in METRIC_TAGS for s in metric_records(run, tag)]
      footer.append(
        f"{run.label}: last iteration {max(s.step for s in records):,}; "
        f"last event {utc(max(s.wall_time for s in records))[:19]} UTC"
      )
    footer.extend(summary["notes"])
    if summary["warnings"]:
      footer.append("Incomplete axis data: see panel annotations and summary JSON.")
    fig.text(
      0.065,
      0.025,
      "\n".join(footer),
      fontsize=8.5,
      color="#53667B",
      linespacing=1.5,
      va="bottom",
    )
  return fig


def generate_reports(
  sources: list[Source],
  output: Path,
  *,
  axes: tuple[str, ...] = ("train-time", "real-steps"),
  window: int = 100,
  refresh: bool = False,
  title: str = "Training curves",
  notes: list[str] | None = None,
) -> list[dict]:
  if window < 1:
    raise ValueError("--smooth-window must be positive")
  runs, provenance = load_snapshot(sources, output, refresh)
  colors = {run.name: COLORS[i % len(COLORS)] for i, run in enumerate(runs)}
  # Validate all requested axes before writing figures; never substitute an axis.
  prepared = {axis: prepare_curves(runs, axis) for axis in dict.fromkeys(axes)}
  reports = []
  for axis, (curves, warnings) in prepared.items():
    for report, panels in PANEL_GROUPS.items():
      selected = (
        runs
        if report == "comparison"
        else [r for r in runs if r.algorithm == "supermimic"]
      )
      if not selected:
        continue
      tags = tuple(tag for panel in panels for tag in panel.tags)
      summary: dict[str, Any] = {
        "snapshot_created_utc": provenance["snapshot_created_utc"],
        "provenance_file": "provenance.json",
        "report": report,
        "x_axis": axis,
        "x_label": AXES[axis],
        "x_scale": "log" if axis == "real-steps" else "linear",
        "smooth_window": window,
        "smoothing": "Trailing record mean, reset at gaps/nonfinite values; LR unsmoothed.",
        "colors": colors,
        "run_labels": {r.name: r.label for r in selected},
        "run_metadata": {r.name: r.metadata for r in selected},
        "panels": [
          {"title": p.title, "unit": p.ylabel, "tags": p.tags} for p in panels
        ],
        "notes": list(notes or []),
        "warnings": warnings,
        "diagnostics": {r.name: provenance["diagnostics"][r.name] for r in selected},
        "metrics": summarize(selected, curves, window, tags),
      }
      if axis == "train-time":
        summary["notes"].append(
          "Time = collection + learning, including logged compilation; "
          "excludes setup, initial collection, saving and downtime."
        )
      figure_title = (
        title if report == "comparison" else f"{title} | SuperMimic diagnostics"
      )
      figure = render_figure(selected, curves, summary, figure_title)
      stem = f"{report}_{axis}"
      figure.savefig(output / f"{stem}.png", dpi=180)
      plt.close(figure)
      write_json(output / f"{stem}.json", summary)
      reports.append(summary)
  return reports


def parse_options(argv: list[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  inputs = parser.add_mutually_exclusive_group()
  inputs.add_argument(
    "--config", type=Path, help="JSON config (paths relative to config)"
  )
  inputs.add_argument(
    "--source",
    type=Path,
    action="append",
    help="TensorBoard directory or CSV; repeat for multiple sources",
  )
  parser.add_argument(
    "--x-axis",
    choices=AXES,
    action="append",
    help="Repeat to select axes; default: train-time and real-steps",
  )
  parser.add_argument("--smooth-window", type=int, default=100)
  parser.add_argument("--output-dir", type=Path)
  parser.add_argument(
    "--refresh", action="store_true", help="Replace saved scalar snapshot"
  )
  return parser.parse_args(argv)


def resolve_inputs(
  args: argparse.Namespace,
) -> tuple[list[Source], Path, str, list[str]]:
  config: dict[str, Any] = {}
  if args.source:
    sources = [Source(p.expanduser().resolve()) for p in args.source]
    default_output = Path("logs/train_plots")
  else:
    path = (args.config or DEFAULT_CONFIG).expanduser().resolve()
    config = json.loads(path.read_text())
    sources = [
      Source(
        (path.parent / Path(s["path"]).expanduser()).resolve(),
        s.get("label"),
        {k: v for k, v in s.items() if k not in ("path", "label")},
      )
      for s in config["sources"]
    ]
    default_output = path.parent / config.get("output_dir", "../logs/train_plots")
  if not sources:
    raise ValueError("At least one source is required")
  output = (args.output_dir or default_output).expanduser().resolve()
  if any(s.path == output / "scalars.csv" for s in sources):
    raise ValueError("Output directory must differ from the input snapshot directory")
  return (
    sources,
    output,
    config.get("title", "Training curves"),
    config.get("notes", []),
  )


def main(argv: list[str] | None = None, *, assets=None) -> int:
  from supermimic.assets import LocalAssets

  assets = assets or LocalAssets()
  args = parse_options(argv)
  if not args.source and not args.config:
    if assets.plot_config:
      args.config = assets.plot_config
    elif assets.plot_sources:
      args.source = list(assets.plot_sources)
    else:
      raise ValueError("Set --source, --config, or local plot_sources first.")
  try:
    sources, output, title, notes = resolve_inputs(args)
    output = assets.protect_output(output, *(source.path for source in sources))
    reports = generate_reports(
      sources,
      output,
      axes=tuple(args.x_axis or ("train-time", "real-steps")),
      window=args.smooth_window,
      refresh=args.refresh,
      title=title,
      notes=notes,
    )
  except (ValueError, OSError, KeyError) as error:
    print(f"Error: {error}", file=sys.stderr)
    return 1
  for warning in dict.fromkeys(w for report in reports for w in report["warnings"]):
    print(f"Warning: {warning}", file=sys.stderr)
  print(
    json.dumps(
      {
        "output_dir": str(output),
        "snapshot_created_utc": reports[0]["snapshot_created_utc"],
        "reports": [f"{r['report']}_{r['x_axis']}" for r in reports],
      },
      indent=2,
    )
  )
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
