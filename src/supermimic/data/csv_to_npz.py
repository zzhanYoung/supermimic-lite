"""Convert LAFAN1 Unitree G1 CSV motions to local tracking NPZ files on CPU."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import mjlab
import mujoco
import numpy as np
import torch
import tyro
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import G1_XML
from mjlab.utils.lab_api.math import quat_apply_inverse

from supermimic.data.motion_dataset import write_motion_manifest
from supermimic.data.qpos_to_npz import (
  _angular_velocity,
  _resample_qpos,
  _validate_finite,
)

# Joint columns after root xyz + xyzw in the LAFAN1 G1 retargeting CSVs.
G1_CSV_JOINT_NAMES = (
  "left_hip_pitch_joint",
  "left_hip_roll_joint",
  "left_hip_yaw_joint",
  "left_knee_joint",
  "left_ankle_pitch_joint",
  "left_ankle_roll_joint",
  "right_hip_pitch_joint",
  "right_hip_roll_joint",
  "right_hip_yaw_joint",
  "right_knee_joint",
  "right_ankle_pitch_joint",
  "right_ankle_roll_joint",
  "waist_yaw_joint",
  "waist_roll_joint",
  "waist_pitch_joint",
  "left_shoulder_pitch_joint",
  "left_shoulder_roll_joint",
  "left_shoulder_yaw_joint",
  "left_elbow_joint",
  "left_wrist_roll_joint",
  "left_wrist_pitch_joint",
  "left_wrist_yaw_joint",
  "right_shoulder_pitch_joint",
  "right_shoulder_roll_joint",
  "right_shoulder_yaw_joint",
  "right_elbow_joint",
  "right_wrist_roll_joint",
  "right_wrist_pitch_joint",
  "right_wrist_yaw_joint",
)


@dataclass(frozen=True)
class CsvConversionConfig:
  """Convert one CSV or a directory of CSVs without a W&B account or GPU."""

  input_file: Path | None = None
  """Single headerless G1 CSV: root xyz, root xyzw, then 29 joint angles."""
  output_file: Path | None = None
  """Destination .npz file; required with --input-file."""
  input_dir: Path | None = None
  """Directory containing G1 *.csv files (non-recursive)."""
  output_dir: Path | None = None
  """Destination directory; required with --input-dir. Includes manifest.json."""
  input_fps: float = 30.0
  """Frame rate of the source CSVs."""
  output_fps: float = 50.0
  """Frame rate of the generated tracking motions."""
  line_range: tuple[int, int] | None = None
  """Optional 1-based, inclusive row range for a single input file."""
  overwrite: bool = False
  """Allow replacement of existing outputs with --overwrite True."""


def _conversion_paths(cfg: CsvConversionConfig) -> tuple[tuple[Path, Path], ...]:
  if any(not math.isfinite(fps) or fps <= 0 for fps in (cfg.input_fps, cfg.output_fps)):
    raise ValueError("input_fps and output_fps must be finite and positive.")
  if (cfg.input_file is None) == (cfg.input_dir is None):
    raise ValueError("Provide exactly one of --input-file or --input-dir.")
  if cfg.input_file is not None:
    if cfg.output_file is None or cfg.output_dir is not None:
      raise ValueError("Use --output-file with --input-file.")
    if cfg.line_range is not None:
      first, last = cfg.line_range
      if first < 1 or last <= first:
        raise ValueError("line_range must select at least two rows with start >= 1.")
    paths = ((cfg.input_file.expanduser(), cfg.output_file.expanduser()),)
  else:
    if cfg.output_dir is None or cfg.output_file is not None:
      raise ValueError("Use --output-dir with --input-dir.")
    if cfg.line_range is not None:
      raise ValueError("--line-range is only supported with --input-file.")
    assert cfg.input_dir is not None
    folder = cfg.input_dir.expanduser()
    if not folder.is_dir():
      raise NotADirectoryError(f"Input folder not found: {folder}")
    paths = tuple(
      (path, cfg.output_dir.expanduser() / f"{path.stem}.npz")
      for path in sorted(folder.glob("*.csv"))
      if path.is_file()
    )
    if not paths:
      raise ValueError(f"Input folder contains no .csv files: {folder}")
  # Check all destinations before writing any part of a batch.
  for source, output in paths:
    if not source.is_file():
      raise FileNotFoundError(f"Input CSV not found: {source}")
    if output.suffix.lower() != ".npz":
      raise ValueError(f"Output must have a .npz extension: {output}")
    if source.resolve() == output.resolve():
      raise ValueError(f"Output must differ from the input CSV: {output}")
    if output.exists() and (not cfg.overwrite or not output.is_file()):
      raise FileExistsError(
        f"Output already exists: {output}. Use --overwrite True to replace files."
      )
  return paths


def _load_csv(path: Path, cfg: CsvConversionConfig) -> tuple[np.ndarray, np.ndarray]:
  first, last = cfg.line_range if cfg.line_range is not None else (1, None)
  try:
    motion = np.loadtxt(
      path,
      delimiter=",",
      dtype=np.float32,
      ndmin=2,
      skiprows=first - 1,
      max_rows=None if last is None else last - first + 1,
    )
    if motion.shape[1] != 36:
      raise ValueError(
        f"expected 36 columns (root xyz + xyzw + 29 G1 joints), got {motion.shape}"
      )
    if last is not None and len(motion) != last - first + 1:
      raise ValueError("line_range extends beyond the available CSV rows")
    _validate_finite(motion, label="CSV data")
    motion[:, 3:7] = motion[:, [6, 3, 4, 5]]  # xyzw -> wxyz.
    qpos = _resample_qpos(torch.from_numpy(motion), cfg.input_fps, cfg.output_fps)
    dt = 1.0 / cfg.output_fps
    linear_velocity = torch.gradient(qpos[:, :3], spacing=dt, dim=0)[0]
    joint_velocity = torch.gradient(qpos[:, 7:], spacing=dt, dim=0)[0]
    angular_velocity = _angular_velocity(qpos[:, 3:7], dt)
    # MuJoCo free-joint angular velocity is expressed in the root body frame.
    angular_velocity_b = quat_apply_inverse(qpos[:, 3:7], angular_velocity)
    qvel = torch.cat((linear_velocity, angular_velocity_b, joint_velocity), dim=-1)
  except (OSError, ValueError) as exc:
    raise ValueError(f"Failed to load CSV motion {path}: {exc}") from exc
  return qpos.numpy(), qvel.numpy()


def _forward_kinematics(
  model: mujoco.MjModel, qpos: np.ndarray, qvel: np.ndarray, fps: float
) -> dict[str, np.ndarray]:
  data = mujoco.MjData(model)
  frames, bodies = len(qpos), model.nbody - 1
  outputs = {
    "fps": np.asarray([fps], dtype=np.float32),
    "joint_pos": qpos[:, 7:].copy(),
    "joint_vel": qvel[:, 6:].copy(),
    "body_pos_w": np.empty((frames, bodies, 3), dtype=np.float32),
    "body_quat_w": np.empty((frames, bodies, 4), dtype=np.float32),
    "body_lin_vel_w": np.empty((frames, bodies, 3), dtype=np.float32),
    "body_ang_vel_w": np.empty((frames, bodies, 3), dtype=np.float32),
  }
  for frame in range(frames):
    data.qpos[:] = qpos[frame]
    data.qvel[:] = qvel[frame]
    mujoco.mj_kinematics(model, data)
    mujoco.mj_comPos(model, data)
    mujoco.mj_comVel(model, data)
    # Exclude the world body, matching Entity.body_names and tracking loaders.
    outputs["body_pos_w"][frame] = data.xpos[1:]
    outputs["body_quat_w"][frame] = data.xquat[1:]
    outputs["body_ang_vel_w"][frame] = data.cvel[1:, :3]
    # Shift cvel from the root subtree COM to each link origin.
    outputs["body_lin_vel_w"][frame] = data.cvel[1:, 3:] + np.cross(
      data.cvel[1:, :3], data.xpos[1:] - data.subtree_com[1]
    )
  return outputs


def convert_csv(cfg: CsvConversionConfig) -> tuple[Path, ...]:
  """Resample G1 CSV motions, compute link kinematics, and save local NPZs."""
  paths = _conversion_paths(cfg)
  model = mujoco.MjModel.from_xml_path(str(G1_XML))
  joint_names = tuple(model.joint(i).name for i in range(1, model.njnt))
  if model.nq != 36 or model.nv != 35 or joint_names != G1_CSV_JOINT_NAMES:
    raise ValueError("The G1 model joint layout does not match the CSV joint order.")
  model.opt.timestep = 1.0 / cfg.output_fps
  for index, (source, output) in enumerate(paths, 1):
    qpos, qvel = _load_csv(source, cfg)
    outputs = _forward_kinematics(model, qpos, qvel, cfg.output_fps)
    for key, array in outputs.items():
      _validate_finite(array, label=f"Converted motion {source} array {key!r}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp.npz")
    try:
      cast(Any, np.savez)(temporary, **outputs)
      temporary.replace(output)
    finally:
      temporary.unlink(missing_ok=True)
    print(
      f"[{index}/{len(paths)}] {source.name} -> {output} "
      f"({len(qpos)} frames at {cfg.output_fps:g} Hz)",
      flush=True,
    )
  if cfg.output_dir is not None:
    manifest = write_motion_manifest(cfg.output_dir)
    print(f"Wrote motion manifest: {manifest}")
  return tuple(output for _, output in paths)


def main(*, assets=None) -> None:
  from supermimic.assets import DEFAULT_ASSETS

  assets = assets or DEFAULT_ASSETS
  cfg = tyro.cli(CsvConversionConfig, config=mjlab.TYRO_FLAGS)
  for name in ("output_file", "output_dir"):
    value = getattr(cfg, name)
    if value is not None:
      setattr(cfg, name, assets.protect_output(value, cfg.input_file, cfg.input_dir))
  # Small per-motion resampling operations do not benefit from many CPU threads.
  torch.set_num_threads(min(4, torch.get_num_threads()))
  convert_csv(cfg)


if __name__ == "__main__":
  main()
