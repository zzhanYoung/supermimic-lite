"""Convert a folder of Unitree G1 qpos trajectories to tracking NPZ files."""

from __future__ import annotations

import logging
import math
import os
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Literal, cast

import mjlab
import numpy as np
import torch
import tyro
from mjlab.asset_zoo.robots import get_g1_robot_cfg
from mjlab.entity import Entity
from mjlab.scene import Scene, SceneCfg
from mjlab.sim import Simulation, SimulationCfg
from mjlab.utils.gpu import select_gpus
from mjlab.utils.lab_api.math import (
  axis_angle_from_quat,
  quat_conjugate,
  quat_mul,
)

from supermimic.data.motion_dataset import write_motion_manifest

_OUTPUT_ATTRIBUTES = {
  "joint_pos": "joint_pos",
  "joint_vel": "joint_vel",
  "body_pos_w": "body_link_pos_w",
  "body_quat_w": "body_link_quat_w",
  "body_lin_vel_w": "body_link_lin_vel_w",
  "body_ang_vel_w": "body_link_ang_vel_w",
}


@dataclass(frozen=True)
class QposConversionConfig:
  input_dir: str
  output_dir: str
  input_fps: float = 50.0
  output_fps: float = 50.0
  gpu_ids: list[int] | Literal["all"] | None = field(default_factory=lambda: [0])
  batch_size: int = 512
  overwrite: bool = False


def _validate_finite(array: np.ndarray, *, label: str) -> None:
  """Raise with the first offending index if an array contains NaN or Inf."""
  try:
    finite = np.isfinite(array)
  except TypeError as exc:
    raise ValueError(f"{label} has non-numeric dtype {array.dtype}.") from exc
  if finite.all():
    return

  first_index = tuple(int(i) for i in np.argwhere(~finite)[0])
  value = array[first_index]
  raise ValueError(
    f"{label} contains non-finite value at index {first_index}: {value!r}."
  )


def _continuous_quaternions(quaternions: torch.Tensor) -> torch.Tensor:
  normalized = torch.nn.functional.normalize(quaternions, dim=-1)
  if normalized.shape[0] < 2:
    return normalized
  flips = torch.where(
    (normalized[1:] * normalized[:-1]).sum(dim=-1) < 0,
    -torch.ones(normalized.shape[0] - 1, device=normalized.device),
    torch.ones(normalized.shape[0] - 1, device=normalized.device),
  )
  signs = torch.cat(
    (torch.ones(1, device=normalized.device), torch.cumprod(flips, dim=0))
  )
  return normalized * signs.unsqueeze(-1)


def _angular_velocity(quaternions: torch.Tensor, dt: float) -> torch.Tensor:
  """Differentiate wxyz orientations into world-frame angular velocity."""
  quaternions = _continuous_quaternions(quaternions)
  if quaternions.shape[0] < 2:
    raise ValueError("A motion must contain at least two frames.")
  first_relative = quat_mul(quaternions[1], quat_conjugate(quaternions[0]))
  first = axis_angle_from_quat(first_relative) / dt
  if quaternions.shape[0] == 2:
    return torch.stack((first, first))
  middle_relative = quat_mul(quaternions[2:], quat_conjugate(quaternions[:-2]))
  middle = axis_angle_from_quat(middle_relative) / (2.0 * dt)
  last_relative = quat_mul(quaternions[-1], quat_conjugate(quaternions[-2]))
  last = axis_angle_from_quat(last_relative) / dt
  return torch.cat((first.unsqueeze(0), middle, last.unsqueeze(0)), dim=0)


def _resampled_frame_count(
  input_frames: int, input_fps: float, output_fps: float
) -> int:
  """Return the number of samples on an output-rate grid within the input clip."""
  if input_frames < 2:
    raise ValueError("A motion must contain at least two frames.")
  if (
    not math.isfinite(input_fps)
    or not math.isfinite(output_fps)
    or input_fps <= 0
    or output_fps <= 0
  ):
    raise ValueError("input_fps and output_fps must be finite and positive.")

  output_intervals = (input_frames - 1) * output_fps / input_fps
  nearest_integer = round(output_intervals)
  tolerance = max(1e-12, 8.0 * math.ulp(output_intervals))
  if abs(output_intervals - nearest_integer) <= tolerance:
    output_intervals = float(nearest_integer)
  return math.floor(output_intervals) + 1


def _slerp_quaternions(
  first: torch.Tensor, second: torch.Tensor, blend: torch.Tensor
) -> torch.Tensor:
  """Vectorized shortest-path SLERP for wxyz quaternions."""
  first = torch.nn.functional.normalize(first, dim=-1)
  second = torch.nn.functional.normalize(second, dim=-1)
  dot = (first * second).sum(dim=-1, keepdim=True)
  second = torch.where(dot < 0.0, -second, second)
  dot = dot.abs().clamp(max=1.0)

  blend = blend.unsqueeze(-1)
  linear = torch.nn.functional.normalize(first + blend * (second - first), dim=-1)
  angle = torch.acos(dot)
  sine = torch.sin(angle)
  safe_sine = sine.clamp_min(torch.finfo(first.dtype).eps)
  spherical = (
    torch.sin((1.0 - blend) * angle) / safe_sine * first
    + torch.sin(blend * angle) / safe_sine * second
  )
  interpolated = torch.where(dot > 0.9995, linear, spherical)
  return torch.nn.functional.normalize(interpolated, dim=-1)


def _resample_qpos(
  qpos: torch.Tensor, input_fps: float, output_fps: float
) -> torch.Tensor:
  """Resample G1 qpos onto a fixed output-rate grid without extrapolation."""
  output_frames = _resampled_frame_count(qpos.shape[0], input_fps, output_fps)
  if output_frames < 2:
    duration = (qpos.shape[0] - 1) / input_fps
    raise ValueError(
      f"Motion duration {duration:.6g}s is shorter than one output timestep "
      f"({1.0 / output_fps:.6g}s)."
    )

  quaternion_norms = torch.linalg.vector_norm(qpos[:, 3:7], dim=-1)
  invalid_quaternion = quaternion_norms <= torch.finfo(qpos.dtype).eps
  if invalid_quaternion.any():
    frame = int(torch.nonzero(invalid_quaternion, as_tuple=False)[0, 0])
    raise ValueError(f"Root quaternion has zero norm at frame {frame}.")
  root_orientation = _continuous_quaternions(qpos[:, 3:7])

  if input_fps == output_fps:
    result = qpos.clone()
    result[:, 3:7] = root_orientation
    return result

  source_frames = torch.arange(
    output_frames, dtype=torch.float64, device=qpos.device
  ) * (input_fps / output_fps)
  index_0 = torch.floor(source_frames).to(torch.long)
  index_1 = torch.clamp(index_0 + 1, max=qpos.shape[0] - 1)
  blend = (source_frames - index_0).to(qpos.dtype)
  linear_blend = blend.unsqueeze(-1)

  root_position = torch.lerp(qpos[index_0, :3], qpos[index_1, :3], linear_blend)
  joint_position = torch.lerp(qpos[index_0, 7:], qpos[index_1, 7:], linear_blend)
  root_orientation = _slerp_quaternions(
    root_orientation[index_0], root_orientation[index_1], blend
  )
  return torch.cat((root_position, root_orientation, joint_position), dim=-1)


def _load_qpos(
  path: Path,
  input_fps: float,
  device: str,
  *,
  output_fps: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
  try:
    with np.load(path, allow_pickle=False) as data:
      if "qpos" not in data:
        raise ValueError("missing required qpos array")
      qpos_array = np.asarray(data["qpos"])
  except (OSError, ValueError) as exc:
    raise ValueError(f"Failed to load qpos motion {path}: {exc}") from exc
  if qpos_array.ndim != 2 or qpos_array.shape[1] != 36:
    raise ValueError(
      f"Qpos motion {path} has shape {qpos_array.shape}; expected [time, 36] "
      "(root xyz + root wxyz + 29 G1 joints)."
    )
  if qpos_array.shape[0] < 2:
    raise ValueError(f"Qpos motion {path} must contain at least two frames.")
  _validate_finite(qpos_array, label=f"Qpos motion {path} qpos")

  qpos = torch.as_tensor(qpos_array, dtype=torch.float32, device=device)
  output_fps = input_fps if output_fps is None else output_fps
  try:
    qpos = _resample_qpos(qpos, input_fps, output_fps)
  except ValueError as exc:
    raise ValueError(f"Failed to resample qpos motion {path}: {exc}") from exc
  root_position = qpos[:, :3]
  root_orientation = qpos[:, 3:7]
  joint_position = qpos[:, 7:]
  dt = 1.0 / output_fps
  root_linear_velocity = torch.gradient(root_position, spacing=dt, dim=0)[0]
  root_angular_velocity = _angular_velocity(root_orientation, dt)
  joint_velocity = torch.gradient(joint_position, spacing=dt, dim=0)[0]
  root_state = torch.cat(
    (
      root_position,
      root_orientation,
      root_linear_velocity,
      root_angular_velocity,
    ),
    dim=-1,
  )
  return root_state, joint_position, joint_velocity, qpos


def _create_scene(batch_size: int, output_fps: float, device: str):
  scene_cfg = SceneCfg(
    num_envs=batch_size,
    entities={"robot": get_g1_robot_cfg()},
  )
  scene = Scene(scene_cfg, device=device)
  sim_cfg = SimulationCfg()
  sim_cfg.mujoco.timestep = 1.0 / output_fps
  sim = Simulation(
    num_envs=batch_size,
    cfg=sim_cfg,
    spec=scene.spec,
    variant_info=scene.collect_variant_info(),
    device=device,
  )
  scene.initialize(sim.mj_model, sim.model, sim.data)
  return sim, scene


def _convert_file(
  input_path: Path,
  output_path: Path,
  input_fps: float,
  output_fps: float,
  batch_size: int,
  sim: Simulation,
  scene: Scene,
) -> None:
  root_state, joint_position, joint_velocity, _ = _load_qpos(
    input_path, input_fps, sim.device, output_fps=output_fps
  )
  robot: Entity = scene["robot"]
  if joint_position.shape[1] != robot.data.joint_pos.shape[1]:
    raise ValueError(
      f"Qpos motion {input_path} has {joint_position.shape[1]} joints; "
      f"the G1 model has {robot.data.joint_pos.shape[1]}."
    )

  frame_count = root_state.shape[0]
  body_count = robot.data.body_link_pos_w.shape[1]
  outputs = {
    "joint_pos": np.full(
      (frame_count, joint_position.shape[1]), np.nan, dtype=np.float32
    ),
    "joint_vel": np.full(
      (frame_count, joint_position.shape[1]), np.nan, dtype=np.float32
    ),
    "body_pos_w": np.full((frame_count, body_count, 3), np.nan, dtype=np.float32),
    "body_quat_w": np.full((frame_count, body_count, 4), np.nan, dtype=np.float32),
    "body_lin_vel_w": np.full((frame_count, body_count, 3), np.nan, dtype=np.float32),
    "body_ang_vel_w": np.full((frame_count, body_count, 3), np.nan, dtype=np.float32),
  }
  scene.reset()
  for start in range(0, frame_count, batch_size):
    stop = min(start + batch_size, frame_count)
    env_ids = torch.arange(stop - start, device=sim.device)
    robot.write_root_state_to_sim(root_state[start:stop], env_ids=env_ids)
    robot.write_joint_state_to_sim(
      joint_position[start:stop], joint_velocity[start:stop], env_ids=env_ids
    )
    sim.forward()
    scene.update(1.0 / output_fps)
    for output_key, data_attribute in _OUTPUT_ATTRIBUTES.items():
      tensor = getattr(robot.data, data_attribute)[env_ids]
      outputs[output_key][start:stop] = tensor.detach().cpu().numpy()

  for output_key, output in outputs.items():
    _validate_finite(
      output,
      label=f"Converted motion {input_path} array {output_key!r}",
    )

  output_path.parent.mkdir(parents=True, exist_ok=True)
  temporary = output_path.with_suffix(".tmp.npz")
  cast(Any, np.savez)(temporary, fps=np.asarray([output_fps], np.float32), **outputs)
  temporary.replace(output_path)


def _conversion_worker(cfg: QposConversionConfig) -> None:
  local_rank = int(os.environ.get("LOCAL_RANK", "0"))
  rank = int(os.environ.get("RANK", "0"))
  world_size = int(os.environ.get("WORLD_SIZE", "1"))
  cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
  device = f"cuda:{local_rank}" if cuda_visible else "cpu"
  if device.startswith("cuda"):
    torch.cuda.set_device(local_rank)

  input_folder = Path(cfg.input_dir).expanduser().resolve()
  output_folder = Path(cfg.output_dir).expanduser().resolve()
  files = tuple(
    path
    for path in sorted(input_folder.glob("*.npz"), key=lambda item: item.name)
    if path.is_file()
  )
  worker_files = files[rank::world_size]
  if not worker_files:
    return
  sim, scene = _create_scene(cfg.batch_size, cfg.output_fps, device)
  for index, input_path in enumerate(worker_files, start=1):
    output_path = output_folder / input_path.name
    if output_path.exists() and not cfg.overwrite:
      raise FileExistsError(
        f"Output motion already exists: {output_path}. Pass --overwrite to replace it."
      )
    print(
      f"[INFO] Rank {rank}: converting {input_path.name} ({index}/{len(worker_files)})"
    )
    _convert_file(
      input_path,
      output_path,
      cfg.input_fps,
      cfg.output_fps,
      cfg.batch_size,
      sim,
      scene,
    )


def _torchrunx_conversion_worker(cfg_values: dict[str, Any]) -> None:
  """Resolve the conversion worker by module name inside each torchrunx process.

  ``python -m`` defines this launch shim in ``__main__``. Keeping its imports local
  prevents cloudpickle from traversing conversion globals that include TorchScript
  quaternion functions, which cannot be pickled by value.
  """
  from importlib import import_module

  conversion_module = import_module("supermimic.data.qpos_to_npz")
  config_type = conversion_module.QposConversionConfig
  worker = conversion_module._conversion_worker
  worker(config_type(**cfg_values))


def launch_conversion(cfg: QposConversionConfig) -> None:
  input_folder = Path(cfg.input_dir).expanduser().resolve()
  if not input_folder.is_dir():
    raise NotADirectoryError(f"Input folder not found: {input_folder}")
  if not any(input_folder.glob("*.npz")):
    raise ValueError(f"Input folder contains no .npz files: {input_folder}")
  if (
    not math.isfinite(cfg.input_fps)
    or not math.isfinite(cfg.output_fps)
    or cfg.input_fps <= 0
    or cfg.output_fps <= 0
  ):
    raise ValueError("input_fps and output_fps must be finite and positive.")
  if cfg.batch_size <= 0:
    raise ValueError("batch_size must be positive.")

  output_folder = Path(cfg.output_dir).expanduser().resolve()
  output_folder.mkdir(parents=True, exist_ok=True)
  selected_gpus, num_gpus = select_gpus(cfg.gpu_ids)
  os.environ["CUDA_VISIBLE_DEVICES"] = (
    "" if selected_gpus is None else ",".join(map(str, selected_gpus))
  )
  if num_gpus <= 1:
    _conversion_worker(cfg)
  else:
    import torchrunx

    logging.basicConfig(level=logging.INFO)
    torchrunx.Launcher(
      hostnames=["localhost"],
      workers_per_host=num_gpus,
      backend=None,
      agent_timeout=300,
      copy_env_vars=torchrunx.DEFAULT_ENV_VARS_FOR_COPY + ("MUJOCO*",),
    ).run(_torchrunx_conversion_worker, asdict(cfg))
  manifest_path = write_motion_manifest(output_folder)
  print(f"[INFO] Wrote motion manifest: {manifest_path}")


def main(*, assets=None) -> None:
  from supermimic.assets import DEFAULT_ASSETS

  assets = assets or DEFAULT_ASSETS
  cfg = tyro.cli(QposConversionConfig, config=mjlab.TYRO_FLAGS)
  cfg = replace(
    cfg,
    output_dir=str(assets.protect_output(Path(cfg.output_dir), Path(cfg.input_dir))),
  )
  launch_conversion(cfg)


if __name__ == "__main__":
  main()
