"""Tests for qpos resampling and conversion math."""

import math
from dataclasses import asdict
from functools import partial
from pathlib import Path

import cloudpickle
import numpy as np
import pytest
import torch
from mjlab.entity.data import EntityData

import supermimic.data.qpos_to_npz as qpos_to_npz
from supermimic.data.qpos_to_npz import (
  QposConversionConfig,
  _angular_velocity,
  _continuous_quaternions,
  _load_qpos,
  _resample_qpos,
  _resampled_frame_count,
)


def test_outputs_map_to_entity_data() -> None:
  expected_attributes = {
    "joint_pos": "joint_pos",
    "joint_vel": "joint_vel",
    "body_pos_w": "body_link_pos_w",
    "body_quat_w": "body_link_quat_w",
    "body_lin_vel_w": "body_link_lin_vel_w",
    "body_ang_vel_w": "body_link_ang_vel_w",
  }

  assert qpos_to_npz._OUTPUT_ATTRIBUTES == expected_attributes
  for attribute in expected_attributes.values():
    assert isinstance(getattr(EntityData, attribute), property)


def test_torchrunx_worker_payload_is_pickleable_when_module_runs_by_value(
  tmp_path: Path,
) -> None:
  cfg = QposConversionConfig(
    input_dir=str(tmp_path / "input"),
    output_dir=str(tmp_path / "output"),
    gpu_ids="all",
  )
  payload = partial(qpos_to_npz._torchrunx_conversion_worker, asdict(cfg))

  cloudpickle.register_pickle_by_value(qpos_to_npz)
  try:
    serialized = cloudpickle.dumps(payload)
  finally:
    cloudpickle.unregister_pickle_by_value(qpos_to_npz)

  assert serialized


def test_quaternion_sign_flips_are_made_continuous() -> None:
  quaternions = torch.tensor(
    [[1.0, 0.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]
  )

  continuous = _continuous_quaternions(quaternions)

  assert torch.all((continuous[1:] * continuous[:-1]).sum(dim=-1) > 0)


def test_angular_velocity_uses_50_hz_timestep() -> None:
  dt = 0.02
  angular_speed = 2.0
  angles = torch.arange(5) * dt * angular_speed
  quaternions = torch.stack(
    (
      torch.cos(angles / 2),
      torch.zeros_like(angles),
      torch.zeros_like(angles),
      torch.sin(angles / 2),
    ),
    dim=-1,
  )

  velocity = _angular_velocity(quaternions, dt)

  torch.testing.assert_close(
    velocity,
    torch.tensor([0.0, 0.0, angular_speed]).expand_as(velocity),
    atol=1e-5,
    rtol=1e-5,
  )
  assert math.isclose(float(velocity[:, 2].mean()), angular_speed, rel_tol=1e-5)


def test_resampled_frame_count_uses_output_grid_without_extrapolation() -> None:
  assert _resampled_frame_count(4, 30.0, 50.0) == 6
  assert _resampled_frame_count(5, 30.0, 50.0) == 7
  assert _resampled_frame_count(31, 30.0, 50.0) == 51
  assert _resampled_frame_count(5, 50.0, 50.0) == 5


def test_resample_qpos_linearly_interpolates_and_keeps_aligned_endpoint() -> None:
  input_times = torch.arange(4, dtype=torch.float32) / 30.0
  qpos = torch.zeros((4, 36), dtype=torch.float32)
  qpos[:, 0] = 2.0 * input_times
  qpos[:, 3] = 1.0
  qpos[:, 7:] = 3.0 * input_times[:, None]

  resampled = _resample_qpos(qpos, 30.0, 50.0)

  output_times = torch.arange(6, dtype=torch.float32) / 50.0
  assert resampled.shape == (6, 36)
  torch.testing.assert_close(resampled[:, 0], 2.0 * output_times)
  torch.testing.assert_close(
    resampled[:, 7:], 3.0 * output_times[:, None].expand(-1, 29)
  )
  torch.testing.assert_close(resampled[-1], qpos[-1])


def test_resample_qpos_same_rate_preserves_frames() -> None:
  qpos = torch.randn((5, 36), dtype=torch.float32)
  qpos[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0])

  resampled = _resample_qpos(qpos, 50.0, 50.0)

  torch.testing.assert_close(resampled, qpos)


def test_load_qpos_resamples_slerp_and_recomputes_velocities(tmp_path: Path) -> None:
  input_times = torch.arange(31, dtype=torch.float32) / 30.0
  linear_speed = 1.25
  angular_speed = 2.0
  joint_speed = 3.0
  qpos = torch.zeros((input_times.shape[0], 36), dtype=torch.float32)
  qpos[:, 0] = linear_speed * input_times
  qpos[:, 3] = torch.cos(angular_speed * input_times / 2.0)
  qpos[:, 6] = torch.sin(angular_speed * input_times / 2.0)
  qpos[1::2, 3:7] *= -1.0
  qpos[:, 7:] = joint_speed * input_times[:, None]
  input_path = tmp_path / "motion.npz"
  np.savez(input_path, qpos=qpos.numpy())

  root_state, joint_position, joint_velocity, resampled = _load_qpos(
    input_path, 30.0, "cpu", output_fps=50.0
  )

  output_times = torch.arange(51, dtype=torch.float32) / 50.0
  assert resampled.shape == (51, 36)
  torch.testing.assert_close(
    resampled[:, 0], linear_speed * output_times, atol=1e-6, rtol=1e-6
  )
  torch.testing.assert_close(
    torch.linalg.vector_norm(resampled[:, 3:7], dim=-1),
    torch.ones(51),
    atol=1e-6,
    rtol=1e-6,
  )
  torch.testing.assert_close(
    root_state[:, 7:10],
    torch.tensor([linear_speed, 0.0, 0.0]).expand(51, -1),
    atol=1e-4,
    rtol=1e-4,
  )
  torch.testing.assert_close(
    root_state[:, 10:13],
    torch.tensor([0.0, 0.0, angular_speed]).expand(51, -1),
    atol=1e-4,
    rtol=1e-4,
  )
  torch.testing.assert_close(
    joint_position,
    joint_speed * output_times[:, None].expand(-1, 29),
    atol=1e-6,
    rtol=1e-6,
  )
  torch.testing.assert_close(
    joint_velocity,
    torch.full_like(joint_velocity, joint_speed),
    atol=1e-4,
    rtol=1e-4,
  )


def test_joint_velocity_is_estimated_separately_at_50_hz(tmp_path: Path) -> None:
  frame_count = 5
  qpos = np.zeros((frame_count, 36), dtype=np.float32)
  qpos[:, 3] = 1.0
  qpos[:, 7:] = np.arange(frame_count, dtype=np.float32)[:, None] / 50.0 * 3.0
  input_path = tmp_path / "motion.npz"
  np.savez(input_path, qpos=qpos)

  _, joint_position, joint_velocity, _ = _load_qpos(input_path, 50.0, "cpu")

  torch.testing.assert_close(joint_position, torch.from_numpy(qpos[:, 7:]))
  torch.testing.assert_close(
    joint_velocity,
    torch.full_like(joint_velocity, 3.0),
    atol=1e-5,
    rtol=1e-5,
  )


def test_load_qpos_rejects_non_finite_input(tmp_path: Path) -> None:
  qpos = np.zeros((3, 36), dtype=np.float32)
  qpos[:, 3] = 1.0
  qpos[2, 8] = np.nan
  input_path = tmp_path / "motion.npz"
  np.savez(input_path, qpos=qpos)

  with pytest.raises(
    ValueError,
    match=r"motion\.npz qpos contains non-finite value at index \(2, 8\)",
  ):
    _load_qpos(input_path, 50.0, "cpu")


def test_resample_qpos_rejects_zero_norm_root_quaternion() -> None:
  qpos = torch.zeros((2, 36), dtype=torch.float32)

  with pytest.raises(ValueError, match="zero norm at frame 0"):
    _resample_qpos(qpos, 30.0, 50.0)


def test_resample_qpos_rejects_too_short_output_grid() -> None:
  qpos = torch.zeros((2, 36), dtype=torch.float32)
  qpos[:, 3] = 1.0

  with pytest.raises(ValueError, match="shorter than one output timestep"):
    _resample_qpos(qpos, 100.0, 50.0)


def test_launch_conversion_accepts_different_frame_rates(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  input_dir = tmp_path / "input"
  input_dir.mkdir()
  qpos = np.zeros((2, 36), dtype=np.float32)
  qpos[:, 3] = 1.0
  np.savez(input_dir / "motion.npz", qpos=qpos)
  converted: list[QposConversionConfig] = []
  monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
  monkeypatch.setattr(qpos_to_npz, "select_gpus", lambda _: (None, 1))
  monkeypatch.setattr(qpos_to_npz, "_conversion_worker", converted.append)
  monkeypatch.setattr(
    qpos_to_npz,
    "write_motion_manifest",
    lambda output_dir: Path(output_dir) / "manifest.json",
  )
  cfg = QposConversionConfig(
    input_dir=str(input_dir),
    output_dir=str(tmp_path / "output"),
    input_fps=30.0,
    output_fps=50.0,
  )

  qpos_to_npz.launch_conversion(cfg)

  assert converted == [cfg]
