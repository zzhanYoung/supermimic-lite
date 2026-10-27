from pathlib import Path

import numpy as np
import pytest

from supermimic.data.play_npz import NpzPlaybackConfig, load_npz_motion


def test_lafan_qpos_without_metadata_defaults_to_30_hz() -> None:
  cfg = NpzPlaybackConfig(input_file="motion.npz")

  assert cfg.fallback_fps == 30.0


def test_load_npz_motion_reads_qpos_and_embedded_fps(tmp_path: Path) -> None:
  path = tmp_path / "qpos.npz"
  qpos = np.arange(72, dtype=np.float32).reshape(2, 36)
  np.savez(path, qpos=qpos, fps=np.array([30.0], dtype=np.float32))

  motion = load_npz_motion(path)

  np.testing.assert_array_equal(motion.qpos, qpos)
  assert motion.embedded_fps == 30.0


def test_load_npz_motion_reconstructs_tracking_qpos(tmp_path: Path) -> None:
  path = tmp_path / "tracking.npz"
  joint_pos = np.arange(58, dtype=np.float32).reshape(2, 29)
  body_pos = np.zeros((2, 3, 3), dtype=np.float32)
  body_pos[:, 0] = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
  body_quat = np.zeros((2, 3, 4), dtype=np.float32)
  body_quat[:, :, 0] = 1.0
  np.savez(
    path,
    joint_pos=joint_pos,
    body_pos_w=body_pos,
    body_quat_w=body_quat,
    fps=np.array([50.0], dtype=np.float32),
  )

  motion = load_npz_motion(path)

  assert motion.qpos.shape == (2, 36)
  np.testing.assert_array_equal(motion.qpos[:, :3], body_pos[:, 0])
  np.testing.assert_array_equal(motion.qpos[:, 3:7], body_quat[:, 0])
  np.testing.assert_array_equal(motion.qpos[:, 7:], joint_pos)
  assert motion.embedded_fps == 50.0


def test_load_npz_motion_accepts_missing_fps(tmp_path: Path) -> None:
  path = tmp_path / "qpos.npz"
  np.savez(path, qpos=np.zeros((1, 36), dtype=np.float32))

  motion = load_npz_motion(path)

  assert motion.embedded_fps is None


def test_load_npz_motion_rejects_wrong_qpos_shape(tmp_path: Path) -> None:
  path = tmp_path / "qpos.npz"
  np.savez(path, qpos=np.zeros((2, 35), dtype=np.float32))

  with pytest.raises(ValueError, match=r"\[frames, 36\]"):
    load_npz_motion(path)
