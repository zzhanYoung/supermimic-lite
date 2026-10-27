"""Tests for converted motion dataset manifests."""

import json
from pathlib import Path

import numpy as np
import pytest

from supermimic.data.motion_dataset import (
  build_motion_manifest,
  load_motion_manifest,
  write_motion_manifest,
)


def _write_motion(path: Path, frames: int = 3) -> None:
  np.savez(
    path,
    fps=np.asarray([50.0], dtype=np.float32),
    joint_pos=np.zeros((frames, 2), dtype=np.float32),
    joint_vel=np.zeros((frames, 2), dtype=np.float32),
    body_pos_w=np.zeros((frames, 3, 3), dtype=np.float32),
    body_quat_w=np.zeros((frames, 3, 4), dtype=np.float32),
    body_lin_vel_w=np.zeros((frames, 3, 3), dtype=np.float32),
    body_ang_vel_w=np.zeros((frames, 3, 3), dtype=np.float32),
  )


def test_manifest_records_schema_and_validates_fingerprint(tmp_path: Path) -> None:
  _write_motion(tmp_path / "walk.npz", frames=7)

  manifest = build_motion_manifest(tmp_path)

  assert manifest.motions[0].path == "walk.npz"
  assert manifest.motions[0].frames == 7
  assert manifest.motions[0].fps == 50.0
  assert manifest.motions[0].arrays is not None
  assert manifest.motions[0].arrays["joint_pos"] == {
    "shape": [7, 2],
    "dtype": "float32",
  }

  path = write_motion_manifest(tmp_path)
  loaded, files = load_motion_manifest(tmp_path)
  assert loaded.fingerprint == manifest.fingerprint
  assert files == (str((tmp_path / "walk.npz").resolve()),)

  raw = json.loads(path.read_text())
  raw["motions"][0]["frames"] = 8
  path.write_text(json.dumps(raw))
  with pytest.raises(ValueError, match="fingerprint mismatch"):
    load_motion_manifest(tmp_path)
