"""CPU integration tests for the documented G1 CSV conversion command."""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def csv_input(tmp_path: Path) -> Path:
  folder = tmp_path / "csv"
  folder.mkdir()
  motion = np.zeros((4, 36), dtype=np.float32)
  motion[:, 0] = np.arange(4, dtype=np.float32) / 30.0
  motion[:, 2] = 1.0
  motion[:, 6] = 1.0  # CSV root quaternion uses xyzw.
  np.savetxt(folder / "walk.csv", motion, delimiter=",")
  return folder


def _run_cli(tmp_path: Path, *args: str) -> subprocess.CompletedProcess[str]:
  return subprocess.run(
    [sys.executable, "-m", "supermimic", "data", "csv", *args],
    cwd=tmp_path,
    env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "MUJOCO_GL": "disable"},
    capture_output=True,
    text=True,
    timeout=120,
    check=False,
  )


@pytest.mark.parametrize("batch", [True, False], ids=["directory", "single-file"])
def test_csv_cli_converts_g1_motion(
  tmp_path: Path, csv_input: Path, batch: bool
) -> None:
  output = tmp_path / "motions" / "walk.npz"
  args = (
    ["--input-dir", str(csv_input), "--output-dir", "motions"]
    if batch
    else ["--input-file", str(csv_input / "walk.csv"), "--output-file", str(output)]
  )
  result = _run_cli(tmp_path, *args, "--input-fps", "30", "--output-fps", "50")

  assert result.returncode == 0, result.stdout + result.stderr
  with np.load(output, allow_pickle=False) as motion:
    assert motion["fps"].tolist() == [50.0]
    for key in ("joint_pos", "joint_vel"):
      assert motion[key].shape == (6, 29)
      np.testing.assert_allclose(motion[key], 0.0)
    for key in ("body_pos_w", "body_lin_vel_w", "body_ang_vel_w"):
      assert motion[key].shape == (6, 30, 3)
      assert np.isfinite(motion[key]).all()
    assert motion["body_quat_w"].shape == (6, 30, 4)
    np.testing.assert_allclose(
      motion["body_pos_w"][:, 0, 0], np.arange(6) / 50.0, atol=1e-7
    )
    np.testing.assert_allclose(motion["body_pos_w"][:, 0, 2], 1.0)
    np.testing.assert_allclose(motion["body_quat_w"][:, 0, 0], 1.0)
    np.testing.assert_allclose(motion["body_quat_w"][:, 0, 1:], 0.0)
    np.testing.assert_allclose(motion["body_lin_vel_w"][:, :, 0], 1.0, atol=1e-6)
    np.testing.assert_allclose(motion["body_ang_vel_w"], 0.0)

  manifest_path = output.parent / "manifest.json"
  if batch:
    manifest = json.loads(manifest_path.read_text())
    assert len(manifest["motions"]) == 1
    entry = manifest["motions"][0]
    assert entry["path"] == "walk.npz"
    assert entry["frames"] == 6
    assert entry["fps"] == 50.0
    assert entry["arrays"]["joint_pos"]["shape"] == [6, 29]
    assert entry["size_bytes"] == output.stat().st_size
    assert len(manifest["dataset_fingerprint"]) == 64
  else:
    assert not manifest_path.exists()


@pytest.mark.parametrize("batch", [True, False], ids=["directory", "single-file"])
def test_csv_cli_rejects_protected_output(
  tmp_path: Path, csv_input: Path, batch: bool
) -> None:
  archive = tmp_path / "archive"
  archive.mkdir()
  sentinel = archive / "preserved.txt"
  sentinel.write_text("preserved input")
  (tmp_path / ".supermimic.local.toml").write_text('protected_roots = ["archive"]\n')
  args = (
    ["--input-dir", str(csv_input), "--output-dir", str(archive / "motions")]
    if batch
    else [
      "--input-file",
      str(csv_input / "walk.csv"),
      "--output-file",
      str(archive / "walk.npz"),
    ]
  )

  result = _run_cli(tmp_path, *args)

  assert result.returncode == 1
  assert "Output would modify preserved input assets" in result.stderr
  assert sentinel.read_text() == "preserved input"
  assert list(archive.iterdir()) == [sentinel]
