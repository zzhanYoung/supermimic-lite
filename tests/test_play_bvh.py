from pathlib import Path

import numpy as np
import pytest

from supermimic.data.play_bvh import joint_positions, load_bvh

_MINIMAL_BVH = """\
HIERARCHY
ROOT Hips
{
  OFFSET 10 20 30
  CHANNELS 6 Xposition Yposition Zposition Zrotation Yrotation Xrotation
  JOINT Knee
  {
    OFFSET 0 1 0
    CHANNELS 3 Zrotation Yrotation Xrotation
    End Site
    {
      OFFSET 0 1 0
    }
  }
}
MOTION
Frames: 2
Frame Time: 0.033333
0 0 0 0 0 0 0 0 0
1 2 3 90 0 0 0 0 0
"""


def _write_bvh(tmp_path: Path, contents: str = _MINIMAL_BVH) -> Path:
  path = tmp_path / "motion.bvh"
  path.write_text(contents)
  return path


def test_load_bvh_preserves_timing_and_hierarchy(tmp_path: Path) -> None:
  motion = load_bvh(_write_bvh(tmp_path))

  assert motion.frames.shape == (2, 9)
  assert motion.frame_time == pytest.approx(0.033333)
  assert motion.fps == pytest.approx(30.0003, rel=1e-4)
  assert motion.duration == pytest.approx(0.066666)
  assert [joint.name for joint in motion.joints] == ["Hips", "Knee", "Knee_end"]
  assert [joint.parent for joint in motion.joints] == [None, 0, 1]


def test_joint_positions_follow_channel_order_and_convert_to_z_up(
  tmp_path: Path,
) -> None:
  motion = load_bvh(_write_bvh(tmp_path))

  positions = joint_positions(motion, 1, z_up=True)

  # Root translation channels are absolute for LAFAN1, so ROOT/OFFSET is not
  # added. A +90 degree source-Z rotation maps the child Y offset to -X.
  np.testing.assert_allclose(
    positions,
    np.array(
      [
        [1.0, -3.0, 2.0],
        [0.0, -3.0, 2.0],
        [-1.0, -3.0, 2.0],
      ]
    ),
    atol=1e-7,
  )


def test_load_bvh_rejects_motion_value_count_mismatch(tmp_path: Path) -> None:
  path = _write_bvh(tmp_path, _MINIMAL_BVH.rsplit(" ", 1)[0])

  with pytest.raises(ValueError, match="but contains"):
    load_bvh(path)
