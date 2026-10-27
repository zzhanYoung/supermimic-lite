from pathlib import Path

import mujoco
import numpy as np
import pytest
from mjlab.entity import Entity

from supermimic.tasks import load_env_cfg


@pytest.fixture(scope="module")
def standing_g1_motion(tmp_path_factory: pytest.TempPathFactory) -> Path:
  cfg = load_env_cfg()
  robot = Entity(cfg.scene.entities["robot"])
  model = robot.compile()
  data = mujoco.MjData(model)
  mujoco.mj_resetDataKeyframe(model, data, 0)
  mujoco.mj_forward(model, data)
  body_ids = [model.body(name).id for name in robot.body_names]
  joint_qpos_ids = [model.joint(name).qposadr[0] for name in robot.joint_names]
  frame = {
    "joint_pos": data.qpos[joint_qpos_ids],
    "joint_vel": np.zeros(len(joint_qpos_ids)),
    "body_pos_w": data.xpos[body_ids],
    "body_quat_w": data.xquat[body_ids],
    "body_lin_vel_w": np.zeros((len(body_ids), 3)),
    "body_ang_vel_w": np.zeros((len(body_ids), 3)),
  }
  motion_file = tmp_path_factory.mktemp("g1_motion") / "standing.npz"
  np.savez(
    motion_file,
    allow_pickle=False,
    fps=np.array(50.0),
    **{
      name: np.repeat(value[None], 5, axis=0).astype(np.float32)
      for name, value in frame.items()
    },
  )
  return motion_file
