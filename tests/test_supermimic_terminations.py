"""SuperMimic Appendix A failure conditions, independent of the torso anchor."""

from dataclasses import asdict
from types import SimpleNamespace
from typing import cast

import pytest
import torch
from mjlab.utils.lab_api.math import quat_from_euler_xyz

from supermimic.envs import SuperMimicEnv as ManagerBasedRlEnv
from supermimic.mdp import (
  MotionCommandCfgBase,
  bad_motion_root_height,
  bad_motion_root_tilt,
)
from supermimic.tasks import load_env_cfg
from supermimic.tasks.g1.env_cfgs import unitree_g1_flat_tracking_env_cfg
from supermimic.tasks.tracking_env_cfg import configure_multi_motion_tracking_profile

TASK = "Mjlab-Tracking-Flat-Unitree-G1-SuperMimic"


@pytest.mark.parametrize("play", [False, True])
def test_supermimic_failure_configuration(play: bool) -> None:
  cfg = load_env_cfg(TASK, play=play)
  original = unitree_g1_flat_tracking_env_cfg(multi_motion=not play, play=play)
  if play:
    original = configure_multi_motion_tracking_profile(original)
  assert cfg.terminations.keys() == original.terminations.keys() - {"ee_body_pos"}
  for name, func, threshold in (
    ("anchor_pos", bad_motion_root_height, 0.25),
    ("anchor_ori", bad_motion_root_tilt, 0.8),
  ):
    term = cfg.terminations[name]
    assert term.func is func
    assert not term.time_out
    assert term.params == {
      "command_name": "motion",
      "root_body_name": "pelvis",
      "threshold": threshold,
    }
  for name in cfg.terminations.keys() - {"anchor_pos", "anchor_ori"}:
    assert asdict(cfg.terminations[name]) == asdict(original.terminations[name])
  assert cast(MotionCommandCfgBase, cfg.commands["motion"]).anchor_body_name == (
    "torso_link"
  )


@pytest.fixture
def root_env() -> tuple[ManagerBasedRlEnv, SimpleNamespace]:
  # Pelvis is deliberately not index zero, and the other bodies disagree strongly
  # with their references. Neither torso nor end-effector errors may end a rollout.
  body_names = ("torso_link", "left_wrist_yaw_link", "pelvis", "left_ankle_roll_link")
  positions = torch.zeros(8, len(body_names), 3, dtype=torch.float64)
  positions[..., 2] = 1.0
  quats = torch.zeros(8, len(body_names), 4, dtype=torch.float64)
  quats[..., 0] = 1.0
  robot_pos = positions + 10.0
  robot_pos[:, 2, 2] = 1.0
  robot_quats = torch.zeros_like(quats)
  robot_quats[..., 1] = 1.0
  robot_quats[:, 2] = quats[:, 2]
  command = SimpleNamespace(
    cfg=SimpleNamespace(body_names=body_names),
    body_pos_w=positions,
    robot_body_pos_w=robot_pos,
    body_quat_w=quats,
    robot_body_quat_w=robot_quats,
    robot=SimpleNamespace(
      data=SimpleNamespace(
        gravity_vec_w=torch.tensor([0.0, 0.0, -1.0], dtype=torch.float64).repeat(8, 1)
      )
    ),
  )
  env = cast(
    ManagerBasedRlEnv,
    SimpleNamespace(
      command_manager=SimpleNamespace(get_term={"motion": command}.__getitem__)
    ),
  )
  return env, command


def test_pelvis_height_threshold_is_strict_and_ignores_other_bodies(
  root_env: tuple[ManagerBasedRlEnv, SimpleNamespace],
) -> None:
  env, command = root_env
  command.robot_body_pos_w[:, 2, 2] += torch.tensor(
    [0.0, 0.25, -0.25, 0.251, -0.251, 0.249, -0.249, 0.0], dtype=torch.float64
  )
  term = load_env_cfg(TASK).terminations["anchor_pos"]
  assert callable(term.func)
  torch.testing.assert_close(
    term.func(env, **term.params),
    torch.tensor([False, False, False, True, True, False, False, False]),
  )


def test_pelvis_tilt_compares_reference_gravity_z_and_ignores_yaw(
  root_env: tuple[ManagerBasedRlEnv, SimpleNamespace],
) -> None:
  env, command = root_env
  reference_roll = torch.tensor(
    [0.0, 0.0, 0.0, 0.0, torch.pi / 2, torch.pi / 2, torch.pi / 3, torch.pi / 3],
    dtype=torch.float64,
  )
  # Gravity-Z errors around the 0.8 threshold, plus tilted references. Opposite
  # 60-degree rolls have equal gravity Z, so the paper's rule does not fail them.
  roll = torch.tensor(
    [0.0, 0.201, 0.199, 0.199, torch.pi / 2, 0.0, -torch.pi / 3, 0.0],
    dtype=torch.float64,
  )
  roll[1:4] = roll[1:4].acos()
  roll[3] *= -1
  yaw = torch.tensor([torch.pi, 0, 0, 0, 2, 0, torch.pi, 0], dtype=torch.float64)
  zeros = torch.zeros_like(roll)
  command.body_quat_w[:, 2] = quat_from_euler_xyz(reference_roll, zeros, zeros)
  command.robot_body_quat_w[:, 2] = quat_from_euler_xyz(roll, zeros, yaw)
  term = load_env_cfg(TASK).terminations["anchor_ori"]
  assert callable(term.func)
  torch.testing.assert_close(
    term.func(env, **term.params),
    torch.tensor([False, False, True, True, False, True, False, False]),
  )
