"""SuperMimic collection, reset sampling, rewards and failure conditions."""

from dataclasses import fields
from typing import cast

from mjlab.asset_zoo.robots.unitree_g1.g1_constants import FULL_COLLISION_WITHOUT_SELF
from mjlab.managers.termination_manager import TerminationTermCfg

from supermimic.mdp import (
  MultiMotionCommandCfg,
  bad_motion_root_height,
  bad_motion_root_tilt,
)
from supermimic.mdp.return_sampler import ReturnMotionCommandCfg
from supermimic.tasks.tracking_env_cfg import TrackingEnvCfg

from .env_cfgs import unitree_g1_flat_tracking_env_cfg


def configure_supermimic_tracking_profile(cfg: TrackingEnvCfg) -> TrackingEnvCfg:
  """Apply SuperMimic collisions, rewards and Appendix A failure conditions."""
  cfg.scene.entities["robot"].collisions = (FULL_COLLISION_WITHOUT_SELF,)
  cfg.refresh_kinematics_before_rewards = True
  cfg.rewards.pop("self_collisions")
  cfg.rewards["joint_limit"].weight = -2.0
  cfg.rewards["motion_global_root_height"].weight = 0.5
  cfg.terminations["anchor_pos"] = TerminationTermCfg(
    func=bad_motion_root_height,
    params={"command_name": "motion", "root_body_name": "pelvis", "threshold": 0.25},
  )
  cfg.terminations["anchor_ori"] = TerminationTermCfg(
    func=bad_motion_root_tilt,
    params={"command_name": "motion", "root_body_name": "pelvis", "threshold": 0.8},
  )
  cfg.terminations.pop("ee_body_pos")
  return cfg


def unitree_g1_supermimic_env_cfg() -> TrackingEnvCfg:
  cfg = unitree_g1_flat_tracking_env_cfg(multi_motion=True)
  command = cast(MultiMotionCommandCfg, cfg.commands["motion"])
  return_command = ReturnMotionCommandCfg(
    **{item.name: getattr(command, item.name) for item in fields(command)}
  )
  return_command.motion_folder = "motions/lafan1_g1_rtgt_50hz"
  cfg.commands["motion"] = return_command
  cfg.events.pop("push_robot")
  cfg.events.pop("record_adaptive_key_body_error")
  cfg.auto_reset = False
  cfg.scene.num_envs = 4096
  return configure_supermimic_tracking_profile(cfg)
