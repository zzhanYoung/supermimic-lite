"""Motion mimic task configuration.

This module defines the base configuration for motion mimic tasks.
Robot-specific configurations are located in the config/ directory.

This is a re-implementation of BeyondMimic (https://beyondmimic.github.io/).

Based on https://github.com/HybridRobotics/whole_body_tracking
Commit: f8e20c880d9c8ec7172a13d3a88a65e3a5a88448
"""

import math
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Literal

from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

from supermimic import mdp
from supermimic.envs import SuperMimicEnvCfg as ManagerBasedRlEnvCfg
from supermimic.mdp import (
  GmtAdaptiveSamplerCfg,
  MotionCommandCfg,
  MultiMotionCommandCfg,
)

VELOCITY_RANGE = {
  "x": (-0.5, 0.5),
  "y": (-0.5, 0.5),
  "z": (-0.2, 0.2),
  "roll": (-0.52, 0.52),
  "pitch": (-0.52, 0.52),
  "yaw": (-0.78, 0.78),
}

TrackingRewardProfile = Literal["deepmimic", "gmt"]


@dataclass(kw_only=True)
class TrackingEnvCfg(ManagerBasedRlEnvCfg):
  """Manager-based tracking configuration with an explicit reward identity."""

  reward_profile: TrackingRewardProfile = field(default="deepmimic", init=False)


def configure_multi_motion_tracking_profile(cfg: TrackingEnvCfg) -> TrackingEnvCfg:
  """Apply multi-motion rewards and observations, including single-motion play.

  The profile is independent of the motion command type: multi-motion policies
  use a single reference motion at playback but retain their training inputs.
  """
  for group_name in ("actor", "critic"):
    cfg.observations[group_name].terms.pop("motion_anchor_pos_b", None)
  if cfg.reward_profile == "deepmimic":
    cfg.rewards = {
      ("motion_global_root_height" if name == "motion_global_root_pos" else name): (
        RewardTermCfg(
          func=mdp.motion_global_anchor_height_error_exp,
          weight=0.1,
          params={"command_name": "motion", "std": 0.3},
        )
        if name == "motion_global_root_pos"
        else term
      )
      for name, term in cfg.rewards.items()
    }
  cfg.terminations["anchor_pos"].params["threshold"] = 0.5
  return cfg


def make_tracking_env_cfg(
  multi_motion: bool = False,
  *,
  reward_profile: TrackingRewardProfile = "deepmimic",
  adaptive_sampler: GmtAdaptiveSamplerCfg | None = None,
) -> TrackingEnvCfg:
  """Create base tracking task configuration."""

  if reward_profile not in ("deepmimic", "gmt"):
    raise ValueError(f"Unsupported tracking reward profile: {reward_profile}")
  if not multi_motion and adaptive_sampler is not None:
    raise ValueError("An adaptive sampler requires a multi-motion configuration.")

  ##
  # Observations
  ##

  actor_terms = {
    "command": ObservationTermCfg(
      func=mdp.generated_commands, params={"command_name": "motion"}
    ),
    "motion_anchor_pos_b": ObservationTermCfg(
      func=mdp.motion_anchor_pos_b,
      params={"command_name": "motion"},
      noise=Unoise(n_min=-0.25, n_max=0.25),
    ),
    "motion_anchor_ori_b": ObservationTermCfg(
      func=mdp.motion_anchor_ori_b,
      params={"command_name": "motion"},
      noise=Unoise(n_min=-0.05, n_max=0.05),
    ),
    "base_lin_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_lin_vel"},
      noise=Unoise(n_min=-0.5, n_max=0.5),
    ),
    "base_ang_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_ang_vel"},
      noise=Unoise(n_min=-0.2, n_max=0.2),
    ),
    "projected_gravity": ObservationTermCfg(
      func=mdp.projected_gravity,
      noise=Unoise(n_min=-0.05, n_max=0.05),
    ),
    "joint_pos": ObservationTermCfg(
      func=mdp.joint_pos_rel,
      noise=Unoise(n_min=-0.01, n_max=0.01),
      params={"biased": True},
    ),
    "joint_vel": ObservationTermCfg(
      func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.5, n_max=0.5)
    ),
    "actions": ObservationTermCfg(func=mdp.last_action),
  }

  critic_terms = {
    "command": ObservationTermCfg(
      func=mdp.generated_commands, params={"command_name": "motion"}
    ),
    "motion_anchor_pos_b": ObservationTermCfg(
      func=mdp.motion_anchor_pos_b, params={"command_name": "motion"}
    ),
    "motion_anchor_ori_b": ObservationTermCfg(
      func=mdp.motion_anchor_ori_b, params={"command_name": "motion"}
    ),
    "body_pos": ObservationTermCfg(
      func=mdp.robot_body_pos_b, params={"command_name": "motion"}
    ),
    "body_ori": ObservationTermCfg(
      func=mdp.robot_body_ori_b, params={"command_name": "motion"}
    ),
    "base_lin_vel": ObservationTermCfg(
      func=mdp.builtin_sensor, params={"sensor_name": "robot/imu_lin_vel"}
    ),
    "base_ang_vel": ObservationTermCfg(
      func=mdp.builtin_sensor, params={"sensor_name": "robot/imu_ang_vel"}
    ),
    "projected_gravity": ObservationTermCfg(func=mdp.projected_gravity),
    "joint_pos": ObservationTermCfg(func=mdp.joint_pos_rel),
    "joint_vel": ObservationTermCfg(func=mdp.joint_vel_rel),
    "actions": ObservationTermCfg(func=mdp.last_action),
  }

  observations = {
    "actor": ObservationGroupCfg(
      terms=actor_terms,
      concatenate_terms=True,
      enable_corruption=True,
      history_length=10,  # Current frame plus the previous 9 frames.
      flatten_history_dim=True,
    ),
    "critic": ObservationGroupCfg(
      terms=critic_terms,
      concatenate_terms=True,
      enable_corruption=False,
    ),
  }

  ##
  # Actions
  ##

  actions: dict[str, ActionTermCfg] = {
    "joint_pos": JointPositionActionCfg(
      entity_name="robot",
      actuator_names=(".*",),
      scale=0.5,
      use_default_offset=True,
    )
  }

  ##
  # Commands
  ##

  motion_command_cfg = (
    MultiMotionCommandCfg(
      entity_name="robot",
      resampling_time_range=(1.0e9, 1.0e9),
      debug_vis=True,
      pose_range={
        "x": (-0.05, 0.05),
        "y": (-0.05, 0.05),
        "z": (-0.01, 0.01),
        "roll": (-0.1, 0.1),
        "pitch": (-0.1, 0.1),
        "yaw": (-0.2, 0.2),
      },
      velocity_range=VELOCITY_RANGE,
      joint_position_range=(-0.1, 0.1),
      motion_folder="",
      adaptive_sampler=deepcopy(adaptive_sampler),
      anchor_body_name="",
      body_names=(),
    )
    if multi_motion
    else MotionCommandCfg(
      entity_name="robot",
      resampling_time_range=(1.0e9, 1.0e9),
      debug_vis=True,
      pose_range={
        "x": (-0.05, 0.05),
        "y": (-0.05, 0.05),
        "z": (-0.01, 0.01),
        "roll": (-0.1, 0.1),
        "pitch": (-0.1, 0.1),
        "yaw": (-0.2, 0.2),
      },
      velocity_range=VELOCITY_RANGE,
      joint_position_range=(-0.1, 0.1),
      # Override in robot cfg.
      motion_file="",
      anchor_body_name="",
      body_names=(),
    )
  )
  if (
    isinstance(motion_command_cfg, MultiMotionCommandCfg)
    and motion_command_cfg.adaptive_sampler is not None
    and motion_command_cfg.sampling_mode == "adaptive"
  ):
    raise ValueError(
      "The GMT adaptive sampler cannot be combined with adaptive failure-bin "
      "start-frame sampling."
    )
  commands: dict[str, CommandTermCfg] = {"motion": motion_command_cfg}

  ##
  # Events
  ##

  events: dict[str, EventTermCfg] = {
    "push_robot": EventTermCfg(
      func=mdp.push_by_setting_velocity,
      mode="interval",
      interval_range_s=(1.0, 3.0),
      params={"velocity_range": VELOCITY_RANGE},
    ),
    "base_com": EventTermCfg(
      mode="startup",
      func=dr.body_com_offset,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=()),  # Set in robot cfg.
        "operation": "add",
        "ranges": {
          0: (-0.025, 0.025),
          1: (-0.05, 0.05),
          2: (-0.05, 0.05),
        },
      },
    ),
    "encoder_bias": EventTermCfg(
      mode="startup",
      func=dr.encoder_bias,
      params={
        "asset_cfg": SceneEntityCfg("robot"),
        "bias_range": (-0.01, 0.01),
      },
    ),
    "foot_friction": EventTermCfg(
      mode="startup",
      func=dr.geom_friction,
      params={
        "asset_cfg": SceneEntityCfg("robot", geom_names=()),  # Set per-robot.
        "operation": "abs",
        "ranges": (0.3, 1.2),
        "shared_random": True,  # All foot geoms share the same friction.
      },
    ),
  }

  if multi_motion:
    events["record_adaptive_key_body_error"] = EventTermCfg(
      func=mdp.record_adaptive_key_body_error,
      mode="step",
      params={
        "command_name": "motion",
        "root_body_name": "",  # Set per-robot.
        "body_names": (),  # Set per-robot.
      },
    )

  ##
  # Rewards
  ##

  if reward_profile == "gmt":
    rewards: dict[str, RewardTermCfg] = {
      "motion_joint_pos": RewardTermCfg(
        func=mdp.motion_joint_position_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 1.0},
      ),
      "motion_joint_vel": RewardTermCfg(
        func=mdp.motion_joint_velocity_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 5.0},
      ),
      "motion_root_pose": RewardTermCfg(
        func=mdp.motion_root_pose_error_exp,
        weight=1.0,
        params={
          "command_name": "motion",
          "root_body_name": "",  # Set in robot cfg.
          "orientation_std": 0.4,
          "height_std": 0.3,
        },
      ),
      "motion_root_velocity": RewardTermCfg(
        func=mdp.motion_root_velocity_error_exp,
        weight=1.0,
        params={
          "command_name": "motion",
          "root_body_name": "",  # Set in robot cfg.
          "linear_std": 1.0,
          "angular_std": math.pi,
        },
      ),
      "motion_key_body_pos": RewardTermCfg(
        func=mdp.motion_local_key_body_position_error_exp,
        weight=1.0,
        params={
          "command_name": "motion",
          "root_body_name": "",  # Set in robot cfg.
          "body_names": (),  # Set in robot cfg.
          "std": 0.3,
        },
      ),
      "joint_velocity_norm": RewardTermCfg(
        func=mdp.joint_velocity_l2_norm,
        weight=-0.01,
        params={"entity_name": "robot"},
      ),
      "joint_acceleration_norm": RewardTermCfg(
        func=mdp.joint_acceleration_l2_norm,
        weight=-1e-4,
        params={"entity_name": "robot"},
      ),
      "action_rate_l2": RewardTermCfg(
        func=mdp.action_rate_l2,
        weight=-0.1,
      ),
      "joint_limit": RewardTermCfg(
        func=mdp.joint_pos_limits,
        weight=-10.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
      ),
      "self_collisions": RewardTermCfg(
        func=mdp.self_collision_cost,
        weight=-10.0,
        params={"sensor_name": "self_collision", "force_threshold": 10.0},
      ),
    }
  else:
    rewards = {
      "motion_global_root_pos": RewardTermCfg(
        func=mdp.motion_global_anchor_position_error_exp,
        weight=0.5,
        params={"command_name": "motion", "std": 0.3},
      ),
      "motion_global_root_ori": RewardTermCfg(
        func=mdp.motion_global_anchor_orientation_error_exp,
        weight=0.5,
        params={"command_name": "motion", "std": 0.4},
      ),
      "motion_body_pos": RewardTermCfg(
        func=mdp.motion_relative_body_position_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 0.3},
      ),
      "motion_body_ori": RewardTermCfg(
        func=mdp.motion_relative_body_orientation_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 0.4},
      ),
      "motion_body_lin_vel": RewardTermCfg(
        func=mdp.motion_global_body_linear_velocity_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 1.0},
      ),
      "motion_body_ang_vel": RewardTermCfg(
        func=mdp.motion_global_body_angular_velocity_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 3.14},
      ),
      "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-1e-1),
      "joint_limit": RewardTermCfg(
        func=mdp.joint_pos_limits,
        weight=-10.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
      ),
      "self_collisions": RewardTermCfg(
        func=mdp.self_collision_cost,
        weight=-10.0,
        params={"sensor_name": "self_collision", "force_threshold": 10.0},
      ),
    }

  ##
  # Terminations
  ##

  terminations: dict[str, TerminationTermCfg] = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    "nan_detection": TerminationTermCfg(func=mdp.nan_detection),
    "anchor_pos": TerminationTermCfg(
      func=mdp.bad_anchor_pos_z_only,
      params={"command_name": "motion", "threshold": 0.25},
    ),
    "anchor_ori": TerminationTermCfg(
      func=mdp.bad_anchor_ori,
      params={
        "asset_cfg": SceneEntityCfg("robot"),
        "command_name": "motion",
        "threshold": 0.8,
      },
    ),
    "ee_body_pos": TerminationTermCfg(
      func=mdp.bad_motion_body_pos_z_only,
      params={
        "command_name": "motion",
        "threshold": 0.25,
        "body_names": (),  # Set per-robot.
      },
    ),
  }
  if multi_motion:
    # A true timeout resets observation histories before another motion is sampled,
    # so histories do not span a discontinuous change in the reference motion.
    terminations["motion_end"] = TerminationTermCfg(
      func=mdp.motion_time_out,
      params={"command_name": "motion"},
      time_out=True,
    )

  ##
  # Assemble and return
  ##

  cfg = TrackingEnvCfg(
    scene=SceneCfg(terrain=TerrainEntityCfg(terrain_type="plane"), num_envs=1),
    observations=observations,
    actions=actions,
    commands=commands,
    events=events,
    rewards=rewards,
    terminations=terminations,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="",  # Set per-robot.
      distance=2.8,
      fovy=55.0,
      elevation=-5.0,
      azimuth=120.0,
    ),
    sim=SimulationCfg(
      nconmax=35,
      njmax=250,
      mujoco=MujocoCfg(
        timestep=0.005,
        iterations=10,
        ls_iterations=20,
      ),
    ),
    decimation=4,
    episode_length_s=10.0,
  )
  cfg.reward_profile = reward_profile
  if multi_motion:
    configure_multi_motion_tracking_profile(cfg)
  return cfg
