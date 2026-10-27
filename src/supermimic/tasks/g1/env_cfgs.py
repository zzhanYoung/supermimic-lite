"""Unitree G1 flat tracking environment configurations."""

from copy import deepcopy

from mjlab.asset_zoo.robots import (
  G1_ACTION_SCALE,
  get_g1_robot_cfg,
)
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg

from supermimic.mdp import (
  GmtAdaptiveSamplerCfg,
  MotionCommandCfgBase,
  MultiMotionCommandCfg,
  privileged_dynamics,
)
from supermimic.tasks.tracking_env_cfg import (
  TrackingEnvCfg,
  TrackingRewardProfile,
  make_tracking_env_cfg,
)

G1_PRIVILEGED_STATE_DIM = 286
G1_PRIVILEGED_DYNAMICS_DIM = 128
G1_PRIVILEGED_OBSERVATION_DIM = G1_PRIVILEGED_STATE_DIM + G1_PRIVILEGED_DYNAMICS_DIM

_G1_FOOT_GEOM_PATTERN = r"^(left|right)_foot[1-7]_collision$"
_G1_DISTURBANCE_BODY_NAMES = (
  "pelvis",
  "torso_link",
  "left_wrist_yaw_link",
  "right_wrist_yaw_link",
)
_G1_GMT_KEY_BODY_NAMES = (
  "left_hip_roll_link",
  "left_knee_link",
  "left_ankle_roll_link",
  "right_hip_roll_link",
  "right_knee_link",
  "right_ankle_roll_link",
  "left_shoulder_roll_link",
  "left_elbow_link",
  "left_wrist_yaw_link",
  "right_shoulder_roll_link",
  "right_elbow_link",
  "right_wrist_yaw_link",
)
_G1_TERMINATION_BODY_NAMES = (
  "left_ankle_roll_link",
  "right_ankle_roll_link",
  "left_wrist_yaw_link",
  "right_wrist_yaw_link",
)


def configure_g1_privileged_teacher_observations(
  cfg: TrackingEnvCfg,
  group_names: tuple[str, ...] = ("actor", "critic"),
) -> TrackingEnvCfg:
  """Add clean current-frame privileged observation groups to a G1 task."""
  if "critic" not in cfg.observations:
    raise ValueError("A critic observation group is required for the teacher state.")
  state_terms = {
    name: deepcopy(term)
    for name, term in cfg.observations["critic"].terms.items()
    if name != "dynamics"
  }
  state_terms["dynamics"] = ObservationTermCfg(
    func=privileged_dynamics,
    params={
      "asset_cfg": SceneEntityCfg(
        "robot",
        joint_names=".*",
        actuator_names=".*",
      ),
      "foot_asset_cfg": SceneEntityCfg(
        "robot",
        geom_names=_G1_FOOT_GEOM_PATTERN,
      ),
      "com_asset_cfg": SceneEntityCfg(
        "robot",
        body_names=("torso_link",),
        preserve_order=True,
      ),
      "wrench_asset_cfg": SceneEntityCfg(
        "robot",
        body_names=_G1_DISTURBANCE_BODY_NAMES,
        preserve_order=True,
      ),
      "joint_friction_scale": 1.5,
      "foot_friction_scale": 1.3,
      "com_scale": (0.025, 0.05, 0.05),
      "force_scale": 200.0,
      "torque_scale": 40.0,
    },
  )
  for group_name in group_names:
    cfg.observations[group_name] = ObservationGroupCfg(
      terms=deepcopy(state_terms),
      concatenate_terms=True,
      enable_corruption=False,
      history_length=None,
      flatten_history_dim=True,
    )
  return cfg


def unitree_g1_flat_tracking_env_cfg(
  has_state_estimation: bool = True,
  play: bool = False,
  actor_base_vel: bool = False,
  multi_motion: bool = False,
  *,
  reward_profile: TrackingRewardProfile = "deepmimic",
  adaptive_sampler: GmtAdaptiveSamplerCfg | None = None,
) -> TrackingEnvCfg:
  """Create Unitree G1 flat terrain tracking configuration."""
  cfg = make_tracking_env_cfg(
    multi_motion=multi_motion,
    reward_profile=reward_profile,
    adaptive_sampler=adaptive_sampler,
  )

  cfg.scene.entities = {"robot": get_g1_robot_cfg()}

  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (self_collision_cfg,)

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = G1_ACTION_SCALE

  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfgBase)
  motion_cmd.anchor_body_name = "torso_link"
  motion_cmd.body_names = (
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
  )

  if reward_profile == "gmt":
    for reward_name in ("motion_root_pose", "motion_root_velocity"):
      cfg.rewards[reward_name].params["root_body_name"] = "pelvis"
    key_body_reward = cfg.rewards["motion_key_body_pos"]
    key_body_reward.params["root_body_name"] = "pelvis"
    key_body_reward.params["body_names"] = _G1_GMT_KEY_BODY_NAMES

  if isinstance(motion_cmd, MultiMotionCommandCfg):
    error_event = cfg.events["record_adaptive_key_body_error"]
    error_event.params["root_body_name"] = "pelvis"
    error_event.params["body_names"] = _G1_GMT_KEY_BODY_NAMES

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = _G1_FOOT_GEOM_PATTERN
  cfg.events["base_com"].params["asset_cfg"].body_names = ("torso_link",)

  cfg.terminations["ee_body_pos"].params["body_names"] = _G1_TERMINATION_BODY_NAMES

  cfg.viewer.body_name = "torso_link"

  # Modify observations if we don't have state estimation.
  if not has_state_estimation:
    actor_observations = cfg.observations["actor"]
    new_actor_terms = {
      k: v for k, v in actor_observations.terms.items() if k != "motion_anchor_pos_b"
    }
    cfg.observations["actor"] = ObservationGroupCfg(
      terms=new_actor_terms,
      concatenate_terms=True,
      enable_corruption=True,
      history_length=actor_observations.history_length,
      flatten_history_dim=actor_observations.flatten_history_dim,
    )

  if not actor_base_vel or not has_state_estimation:
    cfg.observations["actor"].terms.pop("base_lin_vel", None)
  assert "base_lin_vel" in cfg.observations["critic"].terms

  # Apply play mode overrides.
  if play:
    # Effectively infinite episode length.
    cfg.episode_length_s = int(1e9)

    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)

    # Disable RSI randomization.
    motion_cmd.pose_range = {}
    motion_cmd.velocity_range = {}

    motion_cmd.sampling_mode = "start"
    if isinstance(motion_cmd, MultiMotionCommandCfg):
      motion_cmd.adaptive_sampler = None

  return cfg
