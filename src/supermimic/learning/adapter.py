"""Read-only task observations and state for the SuperMimic learner."""

from __future__ import annotations

from dataclasses import asdict
from typing import cast

import torch
from mjlab.utils.lab_api.math import matrix_from_quat
from mjlab.utils.noise import UniformNoiseCfg

from supermimic.compat.checkpoints import canonical_config
from supermimic.envs import SuperMimicEnv as ManagerBasedRlEnv
from supermimic.mdp.return_sampler import ReturnMotionCommand

from .kinematics import conjugate, heading_matrix, rotate, transform


class TrackingAdapter:
  history_length = 10
  actor_obs_dim = 157
  future_reference_dim = 730
  critic_obs_dim = 286
  critic_input_dim = 914
  action_dim = 29
  physical_dim = 71
  state_dim = 100

  def __init__(self, env: ManagerBasedRlEnv):
    self.env = env
    self.command = cast(ReturnMotionCommand, env.command_manager.get_term("motion"))
    self.robot = env.scene["robot"]
    manager = env.observation_manager
    self.names = {group: manager.active_terms[group] for group in ("actor", "critic")}
    self.sizes = {
      group: [
        int(torch.tensor(shape).prod()) // (10 if group == "actor" else 1)
        for shape in manager.group_obs_term_dim[group]
      ]
      for group in self.names
    }
    if (
      sum(self.sizes["actor"]) != self.actor_obs_dim
      or sum(self.sizes["critic"]) != self.critic_obs_dim
    ):
      raise ValueError(
        "SuperMimic requires "
        f"{self.history_length * self.actor_obs_dim}/{self.critic_obs_dim} "
        "tracking inputs."
      )
    self.slices: dict[str, dict[str, slice]] = {}
    for group in self.names:
      offset = 0
      self.slices[group] = {}
      for name, size in zip(self.names[group], self.sizes[group], strict=True):
        self.slices[group][name] = slice(offset, offset + size)
        offset += size
    command_slice = self.slices["actor"]["command"]
    # Future references replace the command history, but do not encode the
    # robot's relative orientation error. Keep that feedback in the decoder.
    self.proprio_features = [
      i
      for i in range(self.actor_obs_dim)
      if not command_slice.start <= i < command_slice.stop
    ]
    self.noise_low = torch.zeros(self.actor_obs_dim, device=env.device)
    self.noise_high = torch.zeros_like(self.noise_low)
    for name in self.names["actor"]:
      cfg = manager.get_term_cfg("actor", name)
      if cfg.delay_max_lag or cfg.clip or cfg.scale is not None:
        raise ValueError(
          "SuperMimic adapter requires unscaled, undelayed observations."
        )
      if cfg.noise is not None and env.cfg.observations["actor"].enable_corruption:
        if not isinstance(cfg.noise, UniformNoiseCfg):
          raise ValueError("SuperMimic tracking supports uniform observation noise.")
        if cfg.noise.operation != "add":
          raise ValueError("SuperMimic tracking expects additive sensor noise.")
        self.noise_low[self.slices["actor"][name]] = torch.as_tensor(
          cfg.noise.n_min, device=env.device
        )
        self.noise_high[self.slices["actor"][name]] = torch.as_tensor(
          cfg.noise.n_max, device=env.device
        )
    foot_ids, _ = self.robot.find_geoms(r"^(left|right)_foot[1-7]_collision$")
    self.foot_ids = self.robot.indexing.geom_ids[foot_ids]
    torso_ids, _ = self.robot.find_bodies("torso_link")
    self.torso_id = self.robot.indexing.body_ids[torso_ids[0]]
    friction_event = env.cfg.events.get("foot_friction")
    if friction_event is not None and not friction_event.params.get("shared_random"):
      raise ValueError("SuperMimic context requires shared foot friction.")
    self.context = self.read_context()
    self.context_dim = self.context.shape[-1]
    self.goal_dim = self.reference(
      torch.zeros(1, dtype=torch.long, device=env.device),
      torch.zeros(1, dtype=torch.long, device=env.device),
    ).shape[-1]

  def read_context(self) -> torch.Tensor:
    model = self.env.sim.model
    friction = model.geom_friction[:, self.foot_ids, 0].expand(self.env.num_envs, -1)
    if not torch.equal(friction, friction[:, :1].expand_as(friction)):
      raise ValueError("SuperMimic context cannot compress non-shared foot friction.")
    com = model.body_ipos[:, self.torso_id].expand(self.env.num_envs, -1)
    default = self.env.sim.get_default_field("body_ipos")[self.torso_id]
    return torch.cat(
      (friction[:, :1], self.robot.data.encoder_bias, com - default), -1
    ).clone()

  def state(self) -> torch.Tensor:
    indexing = self.robot.indexing
    root = self.env.sim.data.qpos[:, indexing.free_joint_q_adr].clone()
    root[:, :3] -= self.env.scene.env_origins
    velocity = self.env.sim.data.qvel[:, indexing.free_joint_v_adr]
    return torch.cat(
      (
        root,
        self.robot.data.joint_pos,
        velocity,
        self.robot.data.joint_vel,
        self.env.action_manager.action,
      ),
      -1,
    )

  def clean_observations(self) -> tuple[torch.Tensor, torch.Tensor]:
    # Calling term functions does not touch manager noise or circular buffers.
    observations = []
    for group in ("actor", "critic"):
      terms = []
      for name in self.names[group]:
        cfg = self.env.observation_manager.get_term_cfg(group, name)
        terms.append(cfg.func(self.env, **cfg.params).clone())
      observations.append(torch.cat(terms, -1))
    return observations[0], observations[1]

  def unpack_history(self, observation: torch.Tensor) -> torch.Tensor:
    terms = observation.split([size * 10 for size in self.sizes["actor"]], -1)
    return torch.cat(
      [
        term.reshape(-1, 10, size)
        for term, size in zip(terms, self.sizes["actor"], strict=True)
      ],
      -1,
    )

  def add_noise(self, clean: torch.Tensor) -> torch.Tensor:
    return (
      clean
      + self.noise_low
      + torch.rand_like(clean) * (self.noise_high - self.noise_low)
    )

  def reference(self, motion: torch.Tensor, frame: torch.Tensor) -> torch.Tensor:
    library = self.command.motion
    frame = torch.minimum(frame, library.time_step_totals[motion] - 1)
    indices = library.frame_offsets[motion] + frame
    return torch.cat(
      [
        getattr(library, name)[indices].flatten(start_dim=indices.ndim)
        for name in (
          "joint_pos",
          "joint_vel",
          "body_pos_w",
          "body_quat_w",
          "body_lin_vel_w",
          "body_ang_vel_w",
        )
      ],
      -1,
    )

  def future_reference(self, motion: torch.Tensor, frame: torch.Tensor) -> torch.Tensor:
    """Vibed's 73 features for f+1..f+10, clamped within each reference clip."""
    library = self.command.motion
    root = self.command.cfg.body_names.index("pelvis")
    frame = torch.minimum(frame, library.time_step_totals[motion] - 1)
    current = library.frame_offsets[motion] + frame
    future = frame[..., None] + torch.arange(1, 11, device=frame.device)
    future = torch.minimum(future, library.time_step_totals[motion, None] - 1)
    indices = library.frame_offsets[motion, None] + future
    p = library.body_pos_w[indices][..., root, :]
    q = library.body_quat_w[indices][..., root, :]
    ht = heading_matrix(library.body_quat_w[current][..., root, :]).transpose(-1, -2)[
      ..., None, :, :
    ]
    dp = transform(ht, p - library.body_pos_w[current][..., root, :].unsqueeze(-2))
    orientation = (ht @ matrix_from_quat(q))[..., :, :2].flatten(-2)
    return torch.cat(
      (
        dp,
        orientation,
        rotate(conjugate(q), library.body_lin_vel_w[indices][..., root, :]),
        rotate(conjugate(q), library.body_ang_vel_w[indices][..., root, :]),
        library.joint_pos[indices],
        library.joint_vel[indices] * 0.1,
      ),
      -1,
    ).flatten(-2)

  def signature(self) -> dict:
    cfg = self.env.cfg
    # Function identity and parameters matter, not just observation dimensions.
    return canonical_config(
      {
        "dataset": self.command.motion.dataset_fingerprint,
        "observations": asdict(cfg.observations["actor"]),
        "critic": asdict(cfg.observations["critic"]),
        "rewards": {key: asdict(value) for key, value in cfg.rewards.items()},
        "terminations": {key: asdict(value) for key, value in cfg.terminations.items()},
        "step_dt": self.env.step_dt,
        "scale_rewards_by_dt": cfg.scale_rewards_by_dt,
        "episode_length_s": cfg.episode_length_s,
        "is_finite_horizon": cfg.is_finite_horizon,
        "refresh_kinematics_before_rewards": cfg.refresh_kinematics_before_rewards,
        "critic_input": "vibed.914.v1",
        "actor_input": "proprio_orientation_history.990_future_reference.vibed.730.v1",
        "context": "shared_friction_encoder_com.33.v1",
        "world_model_input": "raw_physical.71_action.29_context.33",
        "world_model_output": "raw_delta.71_running_norm.geometric_loss.70",
        "world_model_network": "mlp.layer_norm.elu",
        "sampler": asdict(self.command.return_sampler.cfg),
      }
    )
