"""Task-owned environment timing; all simulation infrastructure comes from mjlab."""

from dataclasses import dataclass

import torch
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg, types


@dataclass(kw_only=True)
class SuperMimicEnvCfg(ManagerBasedRlEnvCfg):
  refresh_kinematics_before_rewards: bool = False


class SuperMimicEnv(ManagerBasedRlEnv):
  """Preserve the lite collection contract, including manual terminal resets.

  The step implementation derives from mjlab (Apache-2.0); its extra forward
  call synchronizes analytic model heads with real post-integration rewards.
  """

  cfg: SuperMimicEnvCfg  # pyright: ignore[reportIncompatibleVariableOverride]

  def __init__(self, cfg: SuperMimicEnvCfg, *args, **kwargs):
    super().__init__(cfg, *args, **kwargs)

  def step(self, action: torch.Tensor) -> types.VecEnvStepReturn:
    """Run one environment step: apply actions, simulate, compute RL signals.

    When ``auto_reset=True`` (default), environments that terminate or time out are
    reset in place and the returned observation is the post-reset state. When
    ``auto_reset=False``, the reset is skipped and the returned observation is the
    terminal state; the caller must call ``reset(env_ids=...)`` for done envs before
    the next ``step()``.

    **Forward-call placement.** MuJoCo's ``mj_step`` runs forward kinematics *before*
    integration, so after stepping, derived quantities (``xpos``, ``xquat``,
    ``site_xpos``, ``cvel``, ``sensordata``) lag ``qpos``/``qvel`` by one physics
    substep. Rather than calling ``sim.forward()`` twice (once after the decimation
    loop and once after the reset block), this method calls it **once**, right
    before observation computation. This single call refreshes derived quantities
    for *all* envs: non-reset envs pick up post-decimation kinematics, reset envs
    pick up post-reset kinematics.

    With ``refresh_kinematics_before_rewards=True``, an additional forward call
    refreshes the full control-step state before termination and reward. The
    final call remains in place for event/reset changes. Otherwise the default
    tradeoff is that termination, reward, and step/interval event managers see
    derived quantities that are stale by one physics substep (the last ``mj_step``
    ran ``mj_forward`` from *pre*-integration ``qpos``). In practice, the staleness
    is negligible for reward shaping, termination checks, and event perturbations.
    Critically, the staleness is *consistent*: every env, every step, always sees
    the same lag, so the MDP is well-defined and the value function can learn the
    correct mapping.

    **Auto-reset parity.** Auto-reset reproduces the explicit ``reset()`` flow:
    events fire on the terminal state before the reset, and freshly reset envs
    advance commands with ``dt=0`` so their timers start full.

    .. note::

      Events and reset-path resamples all run *before* ``forward()``, so state
      they write (e.g. a velocity push) is refreshed automatically and visible
      in this step's observations. Command updates run *after*
      the call, so a command that writes sim state must refresh derived
      quantities itself (see ``MotionCommand._update_command``). In all cases,
      do not read derived quantities (``root_link_pose_w``, ``body_link_vel_w``,
      etc.) in the same function that writes state (``write_root_state_to_sim``,
      ``write_joint_state_to_sim``, etc.). See :ref:`faq` for details.
    """
    if not self.cfg.auto_reset and torch.any(self._manual_reset_pending):
      pending_ids = self._manual_reset_pending.nonzero(as_tuple=False).squeeze(-1)
      raise RuntimeError(
        f"Environments {pending_ids.cpu().tolist()} must be reset via "
        "reset(env_ids=...) before calling step() again when auto_reset=False."
      )

    self.extras["log"] = dict()
    self.action_manager.process_action(action.to(self.device))

    for _ in range(self.cfg.decimation):
      self._sim_step_counter += 1
      self.action_manager.apply_action()
      self.scene.write_data_to_sim()
      self.sim.step()
      self.scene.update(dt=self.physics_dt)
      self.metrics_manager.compute_substep()

    # Update env counters.
    self.episode_length_buf += 1
    self.common_step_counter += 1

    if self.cfg.refresh_kinematics_before_rewards:
      self.sim.forward()

    # Check terminations and compute rewards.
    # Unless explicitly refreshed above, derived quantities lag one physics
    # substep. See the docstring for the two timing conventions.
    self.reset_buf = self.termination_manager.compute()
    self.reset_terminated = self.termination_manager.terminated
    self.reset_time_outs = self.termination_manager.time_outs

    self.reward_buf = self.reward_manager.compute(dt=self.step_dt)
    self.metrics_manager.compute()

    # Events fire before auto-reset, on the terminal state, matching the
    # explicit flow. Staleness and forward() rules are the same as for
    # termination/reward (see docstring above).
    if "step" in self.event_manager.available_modes:
      self.event_manager.apply(mode="step", dt=self.step_dt)
    if "interval" in self.event_manager.available_modes:
      self.event_manager.apply(mode="interval", dt=self.step_dt)

    # Reset envs that terminated/timed-out and log the episode info.
    reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
    if self.cfg.auto_reset and len(reset_env_ids) > 0:
      self.recorder_manager.record_pre_reset(reset_env_ids)
      self._reset_idx(reset_env_ids)
      self.scene.write_data_to_sim()

    # Final forward() call: recompute derived quantities from current
    # qpos/qvel for every env. For non-reset envs this resolves the
    # one-substep staleness left by mj_step; for reset envs it picks up
    # the freshly written reset state.
    self.sim.forward()

    # Pass dt=0 for freshly reset envs so their command timers start full,
    # matching reset().
    if self.cfg.auto_reset and len(reset_env_ids) > 0:
      command_dt = self._command_dt
      command_dt.fill_(self.step_dt)
      command_dt[reset_env_ids] = 0.0
      self.command_manager.compute(dt=command_dt)
    else:
      self.command_manager.compute(dt=self.step_dt)

    self.sim.sense()
    self.obs_buf = self.observation_manager.compute(update_history=True)

    if self.cfg.auto_reset and len(reset_env_ids) > 0:
      self.recorder_manager.record_post_reset(reset_env_ids)
    elif len(reset_env_ids) > 0:
      self._manual_reset_pending[reset_env_ids] = True

    self.recorder_manager.record_post_step()

    return (
      self.obs_buf,
      self.reward_buf,
      self.reset_terminated,
      self.reset_time_outs,
      self.extras,
    )
