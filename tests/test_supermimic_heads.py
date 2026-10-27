"""Compare analytic learning signals with independently refreshed MuJoCo data."""

from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
import torch
from test_tracking_observation_contract import standing_g1_motion as standing_g1_motion

from supermimic.envs import SuperMimicEnv as ManagerBasedRlEnv
from supermimic.learning.adapter import TrackingAdapter
from supermimic.learning.heads import AnalyticHeads
from supermimic.learning.kinematics import (
  angle_squared,
  matrix_from_quat,
)
from supermimic.mdp.return_sampler import (
  ReturnMotionCommand,
  ReturnMotionCommandCfg,
)
from supermimic.tasks import load_env_cfg


@pytest.fixture
def analytic_adapter(standing_g1_motion: Path):
  torch.set_num_threads(2)
  cfg = load_env_cfg("Mjlab-Tracking-Flat-Unitree-G1-SuperMimic")
  cfg.scene.num_envs = 2
  cfg.seed = 19
  cast(ReturnMotionCommandCfg, cfg.commands["motion"]).motion_folder = str(
    standing_g1_motion.parent
  )
  env = ManagerBasedRlEnv(cfg, device="cpu")
  try:
    env.reset()
    yield TrackingAdapter(env)
  finally:
    env.close()


def write_state(adapter: TrackingAdapter, state: torch.Tensor) -> None:
  env, idx = adapter.env, adapter.robot.indexing
  root = state[:, :7].clone()
  root[:, :3] += env.scene.env_origins
  env.sim.data.qpos[:, idx.free_joint_q_adr] = root
  env.sim.data.qpos[:, idx.joint_q_adr] = state[:, 7:36]
  env.sim.data.qvel[:, idx.free_joint_v_adr] = state[:, 36:42]
  env.sim.data.qvel[:, idx.joint_v_adr] = state[:, 42:71]
  env.action_manager.action.copy_(state[:, -29:])
  env.sim.forward()
  env.sim.sense()
  adapter.command.update_relative_body_poses()


def test_fk_observations_and_transition_signals_match_simulator(analytic_adapter):
  adapter = analytic_adapter
  env, robot, command = adapter.env, adapter.robot, adapter.command
  heads = AnalyticHeads(adapter)
  assert env.scene.env_origins[1].abs().sum() > 0
  torch.manual_seed(28)
  state = adapter.state()
  state[:, :3] += torch.randn(2, 3) * 0.04
  state[:, 3:7] = torch.nn.functional.normalize(torch.randn(2, 4), dim=-1)
  state[:, 7:36] += torch.randn(2, 29) * 0.15
  state[:, 36:] = torch.randn(2, 64) * 0.2
  write_state(adapter, state)
  links = heads.kinematics(state)
  torch.testing.assert_close(
    links.position + env.scene.env_origins[:, None],
    robot.data.body_link_pos_w,
    atol=2e-6,
    rtol=2e-5,
  )
  torch.testing.assert_close(
    matrix_from_quat(links.quaternion),
    matrix_from_quat(robot.data.body_link_quat_w),
    atol=2e-6,
    rtol=2e-5,
  )
  torch.testing.assert_close(
    links.linear_velocity, robot.data.body_link_lin_vel_w, atol=2e-6, rtol=2e-5
  )
  torch.testing.assert_close(
    links.angular_velocity, robot.data.body_link_ang_vel_w, atol=2e-6, rtol=2e-5
  )
  context = adapter.read_context()
  assert context[:, 1:30].abs().sum() > 0  # Encoder bias is actually exercised.
  assert context[:, 30:33].abs().sum() > 0  # COM randomization is active too.
  goal = adapter.reference(command.motion_ids, command.time_steps)
  analytic = heads.observation(state, goal, context, links)
  torch.testing.assert_close(
    analytic, adapter.clean_observations()[0], atol=2e-6, rtol=2e-5
  )
  action = torch.randn(2, 29) * 0.1
  _, reward, terminated, _, _ = env.step(action)
  after = adapter.state()
  next_links = heads.kinematics(after)
  terms = heads.reward_terms(state, action, after, goal, links, next_links)
  for name, value in terms.items():
    i = env.reward_manager.active_terms.index(name)
    torch.testing.assert_close(
      value, env.reward_manager._step_reward[:, i] * env.step_dt, atol=2e-6, rtol=2e-4
    )
  torch.testing.assert_close(sum(terms.values()), reward, atol=2e-6, rtol=2e-4)
  assert torch.equal(heads.hard_termination(after, goal), terminated)
  assert terminated.any()  # Rotated roots exercise terminal next states.
  next_goal = adapter.reference(command.motion_ids, command.time_steps)
  torch.testing.assert_close(
    heads.observation(after, next_goal, context, next_links),
    adapter.clean_observations()[0],
    atol=2e-6,
    rtol=2e-5,
  )


def test_soft_termination_and_rotation_gradients(analytic_adapter):
  adapter = analytic_adapter
  heads = AnalyticHeads(adapter).double()
  motion = adapter.command.motion_ids
  goal = adapter.reference(motion, adapter.command.time_steps).double()
  state = adapter.state().double()
  root_height = heads.reference_parts(goal)[2][:, heads.ref_root, 2]
  state[:, 2] = root_height + torch.tensor([0.25, 0.250001])
  state[:, 3:7] = heads.reference_parts(goal)[3][:, heads.ref_root]
  assert heads.hard_termination(state, goal).tolist() == [False, True]
  state.requires_grad_()
  probability = heads.soft_termination(state, goal)
  assert torch.all((probability > 0) & (probability < 1))
  gradient = torch.autograd.grad(probability.sum(), state)[0]
  assert torch.isfinite(gradient).all() and gradient[:, 2].abs().min() > 1
  rotations = torch.tensor(
    [[1.0, 0, 0, 0], [0.0, 1, 0, 0]], dtype=torch.float64, requires_grad=True
  )
  error = angle_squared(rotations, rotations.detach())
  assert torch.isfinite(torch.autograd.grad(error.sum(), rotations)[0]).all()


def test_critic_features_match_vibed_numpy_formulas(analytic_adapter):
  adapter = analytic_adapter
  heads = AnalyticHeads(adapter)
  command = adapter.command
  state = adapter.state()
  state[:, :3] += torch.tensor([0.1, -0.2, 0.03])
  state[:, 3:7] = torch.nn.functional.normalize(
    torch.tensor([[0.9, 0.2, -0.1, 0.3], [0.8, -0.1, 0.3, -0.2]]), dim=-1
  )
  state[:, 36:71] = torch.randn(2, 35) * 0.2
  context = adapter.read_context()
  frame = torch.tensor([0, 3])
  motion = command.motion_ids
  # A varying reference distinguishes frame ordering, clipping and feature blocks.
  lib = command.motion
  t = torch.arange(len(lib.joint_pos), dtype=torch.float32)
  lib.joint_pos += t[:, None] * 0.01
  lib.joint_vel += t[:, None] * 0.02
  lib.body_pos_w[:, :, 0] += t[:, None] * 0.03
  lib.body_lin_vel_w[:, :, 0] = 0.3
  lib.body_ang_vel_w[:, :, 2] = 0.2
  goal = adapter.reference(motion, frame)
  future = adapter.future_reference(motion, frame)
  links = heads.kinematics(state)
  actual = heads.critic_input(state, goal, context, future, links)
  assert actual.shape == (2, 914)
  qj, _, bp, bq, _, _ = (x.numpy() for x in heads.reference_parts(goal))
  state_np = state.numpy()
  r = matrix_from_quat(state[:, 3:7]).numpy()
  reference_r = matrix_from_quat(torch.tensor(bq[:, heads.ref_root])).numpy()

  def heading(matrix):
    x, y = matrix[0, 0], matrix[1, 0]
    n = np.sqrt(x * x + y * y + 1e-9)
    return np.array([[x / n, -y / n, 0], [y / n, x / n, 0], [0, 0, 1]])

  expected = []
  for i in range(2):
    p, h = state_np[i, :3], heading(r[i])
    rp = bp[i, heads.ref_root]
    ref_bodies = (
      np.r_[p[:2], rp[2]] + (heading(r[i] @ reference_r[i].T) @ (bp[i] - rp).T).T
    )
    normalized_xi = np.r_[
      (context[i, 0].item() - 0.75) / 0.45,
      context[i, 1:30].numpy() / 0.01,
      context[i, 30:33].numpy() / 0.05,
    ]
    future_parts = []
    current = int(lib.frame_offsets[motion[i]] + frame[i])
    for j in range(1, 11):
      idx = int(
        lib.frame_offsets[motion[i]]
        + min(int(frame[i]) + j, int(lib.time_step_totals[motion[i]]) - 1)
      )
      ref_rotation = matrix_from_quat(lib.body_quat_w[idx, heads.ref_root]).numpy()
      cur_rotation = matrix_from_quat(lib.body_quat_w[current, heads.ref_root]).numpy()
      ht = heading(cur_rotation).T
      future_parts.append(
        np.r_[
          ht
          @ (
            lib.body_pos_w[idx, heads.ref_root]
            - lib.body_pos_w[current, heads.ref_root]
          ).numpy(),
          (ht @ ref_rotation)[:, :2].ravel(),
          ref_rotation.T @ lib.body_lin_vel_w[idx, heads.ref_root].numpy(),
          ref_rotation.T @ lib.body_ang_vel_w[idx, heads.ref_root].numpy(),
          lib.joint_pos[idx].numpy(),
          0.1 * lib.joint_vel[idx].numpy(),
        ]
      )
    expected.append(
      np.r_[
        p[2],
        (h.T @ r[i])[:, :2].ravel(),
        r[i].T @ state_np[i, 36:39],
        state_np[i, 39:42],
        state_np[i, 7:36],
        state_np[i, 42:71],
        qj[i] - state_np[i, 7:36],
        ((h.T @ (ref_bodies - links.position[i, heads.tracked].numpy()).T).T).ravel(),
        h.T @ (rp - p),
        (r[i].T @ reference_r[i])[:, :2].ravel(),
        normalized_xi,
        np.concatenate(future_parts),
      ]
    )
  torch.testing.assert_close(
    actual.double(), torch.tensor(np.array(expected)), atol=2e-6, rtol=2e-5
  )


def test_context_contains_shared_friction_encoder_bias_and_com(analytic_adapter):
  adapter = analytic_adapter
  context = adapter.read_context()
  assert context.shape == (2, 33)
  assert adapter.context_dim == 33
  torch.testing.assert_close(
    context[:, :1], adapter.env.sim.model.geom_friction[:, adapter.foot_ids[:1], 0]
  )
  torch.testing.assert_close(context[:, 1:30], adapter.robot.data.encoder_bias)
  torch.testing.assert_close(
    context[:, 30:33],
    adapter.env.sim.model.body_ipos[:, adapter.torso_id]
    - adapter.env.sim.get_default_field("body_ipos")[adapter.torso_id],
  )
  adapter.env.sim.model.geom_friction[:, adapter.foot_ids[0], 0] += 0.1
  with pytest.raises(ValueError, match="non-shared"):
    adapter.read_context()


def test_future_reference_stays_inside_each_clip():
  adapter = TrackingAdapter.__new__(TrackingAdapter)
  quat = torch.zeros(8, 1, 4)
  quat[..., 0] = 1
  library = SimpleNamespace(
    time_step_totals=torch.tensor([5, 3]),
    frame_offsets=torch.tensor([0, 5]),
    joint_pos=torch.arange(8.0)[:, None].expand(8, 29),
    joint_vel=torch.zeros(8, 29),
    body_pos_w=torch.zeros(8, 1, 3),
    body_quat_w=quat,
    body_lin_vel_w=torch.zeros(8, 1, 3),
    body_ang_vel_w=torch.zeros(8, 1, 3),
  )
  # Only reference metadata is needed; no simulator or learned features enter.
  adapter.command = cast(
    ReturnMotionCommand,
    SimpleNamespace(motion=library, cfg=SimpleNamespace(body_names=("pelvis",))),
  )
  future = adapter.future_reference(torch.tensor([0, 1]), torch.tensor([3, 1]))
  positions = future.reshape(2, 10, 73)[..., 15:44]
  torch.testing.assert_close(positions[0], torch.full((10, 29), 4.0))
  torch.testing.assert_close(positions[1], torch.full((10, 29), 7.0))
