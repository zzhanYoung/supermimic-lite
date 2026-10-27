"""Explicit SuperMimic configuration factories; no mjlab registry side effects."""

from supermimic.learning.config import SuperMimicRunnerCfg
from supermimic.tasks.g1.env_cfgs import unitree_g1_flat_tracking_env_cfg
from supermimic.tasks.g1.supermimic import (
  configure_supermimic_tracking_profile,
  unitree_g1_supermimic_env_cfg,
)
from supermimic.tasks.tracking_env_cfg import configure_multi_motion_tracking_profile

TASK_ID = "Mjlab-Tracking-Flat-Unitree-G1-SuperMimic"


def load_env_cfg(task_name: str = TASK_ID, play: bool = False):
  if task_name != TASK_ID:
    raise ValueError(f"This package only provides {TASK_ID}")
  if play:
    cfg = configure_supermimic_tracking_profile(
      configure_multi_motion_tracking_profile(
        unitree_g1_flat_tracking_env_cfg(play=True)
      )
    )
    cfg.auto_reset = False
    return cfg
  return unitree_g1_supermimic_env_cfg()


def load_rl_cfg(task_name: str = TASK_ID) -> SuperMimicRunnerCfg:
  if task_name != TASK_ID:
    raise ValueError(f"This package only provides {TASK_ID}")
  return SuperMimicRunnerCfg()
