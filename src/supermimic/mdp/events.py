from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .commands import MultiMotionCommand
from .metrics import compute_local_key_body_errors

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def record_adaptive_key_body_error(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  command_name: str,
  body_names: tuple[str, ...],
  root_body_name: str = "pelvis",
) -> None:
  """Record sampler residuals on every step, including terminal states before reset.

  This step event always runs on all environments and does not terminate them.
  """
  command = env.command_manager.get_term(command_name)
  if not isinstance(command, MultiMotionCommand) or command.adaptive_sampler is None:
    return
  errors = compute_local_key_body_errors(command, body_names, root_body_name)
  command.record_adaptive_key_body_error(errors)
