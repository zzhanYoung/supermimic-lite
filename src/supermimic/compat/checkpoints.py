"""Read lite checkpoints without modifying mjlab or rewriting source artifacts."""

from __future__ import annotations

import importlib
import pickle
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import torch

FORMAT_VERSION = 2
_MODULES = {
  f"mjlab.tasks.tracking.mdp.{name}": f"supermimic.mdp.{name}"
  for name in ("observations", "rewards", "terminations", "commands", "return_sampler")
}


def canonical_symbol(value: str) -> str:
  for old, new in _MODULES.items():
    if value.startswith(old + "."):
      return new + value[len(old) :]
  return value


def canonical_config(value: Any) -> Any:
  """Stable data-only identity for runtime configs and inert YAML snapshots."""
  if is_dataclass(value) and not isinstance(value, type):
    return canonical_config(asdict(value))
  if isinstance(value, Enum):
    return {
      "python_type": f"{type(value).__module__}.{type(value).__name__}",
      "args": [canonical_config(value.value)],
    }
  if isinstance(value, slice):
    return {
      "python_type": "builtins.slice",
      "args": [value.start, value.stop, value.step],
    }
  if callable(value):
    return canonical_symbol(f"{value.__module__}.{value.__qualname__}")
  if isinstance(value, dict):
    return {key: canonical_config(item) for key, item in value.items()}
  if isinstance(value, (tuple, list)):
    return [canonical_config(item) for item in value]
  if isinstance(value, str):
    return canonical_symbol(value)
  if value is None or isinstance(value, (int, float, bool)):
    return value
  raise TypeError(f"Unsupported configuration identity: {type(value).__name__}")


class _LegacyUnpickler(pickle.Unpickler):
  def find_class(self, module: str, name: str) -> Any:
    if module in _MODULES:
      return getattr(importlib.import_module(_MODULES[module]), name)
    return super().find_class(module, name)


class _LegacyPickle:
  """Per-load pickle adapter, never a sys.modules or site-packages patch."""

  Unpickler = _LegacyUnpickler
  load = staticmethod(pickle.load)
  loads = staticmethod(pickle.loads)


def load_checkpoint(path: str | Path, *, map_location: Any = "cpu", mmap=True) -> dict:
  """Load a trusted training checkpoint, including legacy callable signatures."""
  state = torch.load(
    path,
    map_location=map_location,
    mmap=mmap,
    weights_only=False,
    pickle_module=_LegacyPickle,
  )
  version = state.get("format_version", 1)
  if version not in (1, FORMAT_VERSION):
    raise ValueError(f"Unsupported checkpoint format: {version}")
  if "signature" in state:
    state["signature"] = canonical_config(state["signature"])
  return state
