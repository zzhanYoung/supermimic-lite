"""Run against an installed wheel, from outside the checkout."""

import importlib
import importlib.metadata
import pkgutil

import mjlab

import supermimic
from supermimic.cli import main
from supermimic.tasks import load_env_cfg, load_rl_cfg

for module in pkgutil.walk_packages(supermimic.__path__, "supermimic."):
  if not module.name.endswith(".__main__"):
    importlib.import_module(module.name)

cfg = load_env_cfg()
assert not cfg.auto_reset and cfg.refresh_kinematics_before_rewards
load_rl_cfg().validate()
assert main(["--help"]) == 0
files = importlib.metadata.files("supermimic-lite") or []
assert not any(str(path).startswith("mjlab/") for path in files)
print(f"SuperMimic: {supermimic.__file__}")
print(f"External mjlab: {mjlab.__file__}")
print("Installed package smoke test passed.")
