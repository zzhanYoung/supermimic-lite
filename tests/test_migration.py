"""External dependency, legacy snapshot and output-isolation contracts."""

import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import mjlab
import pytest
import torch
import yaml
from test_tracking_observation_contract import standing_g1_motion as standing_g1_motion

from supermimic.assets import LocalAssets, inventory, load_assets
from supermimic.cli import train
from supermimic.compat.checkpoints import canonical_config, load_checkpoint
from supermimic.learning.config import SuperMimicRunnerCfg
from supermimic.mdp.observations import motion_anchor_ori_b
from supermimic.tasks import load_env_cfg


def test_legacy_pickle_maps_symbols_without_replacing_upstream(tmp_path):
  checkpoint = tmp_path / "new.pt"
  torch.save(
    {"signature": {"func": motion_anchor_ori_b, "threshold": 0.25}}, checkpoint
  )
  legacy = tmp_path / "legacy.pt"
  # Protocol 2 encodes callable names using GLOBAL. Rebuild the ZIP to simulate
  # the old namespace without ever installing or monkeypatching an old module.
  with ZipFile(checkpoint) as source, ZipFile(legacy, "w") as output:
    for item in source.infolist():
      data = source.read(item)
      if item.filename.endswith("data.pkl"):
        assert b"supermimic.mdp.observations\n" in data
        data = data.replace(
          b"supermimic.mdp.observations\n", b"mjlab.tasks.tracking.mdp.observations\n"
        )
      output.writestr(item, data)
  before = {key: value for key, value in sys.modules.items() if key.startswith("mjlab")}
  expected = load_checkpoint(checkpoint)
  assert load_checkpoint(legacy) == expected
  assert expected["signature"]["threshold"] == 0.25
  for key, value in before.items():
    assert sys.modules[key] is value
  assert "site-packages" in str(Path(mjlab.__file__))
  assert not (Path(__file__).parents[1] / "src/mjlab").exists()


def test_checkpoint_version_and_semantic_identity_stay_strict(tmp_path):
  p = tmp_path / "unsupported.pt"
  torch.save({"format_version": 999}, p)
  with pytest.raises(ValueError, match="Unsupported checkpoint"):
    load_checkpoint(p)
  config = {"func": motion_anchor_ori_b, "history_length": 10}
  renamed = {
    "func": "mjlab.tasks.tracking.mdp.observations.motion_anchor_ori_b",
    "history_length": 10,
  }
  assert canonical_config(config) == canonical_config(renamed)
  renamed["history_length"] = 9
  assert canonical_config(config) != canonical_config(renamed)


def test_asset_config_resolves_independently_of_working_directory(
  tmp_path, monkeypatch
):
  location = tmp_path / "config"
  location.mkdir()
  config = location / "assets.toml"
  config.write_text('motion_folder = "../motions"\nprotected_roots = ["../archive"]\n')
  monkeypatch.chdir(tmp_path)
  assets = load_assets(config)
  assert assets.motion_folder == tmp_path / "motions"
  assert assets.protected_roots == (tmp_path / "archive",)
  config.write_text('typo = "somewhere"\n')
  with pytest.raises(ValueError, match="Unknown asset"):
    load_assets(config)


def test_output_protection_follows_symlinks_and_allows_separate_outputs(tmp_path):
  original = tmp_path / "archive"
  original.mkdir()
  alias = tmp_path / "alias"
  alias.symlink_to(original, target_is_directory=True)
  assets = LocalAssets(protected_roots=(original,))
  with pytest.raises(ValueError, match="preserved input"):
    assets.protect_output(alias / "new-run")
  assert assets.protect_output(tmp_path / "new-run") == tmp_path / "new-run"
  checkpoint = tmp_path / "exported.pt"
  checkpoint.touch()
  assert assets.protect_output(tmp_path / "logs", checkpoint) == tmp_path / "logs"


def test_inventory_preserves_bundle_relationships_without_absolute_paths(tmp_path):
  run = tmp_path / "run"
  (run / "params").mkdir(parents=True)
  for name in (
    "model_5.pt",
    "model_5.rank0.replay.pt",
    "model_5.sampling.npz",
    "params/env.yaml",
    "params/agent.yaml",
  ):
    (run / name).write_bytes(b"asset")
  result = inventory({"checkpoints": tmp_path})
  assert len(result["checkpoints"]) == 1
  bundle = result["checkpoints"][0]
  assert bundle["replay"] == ["run/model_5.rank0.replay.pt"]
  assert bundle["sampling"] == "run/model_5.sampling.npz"
  assert str(tmp_path) not in repr(result)


def test_resume_restores_snapshot_and_requires_replay(standing_g1_motion, tmp_path):
  run = tmp_path / "source-run"
  (run / "params").mkdir(parents=True)
  cfg = load_env_cfg()
  snapshot = yaml.dump(asdict(cfg)).replace(
    "supermimic.mdp.", "mjlab.tasks.tracking.mdp."
  )
  (run / "params/env.yaml").write_text(snapshot)
  checkpoint = run / "model_5000.pt"
  torch.save(
    {"iteration": 5000, "world_size": 1, "config": asdict(SuperMimicRunnerCfg())},
    checkpoint,
  )
  assets = LocalAssets(checkpoint=checkpoint, motion_folder=standing_g1_motion.parent)
  args = [
    "--resume",
    "--device",
    "cpu",
    "--max-iterations",
    "5001",
    "--output-dir",
    str(tmp_path / "new-run"),
  ]
  with pytest.raises(FileNotFoundError, match="replay shard"):
    train.parse_config(args, assets)
  checkpoint.with_name("model_5000.rank0.replay.pt").touch()
  cfg, output = train.parse_config(args, assets)
  assert cfg.agent.max_iterations == 5001
  assert cfg.checkpoint == str(checkpoint)
  assert cfg.env.refresh_kinematics_before_rewards
  assert not cfg.env.auto_reset
  assert asdict(cfg.env.events["base_com"]) == asdict(load_env_cfg().events["base_com"])
  assert output == tmp_path / "new-run"
  assert not output.exists()
  with pytest.raises(ValueError, match="must exceed"):
    train.parse_config(["--resume", "--max-iterations", "5000"], assets)


def test_train_total_target_passes_only_remaining_iterations(tmp_path, monkeypatch):
  calls = []
  env = SimpleNamespace(close=lambda: calls.append("close"))

  class Runner:
    current_learning_iteration = 0
    logger = SimpleNamespace(writer=None)

    def load(self, path):
      self.current_learning_iteration = 5000

    def add_git_repo_to_log(self, path):
      pass

    def learn(self, iterations):
      calls.append(iterations)

  monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
  monkeypatch.setenv("RANK", "0")
  monkeypatch.setattr(train, "SuperMimicEnv", lambda *a, **kw: env)
  monkeypatch.setattr(train, "RslRlVecEnvWrapper", lambda *a, **kw: env)
  monkeypatch.setattr(train, "SuperMimicRunner", lambda *a, **kw: Runner())
  monkeypatch.setattr(train, "dump_yaml", lambda *a: None)
  cfg = train.TrainConfig(
    load_env_cfg(), SuperMimicRunnerCfg(max_iterations=5001), checkpoint="legacy.pt"
  )
  train.run_train(cfg, tmp_path)
  assert calls == [1, "close"]
