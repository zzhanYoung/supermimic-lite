"""Check dependency provenance, hardware, local assets, and resume readiness."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
from typing import Any

from supermimic.assets import DEFAULT_ASSETS, LocalAssets, require_path

MJLAB_COMMIT = "b517e0c489139e7fcee95702cfb2b01931264985"


def main(argv: list[str] | None = None, *, assets: LocalAssets = DEFAULT_ASSETS) -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--json", action="store_true")
  args = parser.parse_args(argv)
  import mjlab
  import torch

  from supermimic.compat.checkpoints import load_checkpoint

  report: dict[str, Any] = {
    "packages": {},
    "mjlab_source": str(Path(mjlab.__file__).resolve()),
    "cuda_available": torch.cuda.is_available(),
    "errors": [],
  }
  errors = report["errors"]
  for name in (
    "supermimic-lite",
    "mjlab",
    "torch",
    "mujoco",
    "mujoco-warp",
    "warp-lang",
    "rsl-rl-lib",
  ):
    report["packages"][name] = importlib.metadata.version(name)
  direct = importlib.metadata.distribution("mjlab").read_text("direct_url.json")
  provenance = json.loads(direct or "{}")
  report["mjlab_commit"] = provenance.get("vcs_info", {}).get("commit_id")
  if report["mjlab_commit"] != MJLAB_COMMIT:
    errors.append(
      "mjlab is not installed from the locked official commit; run uv sync --locked."
    )
  if args.device.startswith("cuda") and not torch.cuda.is_available():
    errors.append("CUDA is unavailable; select --device cpu for CPU checks.")
  if assets.motion_folder is not None:
    try:
      folder = require_path(assets.motion_folder, "motion_folder", directory=True)
      count = len(list(folder.glob("*.npz")))
      report["motions"] = {"path": str(folder), "count": count}
      if not count:
        errors.append("Motion folder contains no NPZ files.")
    except (ValueError, OSError) as error:
      errors.append(str(error))
  else:
    report["motions"] = "Not configured; pass --motion-folder to commands."
  if assets.checkpoint is not None:
    try:
      path = require_path(assets.checkpoint, "checkpoint")
      saved = load_checkpoint(path)
      shards = [
        path.with_name(f"{path.stem}.rank{rank}.replay.pt")
        for rank in range(saved["world_size"])
      ]
      run_dir = assets.run_dir or path.parent
      params = [run_dir / "params" / name for name in ("env.yaml", "agent.yaml")]
      report["checkpoint"] = {
        "path": str(path),
        "iteration": saved["iteration"],
        "world_size": saved["world_size"],
        "inference_ready": all(p.is_file() for p in params),
        "resume_ready": all(p.is_file() for p in shards + params),
        "replay_bytes": sum(p.stat().st_size for p in shards if p.is_file()),
      }
      if not all(p.is_file() for p in params):
        errors.append("Checkpoint configuration snapshots are missing.")
    except (ValueError, OSError, KeyError) as error:
      errors.append(str(error))
  else:
    report["checkpoint"] = "Not configured; pass --checkpoint for inference/resume."
  print(json.dumps(report, indent=2))
  return 1 if errors else 0
