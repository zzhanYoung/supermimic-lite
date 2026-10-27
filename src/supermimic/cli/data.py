"""Validate or convert motion data, and inventory preserved experiment assets."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

from supermimic.assets import DEFAULT_ASSETS, LocalAssets, require_path, write_inventory


def main(argv: list[str] | None = None, *, assets: LocalAssets = DEFAULT_ASSETS) -> int:
  args = list(argv if argv is not None else sys.argv[1:])
  conversions = {
    "csv": "csv_to_npz",
    "qpos": "qpos_to_npz",
    "play-npz": "play_npz",
    "play-bvh": "play_bvh",
  }
  if args and args[0] in conversions:
    # Original data tools retain their tyro option interface.
    old = sys.argv
    sys.argv = [f"supermimic data {args[0]}", *args[1:]]
    try:
      module = importlib.import_module("supermimic.data." + conversions[args[0]])
      module.main(**({"assets": assets} if args[0] in ("csv", "qpos") else {}))
    finally:
      sys.argv = old
    return 0
  parser = argparse.ArgumentParser(
    description=__doc__,
    epilog="Converters/viewers: data csv, data qpos, data play-npz, data play-bvh",
  )
  subs = parser.add_subparsers(dest="command", required=True)
  check = subs.add_parser("check")
  check.add_argument("--motion-folder", type=Path, default=assets.motion_folder)
  listing = subs.add_parser("inventory")
  listing.add_argument("--output", type=Path, default=Path("logs/asset-inventory.json"))
  parsed = parser.parse_args(args)
  if parsed.command == "check":
    from supermimic.cli.evaluate import scan_motions

    folder = require_path(parsed.motion_folder, "--motion-folder", directory=True)
    motions = scan_motions(folder)
    print(
      json.dumps(
        {
          "count": len(motions),
          "frames": sum(m["frames"] for m in motions),
          "fps": 50.0,
          "motions": [m["name"] for m in motions],
        },
        indent=2,
      )
    )
  else:
    roots = {p.name: p for p in assets.protected_roots}
    if not roots:
      raise ValueError("Configure protected_roots to inventory preserved assets.")
    output = assets.protect_output(parsed.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_inventory(output, roots)
    print(output)
  return 0
