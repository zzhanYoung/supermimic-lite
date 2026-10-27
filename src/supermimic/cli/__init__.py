"""Lazy command dispatch: help and configuration need no simulator startup."""

from __future__ import annotations

import argparse
import importlib
import sys

from supermimic.assets import load_assets


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(prog="supermimic", add_help=False)
  parser.add_argument("--asset-config", help="Local TOML asset locations.")
  options, args = parser.parse_known_args(argv if argv is not None else sys.argv[1:])
  commands = {
    "train": "train",
    "eval": "evaluate",
    "play": "play",
    "plot": "plot",
    "data": "data",
  }
  if not args or args[0] in ("-h", "--help"):
    print(
      "Usage: supermimic [--asset-config FILE] <command> [options]\n\n"
      "Commands: train, eval, play, plot, data\n"
      "Run supermimic <command> --help for command options."
    )
    return 0
  if args[0] not in commands:
    parser.error(f"Unknown command: {args[0]}")
  try:
    assets = load_assets(options.asset_config)
    module = importlib.import_module("supermimic.cli." + commands[args[0]])
    return module.main(args[1:], assets=assets) or 0
  except (ValueError, FileNotFoundError) as error:
    print(f"Error: {error}", file=sys.stderr)
    return 1
