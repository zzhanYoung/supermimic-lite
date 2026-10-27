"""Local asset locations, independent of the package and of future hosting."""

from __future__ import annotations

import json
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class LocalAssets:
  motion_folder: Path | None = None
  checkpoint: Path | None = None
  run_dir: Path | None = None
  plot_sources: tuple[Path, ...] = ()
  plot_config: Path | None = None
  protected_roots: tuple[Path, ...] = ()

  def checkpoint_run_dir(self, checkpoint: Path, override: Path | None = None) -> Path:
    if override is not None:
      return override.expanduser().resolve()
    if (
      self.checkpoint is not None and checkpoint.resolve() == self.checkpoint.resolve()
    ):
      return self.run_dir or checkpoint.parent
    return checkpoint.parent

  def protect_output(self, path: Path, *inputs: Path | None) -> Path:
    output = path.expanduser().resolve()
    roots = list(self.protected_roots)
    roots.extend(p.resolve() for p in inputs if p is not None)
    for root in roots:
      if output == root or root in output.parents:
        raise ValueError(f"Output would modify preserved input assets: {output}")
    return output


DEFAULT_ASSETS = LocalAssets()


def load_assets(path: str | Path | None = None) -> LocalAssets:
  config = Path(path or ".supermimic.local.toml").expanduser().resolve()
  if not config.exists() and path is None:
    return LocalAssets()
  with config.open("rb") as stream:
    values = tomllib.load(stream)
  unknown = values.keys() - LocalAssets.__dataclass_fields__.keys()
  if unknown:
    raise ValueError(f"Unknown asset configuration keys: {sorted(unknown)}")

  def resolve(value: str) -> Path:
    return (config.parent / Path(value).expanduser()).resolve()

  resolved: dict[str, Any] = {
    key: tuple(resolve(item) for item in value)
    if key in ("plot_sources", "protected_roots")
    else resolve(value)
    for key, value in values.items()
  }
  return LocalAssets(**resolved)


def require_path(value: Path | str | None, flag: str, *, directory=False) -> Path:
  if value is None:
    raise ValueError(f"Set {flag} or configure .supermimic.local.toml first.")
  path = Path(value).expanduser().resolve()
  if not (path.is_dir() if directory else path.is_file()):
    raise FileNotFoundError(f"{flag}: {path}")
  return path


def inventory(roots: dict[str, Path]) -> dict:
  """Portable inventory: logical roots and relative names, no host paths."""
  files = []
  for label, root in roots.items():
    for path in sorted(root.rglob("*")):
      if path.is_file():
        files.append(
          {
            "root": label,
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
          }
        )
  bundles = []
  for label, root in roots.items():
    for model in sorted(root.rglob("model_*.pt")):
      if ".replay." in model.name:
        continue
      bundles.append(
        {
          "root": label,
          "checkpoint": model.relative_to(root).as_posix(),
          "replay": [
            p.relative_to(root).as_posix()
            for p in sorted(model.parent.glob(model.stem + ".rank*.replay.pt"))
          ],
          "sampling": model.with_suffix(".sampling.npz").relative_to(root).as_posix()
          if model.with_suffix(".sampling.npz").exists()
          else None,
          "config": [
            p.relative_to(root).as_posix()
            for p in sorted((model.parent / "params").glob("*.yaml"))
          ],
        }
      )
  return {"format_version": 1, "files": files, "checkpoints": bundles}


def write_inventory(path: Path, roots: dict[str, Path]) -> None:
  path.write_text(json.dumps(inventory(roots), indent=2) + "\n")
