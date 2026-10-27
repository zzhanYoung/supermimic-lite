"""Manifest metadata for converted motion datasets."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

MANIFEST_FILENAME = "manifest.json"
MANIFEST_VERSION = 1


@dataclass(frozen=True)
class MotionManifestEntry:
  """Metadata for one tracking motion."""

  path: str
  frames: int | None = None
  fps: float | None = None
  arrays: dict[str, dict[str, Any]] | None = None
  size_bytes: int | None = None


@dataclass(frozen=True)
class MotionManifest:
  """A lightweight, relocatable description of a motion dataset."""

  motions: tuple[MotionManifestEntry, ...]
  fingerprint: str
  version: int = MANIFEST_VERSION

  def to_dict(self) -> dict[str, Any]:
    return {
      "version": self.version,
      "dataset_fingerprint": self.fingerprint,
      "motions": [asdict(entry) for entry in self.motions],
    }


def _fingerprint_entries(entries: tuple[MotionManifestEntry, ...]) -> str:
  payload = json.dumps(
    [asdict(entry) for entry in entries],
    sort_keys=True,
    separators=(",", ":"),
  ).encode()
  return hashlib.sha256(payload).hexdigest()


def build_motion_manifest(motion_folder: str | Path) -> MotionManifest:
  """Inspect all top-level NPZ files and build their manifest metadata."""
  folder = Path(motion_folder).expanduser().resolve()
  if not folder.is_dir():
    raise NotADirectoryError(f"Motion folder not found: {folder}")

  entries: list[MotionManifestEntry] = []
  for path in sorted(folder.glob("*.npz"), key=lambda item: item.name):
    if not path.is_file():
      continue
    try:
      with np.load(path, allow_pickle=False) as data:
        arrays = {
          key: {"shape": list(data[key].shape), "dtype": str(data[key].dtype)}
          for key in sorted(data.files)
        }
        if "joint_pos" in data:
          frames = int(data["joint_pos"].shape[0])
        elif "qpos" in data:
          frames = int(data["qpos"].shape[0])
        else:
          frames = None
        fps = float(np.asarray(data["fps"]).reshape(-1)[0]) if "fps" in data else None
    except (OSError, ValueError) as exc:
      raise ValueError(f"Failed to inspect motion file {path}: {exc}") from exc
    entries.append(
      MotionManifestEntry(
        path=path.relative_to(folder).as_posix(),
        frames=frames,
        fps=fps,
        arrays=arrays,
        size_bytes=path.stat().st_size,
      )
    )

  if not entries:
    raise ValueError(f"Motion folder contains no .npz files: {folder}")
  motion_entries = tuple(entries)
  return MotionManifest(
    motions=motion_entries,
    fingerprint=_fingerprint_entries(motion_entries),
  )


def write_motion_manifest(
  motion_folder: str | Path, manifest_path: str | Path | None = None
) -> Path:
  """Build and atomically write a dataset manifest."""
  folder = Path(motion_folder).expanduser().resolve()
  output = (
    Path(manifest_path).expanduser().resolve()
    if manifest_path is not None
    else folder / MANIFEST_FILENAME
  )
  manifest = build_motion_manifest(folder)
  output.parent.mkdir(parents=True, exist_ok=True)
  temporary = output.with_suffix(output.suffix + ".tmp")
  temporary.write_text(json.dumps(manifest.to_dict(), indent=2) + "\n")
  temporary.replace(output)
  return output


def load_motion_manifest(
  motion_folder: str | Path,
  manifest_path: str | Path | None = None,
  *,
  require_manifest: bool = False,
) -> tuple[MotionManifest, tuple[str, ...]]:
  """Load a manifest, or create lightweight entries from the folder listing."""
  folder = Path(motion_folder).expanduser().resolve()
  if not folder.is_dir():
    raise NotADirectoryError(f"Motion folder not found: {folder}")
  path = (
    Path(manifest_path).expanduser().resolve()
    if manifest_path is not None
    else folder / MANIFEST_FILENAME
  )

  if path.is_file():
    try:
      raw = json.loads(path.read_text())
      version = int(raw.get("version", MANIFEST_VERSION))
      if version != MANIFEST_VERSION:
        raise ValueError(
          f"Unsupported manifest version {version}; expected {MANIFEST_VERSION}."
        )
      entries = tuple(MotionManifestEntry(**entry) for entry in raw["motions"])
      recorded_fingerprint = str(
        raw.get("dataset_fingerprint", raw.get("fingerprint", ""))
      )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
      raise ValueError(f"Invalid motion manifest {path}: {exc}") from exc
    if not entries:
      raise ValueError(f"Motion manifest contains no motions: {path}")
    computed_fingerprint = _fingerprint_entries(entries)
    if recorded_fingerprint and recorded_fingerprint != computed_fingerprint:
      raise ValueError(
        f"Motion manifest fingerprint mismatch in {path}: "
        f"recorded {recorded_fingerprint}, computed {computed_fingerprint}."
      )
    fingerprint = recorded_fingerprint or computed_fingerprint
  elif manifest_path is not None or require_manifest:
    raise FileNotFoundError(f"Motion manifest not found: {path}")
  else:
    entries = tuple(
      MotionManifestEntry(path=item.name, size_bytes=item.stat().st_size)
      for item in sorted(folder.glob("*.npz"), key=lambda item: item.name)
      if item.is_file()
    )
    if not entries:
      raise ValueError(f"Motion folder contains no .npz files: {folder}")
    fingerprint = _fingerprint_entries(entries)

  files: list[str] = []
  for entry in entries:
    entry_path = (folder / entry.path).resolve()
    try:
      entry_path.relative_to(folder)
    except ValueError as exc:
      raise ValueError(
        f"Motion manifest entry escapes its dataset folder: {entry.path}"
      ) from exc
    if not entry_path.is_file():
      raise FileNotFoundError(
        f"Motion file listed in manifest does not exist: {entry_path}"
      )
    files.append(str(entry_path))

  return (
    MotionManifest(motions=entries, fingerprint=fingerprint),
    tuple(files),
  )
