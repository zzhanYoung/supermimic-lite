#!/usr/bin/env bash

# Generate type stubs for MuJoCo using pybind11-stubgen.
# These local development files are ignored by Git and excluded from releases.
# make check generates them before running ty and pyright.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
STUB_DIR="$PROJECT_ROOT/typings"
cd "$PROJECT_ROOT"
export MUJOCO_GL="${MUJOCO_GL:-disable}"

# --no-sync runs in the already-synced project environment (so mujoco is
# importable) without disturbing it; the conflicting cpu/cu128 extras mean a
# plain "uv run" would re-sync and churn the environment.
# The generator version is pinned in the dev dependency group and uv.lock.
STUBGEN=(uv run --no-sync pybind11-stubgen)

echo "Generating MuJoCo type stubs..."
"${STUBGEN[@]}" mujoco -o "$STUB_DIR" --ignore-all-errors

# mujoco.viewer pulls in GLFW, which may not import on a headless machine.
# It is optional for the project type checks.
if ! "${STUBGEN[@]}" mujoco.viewer -o "$STUB_DIR" --ignore-all-errors; then
  echo "Warning: could not generate mujoco.viewer stubs (no GLFW?). Skipping."
fi

echo "Post-processing stubs to make them platform-independent..."
uv run --no-sync python "$SCRIPT_DIR/postprocess_stubs.py"

echo "MuJoCo stubs generated successfully in $STUB_DIR/mujoco/"
