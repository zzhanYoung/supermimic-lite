"""Play a BVH motion at its declared frame rate in a browser.

The viewer renders the source skeleton directly, without retargeting or physics.
This makes it useful for checking whether a converted motion has accidentally
changed speed. By default, playback uses the BVH ``Frame Time`` value; ``--fps``
can override it for side-by-side timing experiments.
"""

from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import mjlab
import numpy as np
import numpy.typing as npt
import tyro
import viser


@dataclass(frozen=True)
class BvhJoint:
  """One node in a BVH hierarchy."""

  name: str
  parent: int | None
  offset: npt.NDArray[np.float64]
  channels: tuple[str, ...]
  channel_start: int


@dataclass(frozen=True)
class BvhMotion:
  """Parsed BVH hierarchy and motion samples."""

  path: Path
  joints: tuple[BvhJoint, ...]
  frames: npt.NDArray[np.float64]
  frame_time: float

  @property
  def fps(self) -> float:
    return 1.0 / self.frame_time

  @property
  def duration(self) -> float:
    return len(self.frames) * self.frame_time


@dataclass(frozen=True)
class BvhPlaybackConfig:
  """Command-line configuration for BVH playback."""

  input_file: tyro.conf.Positional[str]
  fps: float | None = None
  """Playback FPS override. By default, use the BVH Frame Time value."""
  scale: float = 0.01
  """Convert BVH position units to meters; LAFAN1 uses centimeters."""
  port: int = 8080
  loop: bool = True
  center_root: bool = True
  """Keep the root horizontally centered so locomotion stays in view."""


@dataclass
class _JointBuilder:
  name: str
  parent: int | None
  offset: npt.NDArray[np.float64]
  channels: tuple[str, ...] = ()
  channel_start: int = 0


_FRAME_COUNT_PATTERN = re.compile(r"^Frames\s*:\s*(\d+)\s*$", re.IGNORECASE)
_FRAME_TIME_PATTERN = re.compile(
  r"^Frame\s+Time\s*:\s*([+\-0-9.eE]+)\s*$", re.IGNORECASE
)
_VALID_CHANNELS = {
  "xposition",
  "yposition",
  "zposition",
  "xrotation",
  "yrotation",
  "zrotation",
}


def load_bvh(path: str | Path) -> BvhMotion:
  """Load a BVH file, preserving its channel order and declared frame time."""
  bvh_path = Path(path).expanduser().resolve()
  if not bvh_path.is_file():
    raise FileNotFoundError(f"BVH file not found: {bvh_path}")

  try:
    lines = bvh_path.read_text().splitlines()
  except UnicodeDecodeError as exc:
    raise ValueError(f"BVH file is not valid text: {bvh_path}") from exc

  try:
    motion_line = next(
      index for index, line in enumerate(lines) if line.strip().upper() == "MOTION"
    )
  except StopIteration as exc:
    raise ValueError(f"BVH file has no MOTION section: {bvh_path}") from exc

  builders: list[_JointBuilder] = []
  stack: list[int] = []
  pending_joint: int | None = None
  channel_count = 0
  end_site_counts: dict[str, int] = {}

  for line_number, raw_line in enumerate(lines[:motion_line], start=1):
    line = raw_line.strip()
    if not line or line.upper() == "HIERARCHY":
      continue
    if line.startswith("ROOT ") or line.startswith("JOINT "):
      name = line.split(maxsplit=1)[1]
      parent = stack[-1] if stack else None
      pending_joint = len(builders)
      builders.append(
        _JointBuilder(
          name=name,
          parent=parent,
          offset=np.zeros(3, dtype=np.float64),
        )
      )
      continue
    if line.lower() == "end site":
      if not stack:
        raise ValueError(f"End Site outside a joint at line {line_number}")
      parent_name = builders[stack[-1]].name
      count = end_site_counts.get(parent_name, 0) + 1
      end_site_counts[parent_name] = count
      suffix = "end" if count == 1 else f"end_{count}"
      pending_joint = len(builders)
      builders.append(
        _JointBuilder(
          name=f"{parent_name}_{suffix}",
          parent=stack[-1],
          offset=np.zeros(3, dtype=np.float64),
        )
      )
      continue
    if line == "{":
      if pending_joint is None:
        raise ValueError(f"Unexpected '{{' at line {line_number}")
      stack.append(pending_joint)
      pending_joint = None
      continue
    if line == "}":
      if not stack:
        raise ValueError(f"Unexpected '}}' at line {line_number}")
      stack.pop()
      continue
    if line.startswith("OFFSET "):
      if not stack:
        raise ValueError(f"OFFSET outside a joint at line {line_number}")
      values = line.split()[1:]
      if len(values) != 3:
        raise ValueError(f"OFFSET must contain 3 values at line {line_number}")
      try:
        builders[stack[-1]].offset = np.asarray(values, dtype=np.float64)
      except ValueError as exc:
        raise ValueError(f"Invalid OFFSET at line {line_number}: {line}") from exc
      continue
    if line.startswith("CHANNELS "):
      if not stack:
        raise ValueError(f"CHANNELS outside a joint at line {line_number}")
      parts = line.split()
      try:
        declared_count = int(parts[1])
      except (IndexError, ValueError) as exc:
        raise ValueError(f"Invalid CHANNELS at line {line_number}: {line}") from exc
      channels = tuple(parts[2:])
      if len(channels) != declared_count:
        raise ValueError(
          f"CHANNELS declares {declared_count} values but lists {len(channels)} "
          f"at line {line_number}"
        )
      unknown = [
        channel for channel in channels if channel.lower() not in _VALID_CHANNELS
      ]
      if unknown:
        raise ValueError(f"Unsupported BVH channels at line {line_number}: {unknown}")
      builder = builders[stack[-1]]
      builder.channels = channels
      builder.channel_start = channel_count
      channel_count += declared_count
      continue
    raise ValueError(f"Unsupported BVH hierarchy syntax at line {line_number}: {line}")

  if stack:
    raise ValueError(f"Unclosed BVH hierarchy in {bvh_path}")
  if not builders or builders[0].parent is not None:
    raise ValueError(f"BVH hierarchy has no root joint: {bvh_path}")
  if channel_count == 0:
    raise ValueError(f"BVH hierarchy contains no animation channels: {bvh_path}")

  motion_headers = [
    (index + motion_line + 2, line.strip())
    for index, line in enumerate(lines[motion_line + 1 :])
    if line.strip()
  ]
  if len(motion_headers) < 3:
    raise ValueError(f"BVH MOTION section is incomplete: {bvh_path}")

  frame_count_match = _FRAME_COUNT_PATTERN.match(motion_headers[0][1])
  frame_time_match = _FRAME_TIME_PATTERN.match(motion_headers[1][1])
  if frame_count_match is None:
    raise ValueError(f"Invalid BVH frame count header: {motion_headers[0][1]}")
  if frame_time_match is None:
    raise ValueError(f"Invalid BVH frame time header: {motion_headers[1][1]}")
  frame_count = int(frame_count_match.group(1))
  frame_time = float(frame_time_match.group(1))
  if frame_count <= 0:
    raise ValueError("BVH frame count must be positive")
  if not np.isfinite(frame_time) or frame_time <= 0:
    raise ValueError("BVH frame time must be finite and positive")

  motion_text = "\n".join(line for _, line in motion_headers[2:])
  values = np.fromstring(motion_text, dtype=np.float64, sep=" ")
  expected_values = frame_count * channel_count
  if values.size != expected_values:
    raise ValueError(
      f"BVH declares {frame_count} frames with {channel_count} channels "
      f"({expected_values} values), but contains {values.size}"
    )
  frames = values.reshape(frame_count, channel_count)
  if not np.isfinite(frames).all():
    first_index = tuple(int(item) for item in np.argwhere(~np.isfinite(frames))[0])
    raise ValueError(f"BVH contains a non-finite motion value at {first_index}")

  joints = tuple(
    BvhJoint(
      name=builder.name,
      parent=builder.parent,
      offset=builder.offset,
      channels=builder.channels,
      channel_start=builder.channel_start,
    )
    for builder in builders
  )
  return BvhMotion(
    path=bvh_path,
    joints=joints,
    frames=frames,
    frame_time=frame_time,
  )


def _axis_rotation(axis: str, angle_degrees: float) -> npt.NDArray[np.float64]:
  angle = np.deg2rad(angle_degrees)
  cosine = float(np.cos(angle))
  sine = float(np.sin(angle))
  if axis == "x":
    return np.array([[1.0, 0.0, 0.0], [0.0, cosine, -sine], [0.0, sine, cosine]])
  if axis == "y":
    return np.array([[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]])
  if axis == "z":
    return np.array([[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]])
  raise ValueError(f"Unknown rotation axis: {axis}")


def joint_positions(
  motion: BvhMotion,
  frame: int,
  *,
  scale: float = 1.0,
  z_up: bool = True,
) -> npt.NDArray[np.float64]:
  """Compute world joint positions for one frame using BVH channel order."""
  if frame < 0 or frame >= len(motion.frames):
    raise IndexError(f"Frame {frame} is outside [0, {len(motion.frames) - 1}]")
  if not np.isfinite(scale) or scale <= 0:
    raise ValueError("scale must be finite and positive")

  frame_values = motion.frames[frame]
  positions = np.zeros((len(motion.joints), 3), dtype=np.float64)
  rotations = np.zeros((len(motion.joints), 3, 3), dtype=np.float64)

  for joint_index, joint in enumerate(motion.joints):
    has_position_channels = any(
      channel.lower().endswith("position") for channel in joint.channels
    )
    # LAFAN1 repeats its first root position in ROOT/OFFSET. Its reference loader
    # treats root position channels as absolute values, so do the same here.
    local_position = (
      np.zeros(3, dtype=np.float64)
      if joint.parent is None and has_position_channels
      else joint.offset.copy()
    )
    local_rotation = np.eye(3, dtype=np.float64)
    for channel_offset, channel in enumerate(joint.channels):
      value = frame_values[joint.channel_start + channel_offset]
      normalized_channel = channel.lower()
      axis = normalized_channel[0]
      axis_index = "xyz".index(axis)
      if normalized_channel.endswith("position"):
        local_position[axis_index] += value
      else:
        local_rotation = local_rotation @ _axis_rotation(axis, value)

    if joint.parent is None:
      positions[joint_index] = local_position
      rotations[joint_index] = local_rotation
    else:
      parent_rotation = rotations[joint.parent]
      positions[joint_index] = (
        positions[joint.parent] + parent_rotation @ local_position
      )
      rotations[joint_index] = parent_rotation @ local_rotation

  if z_up:
    # BVH/LAFAN1 is Y-up. This proper rotation maps (x, y, z) -> (x, -z, y).
    positions = positions[:, (0, 2, 1)]
    positions[:, 1] *= -1.0
  return positions * scale


class BvhViewer:
  """Small Viser-based skeleton player with an explicit playback clock."""

  def __init__(self, motion: BvhMotion, cfg: BvhPlaybackConfig) -> None:
    playback_fps = motion.fps if cfg.fps is None else cfg.fps
    if not np.isfinite(playback_fps) or playback_fps <= 0:
      raise ValueError("Playback FPS must be finite and positive")
    if not np.isfinite(cfg.scale) or cfg.scale <= 0:
      raise ValueError("Scale must be finite and positive")
    if not 1 <= cfg.port <= 65535:
      raise ValueError("Port must be between 1 and 65535")

    self.motion = motion
    self.scale = cfg.scale
    self.server = viser.ViserServer(port=cfg.port, label="BVH Player")
    self.server.scene.set_up_direction("+z")
    self.server.scene.world_axes.visible = True
    self.server.scene.add_grid(
      "/ground",
      width=20.0,
      height=20.0,
      cell_size=0.25,
      section_size=1.0,
      plane="xy",
    )

    self._lock = threading.Lock()
    self._frame = 0
    self._fps = float(playback_fps)
    self._speed = 1.0
    self._playing = True
    self._loop = cfg.loop
    self._center_root = cfg.center_root
    self._anchor_frame = 0.0
    self._anchor_time = time.perf_counter()
    self._last_rendered_frame = -1

    self._edge_indices = np.asarray(
      [
        (joint.parent, index)
        for index, joint in enumerate(motion.joints)
        if joint.parent is not None
      ],
      dtype=np.int64,
    )
    initial_positions = self._display_positions(0)
    self._bones = self.server.scene.add_line_segments(
      "/skeleton/bones",
      initial_positions[self._edge_indices],
      colors=(70, 150, 255),
      thickness=5.0,
      thickness_units="screen",
    )
    self._joints = self.server.scene.add_point_cloud(
      "/skeleton/joints",
      initial_positions,
      colors=(255, 180, 40),
      point_size=0.04,
      point_shape="circle",
      precision="float32",
    )
    self._setup_gui()
    self._setup_camera(initial_positions)
    self._render_frame(0)

  def _setup_gui(self) -> None:
    self._info = self.server.gui.add_html("")
    with self.server.gui.add_folder("Playback"):
      play_pause = self.server.gui.add_button("Play / Pause")

      @play_pause.on_click
      def _(_) -> None:
        with self._lock:
          now = time.perf_counter()
          if self._playing:
            self._frame = self._clock_frame(now)
            self._playing = False
          else:
            self._anchor_frame = float(self._frame)
            self._anchor_time = now
            self._playing = True

      step_buttons = self.server.gui.add_button_group(
        "Step", options=("Previous", "Next")
      )

      @step_buttons.on_click
      def _(event) -> None:
        increment = -1 if event.target.value == "Previous" else 1
        with self._lock:
          self._playing = False
          self._frame = int(
            np.clip(self._frame + increment, 0, len(self.motion.frames) - 1)
          )
          self._anchor_frame = float(self._frame)
          self._anchor_time = time.perf_counter()
          frame = self._frame
        self._render_frame(frame)

      self._frame_slider = self.server.gui.add_slider(
        "Frame",
        min=0,
        max=len(self.motion.frames) - 1,
        step=1,
        initial_value=0,
      )

      @self._frame_slider.on_update
      def _(_) -> None:
        with self._lock:
          self._frame = int(self._frame_slider.value)
          self._anchor_frame = float(self._frame)
          self._anchor_time = time.perf_counter()
          frame = self._frame
        self._render_frame(frame)

      fps_input = self.server.gui.add_number(
        "Playback FPS",
        initial_value=self._fps,
        min=0.1,
        max=1000.0,
        step=0.1,
        hint="Use 30 for source LAFAN1; try 50 to reproduce the suspected speedup.",
      )

      @fps_input.on_update
      def _(_) -> None:
        with self._lock:
          self._frame = self._clock_frame(time.perf_counter())
          self._fps = float(fps_input.value)
          self._anchor_frame = float(self._frame)
          self._anchor_time = time.perf_counter()

      speed_input = self.server.gui.add_dropdown(
        "Speed", options=("0.25x", "0.5x", "1x", "2x", "4x"), initial_value="1x"
      )

      @speed_input.on_update
      def _(_) -> None:
        with self._lock:
          self._frame = self._clock_frame(time.perf_counter())
          self._speed = float(speed_input.value.removesuffix("x"))
          self._anchor_frame = float(self._frame)
          self._anchor_time = time.perf_counter()

      loop_checkbox = self.server.gui.add_checkbox("Loop", initial_value=self._loop)

      @loop_checkbox.on_update
      def _(_) -> None:
        with self._lock:
          self._loop = loop_checkbox.value

      center_checkbox = self.server.gui.add_checkbox(
        "Center root", initial_value=self._center_root
      )

      @center_checkbox.on_update
      def _(_) -> None:
        with self._lock:
          self._center_root = center_checkbox.value
          frame = self._frame
        self._render_frame(frame)

  def _setup_camera(self, positions: npt.NDArray[np.float64]) -> None:
    center = (positions.min(axis=0) + positions.max(axis=0)) / 2.0
    extent = max(float(np.ptp(positions, axis=0).max()), 1.0)

    @self.server.on_client_connect
    def _(client: viser.ClientHandle) -> None:
      client.camera.look_at = center
      client.camera.position = center + extent * np.array([2.0, -2.5, 1.2])
      client.camera.up_direction = np.array([0.0, 0.0, 1.0])

  def _clock_frame(self, now: float) -> int:
    elapsed_frames = (now - self._anchor_time) * self._fps * self._speed
    raw_frame = int(self._anchor_frame + elapsed_frames)
    if self._loop:
      return raw_frame % len(self.motion.frames)
    return min(raw_frame, len(self.motion.frames) - 1)

  def _display_positions(self, frame: int) -> npt.NDArray[np.float64]:
    positions = joint_positions(self.motion, frame, scale=self.scale)
    if self._center_root:
      positions[:, :2] -= positions[0, :2]
    return positions

  def _render_frame(self, frame: int) -> None:
    positions = self._display_positions(frame)
    with self.server.atomic():
      self._bones.points = positions[self._edge_indices].astype(np.float32)
      self._joints.points = positions.astype(np.float32)
      if self._frame_slider.value != frame:
        self._frame_slider.value = frame
      current_time = frame / self._fps
      effective_duration = len(self.motion.frames) / self._fps
      self._info.content = f"""
        <strong>{self.motion.path.name}</strong><br/>
        Frame: {frame + 1} / {len(self.motion.frames)}<br/>
        Time: {current_time:.2f} s / {effective_duration:.2f} s<br/>
        BVH timing: {self.motion.fps:.3f} FPS ({self.motion.frame_time:.6f} s/frame)<br/>
        Playback: {self._fps:.3f} FPS × {self._speed:g}
      """
    self._last_rendered_frame = frame

  def run(self) -> None:
    """Run until interrupted, advancing frames from monotonic wall time."""
    print(
      f"[INFO] Loaded {self.motion.path.name}: {len(self.motion.frames)} frames, "
      f"{self.motion.fps:.3f} FPS, {self.motion.duration:.2f} s"
    )
    print("[INFO] Open the Viser URL above. Press Ctrl+C to stop.")
    try:
      while True:
        with self._lock:
          if self._playing:
            now = time.perf_counter()
            frame = self._clock_frame(now)
            if not self._loop and frame == len(self.motion.frames) - 1:
              self._playing = False
              self._frame = frame
            else:
              self._frame = frame
          else:
            frame = self._frame
        if frame != self._last_rendered_frame:
          self._render_frame(frame)
        time.sleep(0.002)
    except KeyboardInterrupt:
      print("\n[INFO] Shutting down BVH viewer.")
    finally:
      self.server.stop()


def main() -> None:
  cfg = tyro.cli(
    BvhPlaybackConfig,
    description=__doc__,
    config=mjlab.TYRO_FLAGS,
  )
  motion = load_bvh(cfg.input_file)
  BvhViewer(motion, cfg).run()


if __name__ == "__main__":
  main()
