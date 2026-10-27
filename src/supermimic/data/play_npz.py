"""Play a retargeted Unitree G1 NPZ trajectory directly in a browser.

The player writes one recorded ``qpos`` frame at a time into MuJoCo and runs only
forward kinematics. It does not create a tracking environment, execute a policy,
apply controls, or integrate physics. Use ``--fps`` to compare the same retargeted
frames at different assumed sampling rates.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path

import mjlab
import mujoco
import numpy as np
import numpy.typing as npt
import tyro
import viser
from mjlab.asset_zoo.robots import get_g1_robot_cfg
from mjlab.scene import Scene, SceneCfg
from mjviser import ViserMujocoScene


@dataclass(frozen=True)
class NpzMotion:
  """G1 generalized positions and optional timing metadata."""

  path: Path
  qpos: npt.NDArray[np.float64]
  embedded_fps: float | None


@dataclass(frozen=True)
class NpzPlaybackConfig:
  """Command-line configuration for direct G1 NPZ playback."""

  input_file: tyro.conf.Positional[str]
  fps: float | None = None
  """Playback FPS override. It takes precedence over an embedded fps value."""
  fallback_fps: float = 30.0
  """FPS used when the NPZ has no fps field, as in LAFAN1 G1 rtgt files."""
  port: int = 8080
  loop: bool = True
  center_root: bool = True
  """Keep the root horizontally centered so locomotion stays in view."""


def _read_embedded_fps(data: np.lib.npyio.NpzFile, path: Path) -> float | None:
  if "fps" not in data:
    return None
  fps_values = np.asarray(data["fps"])
  if fps_values.size != 1:
    raise ValueError(
      f"NPZ fps must contain one value in {path}, got {fps_values.shape}"
    )
  fps = float(fps_values.reshape(-1)[0])
  if not np.isfinite(fps) or fps <= 0:
    raise ValueError(f"NPZ fps must be finite and positive in {path}, got {fps}")
  return fps


def load_npz_motion(path: str | Path) -> NpzMotion:
  """Load a G1 qpos NPZ or reconstruct qpos from a tracking NPZ."""
  npz_path = Path(path).expanduser().resolve()
  if not npz_path.is_file():
    raise FileNotFoundError(f"NPZ file not found: {npz_path}")

  try:
    with np.load(npz_path, allow_pickle=False) as data:
      embedded_fps = _read_embedded_fps(data, npz_path)
      if "qpos" in data:
        qpos = np.asarray(data["qpos"], dtype=np.float64)
      else:
        required = ("joint_pos", "body_pos_w", "body_quat_w")
        missing = [key for key in required if key not in data]
        if missing:
          raise ValueError(
            f"NPZ {npz_path} must contain qpos or tracking arrays; missing {missing}"
          )
        joint_pos = np.asarray(data["joint_pos"], dtype=np.float64)
        body_pos = np.asarray(data["body_pos_w"], dtype=np.float64)
        body_quat = np.asarray(data["body_quat_w"], dtype=np.float64)
        if joint_pos.ndim != 2:
          raise ValueError(
            f"NPZ joint_pos must have shape [frames, joints], got {joint_pos.shape}"
          )
        frame_count = joint_pos.shape[0]
        if body_pos.ndim != 3 or body_pos.shape != (frame_count, body_pos.shape[1], 3):
          raise ValueError(
            f"NPZ body_pos_w must have shape [frames, bodies, 3], got {body_pos.shape}"
          )
        if body_quat.ndim != 3 or body_quat.shape != (
          frame_count,
          body_pos.shape[1],
          4,
        ):
          raise ValueError(
            f"NPZ body_quat_w must have shape [frames, bodies, 4], got {body_quat.shape}"
          )
        if body_pos.shape[1] == 0:
          raise ValueError(f"NPZ tracking arrays contain no bodies: {npz_path}")
        qpos = np.concatenate((body_pos[:, 0], body_quat[:, 0], joint_pos), axis=1)
  except OSError as exc:
    raise ValueError(f"Failed to load NPZ file {npz_path}: {exc}") from exc

  if qpos.ndim != 2 or qpos.shape[1] != 36:
    raise ValueError(
      f"G1 qpos must have shape [frames, 36] "
      f"(root xyz + root wxyz + 29 joints), got {qpos.shape}"
    )
  if qpos.shape[0] == 0:
    raise ValueError(f"NPZ motion contains no frames: {npz_path}")
  finite = np.isfinite(qpos)
  if not finite.all():
    first_index = tuple(int(item) for item in np.argwhere(~finite)[0])
    raise ValueError(f"NPZ qpos contains a non-finite value at {first_index}")

  return NpzMotion(path=npz_path, qpos=qpos, embedded_fps=embedded_fps)


class NpzViewer:
  """Viser player that applies recorded G1 qpos frames without simulation steps."""

  def __init__(self, motion: NpzMotion, cfg: NpzPlaybackConfig) -> None:
    playback_fps = (
      cfg.fps
      if cfg.fps is not None
      else motion.embedded_fps
      if motion.embedded_fps is not None
      else cfg.fallback_fps
    )
    if not np.isfinite(playback_fps) or playback_fps <= 0:
      raise ValueError("Playback FPS must be finite and positive")
    if not np.isfinite(cfg.fallback_fps) or cfg.fallback_fps <= 0:
      raise ValueError("Fallback FPS must be finite and positive")
    if not 1 <= cfg.port <= 65535:
      raise ValueError("Port must be between 1 and 65535")

    self.motion = motion
    self.server = viser.ViserServer(port=cfg.port, label="G1 NPZ Player")
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

    robot_scene = Scene(
      SceneCfg(num_envs=1, entities={"robot": get_g1_robot_cfg()}), device="cpu"
    )
    self.model = robot_scene.compile()
    if self.model.nq != motion.qpos.shape[1]:
      raise ValueError(
        f"G1 model expects nq={self.model.nq}, but NPZ contains "
        f"{motion.qpos.shape[1]} qpos values"
      )
    self.data = mujoco.MjData(self.model)
    self.scene = ViserMujocoScene(self.server, self.model, num_envs=1)

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

    self._setup_gui()
    self._setup_camera()
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
            np.clip(self._frame + increment, 0, len(self.motion.qpos) - 1)
          )
          self._anchor_frame = float(self._frame)
          self._anchor_time = time.perf_counter()
          frame = self._frame
        self._render_frame(frame)

      self._frame_slider = self.server.gui.add_slider(
        "Frame",
        min=0,
        max=len(self.motion.qpos) - 1,
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
        hint="Try 30 and 50 to compare source-time and assumed retarget-time playback.",
      )

      @fps_input.on_update
      def _(_) -> None:
        with self._lock:
          self._frame = self._clock_frame(time.perf_counter())
          self._fps = float(fps_input.value)
          self._anchor_frame = float(self._frame)
          self._anchor_time = time.perf_counter()
          self._last_rendered_frame = -1

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
          self._last_rendered_frame = -1

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

  def _setup_camera(self) -> None:
    @self.server.on_client_connect
    def _(client: viser.ClientHandle) -> None:
      client.camera.look_at = np.array([0.0, 0.0, 0.9])
      client.camera.position = np.array([3.0, -3.0, 1.8])
      client.camera.up_direction = np.array([0.0, 0.0, 1.0])

  def _clock_frame(self, now: float) -> int:
    elapsed_frames = (now - self._anchor_time) * self._fps * self._speed
    raw_frame = int(self._anchor_frame + elapsed_frames)
    if self._loop:
      return raw_frame % len(self.motion.qpos)
    return min(raw_frame, len(self.motion.qpos) - 1)

  def _render_frame(self, frame: int) -> None:
    qpos = self.motion.qpos[frame].copy()
    if self._center_root:
      qpos[:2] = 0.0
    self.data.qpos[:] = qpos
    self.data.qvel[:] = 0.0
    mujoco.mj_forward(self.model, self.data)
    with self.server.atomic():
      self.scene.update_from_mjdata(self.data)
      if self._frame_slider.value != frame:
        self._frame_slider.value = frame
      current_time = frame / self._fps
      effective_duration = len(self.motion.qpos) / self._fps
      timing_source = (
        f"embedded {self.motion.embedded_fps:g} FPS"
        if self.motion.embedded_fps is not None
        else "no embedded FPS"
      )
      self._info.content = f"""
        <strong>{self.motion.path.name}</strong><br/>
        Frame: {frame + 1} / {len(self.motion.qpos)}<br/>
        Time: {current_time:.2f} s / {effective_duration:.2f} s<br/>
        NPZ timing: {timing_source}<br/>
        Playback: {self._fps:.3f} FPS × {self._speed:g}<br/>
        Mode: direct qpos + forward kinematics (no control or physics step)
      """
    self._last_rendered_frame = frame

  def run(self) -> None:
    """Run until interrupted, advancing qpos frames from monotonic wall time."""
    timing_source = (
      f"embedded {self.motion.embedded_fps:g} FPS"
      if self.motion.embedded_fps is not None
      else "no embedded FPS"
    )
    print(
      f"[INFO] Loaded {self.motion.path.name}: {len(self.motion.qpos)} frames, "
      f"{timing_source}; playing at {self._fps:g} FPS "
      f"({len(self.motion.qpos) / self._fps:.2f} s)"
    )
    print("[INFO] Direct qpos playback: no controller and no physics integration.")
    print("[INFO] Open the Viser URL above. Press Ctrl+C to stop.")
    try:
      while True:
        with self._lock:
          if self._playing:
            frame = self._clock_frame(time.perf_counter())
            if not self._loop and frame == len(self.motion.qpos) - 1:
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
      print("\n[INFO] Shutting down NPZ viewer.")
    finally:
      self.server.stop()


def main() -> None:
  cfg = tyro.cli(
    NpzPlaybackConfig,
    description=__doc__,
    config=mjlab.TYRO_FLAGS,
  )
  motion = load_npz_motion(cfg.input_file)
  NpzViewer(motion, cfg).run()


if __name__ == "__main__":
  main()
