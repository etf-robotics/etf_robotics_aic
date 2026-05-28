"""Small MJPEG server for viewing Isaac Lab camera frames without a GUI display."""

from __future__ import annotations

import json
import os
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Mapping

import numpy as np


DEFAULT_CAMERA_NAMES = ("left_camera", "center_camera", "right_camera")
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8080
DEFAULT_JPEG_QUALITY = 80
DEFAULT_UPDATE_EVERY = 1
DEFAULT_ENV_INDEX = 0
_FALSE_VALUES = {"0", "false", "no", "off"}


class _DaemonThreadingHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def default_camera_stream_enabled() -> bool:
    """Return whether the default camera stream should be enabled."""
    return os.environ.get("AIC_CAMERA_STREAM", "1").strip().lower() not in _FALSE_VALUES


def should_enable_camera_rendering() -> bool:
    """Return whether scripts should pass enable_cameras to Isaac Lab."""
    return default_camera_stream_enabled()


class CameraStreamServer:
    """Serve a labeled camera mosaic as MJPEG and latest JPEG snapshots."""

    def __init__(
        self,
        camera_names: list[str],
        host: str = "0.0.0.0",
        port: int = 8080,
        jpeg_quality: int = 80,
    ):
        self.camera_names = camera_names
        self.host = host
        self.port = port
        self.jpeg_quality = int(np.clip(jpeg_quality, 1, 100))
        self.latest_step = -1
        self.latest_time = 0.0

        self._cv2 = self._import_cv2()
        self._condition = threading.Condition()
        self._latest_jpeg: bytes | None = None
        self._httpd: _DaemonThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        display_host = "localhost" if self.host in ("0.0.0.0", "::") else self.host
        return f"http://{display_host}:{self.port}/"

    def start(self) -> None:
        if self._httpd is not None:
            return

        stream_server = self

        class Handler(BaseHTTPRequestHandler):
            server_version = "AICCameraStream/1.0"

            def do_GET(self) -> None:  # noqa: N802
                if self.path in ("", "/"):
                    self._send_index()
                elif self.path == "/stream.mjpg":
                    self._send_stream()
                elif self.path == "/latest.jpg":
                    self._send_latest_jpeg()
                elif self.path == "/healthz":
                    self._send_healthz()
                else:
                    self.send_error(HTTPStatus.NOT_FOUND, "Not found")

            def log_message(self, format: str, *args) -> None:
                return

            def _send_index(self) -> None:
                camera_list = ", ".join(stream_server.camera_names)
                html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AIC camera stream</title>
  <style>
    html, body {{ margin: 0; background: #111; color: #eee; font-family: sans-serif; }}
    header {{ padding: 10px 14px; background: #1d1d1d; }}
    img {{ display: block; width: 100vw; height: auto; image-rendering: auto; }}
    small {{ color: #aaa; }}
  </style>
</head>
<body>
  <header>AIC cameras <small>{camera_list}</small></header>
  <img src="/stream.mjpg" alt="AIC camera stream">
</body>
</html>
"""
                payload = html.encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(payload)

            def _send_healthz(self) -> None:
                payload = json.dumps(
                    {
                        "ok": True,
                        "has_frame": stream_server.has_frame,
                        "latest_step": stream_server.latest_step,
                        "latest_age_s": max(0.0, time.time() - stream_server.latest_time)
                        if stream_server.latest_time
                        else None,
                    }
                ).encode("ascii")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(payload)

            def _send_latest_jpeg(self) -> None:
                jpeg = stream_server.wait_for_frame(timeout=2.0)
                if jpeg is None:
                    self.send_error(HTTPStatus.SERVICE_UNAVAILABLE, "No camera frame has been published yet.")
                    return
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(jpeg)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(jpeg)

            def _send_stream(self) -> None:
                self.send_response(HTTPStatus.OK)
                self.send_header("Age", "0")
                self.send_header("Cache-Control", "no-cache, private")
                self.send_header("Pragma", "no-cache")
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()

                last_step = -1
                while True:
                    jpeg, step = stream_server.wait_for_new_frame(last_step, timeout=5.0)
                    if jpeg is None:
                        continue
                    last_step = step
                    try:
                        self.wfile.write(b"--frame\r\n")
                        self.wfile.write(b"Content-Type: image/jpeg\r\n")
                        self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode("ascii"))
                        self.wfile.write(jpeg)
                        self.wfile.write(b"\r\n")
                    except (BrokenPipeError, ConnectionResetError, OSError):
                        break

        self._httpd = self._bind_server(Handler)
        self.port = int(self._httpd.server_address[1])
        self._thread = threading.Thread(target=self._httpd.serve_forever, name="aic-camera-stream", daemon=True)
        self._thread.start()

    def close(self) -> None:
        if self._httpd is None:
            return
        self._httpd.shutdown()
        self._httpd.server_close()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._httpd = None
        self._thread = None

    def _bind_server(self, handler_cls: type[BaseHTTPRequestHandler]) -> _DaemonThreadingHTTPServer:
        last_error = None
        candidate_ports = [self.port] if self.port == 0 else range(self.port, self.port + 20)
        for candidate_port in candidate_ports:
            try:
                return _DaemonThreadingHTTPServer((self.host, candidate_port), handler_cls)
            except OSError as exc:
                last_error = exc
        raise RuntimeError(
            f"Could not start camera stream server on {self.host}:{self.port}-{self.port + 19}."
        ) from last_error

    @property
    def has_frame(self) -> bool:
        with self._condition:
            return self._latest_jpeg is not None

    def update(self, frames: Mapping[str, np.ndarray], step: int | None = None) -> None:
        mosaic_rgb = self._make_mosaic(frames)
        mosaic_bgr = self._cv2.cvtColor(mosaic_rgb, self._cv2.COLOR_RGB2BGR)
        ok, encoded = self._cv2.imencode(
            ".jpg",
            mosaic_bgr,
            [int(self._cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
        )
        if not ok:
            raise RuntimeError("OpenCV failed to encode the camera mosaic as JPEG.")

        with self._condition:
            self._latest_jpeg = encoded.tobytes()
            self.latest_step = self.latest_step + 1 if step is None else step
            self.latest_time = time.time()
            self._condition.notify_all()

    def wait_for_frame(self, timeout: float | None = None) -> bytes | None:
        with self._condition:
            if self._latest_jpeg is None:
                self._condition.wait(timeout=timeout)
            return self._latest_jpeg

    def wait_for_new_frame(self, last_step: int, timeout: float | None = None) -> tuple[bytes | None, int]:
        with self._condition:
            if self.latest_step <= last_step:
                self._condition.wait(timeout=timeout)
            return self._latest_jpeg, self.latest_step

    def _make_mosaic(self, frames: Mapping[str, np.ndarray]) -> np.ndarray:
        labeled_frames = []
        for camera_name in self.camera_names:
            if camera_name not in frames:
                available = ", ".join(frames.keys())
                raise KeyError(f"Missing frame for camera '{camera_name}'. Available frames: {available}")
            frame = self._as_rgb_uint8(frames[camera_name]).copy()
            self._cv2.putText(
                frame,
                camera_name,
                (8, 22),
                self._cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                self._cv2.LINE_AA,
            )
            self._cv2.putText(
                frame,
                camera_name,
                (8, 22),
                self._cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (20, 20, 20),
                1,
                self._cv2.LINE_AA,
            )
            labeled_frames.append(frame)
        return np.concatenate(labeled_frames, axis=1)

    @staticmethod
    def _as_rgb_uint8(frame: np.ndarray) -> np.ndarray:
        array = np.asarray(frame)
        if array.ndim != 3 or array.shape[-1] < 3:
            raise ValueError(f"Expected RGB image with shape HxWx3, got {array.shape}.")
        array = array[..., :3]
        if array.dtype == np.uint8:
            return np.ascontiguousarray(array)
        if np.issubdtype(array.dtype, np.floating) and float(np.nanmax(array)) <= 1.0:
            array = array * 255.0
        return np.ascontiguousarray(np.clip(array, 0, 255).astype(np.uint8))

    @staticmethod
    def _import_cv2():
        try:
            import cv2
        except Exception as exc:
            raise RuntimeError(
                "Camera browser streaming requires OpenCV for JPEG encoding. "
                "Install opencv-python-headless/opencv-python in the Isaac Lab Python environment."
            ) from exc
        return cv2


class AttachedCameraStream:
    """Camera stream attached to an Isaac Lab environment via step/close wrappers."""

    def __init__(
        self,
        env: Any,
        camera_names: tuple[str, ...],
        env_index: int,
        update_every: int,
        server: CameraStreamServer,
    ):
        self.env = env
        self.camera_names = camera_names
        self.env_index = env_index
        self.update_every = update_every
        self.server = server
        self.step_count = 0
        self.enabled = True
        self._warned_missing_cameras = False
        self._original_step: Callable[..., Any] | None = None
        self._original_close: Callable[..., Any] | None = None

    @property
    def url(self) -> str:
        return self.server.url

    def install(self) -> None:
        """Start the server and update it automatically after each env.step call."""
        self.server.start()
        self._original_step = self.env.step
        self._original_close = getattr(self.env, "close", None)

        attachment = self

        def step_with_camera_stream(*args, **kwargs):
            result = attachment._original_step(*args, **kwargs)
            attachment.step_count += 1
            if attachment.step_count % attachment.update_every == 0:
                attachment.update()
            return result

        def close_with_camera_stream(*args, **kwargs):
            attachment.close()
            if attachment._original_close is not None:
                return attachment._original_close(*args, **kwargs)
            return None

        self.env.step = step_with_camera_stream
        self.env.close = close_with_camera_stream

    def update(self) -> None:
        if not self.enabled:
            return
        frames = self._read_camera_frames()
        if frames is None:
            self.enabled = False
            return
        self.server.update(frames, step=self.step_count)

    def close(self) -> None:
        self.server.close()

    def _read_camera_frames(self) -> dict[str, np.ndarray] | None:
        unwrapped_env = getattr(self.env, "unwrapped", self.env)
        num_envs = getattr(unwrapped_env, "num_envs", None)
        if num_envs is not None and (self.env_index < 0 or self.env_index >= num_envs):
            self._warn_once(
                "[WARN] Camera stream disabled: AIC_CAMERA_STREAM_ENV_INDEX must be in "
                f"[0, {num_envs - 1}], got {self.env_index}."
            )
            return None

        scene = getattr(unwrapped_env, "scene", None)
        if scene is None or not hasattr(scene, "sensors"):
            self._warn_once("[WARN] Camera stream disabled: environment has no scene sensors.")
            return None

        missing = [name for name in self.camera_names if name not in scene.sensors]
        if missing:
            available = ", ".join(scene.sensors.keys())
            self._warn_once(
                "[WARN] Camera stream disabled: missing camera sensor(s) "
                f"{missing}. Available sensors: {available}"
            )
            return None

        frames = {}
        for camera_name in self.camera_names:
            camera = scene.sensors[camera_name]
            if "rgb" not in camera.data.output:
                self._warn_once(f"[WARN] Camera stream disabled: camera '{camera_name}' has no rgb output.")
                return None
            frames[camera_name] = camera.data.output["rgb"][self.env_index].detach().cpu().numpy()
        return frames

    def _warn_once(self, message: str) -> None:
        if self._warned_missing_cameras:
            return
        print(message)
        self._warned_missing_cameras = True


def attach_default_camera_stream(env: Any) -> AttachedCameraStream | None:
    """Attach the default localhost browser camera stream to an env.

    The default mosaic order is left, center, right. Configure only through env vars:
    AIC_CAMERA_STREAM=0 disables it, while AIC_CAMERA_STREAM_PORT/HOST/EVERY/QUALITY/ENV_INDEX override defaults.
    """
    if not default_camera_stream_enabled():
        return None

    camera_names = tuple(
        name.strip()
        for name in os.environ.get("AIC_CAMERA_STREAM_CAMERAS", ",".join(DEFAULT_CAMERA_NAMES)).split(",")
        if name.strip()
    )
    env_index = _get_int_env("AIC_CAMERA_STREAM_ENV_INDEX", DEFAULT_ENV_INDEX)
    update_every = max(1, _get_int_env("AIC_CAMERA_STREAM_EVERY", DEFAULT_UPDATE_EVERY))
    server = CameraStreamServer(
        list(camera_names),
        host=os.environ.get("AIC_CAMERA_STREAM_HOST", DEFAULT_HOST),
        port=_get_int_env("AIC_CAMERA_STREAM_PORT", DEFAULT_PORT),
        jpeg_quality=_get_int_env("AIC_CAMERA_STREAM_QUALITY", DEFAULT_JPEG_QUALITY),
    )
    attachment = AttachedCameraStream(env, camera_names, env_index, update_every, server)
    attachment.install()
    print(f"[INFO] Browser camera stream ready at: {attachment.url}")
    return attachment


def _get_int_env(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}.") from exc
