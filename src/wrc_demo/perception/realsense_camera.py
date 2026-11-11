"""RealSense backend (D4xx and L515) via pyrealsense2.

On this rig pyrealsense2 comes from the local librealsense L515 fork
(l515-on-development, v2.58.x) installed into the .demo venv. Depth is
aligned to color and converted to float32 meters using the active device's
depth scale.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..config import Cfg
from ..types import Frame
from .camera_base import CameraBase, CameraError


def _matches_model(name: str, model: str) -> bool:
    return name.split()[-1].casefold() == model.casefold() if name else False


def list_realsense_devices(model: str | None = None) -> list[dict]:
    import pyrealsense2 as rs

    context = rs.context()
    devices = []
    for device in context.query_devices():
        name = device.get_info(rs.camera_info.name)
        if not model or _matches_model(name, model):
            devices.append({
                "index": len(devices), "name": name,
                "serial": device.get_info(rs.camera_info.serial_number),
            })
    return devices


def _inverse_brown_rectification_map(rs, intr) -> np.ndarray:
    # Map each output pinhole pixel to its source pixel using librealsense's
    # projection. Use the same map for color and depth aligned to color.
    x = (np.arange(intr.width) - intr.ppx) / intr.fx
    y = (np.arange(intr.height) - intr.ppy) / intr.fy
    project = rs.rs2_project_point_to_pixel
    pixels = np.fromiter(
        (project(intr, (u, v, 1.0)) for v in y for u in x),
        dtype=np.dtype((np.float32, 2)),
        count=intr.width * intr.height,
    )
    return pixels.reshape(intr.height, intr.width, 2)


class RealSenseCamera(CameraBase):
    def __init__(self, cfg: Cfg):
        super().__init__()
        self._cfg = cfg
        self._pipe = None
        self._align = None
        self._depth_scale = 0.001
        self._K = None
        self._intrinsics = None
        self._rectification_map = None
        self._serial = None

    @property
    def has_depth(self) -> bool:
        return True

    @property
    def serial(self) -> str | None:
        return self._serial

    @property
    def K(self) -> np.ndarray | None:
        return self._K

    @property
    def D(self) -> np.ndarray | None:
        intr = self._intrinsics
        if intr is None or self._rectification_map is not None:
            return None
        coeffs = np.asarray(intr.coeffs, dtype=np.float64)
        if not np.any(coeffs) or intr.model == self._rs.distortion.none:
            return None
        if intr.model == self._rs.distortion.brown_conrady:
            return coeffs
        raise CameraError(f"unsupported RealSense color distortion for calibration: {intr.model}")

    def open(self) -> None:
        import pyrealsense2 as rs

        self._rs = rs
        pipe = rs.pipeline()
        config = rs.config()
        serial = self._cfg.get("serial")
        model = self._cfg.get("device_model")
        if model and not serial:
            devices = list_realsense_devices(model)
            if len(devices) != 1:
                raise CameraError(f"expected one RealSense {model}; found {len(devices)}. Set the camera serial.")
            serial = devices[0]["serial"]
        if serial:
            config.enable_device(str(serial))
        w = int(self._cfg.get("width", 1280))
        h = int(self._cfg.get("height", 720))
        fps = int(self._cfg.get("fps", 30))
        match_depth = self._cfg.get("match_depth_resolution", False)
        if match_depth:
            config.enable_stream(rs.stream.depth, w, h, rs.format.z16, fps)
        else:
            config.enable_stream(rs.stream.depth, rs.format.z16, fps)
        config.enable_stream(rs.stream.color, w, h, rs.format.bgr8, fps)
        try:
            profile = pipe.start(config)
        except RuntimeError as exc:
            if match_depth:
                raise CameraError(
                    f"cannot start matching RealSense depth and color streams at {w}x{h}/{fps} FPS: {exc}"
                ) from exc
            # Fall back to fully automatic stream selection (any resolution).
            config = rs.config()
            if serial:
                config.enable_device(str(serial))
            config.enable_stream(rs.stream.depth, rs.format.z16, fps)
            config.enable_stream(rs.stream.color, rs.format.bgr8, fps)
            profile = pipe.start(config)
        try:
            device = profile.get_device()
            name = device.get_info(rs.camera_info.name)
            if model and not _matches_model(name, model):
                raise CameraError(f"expected RealSense {model}, opened {name!r}; check the camera serial")
            self._depth_scale = device.first_depth_sensor().get_depth_scale()
            self._serial = device.get_info(rs.camera_info.serial_number)
            intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
            self._intrinsics = intr
            self._K = np.array(
                [[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]],
                dtype=np.float64,
            )
            self._rectification_map = None
            if np.any(intr.coeffs) and intr.model == rs.distortion.inverse_brown_conrady:
                self._rectification_map = _inverse_brown_rectification_map(rs, intr)
            self._align = rs.align(rs.stream.color)
        except Exception:
            pipe.stop()
            raise
        self._pipe = pipe

    def close(self) -> None:
        if self._pipe is not None:
            self._pipe.stop()
            self._pipe = None

    def _grab(self) -> Frame:
        if self._pipe is None:
            raise CameraError("camera not opened")
        try:
            frames = self._pipe.wait_for_frames(5000)
            frames = self._align.process(frames)
            depth_f = frames.get_depth_frame()
            color_f = frames.get_color_frame()
            if not depth_f or not color_f:
                raise CameraError("incomplete frameset")
        except RuntimeError as e:
            raise CameraError(str(e)) from e

        color = np.asanyarray(color_f.get_data()).copy()
        depth = np.asanyarray(depth_f.get_data()).astype(np.float32)
        if self._rectification_map is not None:
            color = cv2.remap(color, self._rectification_map, None, cv2.INTER_LINEAR)
            depth = cv2.remap(depth, self._rectification_map, None, cv2.INTER_NEAREST)
        depth *= self._depth_scale  # -> meters, 0 stays invalid

        min_d = float(self._cfg.get("min_depth_m", 0.1))
        max_d = float(self._cfg.get("max_depth_m", 3.0))
        depth[(depth < min_d) | (depth > max_d)] = 0.0

        return Frame(rgb=color, depth_m=depth, K=self._K, depth_source="sensor")
