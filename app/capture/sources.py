from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import cv2

from app.capture.windows import WindowInfo, refresh_window
from app.core.models import FramePacket, PixelRect


class FrameSource(ABC):
    @abstractmethod
    def read(self) -> FramePacket | None:
        raise NotImplementedError

    def close(self) -> None:
        pass


class VideoFrameSource(FrameSource):
    def __init__(self, path: Path, start_s: float = 0.0, duration_s: float | None = None):
        self.path = path
        self.capture = cv2.VideoCapture(str(path))
        if not self.capture.isOpened():
            raise RuntimeError(f"Could not open video: {path}")
        self.capture.set(cv2.CAP_PROP_POS_MSEC, max(0.0, start_s) * 1000.0)
        self.end_ms = None if duration_s is None else (max(0.0, start_s) + duration_s) * 1000.0
        self.sequence = 0

    def read(self) -> FramePacket | None:
        ok, frame = self.capture.read()
        if not ok:
            return None
        timestamp_ms = float(self.capture.get(cv2.CAP_PROP_POS_MSEC))
        if self.end_ms is not None and timestamp_ms > self.end_ms:
            return None
        height, width = frame.shape[:2]
        packet = FramePacket(
            frame_bgr=frame,
            timestamp_ms=timestamp_ms,
            source_name=str(self.path),
            sequence=self.sequence,
            window_rect=PixelRect(0, 0, width, height),
        )
        self.sequence += 1
        return packet

    def close(self) -> None:
        self.capture.release()


class DxcamWindowSource(FrameSource):
    def __init__(self, window: WindowInfo):
        try:
            import dxcam
        except ImportError as exc:
            raise RuntimeError("DXcam is required for live capture") from exc
        self.window = window
        self.camera = dxcam.create(output_color="BGR")
        self.sequence = 0

    def read(self) -> FramePacket | None:
        current = refresh_window(self.window)
        if current is None:
            return None
        self.window = current
        rect = current.client_rect
        frame = self.camera.grab(region=(rect.left, rect.top, rect.right, rect.bottom))
        if frame is None:
            return None
        packet = FramePacket(
            frame_bgr=frame,
            timestamp_ms=float(cv2.getTickCount() * 1000.0 / cv2.getTickFrequency()),
            source_name=f"hwnd:{current.hwnd}",
            sequence=self.sequence,
            window_rect=rect,
        )
        self.sequence += 1
        return packet

    def close(self) -> None:
        try:
            self.camera.stop()
        except Exception:
            pass
