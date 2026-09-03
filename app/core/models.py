from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


@dataclass(frozen=True)
class NormalizedRect:
    left: float
    top: float
    right: float
    bottom: float

    def pixels(self, width: int, height: int) -> "PixelRect":
        return PixelRect(
            round(self.left * width),
            round(self.top * height),
            round(self.right * width),
            round(self.bottom * height),
        ).clamped(width, height)


@dataclass(frozen=True)
class PixelRect:
    left: int
    top: int
    right: int
    bottom: int

    @property
    def width(self) -> int:
        return max(0, self.right - self.left)

    @property
    def height(self) -> int:
        return max(0, self.bottom - self.top)

    @property
    def center_x(self) -> float:
        return (self.left + self.right) / 2.0

    def clamped(self, width: int, height: int) -> "PixelRect":
        return PixelRect(
            max(0, min(width, self.left)),
            max(0, min(height, self.top)),
            max(0, min(width, self.right)),
            max(0, min(height, self.bottom)),
        )


@dataclass(frozen=True)
class FramePacket:
    frame_bgr: np.ndarray
    timestamp_ms: float
    source_name: str
    sequence: int
    window_rect: PixelRect


class LifecycleState(str, Enum):
    IDLE = "IDLE"
    PREPARING = "PREPARING"
    CASTING = "CASTING"
    SHAKING = "SHAKING"
    MINIGAME = "MINIGAME"
    RESULT = "RESULT"
    RECOVERY = "RECOVERY"
    STOPPED = "STOPPED"
    ERROR = "ERROR"


class CommandAction(str, Enum):
    NEUTRAL = "NEUTRAL"
    HOLD = "HOLD"
    RELEASE = "RELEASE"


@dataclass(frozen=True)
class ControlCommand:
    action: CommandAction
    generation: int
    reason: str
    target_x: float | None = None
    error_px: float | None = None
    confidence: float = 0.0


@dataclass
class DetectionSnapshot:
    timestamp_ms: float
    source_name: str
    frame_width: int
    frame_height: int
    minigame_roi: PixelRect
    minigame_visible: bool = False
    shake_visible: bool = False
    result_visible: bool = False
    rail: PixelRect | None = None
    bar: PixelRect | None = None
    stick: PixelRect | None = None
    bar_confidence: float = 0.0
    stick_confidence: float = 0.0
    detector_state: str = "SEARCHING"
    rejection_reason: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def bar_center_x(self) -> float | None:
        return None if self.bar is None else self.bar.center_x

    @property
    def stick_center_x(self) -> float | None:
        return None if self.stick is None else self.stick.center_x

    @property
    def stick_center_inside_bar(self) -> bool | None:
        if self.bar is None or self.stick is None:
            return None
        return self.bar.left <= self.stick.center_x <= self.bar.right

    def overlap_pixels(self) -> int | None:
        if self.bar is None or self.stick is None:
            return None
        return max(0, min(self.bar.right, self.stick.right) - max(self.bar.left, self.stick.left))


@dataclass
class PipelineResult:
    frame: FramePacket
    detection: DetectionSnapshot
    lifecycle: LifecycleState
    command: ControlCommand
    perfect_now: bool | None
    perfect_margin_px: float | None

