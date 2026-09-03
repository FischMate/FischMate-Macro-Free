from __future__ import annotations

import datetime as dt
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Callable

from app.capture.sources import DxcamWindowSource, FrameSource
from app.capture.windows import WindowInfo, is_foreground_window
from app.diagnostics.session import DiagnosticSession
from app.pipeline import MacroPipeline
from app.executor.windows import InterruptibleInputExecutor, WindowsSendInputBackend
from app.core.models import DetectionSnapshot


def noiseform_diagnostic_mode(detection: DetectionSnapshot) -> str:
    """Describe which Noiseform control path owns the current frame."""
    explicit_mode = str(detection.extra.get("noiseform_mode", ""))
    if explicit_mode == "COLOR_TASK":
        return "COLOR-BLOCK TASK"
    if explicit_mode == "RECOVERY":
        return "DETACHED-BAR RECOVERY"
    if explicit_mode == "NORMAL":
        return "NORMAL TRACKING"
    if bool(detection.extra.get("noiseform_task_active", False)):
        return "COLOR-BLOCK TASK"

    if detection.bar is None or detection.stick is None:
        return "DETACHED-BAR RECOVERY"
    if detection.stick_center_inside_bar:
        return "NORMAL TRACKING"
    bar_source = str(detection.extra.get("bar_geometry_source", ""))
    if bar_source in {
        "noiseform_black_arrow_pair",
        "noiseform_black_outline_pair",
        "noiseform_bar_unconfirmed",
        "noiseform_input_response_rejected",
        "noiseform_rejected_frozen_center",
        "noiseform_frozen_weak_geometry",
    }:
        return "DETACHED-BAR RECOVERY"
    return "NORMAL TRACKING"


@dataclass(frozen=True)
class LiveStatus:
    running: bool
    lifecycle: str
    detector_state: str
    fps: float
    bar_center: float | None
    stick_center: float | None
    error_px: float | None
    command: str
    input_enabled: bool = False
    input_event: str = "none"
    input_event_count: int = 0
    message: str = ""
    diagnostic_mode: str = ""
    command_reason: str = ""
    rejection_reason: str = ""
    bar_source: str = ""
    stick_source: str = ""
    mouse_is_down: bool = False
    noiseform_mode_reason: str = ""
    noiseform_measured_bar_width: float | None = None
    noiseform_bar_width_confidence: float = 0.0


class LiveDetectionEngine:
    """Runs the exact same MacroPipeline used by recorded-video replay."""

    def __init__(
        self,
        project_root: Path,
        status_callback: Callable[[LiveStatus], None],
        source_factory: Callable[[WindowInfo], FrameSource] = DxcamWindowSource,
    ):
        self.project_root = project_root
        self.status_callback = status_callback
        self.source_factory = source_factory
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._executor: InterruptibleInputExecutor | None = None

    @property
    def running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def start(self, window: WindowInfo, profile: dict, automation_enabled: bool = False) -> None:
        if self.running:
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run,
            args=(window, profile, automation_enabled),
            name="fischmate-live-detection",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self.emergency_release()
        if self._thread and self._thread is not threading.current_thread():
            self._thread.join(timeout=2.0)

    def emergency_release(self) -> None:
        # Set the stop flag first so a later detector frame cannot reassert HOLD.
        self._stop.set()
        executor = self._executor
        if executor is not None:
            executor.set_enabled(False)

    def _run(self, window: WindowInfo, profile: dict, automation_enabled: bool) -> None:
        source: FrameSource | None = None
        session: DiagnosticSession | None = None
        try:
            if automation_enabled and not is_foreground_window(window):
                raise RuntimeError("Select Roblox and keep it in the foreground before starting automation")
            source = self.source_factory(window)
            enabled_mechanics = profile.get("mechanics", {}).get("enabled", [])
            maximum_mouse_down_bpm = (
                float(
                    profile["input"].get(
                        "maximum_mouse_transition_bpm",
                        profile["input"].get("maximum_mouse_down_bpm"),
                    )
                )
                if "rate_limited_clicking" in enabled_mechanics
                else None
            )
            input_config = profile.get("input", {})
            noiseform_profile = (
                str(profile.get("detection", {}).get("detector", "")).lower()
                == "noiseform"
            )
            executor = InterruptibleInputExecutor(
                WindowsSendInputBackend(),
                enabled=automation_enabled,
                maximum_mouse_down_bpm=maximum_mouse_down_bpm,
                mouse_down_timing=str(
                    input_config.get("mouse_down_timing", "minimum_interval")
                ),
                mouse_down_beat_window_ms=float(
                    input_config.get("mouse_down_beat_window_ms", 70)
                ),
            )
            self._executor = executor
            pipeline = MacroPipeline(profile, executor=executor, automate_lifecycle=automation_enabled)
            if bool(profile.get("diagnostics", {}).get("live_enabled", False)):
                stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
                output = self.project_root / "diagnostics" / f"live_{stamp}"
                session = DiagnosticSession(
                    output,
                    {
                        "mode": "live_automation" if automation_enabled else "live_detection_only",
                        "profile": profile["profile"]["display_name"],
                        "window": window.label,
                        "input_enabled": automation_enabled,
                    },
                )
            frames = 0
            report_frames = 0
            started = report_started = time.perf_counter()
            last_result = None
            automation_started = False
            while not self._stop.is_set():
                if automation_enabled and not is_foreground_window(window):
                    raise RuntimeError("Automation stopped because Roblox lost foreground focus")
                packet = source.read()
                if packet is None:
                    time.sleep(0.002)
                    continue
                if automation_enabled and not automation_started:
                    pipeline.start_automation(packet.timestamp_ms)
                    automation_started = True
                last_result = pipeline.process(packet)
                last_result.detection.extra["input_event"] = executor.last_input_event
                last_result.detection.extra["input_event_count"] = executor.input_event_count
                if session is not None:
                    session.record(last_result)
                frames += 1
                report_frames += 1
                now = time.perf_counter()
                if now - report_started >= 0.1:
                    fps = report_frames / max(0.001, now - report_started)
                    detection = last_result.detection
                    diagnostic_mode = (
                        noiseform_diagnostic_mode(detection)
                        if noiseform_profile and last_result.lifecycle.value == "MINIGAME"
                        else ""
                    )
                    self.status_callback(
                        LiveStatus(
                            True,
                            last_result.lifecycle.value,
                            detection.detector_state,
                            fps,
                            detection.bar_center_x,
                            detection.stick_center_x,
                            last_result.command.error_px,
                            last_result.command.action.value,
                            input_enabled=automation_enabled,
                            input_event=executor.last_input_event,
                            input_event_count=executor.input_event_count,
                            diagnostic_mode=diagnostic_mode,
                            command_reason=last_result.command.reason,
                            rejection_reason=detection.rejection_reason,
                            bar_source=str(detection.extra.get("bar_geometry_source", "")),
                            stick_source=str(detection.extra.get("noiseform_stick_source", "")),
                            mouse_is_down=executor.mouse_is_down,
                            noiseform_mode_reason=str(
                                detection.extra.get("noiseform_mode_reason", "")
                            ),
                            noiseform_measured_bar_width=(
                                None
                                if detection.extra.get("noiseform_measured_bar_width", "") == ""
                                else float(detection.extra["noiseform_measured_bar_width"])
                            ),
                            noiseform_bar_width_confidence=float(
                                detection.extra.get("noiseform_bar_width_confidence", 0.0)
                            ),
                        )
                    )
                    report_started = now
                    report_frames = 0
            elapsed = time.perf_counter() - started
            self.status_callback(
                LiveStatus(
                    False,
                    "STOPPED",
                    "STOPPED",
                    frames / max(0.001, elapsed),
                    None,
                    None,
                    None,
                    "NEUTRAL",
                    input_enabled=automation_enabled,
                    input_event=executor.last_input_event,
                    input_event_count=executor.input_event_count,
                )
            )
        except Exception as exc:
            self.emergency_release()
            self.status_callback(
                LiveStatus(
                    False,
                    "ERROR",
                    "ERROR",
                    0.0,
                    None,
                    None,
                    None,
                    "NEUTRAL",
                    input_enabled=False,
                    message=str(exc),
                )
            )
        finally:
            self.emergency_release()
            if session is not None:
                session.close()
            if source is not None:
                source.close()
            self._executor = None
