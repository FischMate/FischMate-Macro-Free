from __future__ import annotations

from typing import Protocol

from app.core.models import ControlCommand, DetectionSnapshot, LifecycleState
from app.lifecycle.machine import LifecycleMachine


class LifecycleExecutor(Protocol):
    def begin_cast(self) -> None: ...
    def tap_key(self, key: str) -> None: ...
    def release_phase_inputs(self) -> None: ...
    def submit(self, command: ControlCommand) -> None: ...
    def emergency_release(self) -> None: ...


class FishingCoordinator:
    """Coordinates phase input from lifecycle evidence.

    The detector only produces observations. This class owns casting and shake
    scheduling. The controller remains responsible only for minigame intent.
    """

    def __init__(self, profile: dict, machine: LifecycleMachine, executor: LifecycleExecutor):
        self.profile = profile
        self.machine = machine
        self.executor = executor
        self.cast_release_ms: float | None = None
        self.preparation_end_ms: float | None = None
        self.next_navigation_ms: float | None = None
        self.recovery_end_ms: float | None = None
        self.shake_was_visible = False
        self.active = False

    def reset(self, now_ms: float) -> None:
        self.active = False
        self.cast_release_ms = None
        self.preparation_end_ms = None
        self.next_navigation_ms = None
        self.recovery_end_ms = None
        self.shake_was_visible = False
        self.executor.emergency_release()
        self.machine.reset(now_ms)

    def start(self, now_ms: float) -> None:
        self._start_cycle(now_ms)

    def _start_cycle(self, now_ms: float) -> None:
        self.executor.emergency_release()
        self.cast_release_ms = None
        self.preparation_end_ms = None
        self.machine.reset(now_ms)
        self.machine.start(now_ms)
        if (
            self.profile["shake"].get("mode") == "navigation"
            and self.profile["shake"].get("navigation_enable_before_cast", True)
        ):
            self.executor.tap_key(self.profile["shake"]["navigation_key"])
            self.preparation_end_ms = now_ms + float(
                self.profile["shake"].get("navigation_enable_settle_ms", 80)
            )
        else:
            self._begin_cast(now_ms)
        self.next_navigation_ms = None
        self.recovery_end_ms = None
        self.shake_was_visible = False
        self.active = True

    def stop(self, now_ms: float, reason: str = "emergency_stop") -> None:
        self.active = False
        self.cast_release_ms = None
        self.preparation_end_ms = None
        self.next_navigation_ms = None
        self.recovery_end_ms = None
        self.shake_was_visible = False
        self.executor.emergency_release()
        self.machine.stop(now_ms, reason)

    def update(self, observation: DetectionSnapshot) -> LifecycleState:
        now_ms = observation.timestamp_ms
        previous = self.machine.state
        state = self.machine.update(observation)
        if not self.active:
            return state

        if (
            state == LifecycleState.RECOVERY
            and bool(
                self.profile.get("loop", {}).get(
                    "reacquire_minigame_during_recovery", False
                )
            )
        ):
            confirm_frames = int(
                self.profile["detection"].get("minigame_confirm_frames", 3)
            )
            if self.machine.visible_frames >= confirm_frames:
                # A difficult visual effect can briefly hide every component.
                # If the same minigame becomes strongly visible during the
                # post-catch grace period, cancel the recast and resume control.
                self.recovery_end_ms = None
                self.executor.release_phase_inputs()
                self.machine.transition(
                    LifecycleState.MINIGAME,
                    now_ms,
                    "minigame_reacquired_during_recovery",
                )
                timing_hook = getattr(
                    self.executor,
                    "begin_minigame_input_timing",
                    None,
                )
                if timing_hook is not None:
                    timing_hook()
                return self.machine.state

        if state == LifecycleState.MINIGAME:
            if previous != LifecycleState.MINIGAME:
                # End cast/shake inputs before the controller gets this frame.
                self.executor.release_phase_inputs()
                timing_hook = getattr(
                    self.executor, "begin_minigame_input_timing", None
                )
                if timing_hook is not None:
                    timing_hook()
            return state

        if (
            state == LifecycleState.CASTING
            and bool(self.profile["casting"].get("early_minigame_acquisition", False))
        ):
            confirm_frames = int(self.profile["detection"].get("minigame_confirm_frames", 3))
            if self.machine.visible_frames >= confirm_frames:
                self.executor.release_phase_inputs()
                self.machine.transition(
                    LifecycleState.MINIGAME,
                    now_ms,
                    "minigame_visual_evidence_during_cast",
                )
                timing_hook = getattr(
                    self.executor,
                    "begin_minigame_input_timing",
                    None,
                )
                if timing_hook is not None:
                    timing_hook()
                return self.machine.state

        if state == LifecycleState.PREPARING and self.preparation_end_ms is not None:
            if now_ms >= self.preparation_end_ms:
                self._begin_cast(now_ms)
                state = self.machine.state

        if state == LifecycleState.CASTING and self.cast_release_ms is not None:
            if now_ms >= self.cast_release_ms:
                self.executor.release_phase_inputs()
                self.machine.casting_finished(now_ms)
                state = self.machine.state
                # Navigation mode must be enabled when shaking actually starts.
                # The pre-cast tap can be consumed by casting/focus changes.
                if self.profile["shake"].get("mode") == "navigation":
                    self.executor.tap_key(self.profile["shake"]["navigation_key"])
                    self.next_navigation_ms = now_ms + float(
                        self.profile["shake"].get("navigation_shake_settle_ms", 30)
                    )
                else:
                    self.next_navigation_ms = now_ms

        if state == LifecycleState.SHAKING:
            timeout_s = float(self.profile["shake"].get("safety_timeout_s", 0))
            if timeout_s > 0 and now_ms - self.machine.entered_ms >= timeout_s * 1000.0:
                # A missed cast or absent shake prompt must not strand an AFK
                # session in navigation input forever. The configured timeout
                # is deliberately only a safeguard; visual acquisition remains
                # the normal way out of SHAKING.
                self.executor.release_phase_inputs()
                loop_config = self.profile.get("loop", {})
                if bool(loop_config.get("enabled", True)):
                    self.recovery_end_ms = now_ms + float(
                        loop_config.get("restart_delay_ms", 1500)
                    )
                    self.machine.transition(
                        LifecycleState.RECOVERY,
                        now_ms,
                        "shake_safety_timeout",
                    )
                    state = self.machine.state
                else:
                    self.active = False
                    self.machine.transition(
                        LifecycleState.ERROR,
                        now_ms,
                        "shake_safety_timeout",
                    )
                    state = self.machine.state

        if (
            state == LifecycleState.SHAKING
            and self.profile["shake"].get("mode") == "navigation"
        ):
            # Keep navigation shaking human-scale even when an imported V13
            # profile contains the old 10 ms spam delay. Profiles may choose
            # a slower cadence, but cannot exceed the shared safety rate.
            configured_interval = max(
                float(self.profile["shake"].get("minimum_confirmation_interval_ms", 500)),
                float(self.profile["shake"]["navigation_interval_ms"]),
            )
            # Roughly eight confirmations per second resembles normal Enter
            # spamming while retaining a shared floor against 10 ms input.
            safety_floor = max(
                100.0,
                float(self.profile["shake"].get("navigation_safety_floor_ms", 125)),
            )
            interval = max(safety_floor, configured_interval)
            # Match V13.4 navigation behavior: once navigation mode is active,
            # Enter is harmlessly repeated until visual minigame confirmation.
            # Gating input on a perfect shake-button detection made one missed
            # frame stall the entire fishing lifecycle.
            if self.next_navigation_ms is None:
                self.next_navigation_ms = now_ms
            if now_ms >= self.next_navigation_ms:
                self.executor.tap_key(self.profile["shake"].get("confirmation_key", "ENTER"))
                skipped = int((now_ms - self.next_navigation_ms) // interval)
                self.next_navigation_ms += (skipped + 1) * interval
            self.shake_was_visible = observation.shake_visible

        if state == LifecycleState.RESULT:
            self.executor.release_phase_inputs()
            loop_config = self.profile.get("loop", {})
            if bool(loop_config.get("enabled", True)):
                self.recovery_end_ms = now_ms + float(
                    loop_config.get("restart_delay_ms", 1500)
                )
                self.machine.transition(LifecycleState.RECOVERY, now_ms, "post_catch_wait")
                state = self.machine.state

        if state == LifecycleState.RECOVERY:
            self.executor.release_phase_inputs()
            if self.recovery_end_ms is not None and now_ms >= self.recovery_end_ms:
                self._start_cycle(now_ms)
                state = self.machine.state

        if state == LifecycleState.ERROR:
            self.executor.release_phase_inputs()
        return state

    def _begin_cast(self, now_ms: float) -> None:
        self.preparation_end_ms = None
        self.machine.casting_started(now_ms)
        self.cast_release_ms = now_ms + float(self.profile["casting"]["hold_ms"])
        self.executor.begin_cast()
