from __future__ import annotations

from dataclasses import dataclass

from app.core.models import DetectionSnapshot, LifecycleState


@dataclass
class LifecycleMachine:
    """Evidence-driven fishing lifecycle.

    Durations are safeguards only. A low-lure-speed rod remains in SHAKING for
    as long as necessary until the shared detector reports the minigame.
    """

    profile: dict
    state: LifecycleState = LifecycleState.IDLE
    entered_ms: float = 0.0
    reason: str = "created"
    visible_frames: int = 0
    missing_frames: int = 0
    missing_since_ms: float | None = None

    def transition(self, state: LifecycleState, now_ms: float, reason: str) -> None:
        if state != self.state:
            self.state = state
            self.entered_ms = now_ms
            self.reason = reason

    def start(self, now_ms: float) -> None:
        self.transition(LifecycleState.PREPARING, now_ms, "user_start")

    def casting_started(self, now_ms: float) -> None:
        self.visible_frames = 0
        self.missing_frames = 0
        self.missing_since_ms = None
        self.transition(LifecycleState.CASTING, now_ms, "executor_cast_started")

    def casting_finished(self, now_ms: float) -> None:
        self.visible_frames = 0
        self.missing_frames = 0
        self.missing_since_ms = None
        self.transition(LifecycleState.SHAKING, now_ms, "cast_released")

    def stop(self, now_ms: float, reason: str = "emergency_stop") -> None:
        self.transition(LifecycleState.STOPPED, now_ms, reason)

    def reset(self, now_ms: float) -> None:
        self.visible_frames = 0
        self.missing_frames = 0
        self.missing_since_ms = None
        self.transition(LifecycleState.IDLE, now_ms, "reset")

    def update(self, observation: DetectionSnapshot) -> LifecycleState:
        now_ms = observation.timestamp_ms
        strong_visible = observation.minigame_visible
        partial_visible = observation.bar is not None or observation.stick is not None
        if self.profile["detection"].get(
            "minigame_end_requires_strong_visibility", False
        ):
            # Specialized minigames can have a reliable full-screen signature.
            # For those profiles, unrelated bar-like UI after completion must
            # not keep the completed round alive.
            partial_visible = False
        if self.profile["detection"].get("minigame_end_ignore_stick_only", False):
            # Some rod outlines leave a persistent stick-shaped edge after the
            # minigame vanishes. For opted-in profiles, a stick without any bar
            # evidence cannot indefinitely keep the completed minigame alive.
            partial_visible = observation.bar is not None

        # Once a minigame has been positively acquired, temporary loss of one
        # component means tracking is partial; it does not mean the game ended.
        # Completion requires sustained total loss of bar and stick evidence.
        if self.state == LifecycleState.MINIGAME:
            if strong_visible or partial_visible:
                self.missing_frames = 0
                self.missing_since_ms = None
            else:
                self.missing_frames += 1
                if self.missing_since_ms is None:
                    self.missing_since_ms = now_ms
            release_frames = int(self.profile["detection"].get("minigame_release_frames", 18))
            release_ms = float(self.profile["detection"].get("minigame_release_ms", 650))
            missing_ms = (
                0.0 if self.missing_since_ms is None else now_ms - self.missing_since_ms
            )
            if self.missing_frames >= release_frames and missing_ms >= release_ms:
                self.transition(LifecycleState.RESULT, now_ms, "minigame_visuals_sustained_absence")
            return self.state

        if strong_visible:
            self.visible_frames += 1
            self.missing_frames = 0
            self.missing_since_ms = None
        else:
            self.missing_frames += 1
            self.visible_frames = 0
            if self.missing_since_ms is None:
                self.missing_since_ms = now_ms
        confirm_frames = int(self.profile["detection"].get("minigame_confirm_frames", 3))
        if self.state in {LifecycleState.IDLE, LifecycleState.STOPPED, LifecycleState.ERROR}:
            # Replay can enter the minigame without simulating input lifecycle.
            if self.visible_frames >= confirm_frames:
                self.transition(LifecycleState.MINIGAME, now_ms, "minigame_visual_evidence")
            return self.state
        if self.state == LifecycleState.SHAKING and self.visible_frames >= confirm_frames:
            self.transition(LifecycleState.MINIGAME, now_ms, "minigame_visual_evidence")
            return self.state
        if self.state == LifecycleState.RESULT and observation.result_visible:
            self.transition(LifecycleState.RECOVERY, now_ms, "result_visual_evidence")
        return self.state
