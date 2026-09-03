from __future__ import annotations

from app.controller.standard import StandardController
from app.controller.special_standard import SpecialStandardController
from app.controller.noiseform import NoiseformController
from app.controller.ruinous_legacy import RuinousLegacyController
from app.core.models import FramePacket, LifecycleState, PipelineResult
from app.detection.standard import StandardDetector
from app.detection.special_standard import SpecialStandardDetector
from app.detection.crowbar import CrowbarDetector
from app.detection.darkheart import DarkheartDetector
from app.detection.noiseform import NoiseformDetector
from app.detection.ruinous_oath import RuinousOathDetector
from app.detection.ruinous_legacy import RuinousLegacyDetector
from app.detection.tranquility import TranquilityDetector
from app.detection.pinions_aria import PinionsAriaDetector
from app.executor.safe import InputDisabledExecutor
from app.lifecycle.coordinator import FishingCoordinator, LifecycleExecutor
from app.lifecycle.machine import LifecycleMachine


class MacroPipeline:
    """The single processing path used by replay and live capture."""

    def __init__(
        self,
        profile: dict,
        executor: LifecycleExecutor | None = None,
        automate_lifecycle: bool = False,
    ):
        self.profile = profile
        detection_config = profile.get("detection", {})
        detector_name = detection_config.get("detector", "standard")
        legacy_ruinous = profile.get("_profile_name") == "ror"
        detector_types = {
            "crowbar": CrowbarDetector,
            "darkheart": DarkheartDetector,
            "noiseform": NoiseformDetector,
            "ruinous_oath": RuinousOathDetector,
            "standard": StandardDetector,
            "tranquility": TranquilityDetector,
            "pinions_aria": PinionsAriaDetector,
        }
        detector_type = (
            RuinousLegacyDetector
            if legacy_ruinous
            else detector_types.get(detector_name, StandardDetector)
        )
        if (
            not legacy_ruinous
            and detector_name == "standard"
            and self._requires_special_standard_detector(profile)
        ):
            detector_type = SpecialStandardDetector
        self.detector = detector_type(profile)
        if legacy_ruinous:
            controller_type = RuinousLegacyController
        elif detector_name == "noiseform":
            controller_type = NoiseformController
        else:
            controller_type = (
                SpecialStandardController
                if self._requires_special_standard_controller(profile)
                else StandardController
            )
        self.controller = controller_type(profile)
        self.executor = executor or InputDisabledExecutor()
        self.lifecycle = LifecycleMachine(profile)
        self.automate_lifecycle = automate_lifecycle
        self.coordinator = FishingCoordinator(profile, self.lifecycle, self.executor)

    @staticmethod
    def _requires_special_standard_detector(profile: dict) -> bool:
        """Keep post-backup detection features confined to rods that request them."""
        detection = profile.get("detection", {})
        bar_width = detection.get("bar_width", {})
        return any(
            (
                bool(detection.get("bar_color_relationships")),
                bool(detection.get("stick_detection")),
                bool(detection.get("rail_fallback")),
                bool(detection.get("bar_candidate_center_y")),
                detection.get("bar_candidate_maximum_vertical_shift_ratio") is not None,
                detection.get("effect_stick_tolerance") is not None,
                detection.get("effect_stick_height_ratio") is not None,
                bool(bar_width.get("learning_requires_lock", False)),
                bool(bar_width.get("allow_below_learned_width", False)),
                bool(bar_width.get("reject_fragment_too_small", False)),
            )
        )

    @staticmethod
    def _requires_special_standard_controller(profile: dict) -> bool:
        """Preserve only explicitly requested special-rod controller branches."""
        detector_name = profile.get("detection", {}).get("detector", "standard")
        controller = profile.get("controller", {})
        enabled_mechanics = profile.get("mechanics", {}).get("enabled", [])
        return any(
            (
                MacroPipeline._requires_special_standard_detector(profile),
                detector_name in {
                    "noiseform",
                    "pinions_aria",
                    "ruinous_oath",
                    "tranquility",
                },
                "rail_end_latching" in enabled_mechanics,
                controller.get("partial_detection_hold_ms") is not None,
                controller.get("arrow_fallback", {}).get("mode")
                == "width_aware_predictive",
            )
        )

    def reset(self, now_ms: float = 0.0) -> None:
        self.detector.reset()
        self.controller.reset()
        self.coordinator.reset(now_ms)

    def start_automation(self, now_ms: float) -> None:
        if not self.automate_lifecycle:
            raise RuntimeError("This pipeline was not created for live automation")
        self.coordinator.start(now_ms)

    def stop_automation(self, now_ms: float, reason: str = "emergency_stop") -> None:
        self.coordinator.stop(now_ms, reason)

    def process(self, packet: FramePacket) -> PipelineResult:
        observation = self.detector.detect(packet)
        previous_state = self.lifecycle.state
        state = (
            self.coordinator.update(observation)
            if self.automate_lifecycle
            else self.lifecycle.update(observation)
        )
        if previous_state == LifecycleState.RECOVERY and state in {
            LifecycleState.PREPARING,
            LifecycleState.CASTING,
        }:
            # Each fish begins with clean per-minigame geometry and motion
            # history. Wait-mode rods skip PREPARING and enter CASTING directly,
            # so both valid cycle-start transitions must clear stale tracking.
            self.detector.reset()
            self.controller.reset()
        command = self.controller.decide(observation)
        if self.automate_lifecycle and isinstance(self.detector, NoiseformDetector):
            self.detector.observe_command(command, observation)
        perfect_now, margin = self.controller.perfect_measurement(observation)
        if not self.automate_lifecycle or state == LifecycleState.MINIGAME:
            self.executor.submit(command)
        if self.automate_lifecycle and state == LifecycleState.MINIGAME:
            for key in observation.extra.get("rhythm_taps", []):
                self.executor.tap_key(str(key))
        return PipelineResult(packet, observation, state, command, perfect_now, margin)
