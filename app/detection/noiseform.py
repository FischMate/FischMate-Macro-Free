from __future__ import annotations

import cv2
import numpy as np

from app.core.models import (
    CommandAction,
    ControlCommand,
    DetectionSnapshot,
    FramePacket,
    NormalizedRect,
    PixelRect,
)
from app.detection.base import Detector
from app.detection.live_bar_width import LiveBarWidthTracker
from app.detection.trained_rod_vision import (
    TrainedRodVisionAssist,
    TrainedVisionFrame,
)


def _green_strength(image: np.ndarray) -> np.ndarray:
    pixels = image.astype(np.int16)
    blue, green, red = cv2.split(pixels)
    chroma = np.maximum(0, green - np.maximum(red, blue) - 12)
    brightness = np.maximum(0, green - 72)
    return chroma.astype(np.float32) + brightness.astype(np.float32) * 0.35


def _runs(values: np.ndarray) -> list[tuple[int, int]]:
    indexes = np.flatnonzero(values)
    if indexes.size == 0:
        return []
    result: list[tuple[int, int]] = []
    start = previous = int(indexes[0])
    for raw_value in indexes[1:]:
        value = int(raw_value)
        if value > previous + 1:
            result.append((start, previous + 1))
            start = value
        previous = value
    result.append((start, previous + 1))
    return result


class NoiseformDetector(Detector):
    """Video-derived tracker for Noiseform's animated green minigame.

    The five decorative arches are stationary while the filled control body
    moves beneath them. A per-catch minimum background removes those arches;
    the remaining moving green energy identifies the body even during its
    gradients, trigger tiles, and vertical flashes.
    """

    def __init__(self, profile: dict):
        self.profile = profile
        # Trained vision is deliberately task-only for Noiseform.  Normal bar
        # and stick geometry remains owned by the known-good classical tracker.
        self.trained_vision = TrainedRodVisionAssist(profile, "noiseform")
        self.live_bar_width = LiveBarWidthTracker(
            profile.get("detection", {}).get("fish_live_width")
        )
        self.active = False
        self.background_strength: np.ndarray | None = None
        self.background_luma: np.ndarray | None = None
        self.previous_bar_center: float | None = None
        self.bar_velocity_px_per_ms = 0.0
        self.previous_timestamp_ms: float | None = None
        self.previous_stick_x: float | None = None
        self.last_stick_timestamp_ms: float | None = None
        self.task_cue_hue: float | None = None
        self.task_cue_mode: str | None = None
        self.task_cue_value: float | None = None
        self.last_task_cue_timestamp_ms: float | None = None
        self.last_task_cue_scan_timestamp_ms: float | None = None
        self.task_cue_source = ""
        self.pending_task_cue: tuple[str, float | None, float | None] | None = None
        self.pending_task_cue_source = ""
        self.pending_task_cue_count = 0
        self.pending_task_cue_timestamp_ms: float | None = None
        self.pending_task_cue_saw_gap = False
        self.previous_task_cue_gray: np.ndarray | None = None
        self.task_cue_candidate_debug = ""
        self.task_target_x: float | None = None
        self.task_target_detected_timestamp_ms: float | None = None
        self.task_target_started_timestamp_ms: float | None = None
        self.last_task_target_timestamp_ms: float | None = None
        # Unlike last_task_target_timestamp_ms, this survives target release.
        # Width penalties become authoritative only after a confirmed rail task
        # has ended; task tiles themselves contain nested rectangles that must
        # never be promoted as the movable bar while the task is active.
        self.last_confirmed_task_observation_timestamp_ms: float | None = None
        self.last_visible_task_tile_row_timestamp_ms: float | None = None
        self.last_consumed_width_task_timestamp_ms: float | None = None
        self.last_task_tile_scan_timestamp_ms: float | None = None
        self.pending_task_target_x: float | None = None
        self.pending_task_target_count = 0
        self.pending_task_target_first_seen_timestamp_ms: float | None = None
        self.last_body_coverage = 0.0
        self.last_body_candidate_count = 0
        self.last_command_action = CommandAction.NEUTRAL
        self.command_anchor_timestamp_ms: float | None = None
        self.command_anchor_bar_center: float | None = None
        self.last_command_error_px: float | None = None
        self.last_command_target_x: float | None = None
        self.rejected_bar_center: float | None = None
        self.rejected_bar_until_ms: float | None = None
        self.bootstrap_started_timestamp_ms: float | None = None
        self.short_bar_recovery_enabled = False
        self.tracking_mode = "NORMAL"
        self.tracking_mode_since_ms: float | None = None
        self.tracking_mode_reason = "bootstrap"
        self.normal_mode_confirm_count = 0
        self.recovery_mode_confirm_count = 0
        self.current_bar_width: float | None = None
        self.pending_bar_width: float | None = None
        self.pending_bar_width_count = 0
        self.last_measured_bar_width: float | None = None
        self.last_bar_width_confidence = 0.0
        self.last_bar_width_source = "nominal"
        self.last_reliable_bar_center: float | None = None
        self.last_reliable_bar_timestamp_ms: float | None = None
        self.task_target_missing_scans = 0
        self.last_dynamic_width_state = "uninitialized"
        self.last_dynamic_width_rejection = ""
        self.last_dynamic_center_agreement = 0.0
        self.last_stick_jump_rejections = 0
        # Noiseform bar state is deliberately split into raw perception and
        # trusted playable geometry.  Only trusted observations are allowed to
        # feed motion/width history or reach the controller.
        self.last_raw_bar_center: float | None = None
        self.last_raw_bar_width: float | None = None
        self.last_raw_bar_source = ""
        self.last_raw_bar_confidence = 0.0
        self.last_bar_trust_state = "missing"
        self.last_bar_trust_reason = "uninitialized"
        self.last_bar_history_updated = False
        self.last_previous_trusted_bar_center: float | None = None
        self.last_structure_body_coverage = 0.0
        self.last_structure_left_boundary = 0.0
        self.last_structure_right_boundary = 0.0
        self.last_structure_top_coverage = 0.0
        self.last_structure_bottom_coverage = 0.0
        self.last_structure_green_range = 0.0
        self.last_width_transition_considered = False
        self.last_width_transition_accepted = False
        self.last_width_transition_reason = ""
        self.pending_edge_reacquire_center: float | None = None
        self.pending_edge_reacquire_count = 0
        self.last_edge_reacquire_confirmed = False
        self.pending_lower_outline_center: float | None = None
        self.pending_lower_outline_width: float | None = None
        self.pending_lower_outline_count = 0
        self.pending_lower_outline_motion_min: float | None = None
        self.pending_lower_outline_motion_max: float | None = None
        self.lower_outline_untrusted_count = 0
        self.last_lower_outline_center: float | None = None
        self.last_lower_outline_width: float | None = None
        self.last_lower_outline_score = 0.0
        self.last_lower_outline_state = "uninitialized"
        self.last_lower_outline_reason = ""

    def reset(self) -> None:
        self.live_bar_width.reset()
        self.trained_vision.reset()
        self.active = False
        self.background_strength = None
        self.background_luma = None
        self.previous_bar_center = None
        self.bar_velocity_px_per_ms = 0.0
        self.previous_timestamp_ms = None
        self.previous_stick_x = None
        self.last_stick_timestamp_ms = None
        self.task_cue_hue = None
        self.task_cue_mode = None
        self.task_cue_value = None
        self.last_task_cue_timestamp_ms = None
        self.last_task_cue_scan_timestamp_ms = None
        self.task_cue_source = ""
        self.pending_task_cue = None
        self.pending_task_cue_source = ""
        self.pending_task_cue_count = 0
        self.pending_task_cue_timestamp_ms = None
        self.pending_task_cue_saw_gap = False
        self.previous_task_cue_gray = None
        self.task_cue_candidate_debug = ""
        self.task_target_x = None
        self.task_target_detected_timestamp_ms = None
        self.task_target_started_timestamp_ms = None
        self.last_task_target_timestamp_ms = None
        self.last_confirmed_task_observation_timestamp_ms = None
        self.last_visible_task_tile_row_timestamp_ms = None
        self.last_consumed_width_task_timestamp_ms = None
        self.last_task_tile_scan_timestamp_ms = None
        self.pending_task_target_x = None
        self.pending_task_target_count = 0
        self.pending_task_target_first_seen_timestamp_ms = None
        self.last_body_coverage = 0.0
        self.last_body_candidate_count = 0
        self.last_command_action = CommandAction.NEUTRAL
        self.command_anchor_timestamp_ms = None
        self.command_anchor_bar_center = None
        self.last_command_error_px = None
        self.last_command_target_x = None
        self.rejected_bar_center = None
        self.rejected_bar_until_ms = None
        self.bootstrap_started_timestamp_ms = None
        self.short_bar_recovery_enabled = False
        self.tracking_mode = "NORMAL"
        self.tracking_mode_since_ms = None
        self.tracking_mode_reason = "reset"
        self.normal_mode_confirm_count = 0
        self.recovery_mode_confirm_count = 0
        self.current_bar_width = None
        self.pending_bar_width = None
        self.pending_bar_width_count = 0
        self.last_measured_bar_width = None
        self.last_bar_width_confidence = 0.0
        self.last_bar_width_source = "nominal"
        self.last_reliable_bar_center = None
        self.last_reliable_bar_timestamp_ms = None
        self.task_target_missing_scans = 0
        self.last_dynamic_width_state = "reset"
        self.last_dynamic_width_rejection = ""
        self.last_dynamic_center_agreement = 0.0
        self.last_stick_jump_rejections = 0
        self.last_raw_bar_center = None
        self.last_raw_bar_width = None
        self.last_raw_bar_source = ""
        self.last_raw_bar_confidence = 0.0
        self.last_bar_trust_state = "missing"
        self.last_bar_trust_reason = "reset"
        self.last_bar_history_updated = False
        self.last_previous_trusted_bar_center = None
        self.last_structure_body_coverage = 0.0
        self.last_structure_left_boundary = 0.0
        self.last_structure_right_boundary = 0.0
        self.last_structure_top_coverage = 0.0
        self.last_structure_bottom_coverage = 0.0
        self.last_structure_green_range = 0.0
        self.last_width_transition_considered = False
        self.last_width_transition_accepted = False
        self.last_width_transition_reason = ""
        self.pending_edge_reacquire_center = None
        self.pending_edge_reacquire_count = 0
        self.last_edge_reacquire_confirmed = False
        self.pending_lower_outline_center = None
        self.pending_lower_outline_width = None
        self.pending_lower_outline_count = 0
        self.pending_lower_outline_motion_min = None
        self.pending_lower_outline_motion_max = None
        self.lower_outline_untrusted_count = 0
        self.last_lower_outline_center = None
        self.last_lower_outline_width = None
        self.last_lower_outline_score = 0.0
        self.last_lower_outline_state = "reset"
        self.last_lower_outline_reason = ""

    def observe_command(
        self,
        command: ControlCommand,
        observation: DetectionSnapshot,
    ) -> None:
        """Remember live input so frozen visual locks can be invalidated."""
        if command.action == CommandAction.NEUTRAL:
            self.last_command_action = command.action
            self.command_anchor_timestamp_ms = None
            self.command_anchor_bar_center = None
            self.last_command_error_px = command.error_px
            self.last_command_target_x = command.target_x
            return
        target_jump = (
            command.target_x is not None
            and self.last_command_target_x is not None
            and abs(command.target_x - self.last_command_target_x)
            >= observation.frame_width
            * float(
                self.profile["detection"]["noiseform"].get(
                    "bar_response_target_jump_reset_ratio",
                    0.04,
                )
            )
        )
        if command.action != self.last_command_action or target_jump:
            self.command_anchor_timestamp_ms = observation.timestamp_ms
            self.command_anchor_bar_center = (
                None if observation.bar is None else observation.bar.center_x
            )
        elif self.command_anchor_bar_center is None and observation.bar is not None:
            self.command_anchor_timestamp_ms = observation.timestamp_ms
            self.command_anchor_bar_center = observation.bar.center_x
        self.last_command_action = command.action
        self.last_command_error_px = command.error_px
        self.last_command_target_x = command.target_x

    def detect(self, packet: FramePacket) -> DetectionSnapshot:
        frame = packet.frame_bgr
        height, width = frame.shape[:2]
        roi = NormalizedRect(
            *map(float, self.profile["detection"]["minigame_roi"])
        ).pixels(width, height)
        snapshot = DetectionSnapshot(
            timestamp_ms=packet.timestamp_ms,
            source_name=packet.source_name,
            frame_width=width,
            frame_height=height,
            minigame_roi=roi,
        )
        trained = self.trained_vision.detect(frame, packet.timestamp_ms)
        self._record_trained_telemetry(snapshot, trained)
        config = self.profile["detection"]["noiseform"]
        rail = NormalizedRect(*map(float, config["rail_rect"])).pixels(width, height)
        if rail.width <= 0 or rail.height <= 0:
            snapshot.rejection_reason = "noiseform_empty_rail"
            return snapshot

        rail_crop = frame[rail.top : rail.bottom, rail.left : rail.right]
        strength = _green_strength(rail_crop)
        luma = cv2.cvtColor(rail_crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
        pixels = rail_crop.astype(np.int16)
        blue, green, red = cv2.split(pixels)
        signature_mask = (
            (green >= int(config.get("signature_green_min", 90)))
            & ((green - red) >= int(config.get("signature_green_red_gap", 25)))
            & ((green - blue) >= int(config.get("signature_green_blue_gap", 15)))
        )
        signature_pixels = int(np.count_nonzero(signature_mask))
        signature_threshold = max(
            120,
            round(
                rail.width
                * rail.height
                * float(config.get("signature_minimum_fraction", 0.04))
            ),
        )
        signature_visible = signature_pixels >= signature_threshold
        snapshot.extra["noiseform_signature_pixels"] = signature_pixels
        snapshot.extra["noiseform_signature_threshold"] = signature_threshold

        if not signature_visible:
            self.reset()
            snapshot.rejection_reason = "noiseform_signature_missing"
            snapshot.detector_state = "SEARCHING"
            return snapshot

        snapshot.rail = rail
        if (
            not self.active
            or self.background_strength is None
            or self.background_strength.shape != strength.shape
            or self.background_luma is None
            or self.background_luma.shape != luma.shape
        ):
            self.active = True
            self.background_strength = strength.copy()
            self.background_luma = luma.copy()
            # Rail center is only a bootstrap search hint, never a trusted bar
            # observation.  Starting history here made later weak candidates
            # look continuous even though no playable bar had been seen.
            self.previous_bar_center = None
            self.bar_velocity_px_per_ms = 0.0
            self.previous_timestamp_ms = None
            self.bootstrap_started_timestamp_ms = packet.timestamp_ms

        nominal_bar_width = max(
            40,
            min(
                rail.width,
                round(width * float(config.get("bar_width_normalized", 0.1198))),
            ),
        )
        if self.current_bar_width is None:
            self.current_bar_width = float(nominal_bar_width)
            self.last_bar_width_source = "nominal_bootstrap"
        predicted_center = self.previous_bar_center or rail.center_x
        # Width is part of the live playable geometry. Noiseform may shorten
        # immediately after a missed color task, so width observation must stay
        # active while the cue and task tiles are on screen as well as during
        # ordinary tracking.
        measured_geometry = self._find_dynamic_bar_outline(
            rail_crop,
            strength,
            rail,
            nominal_bar_width,
            predicted_center,
            config,
        )
        if measured_geometry is not None:
            measured_center, measured_width, width_confidence = measured_geometry
        else:
            measured_center = None
            measured_width = None
            width_confidence = 0.0
            self.last_measured_bar_width = None
            self.last_bar_width_confidence = 0.0
            self.last_bar_width_source = "live_width_unobserved"
        bar_width = max(40, min(rail.width, round(self.current_bar_width)))
        preliminary_left = round(predicted_center - bar_width / 2.0)
        preliminary_left = max(
            rail.left,
            min(rail.right - bar_width, preliminary_left),
        )
        preliminary_bar = PixelRect(
            preliminary_left,
            rail.top,
            preliminary_left + bar_width,
            rail.bottom,
        )
        # The stick detector does not depend on the final bar candidate. Find it
        # first so competing bar geometries can be validated against the real
        # target instead of reinforcing one another around a decorative arch.
        stick_x, stick_confidence, stick_source = self._find_stick(
            frame,
            strength,
            rail,
            preliminary_bar,
            packet.timestamp_ms,
            config,
        )
        task_target_x, task_confidence = self._update_color_task(
            frame,
            packet.timestamp_ms,
            config,
            preliminary_bar,
            trained,
        )
        # Reset per-frame dynamic/transition telemetry before either bar path
        # runs.  The lower-outline tracker may accept a genuine persistent
        # penalty width; resetting these fields afterward would erase that
        # accepted transition from diagnostics even though the width state had
        # already changed.
        dynamic_width_accepted = False
        dynamic_center_trusted = False
        self.last_dynamic_width_state = "unobserved"
        self.last_dynamic_width_rejection = ""
        self.last_dynamic_center_agreement = 0.0
        self.last_width_transition_considered = False
        self.last_width_transition_accepted = False
        self.last_width_transition_reason = "unobserved"
        self.last_edge_reacquire_confirmed = False
        track_previous_center = self.previous_bar_center
        track_previous_timestamp_ms = self.previous_timestamp_ms
        lower_outline = self._find_lower_outline_bar(
            frame,
            rail,
            nominal_bar_width,
            predicted_center,
            stick_x,
            packet.timestamp_ms,
            config,
        )
        if lower_outline is not None:
            lower_center, lower_width, lower_score, lower_reason = lower_outline
            bar_center = lower_center
            bar_width = max(40, min(rail.width, round(lower_width)))
            residual_score = lower_score
            geometry_source = "noiseform_lower_outline_pair"
            self.last_raw_bar_center = lower_center
            self.last_raw_bar_width = lower_width
            self.last_raw_bar_source = geometry_source
            self.last_raw_bar_confidence = min(1.0, lower_score / 100.0)
            self.last_bar_trust_state = "trusted"
            self.last_bar_trust_reason = lower_reason
            self.last_bar_history_updated = False
            self.last_previous_trusted_bar_center = self.previous_bar_center
            self._update_trusted_bar_history(
                lower_center,
                packet.timestamp_ms,
                config,
            )
            self.last_reliable_bar_center = lower_center
            self.last_reliable_bar_timestamp_ms = packet.timestamp_ms
        else:
            bar_center, residual_score, geometry_source = self._track_bar(
                strength,
                luma,
                rail_crop,
                rail,
                bar_width,
                packet.timestamp_ms,
                config,
                stick_x,
            )
        if measured_center is not None and measured_width is not None:
            self.last_measured_bar_width = measured_width
            self.last_bar_width_confidence = width_confidence
            acceptance = float(config.get("bar_dynamic_outline_acceptance", 0.72))
            reference_center = (
                bar_center if bar_center is not None else track_previous_center
            )
            agreement = (
                0.0
                if reference_center is None
                else abs(measured_center - reference_center)
            )
            self.last_dynamic_center_agreement = agreement
            agreement_limit = max(
                8.0,
                nominal_bar_width
                * float(config.get("bar_dynamic_center_agreement_ratio", 0.34)),
            )
            elapsed = (
                0.0
                if track_previous_timestamp_ms is None
                else max(0.0, packet.timestamp_ms - track_previous_timestamp_ms)
            )
            temporal_limit = (
                nominal_bar_width
                * float(config.get("bar_dynamic_maximum_center_step_ratio", 0.20))
                + float(config.get("maximum_speed_px_per_ms", 0.85))
                * min(
                    elapsed,
                    float(config.get("bar_dynamic_center_step_growth_cap_ms", 240)),
                )
            )
            temporal_jump = (
                track_previous_center is not None
                and abs(measured_center - track_previous_center) > temporal_limit
            )
            structure = self._bar_structure_metrics(
                rail_crop,
                rail,
                measured_center,
                measured_width,
                config,
            )
            structure_ok, structure_reason = self._bar_structure_is_trusted(
                structure,
                config,
            )
            track_is_trusted = self.last_bar_trust_state == "trusted"
            # An internal arrow pair can reveal the correct center while
            # under-measuring width.  Rail-edge identity therefore uses the
            # larger trusted width, while the measured width remains blocked
            # from width history unless a color-task transition confirms it.
            edge_identity_width = max(
                measured_width,
                self.current_bar_width or float(nominal_bar_width),
            )
            measured_left = measured_center - edge_identity_width / 2.0
            measured_right = measured_center + edge_identity_width / 2.0
            edge_margin = nominal_bar_width * float(
                config.get("bar_dynamic_edge_reacquire_margin_ratio", 0.08)
            )
            near_rail_edge = (
                measured_left <= rail.left + edge_margin
                or measured_right >= rail.right - edge_margin
            )
            edge_reacquire_confirmed = False
            if temporal_jump and structure_ok and near_rail_edge:
                confirmation_tolerance = nominal_bar_width * float(
                    config.get("bar_dynamic_edge_reacquire_center_tolerance_ratio", 0.12)
                )
                if (
                    self.pending_edge_reacquire_center is not None
                    and abs(measured_center - self.pending_edge_reacquire_center)
                    <= confirmation_tolerance
                ):
                    self.pending_edge_reacquire_count += 1
                    self.pending_edge_reacquire_center += 0.30 * (
                        measured_center - self.pending_edge_reacquire_center
                    )
                else:
                    self.pending_edge_reacquire_center = measured_center
                    self.pending_edge_reacquire_count = 1
                edge_reacquire_confirmed = self.pending_edge_reacquire_count >= int(
                    config.get("bar_dynamic_edge_reacquire_confirm_frames", 3)
                )
                if edge_reacquire_confirmed:
                    temporal_jump = False
                    self.last_edge_reacquire_confirmed = True
            else:
                self.pending_edge_reacquire_center = None
                self.pending_edge_reacquire_count = 0
            if width_confidence < acceptance:
                self.last_dynamic_width_state = "rejected"
                self.last_dynamic_width_rejection = "confidence"
                self.last_width_transition_reason = "outline_confidence"
            elif not structure_ok:
                self.last_dynamic_width_state = "rejected"
                self.last_dynamic_width_rejection = structure_reason
                self.last_width_transition_reason = structure_reason
            elif temporal_jump:
                # A structurally complete outline still cannot teleport across
                # the rail.  Wait for trusted continuity to grow rather than
                # allowing an arch/tile pair to poison the next frame.
                self.last_dynamic_width_state = "rejected"
                self.last_dynamic_width_rejection = "implausible_trusted_center_jump"
                self.last_width_transition_reason = "implausible_trusted_center_jump"
            elif (
                track_is_trusted
                and reference_center is not None
                and agreement > agreement_limit
            ):
                self.last_dynamic_width_state = "rejected"
                self.last_dynamic_width_rejection = "trusted_center_disagreement"
                self.last_width_transition_reason = "trusted_center_disagreement"
            else:
                bootstrap_center_allowed = track_previous_center is not None or (
                    stick_x is not None
                    and stick_source
                    not in {"noiseform_temporary_prediction", "noiseform_stick_missing"}
                    and abs(measured_center - stick_x)
                    <= nominal_bar_width
                    * float(config.get("bar_dynamic_bootstrap_stick_agreement_ratio", 0.25))
                )
                if not bootstrap_center_allowed:
                    self.last_dynamic_width_state = "rejected"
                    self.last_dynamic_width_rejection = "bootstrap_without_real_stick_agreement"
                    self.last_width_transition_reason = (
                        "bootstrap_without_real_stick_agreement"
                    )
                else:
                    dynamic_center_trusted = True
                    dynamic_width_accepted = self._consider_trusted_bar_width(
                        measured_width,
                        nominal_bar_width,
                        packet.timestamp_ms,
                        config,
                    )
                    self.last_dynamic_width_state = (
                        "trusted_width_accepted"
                        if dynamic_width_accepted
                        else "trusted_width_pending"
                    )
        if (
            measured_center is not None
            and measured_width is not None
            and dynamic_center_trusted
            and self.last_bar_trust_state != "trusted"
        ):
            # A confirmed width transition is itself a multi-frame, complete
            # outline observation.  It may reacquire the bar, but only after
            # that confirmation; pending/weak outline frames remain PARTIAL.
            bar_center = measured_center
            bar_width = max(40, min(rail.width, round(self.current_bar_width)))
            residual_score = max(residual_score, 36.0 * width_confidence)
            geometry_source = (
                "noiseform_dynamic_outline"
                if dynamic_width_accepted
                else "noiseform_dynamic_center_only"
            )
            self.last_raw_bar_center = measured_center
            self.last_raw_bar_width = measured_width
            self.last_raw_bar_source = geometry_source
            self.last_raw_bar_confidence = width_confidence
            self.last_bar_trust_state = "trusted"
            self.last_bar_trust_reason = (
                "confirmed_structural_width_transition"
                if dynamic_width_accepted
                else (
                    "confirmed_rail_edge_reacquire"
                    if self.last_edge_reacquire_confirmed
                    else "trusted_center_width_retained"
                )
            )
            self._update_trusted_bar_history(measured_center, packet.timestamp_ms, config)
            self.last_reliable_bar_center = measured_center
            self.last_reliable_bar_timestamp_ms = packet.timestamp_ms
        if bar_center is None:
            self.lower_outline_untrusted_count += 1
        else:
            self.lower_outline_untrusted_count = 0
        if bar_center is not None:
            bar_left = round(bar_center - bar_width / 2.0)
            bar_left = max(rail.left, min(rail.right - bar_width, bar_left))
            snapshot.bar = PixelRect(
                bar_left,
                rail.top,
                bar_left + bar_width,
                rail.bottom,
            )
            snapshot.bar_confidence = min(1.0, 0.40 + residual_score / 80.0)
            snapshot.extra["raw_bar_left"] = bar_left
            snapshot.extra["raw_bar_right"] = bar_left + bar_width
        else:
            snapshot.extra["raw_bar_left"] = ""
            snapshot.extra["raw_bar_right"] = ""
            if bool(config.get("task_bar_prediction_enabled", True)) and (
                self.task_target_x is not None
            ):
                task_bar_center = (
                    self.last_reliable_bar_center
                    if self.last_reliable_bar_center is not None
                    else self.previous_bar_center
                )
                if task_bar_center is not None:
                    task_bar_left = round(task_bar_center - bar_width / 2.0)
                    task_bar_left = max(
                        rail.left,
                        min(rail.right - bar_width, task_bar_left),
                    )
                    snapshot.bar = PixelRect(
                        task_bar_left,
                        rail.top,
                        task_bar_left + bar_width,
                        rail.bottom,
                    )
                    snapshot.bar_confidence = float(
                        config.get("task_bar_prediction_confidence", 0.42)
                    )
                    geometry_source = "noiseform_color_task_prediction"
        snapshot.extra["raw_bar_width"] = bar_width
        snapshot.extra["learned_bar_width"] = bar_width
        snapshot.extra["noiseform_nominal_bar_width"] = nominal_bar_width
        snapshot.extra["noiseform_measured_bar_width"] = (
            "" if measured_width is None else round(measured_width, 2)
        )
        snapshot.extra["noiseform_bar_width_confidence"] = round(width_confidence, 3)
        snapshot.extra["noiseform_bar_width_source"] = self.last_bar_width_source
        snapshot.extra["live_width_enabled"] = self.live_bar_width.enabled
        snapshot.extra["live_bar_width"] = bar_width
        snapshot.extra["live_width_state"] = self.last_dynamic_width_state
        snapshot.extra["live_width_rejection_reason"] = (
            self.last_dynamic_width_rejection
        )
        snapshot.extra["noiseform_dynamic_center_agreement_px"] = round(
            self.last_dynamic_center_agreement,
            2,
        )
        snapshot.extra["noiseform_lower_outline_center"] = (
            ""
            if self.last_lower_outline_center is None
            else round(self.last_lower_outline_center, 2)
        )
        snapshot.extra["noiseform_lower_outline_width"] = (
            ""
            if self.last_lower_outline_width is None
            else round(self.last_lower_outline_width, 2)
        )
        snapshot.extra["noiseform_lower_outline_score"] = round(
            self.last_lower_outline_score,
            3,
        )
        snapshot.extra["noiseform_lower_outline_state"] = self.last_lower_outline_state
        snapshot.extra["noiseform_lower_outline_reason"] = self.last_lower_outline_reason
        snapshot.extra["noiseform_lower_outline_pending_count"] = (
            self.pending_lower_outline_count
        )
        snapshot.extra["noiseform_lower_outline_untrusted_count"] = (
            self.lower_outline_untrusted_count
        )
        snapshot.extra["bar_geometry_source"] = geometry_source
        snapshot.extra["noiseform_residual_score"] = round(residual_score, 3)
        snapshot.extra["noiseform_body_coverage"] = round(
            self.last_body_coverage,
            3,
        )
        snapshot.extra["noiseform_body_candidates"] = self.last_body_candidate_count
        snapshot.extra["noiseform_raw_bar_center"] = (
            "" if self.last_raw_bar_center is None else round(self.last_raw_bar_center, 2)
        )
        snapshot.extra["noiseform_raw_bar_width"] = (
            "" if self.last_raw_bar_width is None else round(self.last_raw_bar_width, 2)
        )
        snapshot.extra["noiseform_raw_bar_source"] = self.last_raw_bar_source
        snapshot.extra["noiseform_raw_bar_confidence"] = round(
            self.last_raw_bar_confidence,
            3,
        )
        snapshot.extra["noiseform_bar_trust_state"] = self.last_bar_trust_state
        snapshot.extra["noiseform_bar_trust_reason"] = self.last_bar_trust_reason
        snapshot.extra["noiseform_bar_history_updated"] = self.last_bar_history_updated
        snapshot.extra["noiseform_previous_trusted_bar_center"] = (
            ""
            if self.last_previous_trusted_bar_center is None
            else round(self.last_previous_trusted_bar_center, 2)
        )
        snapshot.extra["noiseform_trusted_bar_center"] = (
            "" if bar_center is None else round(bar_center, 2)
        )
        snapshot.extra["noiseform_trusted_bar_width"] = (
            "" if bar_center is None else bar_width
        )
        snapshot.extra["noiseform_structure_body_coverage"] = round(
            self.last_structure_body_coverage,
            3,
        )
        snapshot.extra["noiseform_structure_left_boundary"] = round(
            self.last_structure_left_boundary,
            3,
        )
        snapshot.extra["noiseform_structure_right_boundary"] = round(
            self.last_structure_right_boundary,
            3,
        )
        snapshot.extra["noiseform_structure_top_coverage"] = round(
            self.last_structure_top_coverage,
            3,
        )
        snapshot.extra["noiseform_structure_bottom_coverage"] = round(
            self.last_structure_bottom_coverage,
            3,
        )
        snapshot.extra["noiseform_structure_green_range"] = round(
            self.last_structure_green_range,
            3,
        )
        snapshot.extra["noiseform_width_transition_considered"] = (
            self.last_width_transition_considered
        )
        snapshot.extra["noiseform_width_transition_accepted"] = (
            self.last_width_transition_accepted
        )
        snapshot.extra["noiseform_width_transition_reason"] = (
            self.last_width_transition_reason
        )
        snapshot.extra["noiseform_edge_reacquire_count"] = (
            self.pending_edge_reacquire_count
        )
        snapshot.extra["noiseform_edge_reacquire_confirmed"] = (
            self.last_edge_reacquire_confirmed
        )

        tracking_center = (
            bar_center
            if bar_center is not None
            else self.previous_bar_center or rail.center_x
        )
        tracking_left = round(tracking_center - bar_width / 2.0)
        tracking_left = max(rail.left, min(rail.right - bar_width, tracking_left))
        tracking_bar = PixelRect(
            tracking_left,
            rail.top,
            tracking_left + bar_width,
            rail.bottom,
        )

        if stick_x is not None:
            stick_half = max(2, round(width * 0.0016))
            stick_top = max(0, rail.top - round(height * 0.006))
            stick_bottom = min(height, rail.bottom + round(height * 0.006))
            snapshot.stick = PixelRect(
                round(stick_x) - stick_half,
                stick_top,
                round(stick_x) + stick_half,
                stick_bottom,
            )
            snapshot.stick_confidence = stick_confidence
            snapshot.extra["noiseform_stick_source"] = stick_source
        snapshot.extra["noiseform_real_stick_x"] = (
            "" if stick_x is None else round(stick_x, 2)
        )
        snapshot.extra["noiseform_real_stick_confidence"] = round(stick_confidence, 3)
        snapshot.extra["noiseform_real_stick_source"] = stick_source
        snapshot.extra["noiseform_stick_jump_rejections"] = (
            self.last_stick_jump_rejections
        )

        snapshot.extra["noiseform_task_cue_hue"] = (
            "" if self.task_cue_hue is None else round(self.task_cue_hue, 2)
        )
        snapshot.extra["noiseform_task_cue_mode"] = self.task_cue_mode or ""
        snapshot.extra["noiseform_task_cue_value"] = (
            "" if self.task_cue_value is None else round(self.task_cue_value, 2)
        )
        snapshot.extra["noiseform_task_cue_source"] = self.task_cue_source
        snapshot.extra["noiseform_task_cue_candidate"] = self.task_cue_candidate_debug
        task_present = self.task_target_x is not None
        task_response_ready = task_target_x is not None
        snapshot.extra["noiseform_task_active"] = task_present
        snapshot.extra["noiseform_task_response_ready"] = task_response_ready
        snapshot.extra["noiseform_task_target_x"] = (
            "" if self.task_target_x is None else round(self.task_target_x, 2)
        )
        if self.task_target_detected_timestamp_ms is None:
            snapshot.extra["noiseform_task_wait_remaining_ms"] = ""
        else:
            response_delay_ms = float(config.get("task_response_delay_ms", 0))
            snapshot.extra["noiseform_task_wait_remaining_ms"] = round(
                max(
                    0.0,
                    response_delay_ms
                    - (packet.timestamp_ms - self.task_target_detected_timestamp_ms),
                ),
                1,
            )
        if task_target_x is not None:
            snapshot.extra["noiseform_tracking_stick_x"] = (
                "" if snapshot.stick is None else snapshot.stick.center_x
            )
            task_half = max(2, round(width * 0.0016))
            task_top = max(0, rail.top - round(height * 0.006))
            task_bottom = min(height, rail.bottom + round(height * 0.006))
            snapshot.stick = PixelRect(
                round(task_target_x) - task_half,
                task_top,
                round(task_target_x) + task_half,
                task_bottom,
            )
            snapshot.stick_confidence = task_confidence
            snapshot.extra["noiseform_stick_source"] = "noiseform_color_task"

        real_stick_rect = None
        if stick_x is not None:
            real_stick_rect = PixelRect(
                round(stick_x) - stick_half,
                stick_top,
                round(stick_x) + stick_half,
                stick_bottom,
            )
        self._update_tracking_mode(
            packet.timestamp_ms,
            task_present,
            snapshot.bar,
            real_stick_rect,
            geometry_source,
            config,
        )
        snapshot.extra["noiseform_mode"] = self.tracking_mode
        snapshot.extra["noiseform_mode_reason"] = self.tracking_mode_reason
        snapshot.extra["noiseform_mode_since_ms"] = (
            "" if self.tracking_mode_since_ms is None else self.tracking_mode_since_ms
        )

        # The Noiseform signature and stick are sufficient to keep lifecycle
        # ownership while the bar is deliberately marked lost and reacquired.
        snapshot.minigame_visible = snapshot.stick is not None
        snapshot.detector_state = (
            "LOCKED"
            if snapshot.bar is not None and snapshot.stick is not None
            else "PARTIAL"
        )
        if snapshot.bar is None:
            snapshot.rejection_reason = "noiseform_bar_unconfirmed"
        if snapshot.stick is None:
            snapshot.rejection_reason = "noiseform_stick_temporarily_missing"
        snapshot.extra["bar_candidate_count"] = int(snapshot.bar is not None)
        snapshot.extra["stick_candidate_count"] = int(snapshot.stick is not None)
        return snapshot

    @staticmethod
    def _record_trained_telemetry(
        snapshot: DetectionSnapshot,
        trained: TrainedVisionFrame,
    ) -> None:
        """Expose the task-only model without implying geometry authority."""
        snapshot.extra["trained_vision_role"] = "noiseform_task_tiles_only"
        snapshot.extra["trained_vision_available"] = trained.available
        snapshot.extra["trained_vision_ran"] = trained.ran_inference
        snapshot.extra["trained_vision_inference_ms"] = round(trained.inference_ms, 3)
        snapshot.extra["trained_vision_error"] = trained.error
        snapshot.extra["trained_vision_raw"] = ";".join(
            f"{item.class_name}:{item.confidence:.3f}:{item.center_x:.1f},{item.center_y:.1f}"
            for item in trained.raw
        )
        snapshot.extra["trained_vision_trusted"] = ";".join(
            f"{item.class_name}:{item.confidence:.3f}:{item.center_x:.1f},{item.center_y:.1f}"
            for item in trained.trusted
        )

    def _set_tracking_mode(
        self,
        mode: str,
        timestamp_ms: float,
        reason: str,
    ) -> None:
        if mode == self.tracking_mode:
            self.tracking_mode_reason = reason
            if self.tracking_mode_since_ms is None:
                self.tracking_mode_since_ms = timestamp_ms
            return
        self.tracking_mode = mode
        self.tracking_mode_since_ms = timestamp_ms
        self.tracking_mode_reason = reason
        self.normal_mode_confirm_count = 0
        self.recovery_mode_confirm_count = 0
        # A mode hand-off invalidates velocity learned under a different target.
        # Position remains available for continuity, but momentum must be relearned.
        self.bar_velocity_px_per_ms = 0.0
        self.command_anchor_timestamp_ms = None
        self.command_anchor_bar_center = None

    def _update_tracking_mode(
        self,
        timestamp_ms: float,
        task_active: bool,
        bar: PixelRect | None,
        real_stick: PixelRect | None,
        geometry_source: str,
        config: dict,
    ) -> None:
        if task_active:
            self._set_tracking_mode("COLOR_TASK", timestamp_ms, "validated_task_tiles")
            return

        if self.tracking_mode == "COLOR_TASK":
            self._set_tracking_mode("RECOVERY", timestamp_ms, "task_tiles_disappeared")
            return

        weak_sources = {
            "noiseform_bar_unconfirmed",
            "noiseform_rejected_frozen_center",
            "noiseform_frozen_weak_geometry",
        }
        detached = (
            bar is None
            or real_stick is None
            or not (bar.left <= real_stick.center_x <= bar.right)
            or geometry_source in weak_sources
        )
        if detached:
            self.normal_mode_confirm_count = 0
            self.recovery_mode_confirm_count += 1
            confirmations = int(config.get("mode_recovery_confirm_frames", 2))
            if self.tracking_mode == "RECOVERY" or self.recovery_mode_confirm_count >= confirmations:
                reason = (
                    "bar_or_stick_unconfirmed"
                    if bar is None or real_stick is None
                    else "bar_detached_from_real_stick"
                )
                self._set_tracking_mode("RECOVERY", timestamp_ms, reason)
            return

        self.recovery_mode_confirm_count = 0
        self.normal_mode_confirm_count += 1
        confirmations = int(config.get("mode_normal_confirm_frames", 3))
        if self.tracking_mode == "NORMAL" or self.normal_mode_confirm_count >= confirmations:
            self._set_tracking_mode("NORMAL", timestamp_ms, "coherent_bar_and_stick")

    def _recent_completed_color_task(
        self,
        timestamp_ms: float,
        config: dict,
    ) -> bool:
        """Whether a confirmed rail task ended recently enough to explain width loss.

        A cue alone is insufficient, and an active task is deliberately excluded:
        its rail tiles are the exact nested rectangles that previously caused a
        false 231px -> 160px playable-width transition.
        """
        observed = self.last_confirmed_task_observation_timestamp_ms
        return (
            self.task_target_x is None
            and observed is not None
            and 0.0 <= timestamp_ms - observed
            <= float(config.get("bar_width_recent_task_window_ms", 3600))
        )

    def _consider_trusted_bar_width(
        self,
        measured_width: float,
        nominal_width: int,
        timestamp_ms: float,
        config: dict,
    ) -> bool:
        """Update width only from a complete, already trusted outline.

        Stable measurements may refine the current width immediately.  A real
        shrink/growth is a state transition and must persist; one merged arch,
        tile, or fishing-line split therefore cannot resize the playable bar.
        The return value says whether the current outline is trusted for this
        frame, not merely whether a width transition was completed.
        """
        if self.current_bar_width is None:
            self.current_bar_width = float(nominal_width)
        tolerance = max(
            3.0,
            nominal_width * float(config.get("bar_width_stability_tolerance_ratio", 0.035)),
        )
        minimum_width = nominal_width * float(
            config.get("bar_width_state_minimum_ratio", 0.50)
        )
        maximum_width = nominal_width * float(
            config.get("bar_width_state_maximum_ratio", 1.08)
        )
        if not minimum_width <= measured_width <= maximum_width:
            self.pending_bar_width = None
            self.pending_bar_width_count = 0
            self.last_bar_width_source = "trusted_outline_rejected"
            self.last_width_transition_reason = "width_outside_playable_range"
            return False
        if abs(measured_width - self.current_bar_width) <= tolerance:
            smoothing = float(config.get("bar_width_smoothing", 0.25))
            self.current_bar_width += smoothing * (measured_width - self.current_bar_width)
            self.pending_bar_width = None
            self.pending_bar_width_count = 0
            self.last_bar_width_source = "trusted_outline_stable"
            self.last_width_transition_reason = "stable_trusted_width"
            return True

        self.last_width_transition_considered = True
        if (
            self.pending_bar_width is not None
            and abs(measured_width - self.pending_bar_width) <= tolerance
        ):
            self.pending_bar_width_count += 1
            self.pending_bar_width += 0.35 * (measured_width - self.pending_bar_width)
        else:
            self.pending_bar_width = measured_width
            self.pending_bar_width_count = 1
        recent_task = (
            self.task_target_x is not None
            and self.last_task_target_timestamp_ms is not None
        ) or self._recent_completed_color_task(timestamp_ms, config)
        shrinking = measured_width < self.current_bar_width
        if bool(config.get("bar_width_transition_requires_recent_task", True)) and not recent_task:
            self.pending_bar_width = None
            self.pending_bar_width_count = 0
            self.last_bar_width_source = "trusted_outline_transition_blocked"
            self.last_width_transition_reason = "no_recent_color_task"
            return False
        if shrinking:
            required = int(
                config.get(
                    "bar_width_shrink_confirm_frames_after_task"
                    if recent_task
                    else "bar_width_shrink_confirm_frames",
                    3 if recent_task else 4,
                )
            )
        else:
            required = int(
                config.get(
                    "bar_width_growth_confirm_frames_after_task"
                    if recent_task
                    else "bar_width_growth_confirm_frames",
                    4 if recent_task else 6,
                )
            )
        if self.pending_bar_width_count >= required:
            self.current_bar_width = float(self.pending_bar_width)
            self.pending_bar_width = None
            self.pending_bar_width_count = 0
            self.last_bar_width_source = "trusted_outline_transition"
            self.last_width_transition_accepted = True
            self.last_width_transition_reason = (
                "persistent_shrink_accepted"
                if shrinking
                else "persistent_growth_accepted"
            )
            return True
        else:
            self.last_bar_width_source = "trusted_outline_transition_pending"
            self.last_width_transition_reason = (
                f"{'shrink' if shrinking else 'growth'}_pending_"
                f"{self.pending_bar_width_count}_of_{required}"
            )
            return False

    @staticmethod
    def _bar_structure_metrics(
        rail_crop: np.ndarray,
        rail: PixelRect,
        center_x: float,
        candidate_width: float,
        config: dict,
    ) -> dict[str, float]:
        """Measure whether a proposal is one complete horizontal bar body."""
        gray = cv2.cvtColor(rail_crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
        if gray.size == 0:
            return {
                "body_coverage": 0.0,
                "left_boundary": 0.0,
                "right_boundary": 0.0,
                "top_coverage": 0.0,
                "bottom_coverage": 0.0,
                "green_range": 999.0,
            }
        local_center = center_x - rail.left
        left = max(0, round(local_center - candidate_width / 2.0))
        right = min(gray.shape[1], round(local_center + candidate_width / 2.0))
        if right - left < 16:
            return {
                "body_coverage": 0.0,
                "left_boundary": 0.0,
                "right_boundary": 0.0,
                "top_coverage": 0.0,
                "bottom_coverage": 0.0,
                "green_range": 999.0,
            }

        vertical_margin = max(2, round(gray.shape[0] * 0.16))
        middle = gray[vertical_margin : gray.shape[0] - vertical_margin]
        boundary_half = max(2, round(gray.shape[0] * 0.07))
        dark_max = int(config.get("bar_structure_boundary_dark_max", 75))

        def boundary_fraction(x: int) -> float:
            start = max(0, x - boundary_half)
            end = min(gray.shape[1], x + boundary_half + 1)
            patch = middle[:, start:end]
            return 0.0 if patch.size == 0 else float(np.mean(patch <= dark_max))

        inner_pad = max(4, round((right - left) * 0.025))
        body = middle[:, left + inner_pad : right - inner_pad]
        if body.size == 0:
            body_coverage = 0.0
            green_range = 999.0
        else:
            column_median = np.median(body, axis=0)
            body_coverage = float(
                np.mean(
                    column_median
                    >= float(config.get("bar_structure_body_luma_minimum", 70.0))
                )
            )
            pixels = rail_crop[
                vertical_margin : rail_crop.shape[0] - vertical_margin,
                left + inner_pad : right - inner_pad,
            ].astype(np.int16)
            blue, green, red = cv2.split(pixels)
            green_strength = np.maximum(
                0,
                green - np.maximum(red, blue) - 12,
            ).astype(np.float32) + np.maximum(0, green - 72).astype(np.float32) * 0.35
            green_columns = np.mean(green_strength, axis=0)
            green_range = float(
                np.percentile(green_columns, 90) - np.percentile(green_columns, 10)
            )

        edge_band = max(2, round(gray.shape[0] * 0.14))
        compare_band = max(2, round(gray.shape[0] * 0.18))
        top_edge = np.mean(gray[:edge_band, left:right], axis=0)
        top_inside = np.mean(
            gray[edge_band : edge_band + compare_band, left:right],
            axis=0,
        )
        bottom_edge = np.mean(gray[-edge_band:, left:right], axis=0)
        bottom_inside = np.mean(
            gray[-edge_band - compare_band : -edge_band, left:right],
            axis=0,
        )
        edge_delta = float(config.get("bar_structure_horizontal_edge_delta", 12.0))
        top_coverage = float(np.mean(np.abs(top_inside - top_edge) >= edge_delta))
        bottom_coverage = float(np.mean(np.abs(bottom_inside - bottom_edge) >= edge_delta))
        return {
            "body_coverage": body_coverage,
            "left_boundary": boundary_fraction(left),
            "right_boundary": boundary_fraction(right),
            "top_coverage": top_coverage,
            "bottom_coverage": bottom_coverage,
            "green_range": green_range,
        }

    def _bar_structure_is_trusted(
        self,
        metrics: dict[str, float],
        config: dict,
    ) -> tuple[bool, str]:
        self.last_structure_body_coverage = metrics["body_coverage"]
        self.last_structure_left_boundary = metrics["left_boundary"]
        self.last_structure_right_boundary = metrics["right_boundary"]
        self.last_structure_top_coverage = metrics["top_coverage"]
        self.last_structure_bottom_coverage = metrics["bottom_coverage"]
        self.last_structure_green_range = metrics["green_range"]
        if metrics["body_coverage"] < float(
            config.get("bar_structure_minimum_body_coverage", 0.90)
        ):
            return False, "incomplete_bar_body"
        minimum_boundary = float(
            config.get("bar_structure_minimum_boundary_coverage", 0.06)
        )
        if metrics["left_boundary"] < minimum_boundary:
            return False, "missing_left_outer_edge"
        if metrics["right_boundary"] < minimum_boundary:
            return False, "missing_right_outer_edge"
        if max(metrics["top_coverage"], metrics["bottom_coverage"]) < float(
            config.get("bar_structure_minimum_horizontal_edge_coverage", 0.10)
        ):
            return False, "missing_horizontal_body_edges"
        if metrics["green_range"] > float(
            config.get("bar_structure_maximum_green_range", 85.0)
        ):
            return False, "merged_or_sparse_green_fragments"
        return True, "complete_bar_body"

    def _find_lower_outline_bar(
        self,
        frame: np.ndarray,
        rail: PixelRect,
        nominal_width: int,
        predicted_center: float,
        stick_x: float | None,
        timestamp_ms: float,
        config: dict,
    ) -> tuple[float, float, float, str] | None:
        """Find the real Noiseform rectangle from its lower full-height borders.

        The ordinary tracker intentionally scans the bright upper portion of the
        minigame, where the green arches can obscure one or both bar edges.  The
        actual movable rectangle has a second, highly discriminative signature
        lower on screen: two straight dark vertical borders surrounding a
        comparatively continuous, low-green-range body.  This proposal is
        Noiseform-only and never updates history until continuity or a
        multi-frame reacquisition/width transition is confirmed.
        """
        self.last_lower_outline_center = None
        self.last_lower_outline_width = None
        self.last_lower_outline_score = 0.0
        self.last_lower_outline_state = "missing"
        self.last_lower_outline_reason = "no_lower_outline_candidate"
        if not bool(config.get("bar_lower_outline_enabled", True)):
            self.last_lower_outline_reason = "disabled"
            return None
        if self.task_target_x is not None:
            # The lower band is occupied by the color-task tiles while a task
            # is active.  Their bounded dark rectangles are visually valid
            # outline pairs but are not the player-controlled bar.  Keep this
            # normal-tracker assist isolated until the task row has vanished;
            # the established whole-body outline path still measures genuine
            # live width during the task.
            self.last_lower_outline_reason = "active_color_task_tiles_excluded"
            return None
        if self._task_glyph_row_visibly_present(frame, config):
            # Do not depend on the color-task latch for this exclusion.  It can
            # release a few frames before the rendered tiles disappear.  Their
            # white symbols remain a reliable, color-agnostic signal that the
            # lower outline band is not safe for normal bar identity.
            self.last_visible_task_tile_row_timestamp_ms = timestamp_ms
            self.last_lower_outline_reason = "visible_color_task_tiles_excluded"
            return None

        height, width = frame.shape[:2]
        rect = NormalizedRect(
            *map(
                float,
                config.get(
                    "bar_lower_outline_rect",
                    [0.296, 0.837, 0.704, 0.875],
                ),
            )
        ).pixels(width, height)
        crop = frame[rect.top : rect.bottom, rect.left : rect.right]
        if crop.size == 0 or crop.shape[0] < 12 or crop.shape[1] < nominal_width:
            self.last_lower_outline_reason = "empty_lower_outline_rect"
            return None

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        vertical_margin = max(2, round(gray.shape[0] * 0.16))
        middle = gray[vertical_margin : gray.shape[0] - vertical_margin]
        if middle.size == 0:
            self.last_lower_outline_reason = "empty_lower_outline_middle"
            return None

        dark_fraction = np.mean(
            middle <= int(config.get("bar_lower_outline_dark_max", 75)),
            axis=0,
        ).astype(np.float32)
        boundary_half = max(2, round(middle.shape[0] * 0.07))
        boundary_kernel = np.ones(boundary_half * 2 + 1, dtype=np.float32)
        boundary_kernel /= float(boundary_kernel.size)
        boundary = np.convolve(dark_fraction, boundary_kernel, mode="same")
        bright = (
            np.median(middle, axis=0)
            >= float(config.get("bar_lower_outline_body_luma_minimum", 70.0))
        ).astype(np.float32)
        bright_prefix = np.concatenate(
            (np.zeros(1, dtype=np.float32), np.cumsum(bright, dtype=np.float32))
        )

        current_width = float(self.current_bar_width or nominal_width)
        stable_width = max(40, min(rect.width, round(current_width)))
        widths = {stable_width}
        task_markers = [
            marker
            for marker in (
                self.last_visible_task_tile_row_timestamp_ms,
                self.last_confirmed_task_observation_timestamp_ms,
            )
            if marker is not None
        ]
        task_marker = max(task_markers, default=None)
        task_elapsed = (
            None if task_marker is None else timestamp_ms - float(task_marker)
        )
        settle_ms = float(config.get("bar_lower_outline_post_task_settle_ms", 450))
        recent_task = (
            task_marker is not None
            and task_elapsed is not None
            and settle_ms <= task_elapsed
            <= float(config.get("bar_width_recent_task_window_ms", 3600))
            and (
                self.last_consumed_width_task_timestamp_ms is None
                or task_marker > self.last_consumed_width_task_timestamp_ms
            )
        )
        # Only the color mechanic can create a persistent reduced width.  The
        # visually observed tile row is an independent trigger even when the
        # color-target latch misses or releases early; ordinary normal frames
        # cannot arm alternate-width search.
        transition_search = recent_task
        if transition_search:
            configured_minimum_width = max(
                40,
                round(
                    nominal_width
                    * float(config.get("bar_width_state_minimum_ratio", 0.50))
                ),
            )
            minimum_width = (
                configured_minimum_width
                if recent_task
                else max(
                    configured_minimum_width,
                    round(
                        current_width
                        * float(
                            config.get(
                                "bar_lower_outline_no_task_minimum_width_ratio",
                                0.68,
                            )
                        )
                    ),
                )
            )
            transition_maximum = min(
                stable_width - 1,
                round(
                    current_width
                    * float(config.get("bar_lower_outline_transition_maximum_ratio", 0.90))
                ),
            )
            if transition_maximum >= minimum_width:
                widths.update(
                    int(round(value))
                    for value in np.linspace(
                        minimum_width,
                        transition_maximum,
                        int(config.get("bar_lower_outline_width_hypotheses", 14)),
                    )
                )

        minimum_body = float(config.get("bar_lower_outline_minimum_body_coverage", 0.72))
        proposal_boundary = float(
            config.get("bar_lower_outline_proposal_boundary_coverage", 0.55)
        )
        trusted_boundary = float(
            config.get("bar_lower_outline_trusted_boundary_coverage", 0.65)
        )
        maximum_green_range = float(
            config.get("bar_lower_outline_maximum_green_range", 60.0)
        )
        candidates: list[tuple[float, float, float, dict[str, float]]] = []
        for candidate_width in sorted(widths):
            if candidate_width >= rect.width:
                continue
            lefts = np.arange(0, rect.width - candidate_width + 1, dtype=np.int32)
            rights = lefts + candidate_width
            right_indexes = np.minimum(rights, rect.width - 1)
            body_coverage = (
                bright_prefix[rights] - bright_prefix[lefts]
            ) / max(1, candidate_width)
            paired_boundary = np.minimum(boundary[lefts], boundary[right_indexes])
            valid = (body_coverage >= minimum_body) & (
                paired_boundary >= proposal_boundary
            )
            valid_indexes = np.flatnonzero(valid)
            if not valid_indexes.size:
                continue
            centers = rect.left + lefts.astype(np.float32) + candidate_width / 2.0
            preliminary = paired_boundary * 80.0 + body_coverage * 15.0
            preliminary -= (
                np.abs(centers - predicted_center)
                / max(1.0, rail.width)
                * float(config.get("bar_lower_outline_continuity_penalty", 2.0))
            )
            ranked = valid_indexes[
                np.argsort(preliminary[valid_indexes])[
                    -int(config.get("bar_lower_outline_candidates_per_width", 12)) :
                ]
            ]
            for index in ranked:
                center = float(centers[index])
                metrics = self._bar_structure_metrics(
                    crop,
                    rect,
                    center,
                    float(candidate_width),
                    config,
                )
                paired = min(metrics["left_boundary"], metrics["right_boundary"])
                if (
                    metrics["body_coverage"] < minimum_body
                    or paired < trusted_boundary
                    or metrics["green_range"] > maximum_green_range
                ):
                    continue
                score = (
                    metrics["body_coverage"] * 15.0
                    + paired * 80.0
                    + max(metrics["top_coverage"], metrics["bottom_coverage"]) * 4.0
                    - max(0.0, metrics["green_range"] - 52.0) * 0.20
                    - abs(center - predicted_center)
                    / max(1.0, rail.width)
                    * float(config.get("bar_lower_outline_continuity_penalty", 2.0))
                )
                candidates.append((score, center, float(candidate_width), metrics))

        if not candidates:
            return None

        stable_tolerance = max(
            3.0,
            nominal_width * float(config.get("bar_width_stability_tolerance_ratio", 0.035)),
        )
        stable_candidates = [
            item for item in candidates if abs(item[2] - current_width) <= stable_tolerance
        ]
        stable_candidate = max(stable_candidates, default=None, key=lambda item: item[0])
        transition_candidates = [
            item
            for item in candidates
            if item[2] < current_width - stable_tolerance
        ]
        transition_candidate = None
        if transition_search and transition_candidates:
            best_transition_score = max(item[0] for item in transition_candidates)
            score_margin = float(
                config.get("bar_lower_outline_transition_score_margin", 2.5)
            )
            competitive = [
                item
                for item in transition_candidates
                if item[0] >= best_transition_score - score_margin
            ]
            # Internal arrow edges produce many nested rectangles.  The real
            # outer body is the widest strongly supported member of the same
            # high-scoring cluster.
            transition_candidate = max(
                competitive,
                key=lambda item: (item[2], item[0]),
            )

        chosen = stable_candidate
        is_transition = False
        if chosen is None and transition_candidate is not None:
            chosen = transition_candidate
            is_transition = True
        if chosen is None:
            return None

        score, center, measured_width, metrics = chosen
        self.last_lower_outline_center = center
        self.last_lower_outline_width = measured_width
        self.last_lower_outline_score = score
        self.last_structure_body_coverage = metrics["body_coverage"]
        self.last_structure_left_boundary = metrics["left_boundary"]
        self.last_structure_right_boundary = metrics["right_boundary"]
        self.last_structure_top_coverage = metrics["top_coverage"]
        self.last_structure_bottom_coverage = metrics["bottom_coverage"]
        self.last_structure_green_range = metrics["green_range"]

        elapsed = (
            0.0
            if self.previous_timestamp_ms is None
            else max(0.0, timestamp_ms - self.previous_timestamp_ms)
        )
        continuity_limit = (
            current_width
            * float(config.get("bar_trusted_maximum_center_step_ratio", 0.16))
            + float(config.get("maximum_speed_px_per_ms", 0.85))
            * min(
                elapsed,
                float(config.get("bar_trusted_center_step_growth_cap_ms", 360)),
            )
        )
        continuous = self.previous_bar_center is not None and (
            abs(center - predicted_center) <= continuity_limit
        )
        if not is_transition and continuous:
            self.pending_lower_outline_center = None
            self.pending_lower_outline_width = None
            self.pending_lower_outline_count = 0
            self.pending_lower_outline_motion_min = None
            self.pending_lower_outline_motion_max = None
            self.last_lower_outline_state = "trusted"
            self.last_lower_outline_reason = "stable_complete_lower_outline"
            return center, current_width, score, self.last_lower_outline_reason

        center_tolerance = nominal_width * float(
            config.get("bar_lower_outline_confirmation_center_ratio", 0.22)
        )
        width_tolerance = nominal_width * float(
            config.get("bar_lower_outline_confirmation_width_ratio", 0.06)
        )
        if (
            self.pending_lower_outline_center is not None
            and self.pending_lower_outline_width is not None
            and abs(center - self.pending_lower_outline_center) <= center_tolerance
            and abs(measured_width - self.pending_lower_outline_width) <= width_tolerance
        ):
            self.pending_lower_outline_count += 1
            self.pending_lower_outline_motion_min = min(
                center,
                self.pending_lower_outline_motion_min
                if self.pending_lower_outline_motion_min is not None
                else center,
            )
            self.pending_lower_outline_motion_max = max(
                center,
                self.pending_lower_outline_motion_max
                if self.pending_lower_outline_motion_max is not None
                else center,
            )
            self.pending_lower_outline_center += 0.30 * (
                center - self.pending_lower_outline_center
            )
            self.pending_lower_outline_width += 0.30 * (
                measured_width - self.pending_lower_outline_width
            )
        else:
            self.pending_lower_outline_center = center
            self.pending_lower_outline_width = measured_width
            self.pending_lower_outline_count = 1
            self.pending_lower_outline_motion_min = center
            self.pending_lower_outline_motion_max = center

        if is_transition:
            required = int(
                config.get(
                    "bar_lower_outline_transition_confirm_frames"
                    if recent_task
                    else "bar_lower_outline_no_task_transition_confirm_frames",
                    3 if recent_task else 4,
                )
            )
        else:
            required = int(config.get("bar_lower_outline_reacquire_confirm_frames", 3))
        if self.previous_bar_center is None and not is_transition:
            required = int(config.get("bar_lower_outline_bootstrap_confirm_frames", 2))
        if self.pending_lower_outline_count < required:
            self.last_lower_outline_state = "partial"
            self.last_lower_outline_reason = (
                f"{'width_transition' if is_transition else 'reacquire'}_pending_"
                f"{self.pending_lower_outline_count}_of_{required}"
            )
            return None

        if is_transition and not recent_task:
            motion_span = (
                0.0
                if self.pending_lower_outline_motion_min is None
                or self.pending_lower_outline_motion_max is None
                else self.pending_lower_outline_motion_max
                - self.pending_lower_outline_motion_min
            )
            if motion_span < float(
                config.get("bar_lower_outline_no_task_minimum_motion_px", 5.0)
            ):
                self.last_lower_outline_state = "partial"
                self.last_lower_outline_reason = "no_task_width_transition_without_motion"
                return None

        confirmed_center = float(self.pending_lower_outline_center)
        confirmed_width = float(self.pending_lower_outline_width)
        self.pending_lower_outline_center = None
        self.pending_lower_outline_width = None
        self.pending_lower_outline_count = 0
        self.pending_lower_outline_motion_min = None
        self.pending_lower_outline_motion_max = None
        if is_transition:
            self.current_bar_width = confirmed_width
            if task_marker is not None:
                self.last_consumed_width_task_timestamp_ms = float(task_marker)
            self.pending_bar_width = None
            self.pending_bar_width_count = 0
            self.last_bar_width_source = "trusted_lower_outline_transition"
            self.last_width_transition_considered = True
            self.last_width_transition_accepted = True
            self.last_width_transition_reason = "persistent_penalty_width_accepted"
            self.last_lower_outline_reason = "confirmed_penalty_width_transition"
        else:
            self.last_lower_outline_reason = "confirmed_complete_lower_outline_reacquire"
        self.last_lower_outline_state = "trusted"
        return (
            confirmed_center,
            confirmed_width if is_transition else current_width,
            score,
            self.last_lower_outline_reason,
        )

    @staticmethod
    def _task_glyph_row_visibly_present(frame: np.ndarray, config: dict) -> bool:
        """Return True when at least two task symbols occupy one rail row."""
        height, width = frame.shape[:2]
        task_rect = NormalizedRect(
            *map(float, config.get("task_rail_rect", [0.20, 0.75, 0.80, 0.88]))
        ).pixels(width, height)
        crop = frame[task_rect.top : task_rect.bottom, task_rect.left : task_rect.right]
        if crop.size == 0:
            return False
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        white = (
            (hsv[:, :, 2] >= int(config.get("task_glyph_value_minimum", 145)))
            & (hsv[:, :, 1] <= int(config.get("task_glyph_saturation_maximum", 75)))
        ).astype(np.uint8)
        count, _, stats, centroids = cv2.connectedComponentsWithStats(white, 8)
        minimum_width = max(
            4,
            round(width * float(config.get("task_glyph_minimum_width_ratio", 0.004))),
        )
        maximum_width = max(
            minimum_width,
            round(width * float(config.get("task_glyph_maximum_width_ratio", 0.030))),
        )
        minimum_height = max(
            6,
            round(height * float(config.get("task_glyph_minimum_height_ratio", 0.008))),
        )
        maximum_height = max(
            minimum_height,
            round(height * float(config.get("task_glyph_maximum_height_ratio", 0.045))),
        )
        minimum_area = max(
            30,
            round(
                width
                * height
                * float(config.get("task_glyph_minimum_area_fraction", 0.000035))
            ),
        )
        centers: list[tuple[float, float]] = []
        for index in range(1, count):
            component_width = int(stats[index, cv2.CC_STAT_WIDTH])
            component_height = int(stats[index, cv2.CC_STAT_HEIGHT])
            area = int(stats[index, cv2.CC_STAT_AREA])
            center_x = task_rect.left + float(centroids[index, 0])
            center_y = task_rect.top + float(centroids[index, 1])
            if (
                minimum_width <= component_width <= maximum_width
                and minimum_height <= component_height <= maximum_height
                and area >= minimum_area
                and height
                * float(
                    config.get("bar_lower_outline_task_gate_center_y_minimum", 0.84)
                )
                <= center_y
                <= height * float(config.get("task_glyph_center_y_maximum", 0.86))
            ):
                half_width = max(
                    12,
                    round(width * float(config.get("task_tile_half_width", 0.024))),
                )
                half_height = max(
                    8,
                    round(height * float(config.get("task_tile_half_height", 0.017))),
                )
                left = max(0, round(center_x) - half_width)
                right = min(width, round(center_x) + half_width)
                top = max(0, round(center_y) - half_height)
                bottom = min(height, round(center_y) + half_height)
                tile_gray = cv2.cvtColor(
                    frame[top:bottom, left:right], cv2.COLOR_BGR2GRAY
                )
                if tile_gray.size == 0:
                    continue
                edge_width = max(2, round(tile_gray.shape[1] * 0.045))
                edge_pixels = np.concatenate(
                    (tile_gray[:, :edge_width].ravel(), tile_gray[:, -edge_width:].ravel())
                )
                dark_fraction = float(
                    np.mean(
                        tile_gray
                        <= int(config.get("bar_lower_outline_task_gate_dark_max", 95))
                    )
                )
                dark_edge_fraction = float(
                    np.mean(
                        edge_pixels
                        <= int(config.get("bar_lower_outline_task_gate_edge_dark_max", 80))
                    )
                )
                if (
                    dark_fraction
                    >= float(
                        config.get(
                            "bar_lower_outline_task_gate_minimum_dark_fraction", 0.25
                        )
                    )
                    and dark_edge_fraction
                    >= float(
                        config.get(
                            "bar_lower_outline_task_gate_minimum_edge_dark_fraction",
                            0.30,
                        )
                    )
                ):
                    centers.append((center_x, center_y))
        required = max(
            3,
            int(config.get("bar_lower_outline_task_gate_minimum_glyph_count", 3)),
        )
        tolerance = max(
            6.0,
            height
            * float(config.get("bar_lower_outline_task_gate_row_tolerance_ratio", 0.008)),
        )
        minimum_span = width * float(
            config.get("bar_lower_outline_task_gate_minimum_x_span_ratio", 0.18)
        )
        for _, center_y in centers:
            row = [
                center_x
                for center_x, other_y in centers
                if abs(other_y - center_y) <= tolerance
            ]
            groups: list[list[float]] = []
            merge_gap = width * float(
                config.get("task_glyph_component_merge_gap_ratio", 0.045)
            )
            for center_x in sorted(row):
                if not groups or center_x - groups[-1][-1] > merge_gap:
                    groups.append([center_x])
                else:
                    groups[-1].append(center_x)
            group_centers = [float(np.mean(group)) for group in groups]
            if (
                len(group_centers) >= required
                and max(group_centers) - min(group_centers) >= minimum_span
            ):
                return True
        return False

    def _update_trusted_bar_history(
        self,
        center: float,
        timestamp_ms: float,
        config: dict,
    ) -> None:
        previous_center = self.previous_bar_center
        previous_timestamp = self.previous_timestamp_ms
        self.last_previous_trusted_bar_center = previous_center
        self.last_bar_history_updated = True
        if previous_center is not None and previous_timestamp is not None:
            elapsed = max(1.0, timestamp_ms - previous_timestamp)
            if elapsed <= 180:
                maximum_speed = float(config.get("maximum_speed_px_per_ms", 0.85))
                instant_velocity = max(
                    -maximum_speed,
                    min(maximum_speed, (center - previous_center) / elapsed),
                )
                smoothing = float(config.get("velocity_smoothing", 0.35))
                self.bar_velocity_px_per_ms += smoothing * (
                    instant_velocity - self.bar_velocity_px_per_ms
                )
            else:
                self.bar_velocity_px_per_ms = 0.0
        else:
            self.bar_velocity_px_per_ms = 0.0
        self.previous_bar_center = center
        self.previous_timestamp_ms = timestamp_ms

    def _find_dynamic_bar_outline(
        self,
        rail_crop: np.ndarray,
        strength: np.ndarray,
        rail: PixelRect,
        nominal_width: int,
        predicted_center: float,
        config: dict,
    ) -> tuple[float, float, float] | None:
        """Find the real control body's two borders without assuming its width.

        Falling arrows contain dark pixels, but they form broad patches rather
        than two narrow full-height borders with brighter body pixels inside.
        Measuring both borders lets a shortened bar be tracked instead of
        lowering coverage for an object that is still incorrectly fixed-width.
        """
        if rail_crop.size == 0:
            return None
        gray = cv2.cvtColor(rail_crop, cv2.COLOR_BGR2GRAY)
        margin = max(1, round(gray.shape[0] * 0.12))
        middle = gray[margin : max(margin + 1, gray.shape[0] - margin), :]
        dark_max = int(config.get("bar_dynamic_outline_dark_max", 58))
        dark_counts = np.sum(middle <= dark_max, axis=0)
        minimum_rows = max(
            6,
            round(
                middle.shape[0]
                * float(config.get("bar_dynamic_outline_minimum_height_ratio", 0.72))
            ),
        )
        minimum_stem_width = int(config.get("bar_dynamic_outline_minimum_width", 2))
        maximum_stem_width = int(config.get("bar_dynamic_outline_maximum_width", 12))
        patch_half = max(8, round(rail.height * 0.42))
        maximum_patch_fraction = float(
            config.get("bar_dynamic_outline_maximum_dark_patch_fraction", 0.20)
        )
        stems: list[tuple[float, float, float]] = []
        for start, end in _runs(dark_counts >= minimum_rows):
            stem_width = end - start
            if not minimum_stem_width <= stem_width <= maximum_stem_width:
                continue
            center = (start + end) / 2.0
            patch_left = max(0, round(center) - patch_half)
            patch_right = min(gray.shape[1], round(center) + patch_half + 1)
            patch_fraction = float(
                np.mean(gray[:, patch_left:patch_right] <= dark_max)
            )
            if patch_fraction > maximum_patch_fraction:
                continue
            height_quality = float(np.max(dark_counts[start:end])) / max(1, middle.shape[0])
            stems.append((center, height_quality, patch_fraction))
        # Search all physically valid Noiseform widths every frame.  Acceptance
        # remains stateful below, so a short proposal is observable without a
        # cue but cannot resize history from one accidental outline pair.
        minimum_width_ratio = float(
            config.get("bar_width_state_minimum_ratio", 0.50)
        )
        maximum_width_ratio = float(
            config.get(
                "bar_width_search_maximum_ratio",
                config.get("bar_width_maximum_ratio", 1.10),
            )
        )
        minimum_width = nominal_width * minimum_width_ratio
        maximum_width = nominal_width * maximum_width_ratio
        strip = max(4, round(nominal_width * 0.035))
        gray_float = gray.astype(np.float32)
        current_width = self.current_bar_width or float(nominal_width)
        best: tuple[float, float, float, float] | None = None
        for index, (left_stem, left_height, left_patch) in enumerate(stems[:-1]):
            for right_stem, right_height, right_patch in stems[index + 1 :]:
                measured_width = right_stem - left_stem
                if not minimum_width <= measured_width <= maximum_width:
                    continue
                left = round(left_stem)
                right = round(right_stem)
                if left - strip < 0 or right + strip >= gray.shape[1]:
                    continue
                inside_left = gray_float[:, left + 1 : left + 1 + strip]
                inside_right = gray_float[:, right - strip : right]
                outside_left = gray_float[:, left - strip : left]
                outside_right = gray_float[:, right + 1 : right + 1 + strip]
                if not all(
                    region.size
                    for region in (inside_left, inside_right, outside_left, outside_right)
                ):
                    continue
                luma_contrast = min(
                    float(np.mean(inside_left) - np.mean(outside_left)),
                    float(np.mean(inside_right) - np.mean(outside_right)),
                )
                inside_strength = strength[:, left + 1 : right]
                outside_strength = np.concatenate(
                    (strength[:, left - strip : left], strength[:, right + 1 : right + 1 + strip]),
                    axis=1,
                )
                green_contrast = float(np.mean(inside_strength) - np.mean(outside_strength))
                combined_contrast = luma_contrast + max(0.0, green_contrast) * 0.22
                center_local = (left_stem + right_stem) / 2.0
                center_x = rail.left + center_local
                minimum_contrast = float(
                    config.get("bar_dynamic_outline_minimum_contrast", 5.0)
                )
                if combined_contrast < minimum_contrast:
                    # At a rail end, the real detached bar can be darker than
                    # the glow immediately outside it.  Permit that one special
                    # contrast inversion only when a full bar structure ends at
                    # the playable rail boundary.  Decorative center arches do
                    # not satisfy the boundary relationship.
                    edge_margin = nominal_width * float(
                        config.get("bar_dynamic_edge_reacquire_margin_ratio", 0.08)
                    )
                    near_rail_edge = (
                        left_stem <= edge_margin
                        or rail.width - right_stem <= edge_margin
                    )
                    edge_metrics = self._bar_structure_metrics(
                        rail_crop,
                        rail,
                        center_x,
                        measured_width,
                        config,
                    )
                    strong_edge_structure = (
                        edge_metrics["body_coverage"]
                        >= float(config.get("bar_structure_minimum_body_coverage", 0.90))
                        and edge_metrics["green_range"]
                        <= float(config.get("bar_structure_maximum_green_range", 85.0))
                        and max(
                            edge_metrics["top_coverage"],
                            edge_metrics["bottom_coverage"],
                        )
                        >= float(
                            config.get(
                                "bar_structure_minimum_horizontal_edge_coverage",
                                0.10,
                            )
                        )
                    )
                    if not (
                        near_rail_edge
                        and strong_edge_structure
                        and combined_contrast
                        >= float(
                            config.get(
                                "bar_dynamic_edge_reacquire_minimum_contrast",
                                -20.0,
                            )
                        )
                    ):
                        continue
                continuity = abs(center_x - predicted_center) / max(1.0, rail.width)
                width_error = abs(measured_width - current_width) / max(1.0, nominal_width)
                score = (
                    combined_contrast
                    + min(left_height, right_height) * 16.0
                    - (left_patch + right_patch) * 10.0
                    - width_error * float(config.get("bar_dynamic_width_penalty", 3.0))
                    - continuity
                    * float(
                        config.get(
                            "bar_dynamic_recovery_continuity_penalty"
                            if self.tracking_mode in {"RECOVERY", "COLOR_TASK"}
                            else "bar_dynamic_continuity_penalty",
                            0.8 if self.tracking_mode in {"RECOVERY", "COLOR_TASK"} else 2.5,
                        )
                    )
                )
                confidence = min(
                    1.0,
                    0.48
                    + max(0.0, combined_contrast) / 32.0
                    + min(left_height, right_height) * 0.18,
                )
                candidate = (score, center_x, measured_width, confidence)
                if best is None or candidate[0] > best[0]:
                    best = candidate
        if best is None:
            # The animated arches frequently hide one vertical border. Reuse
            # the existing full outline-pair recognizer across several widths;
            # only true border pairs may change width. Internal/falling arrow
            # pairs can recover a center elsewhere, but cannot teach width.
            minimum_ratio = float(config.get("bar_width_state_minimum_ratio", 0.50))
            ratios = np.linspace(
                minimum_ratio,
                float(
                    config.get(
                        "bar_width_search_maximum_ratio",
                        config.get("bar_width_maximum_ratio", 1.10),
                    )
                ),
                int(config.get("bar_width_hypothesis_count", 8)),
            )
            outline_best: tuple[float, float, float, float] | None = None
            for ratio in ratios:
                candidate_width = max(40, round(nominal_width * float(ratio)))
                candidate_center, candidate_confidence, candidate_source = (
                    self._find_bar_arrow_center(
                        rail_crop,
                        rail,
                        candidate_width,
                        predicted_center,
                        config,
                    )
                )
                if (
                    candidate_center is None
                    or candidate_source != "noiseform_black_outline_pair"
                ):
                    continue
                width_error = abs(candidate_width - (self.current_bar_width or nominal_width))
                score = candidate_confidence - width_error / max(1.0, nominal_width) * 0.08
                candidate = (
                    score,
                    candidate_center,
                    float(candidate_width),
                    candidate_confidence,
                )
                if outline_best is None or candidate[0] > outline_best[0]:
                    outline_best = candidate
            if outline_best is None:
                return None
            _, center_x, measured_width, confidence = outline_best
            learned_maximum = nominal_width * float(
                config.get("bar_width_maximum_ratio", 1.03)
            )
            return center_x, min(measured_width, learned_maximum), confidence
        _, center_x, measured_width, confidence = best
        learned_maximum = nominal_width * float(
            config.get("bar_width_maximum_ratio", 1.03)
        )
        return center_x, min(measured_width, learned_maximum), confidence

    def _update_color_task(
        self,
        frame: np.ndarray,
        timestamp_ms: float,
        config: dict,
        bar: PixelRect,
        trained: TrainedVisionFrame | None = None,
    ) -> tuple[float | None, float]:
        scan_interval_ms = float(config.get("task_cue_scan_interval_ms", 50))
        scan_due = (
            self.pending_task_cue is not None
            or self.last_task_cue_scan_timestamp_ms is None
            or timestamp_ms - self.last_task_cue_scan_timestamp_ms >= scan_interval_ms
        )
        cue_color: tuple[str, float | None, float | None] | None = None
        cue_source = ""
        confirmed_cue = False
        # A model-confirmed cue is authoritative until its tile row arrives.
        # The classical fallback is intentionally refreshed by later centered
        # pulses: the missed-model green/black prompts animate through several
        # phases, and the last real pulse must be able to replace an earlier
        # background change before the tiles appear.
        if self.task_target_x is None and scan_due:
            trained_cue = self._find_trained_task_cue_color(frame, trained, config)
            if trained_cue is not None:
                # The human-reviewed model supplies cue geometry only.  Color is
                # still sampled from the current pixels, so the model cannot
                # invent a target color.  A validated model cue may replace
                # stale classical cue memory left by an unrelated scene pulse.
                cue_color = trained_cue
                cue_source = "trained_center_cue"
                self.last_task_cue_scan_timestamp_ms = timestamp_ms
            elif self.task_cue_mode is None or (
                bool(config.get("task_refresh_classical_cue_while_pending", False))
                and self.task_cue_source != "trained_center_cue"
            ):
                # Every Noiseform prompt is a large centered flashing shape.  The
                # prompt itself has no white symbol; white symbols belong only to
                # the later rail tiles.  Use temporal shape onset for saturated,
                # black, gray, and white prompts alike so persistent world/UI
                # colors cannot arm a task.
                cue_color = self._find_center_task_cue_color(frame, config)
                cue_source = "temporal_center_shape"
                self.last_task_cue_scan_timestamp_ms = timestamp_ms
        if cue_color is not None:
            if cue_source == "trained_center_cue":
                self.task_cue_mode, self.task_cue_hue, self.task_cue_value = cue_color
                self.task_cue_source = cue_source
                self.pending_task_cue = None
                self.pending_task_cue_source = ""
                self.pending_task_cue_count = 0
                self.pending_task_cue_timestamp_ms = None
                self.pending_task_cue_saw_gap = False
                confirmed_cue = True
            else:
                confirmation_window_ms = float(
                    config.get("task_cue_confirmation_window_ms", 650)
                )
                if (
                    self.pending_task_cue is not None
                    and self.pending_task_cue_timestamp_ms is not None
                    and timestamp_ms - self.pending_task_cue_timestamp_ms
                    > confirmation_window_ms
                ):
                    self.pending_task_cue = None
                    self.pending_task_cue_source = ""
                    self.pending_task_cue_count = 0
                    self.pending_task_cue_timestamp_ms = None
                    self.pending_task_cue_saw_gap = False
                if self._task_cue_matches_pending(cue_color, config):
                    pulse_interval_ms = (
                        0.0
                        if self.pending_task_cue_timestamp_ms is None
                        else timestamp_ms - self.pending_task_cue_timestamp_ms
                    )
                    if (
                        self.pending_task_cue_saw_gap
                        and pulse_interval_ms
                        >= float(config.get("task_cue_minimum_pulse_interval_ms", 150))
                    ):
                        self.pending_task_cue_count += 1
                        self.pending_task_cue_timestamp_ms = timestamp_ms
                        self.pending_task_cue_saw_gap = False
                elif self.pending_task_cue is None:
                    self.pending_task_cue = cue_color
                    self.pending_task_cue_source = cue_source
                    self.pending_task_cue_count = 1
                    self.pending_task_cue_timestamp_ms = timestamp_ms
                    self.pending_task_cue_saw_gap = False
                else:
                    # A different observation between matching pulses is the
                    # required off-phase.  Do not let it overwrite cue memory.
                    self.pending_task_cue_saw_gap = True
                if self.pending_task_cue_count >= int(
                    config.get("task_cue_confirm_scans", 2)
                ):
                    assert self.pending_task_cue is not None
                    self.task_cue_mode, self.task_cue_hue, self.task_cue_value = (
                        self.pending_task_cue
                    )
                    self.task_cue_source = self.pending_task_cue_source
                    self.pending_task_cue = None
                    self.pending_task_cue_source = ""
                    self.pending_task_cue_count = 0
                    self.pending_task_cue_timestamp_ms = None
                    self.pending_task_cue_saw_gap = False
                    confirmed_cue = True
        elif self.pending_task_cue is not None:
            self.pending_task_cue_saw_gap = True
            confirmation_window_ms = float(
                config.get("task_cue_confirmation_window_ms", 650)
            )
            if (
                self.pending_task_cue_timestamp_ms is None
                or timestamp_ms - self.pending_task_cue_timestamp_ms
                > confirmation_window_ms
            ):
                self.pending_task_cue = None
                self.pending_task_cue_source = ""
                self.pending_task_cue_count = 0
                self.pending_task_cue_timestamp_ms = None
                self.pending_task_cue_saw_gap = False
        if confirmed_cue:
            # A missed Noiseform task permanently shortens the control body
            # for the rest of the catch.  Once any task has appeared, retain a
            # lower-coverage recovery path even after its tiles disappear.
            self.short_bar_recovery_enabled = True
            self.last_task_cue_timestamp_ms = timestamp_ms

        pending_hold_ms = float(config.get("task_cue_pending_hold_ms", 4500))
        cue_recent = (
            self.task_cue_mode is not None
            and self.last_task_cue_timestamp_ms is not None
            and timestamp_ms - self.last_task_cue_timestamp_ms <= pending_hold_ms
        )
        tile_scan_interval_ms = float(config.get("task_tile_scan_interval_ms", 50))
        tile_scan_due = (
            self.last_task_tile_scan_timestamp_ms is None
            or timestamp_ms - self.last_task_tile_scan_timestamp_ms
            >= tile_scan_interval_ms
        )
        if (
            self.task_target_x is not None
            and self.task_target_started_timestamp_ms is not None
            and timestamp_ms - self.task_target_started_timestamp_ms
            > float(config.get("task_target_maximum_duration_ms", 1250))
        ):
            self.task_target_x = None
            self.task_target_detected_timestamp_ms = None
            self.task_target_started_timestamp_ms = None
            self.last_task_target_timestamp_ms = None
            self.last_task_tile_scan_timestamp_ms = None
            self.pending_task_target_x = None
            self.pending_task_target_count = 0
            self.pending_task_target_first_seen_timestamp_ms = None
            self.task_target_missing_scans = 0
            self.task_cue_hue = None
            self.task_cue_mode = None
            self.task_cue_value = None
            self.task_cue_source = ""
            self.last_task_cue_timestamp_ms = None
            return None, 0.0
        if (cue_recent or self.task_target_x is not None) and tile_scan_due:
            target = self._find_task_tile_x(
                frame,
                self.task_cue_mode or "hue",
                self.task_cue_hue,
                self.task_cue_value,
                config,
                self.task_target_x,
                trained,
            )
            self.last_task_tile_scan_timestamp_ms = timestamp_ms
            if target is not None:
                self.task_target_missing_scans = 0
                target_x, confidence = target
                if self.task_target_x is None:
                    stability_tolerance = frame.shape[1] * float(
                        config.get("task_target_stability_tolerance", 0.012)
                    )
                    if (
                        self.pending_task_target_x is not None
                        and abs(target_x - self.pending_task_target_x)
                        <= stability_tolerance
                    ):
                        self.pending_task_target_count += 1
                        self.pending_task_target_x += 0.35 * (
                            target_x - self.pending_task_target_x
                        )
                    else:
                        self.pending_task_target_x = target_x
                        self.pending_task_target_count = 1
                        self.pending_task_target_first_seen_timestamp_ms = timestamp_ms
                    confirm_scans = int(config.get("task_target_confirm_scans", 2))
                    if (
                        self.task_cue_mode == "achromatic"
                        and self.task_cue_value is not None
                        and self.task_cue_value
                        <= float(config.get("task_tile_black_cue_value_maximum", 70))
                    ):
                        confirm_scans = int(
                            config.get("task_target_black_confirm_scans", 3)
                        )
                    if self.pending_task_target_count < confirm_scans:
                        return None, 0.0
                    self.task_target_x = self.pending_task_target_x
                    self.task_target_detected_timestamp_ms = (
                        self.pending_task_target_first_seen_timestamp_ms
                        if self.pending_task_target_first_seen_timestamp_ms is not None
                        else timestamp_ms
                    )
                    self.task_target_started_timestamp_ms = None
                    self.pending_task_target_x = None
                    self.pending_task_target_count = 0
                    self.pending_task_target_first_seen_timestamp_ms = None
                # Task blocks are screen-fixed.  Re-detection only proves the
                # latched block is still present; allowing its center to drift
                # can walk the target toward neighboring patterned glyphs.
                self.last_task_target_timestamp_ms = timestamp_ms
                self.last_confirmed_task_observation_timestamp_ms = timestamp_ms
                if self._task_response_ready(timestamp_ms, config):
                    return self.task_target_x, confidence
                return None, 0.0
            if self.task_target_x is not None:
                # The trained detector may temporarily omit the specific tile
                # underneath effects while still seeing another validated tile
                # in the same row.  That proves the task is still on screen, so
                # keep the already-confirmed target instead of clearing it and
                # re-reading ordinary scene pulses as a new cue.
                height, width = frame.shape[:2]
                task_rect = NormalizedRect(
                    *map(
                        float,
                        config.get("task_rail_rect", [0.285, 0.78, 0.715, 0.845]),
                    )
                ).pixels(width, height)
                row_anchor_visible = bool(
                    self._trained_task_tile_candidates(
                        frame,
                        trained,
                        task_rect,
                        config,
                    )
                )
                if row_anchor_visible:
                    self.task_target_missing_scans = 0
                    self.last_task_target_timestamp_ms = timestamp_ms
                    if self._task_response_ready(timestamp_ms, config):
                        return self.task_target_x, 0.68
                    return None, 0.0
                self.task_target_missing_scans += 1
            else:
                # An unlatched proposal must be continuously present.  Do not
                # let a single stale tile observation combine with a later row.
                self.pending_task_target_x = None
                self.pending_task_target_count = 0
                self.pending_task_target_first_seen_timestamp_ms = None

        if self.task_target_x is not None and not self._task_response_ready(
            timestamp_ms,
            config,
        ):
            # The answer is latched from the first stable tile row, but normal
            # stick tracking owns the requested one-second observation period.
            # Tile effects may obscure later scans, so do not discard the
            # remembered answer before that response window opens.
            return None, 0.0

        minimum_response_hold_ms = float(
            config.get("task_response_minimum_hold_ms", 600)
        )
        if (
            self.task_target_x is not None
            and self.task_target_started_timestamp_ms is not None
            and timestamp_ms - self.task_target_started_timestamp_ms
            <= minimum_response_hold_ms
        ):
            return self.task_target_x, 0.72

        release_hold_ms = float(config.get("task_target_release_hold_ms", 180))
        missing_scan_confirmed = self.task_target_missing_scans >= int(
            config.get("task_target_missing_scans", 3)
        )
        if (
            self.task_target_x is not None
            and self.last_task_target_timestamp_ms is not None
            and (
                not missing_scan_confirmed
                or timestamp_ms - self.last_task_target_timestamp_ms <= release_hold_ms
            )
        ):
            return self.task_target_x, 0.72

        if self.task_target_x is not None:
            self.task_target_x = None
            self.task_target_detected_timestamp_ms = None
            self.task_target_started_timestamp_ms = None
            self.last_task_target_timestamp_ms = None
            self.last_task_tile_scan_timestamp_ms = None
            self.pending_task_target_x = None
            self.pending_task_target_count = 0
            self.pending_task_target_first_seen_timestamp_ms = None
            self.task_target_missing_scans = 0
            self.task_cue_hue = None
            self.task_cue_mode = None
            self.task_cue_value = None
            self.task_cue_source = ""
            self.last_task_cue_timestamp_ms = None
        elif not cue_recent:
            self.task_cue_hue = None
            self.task_cue_mode = None
            self.task_cue_value = None
            self.task_cue_source = ""
            self.last_task_cue_timestamp_ms = None
        return None, 0.0

    def _task_response_ready(self, timestamp_ms: float, config: dict) -> bool:
        """Open control ownership only after the configured tile observation delay."""
        if self.task_target_x is None:
            return False
        if self.task_target_detected_timestamp_ms is None:
            self.task_target_detected_timestamp_ms = (
                self.last_task_target_timestamp_ms
                if self.last_task_target_timestamp_ms is not None
                else timestamp_ms
            )
        delay_ms = max(0.0, float(config.get("task_response_delay_ms", 0)))
        if timestamp_ms - self.task_target_detected_timestamp_ms < delay_ms:
            return False
        if self.task_target_started_timestamp_ms is None:
            self.task_target_started_timestamp_ms = timestamp_ms
        return True

    def _task_cue_matches_pending(
        self,
        cue: tuple[str, float | None, float | None],
        config: dict,
    ) -> bool:
        pending = self.pending_task_cue
        if pending is None or cue[0] != pending[0]:
            return False
        if cue[0] == "hue":
            if cue[1] is None or pending[1] is None:
                return False
            delta = abs(cue[1] - pending[1])
            delta = min(delta, 180.0 - delta)
            return delta <= float(config.get("task_cue_confirm_hue_tolerance", 10))
        if cue[2] is None or pending[2] is None:
            return False
        return abs(cue[2] - pending[2]) <= float(
            config.get("task_cue_confirm_value_tolerance", 32)
        )

    def _find_center_task_cue_color(
        self,
        frame: np.ndarray,
        config: dict,
    ) -> tuple[str, float | None, float | None] | None:
        """Read a flashing centered cue by temporal shape and glyph identity."""
        self.task_cue_candidate_debug = ""
        height, width = frame.shape[:2]
        cue_rect = NormalizedRect(
            *map(float, config.get("task_cue_rect", [0.35, 0.18, 0.65, 0.72]))
        ).pixels(width, height)
        crop = frame[cue_rect.top : cue_rect.bottom, cue_rect.left : cue_rect.right]
        if crop.size == 0:
            return None
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        previous = self.previous_task_cue_gray
        self.previous_task_cue_gray = gray.copy()
        if previous is None or previous.shape != gray.shape:
            return None
        difference = cv2.absdiff(gray, previous)
        threshold = int(config.get("task_cue_shape_difference_minimum", 18))
        raw_changed = difference >= threshold
        changed = raw_changed.astype(np.uint8) * 255
        kernel_size = max(
            5,
            round(min(width, height) * float(
                config.get("task_cue_shape_close_kernel_ratio", 0.012)
            )),
        )
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (kernel_size, kernel_size),
        )
        changed = cv2.morphologyEx(changed, cv2.MORPH_CLOSE, kernel)
        changed = cv2.dilate(changed, kernel, iterations=1)
        count, _, stats, centroids = cv2.connectedComponentsWithStats(changed, 8)
        best: tuple[int, int, int, int] | None = None
        best_score = -1.0
        for index in range(1, count):
            component_left = int(stats[index, cv2.CC_STAT_LEFT])
            component_top = int(stats[index, cv2.CC_STAT_TOP])
            component_width = int(stats[index, cv2.CC_STAT_WIDTH])
            component_height = int(stats[index, cv2.CC_STAT_HEIGHT])
            area = int(stats[index, cv2.CC_STAT_AREA])
            center_x = cue_rect.left + float(centroids[index, 0])
            center_y = cue_rect.top + float(centroids[index, 1])
            aspect = component_width / max(1.0, component_height)
            extent = area / max(1.0, component_width * component_height)
            if not (
                width * float(config.get("task_cue_shape_width_minimum", 0.09))
                <= component_width
                <= width * float(config.get("task_cue_shape_width_maximum", 0.32))
                and height * float(config.get("task_cue_shape_height_minimum", 0.14))
                <= component_height
                <= height * float(config.get("task_cue_shape_height_maximum", 0.48))
                and float(config.get("task_cue_shape_aspect_minimum", 0.62))
                <= aspect
                <= float(config.get("task_cue_shape_aspect_maximum", 1.45))
                and extent >= float(config.get("task_cue_shape_extent_minimum", 0.20))
            and width * float(config.get("task_cue_center_x_minimum", 0.39))
            <= center_x
            <= width * float(config.get("task_cue_center_x_maximum", 0.61))
            and height * float(config.get("task_cue_center_y_minimum", 0.30))
            <= center_y
            <= height * float(config.get("task_cue_center_y_maximum", 0.70))
            ):
                continue
            centeredness = abs(center_x - width / 2.0) / max(1.0, width)
            score = area - centeredness * width * height * 0.10
            if score > best_score:
                best_score = score
                best = (
                    component_left,
                    component_top,
                    component_width,
                    component_height,
                )
        if best is None:
            fixed_glyph = NoiseformDetector._find_fixed_glyph_task_cue(
                frame,
                config,
            )
            if fixed_glyph is not None:
                return fixed_glyph
            return NoiseformDetector._find_fixed_black_task_cue(
                crop,
                previous,
                width,
                height,
                cue_rect,
                config,
            )
        component_left, component_top, component_width, component_height = best
        self.task_cue_candidate_debug = (
            f"x={cue_rect.left + component_left}:"
            f"y={cue_rect.top + component_top}:"
            f"w={component_width}:h={component_height}:score={best_score:.1f}"
        )
        shape = crop[
            component_top : component_top + component_height,
            component_left : component_left + component_width,
        ]
        shape_change_mask = raw_changed[
            component_top : component_top + component_height,
            component_left : component_left + component_width,
        ]
        if shape.size == 0:
            return None
        glyph_shape = shape
        # Ignore the changing outline and sample the cue body's interior.
        inset_x = max(2, round(component_width * 0.18))
        inset_y = max(2, round(component_height * 0.18))
        if component_width > inset_x * 2 and component_height > inset_y * 2:
            shape = shape[inset_y:-inset_y, inset_x:-inset_x]
            shape_change_mask = shape_change_mask[
                inset_y:-inset_y,
                inset_x:-inset_x,
            ]
        classified = NoiseformDetector._classify_task_cue_shape(
            shape,
            config,
            sample_mask=shape_change_mask,
        )
        if classified is not None:
            if not bool(config.get("task_cue_require_contrast_glyph", False)):
                return classified
            glyph_hsv = cv2.cvtColor(glyph_shape, cv2.COLOR_BGR2HSV)
            if NoiseformDetector._has_task_cue_glyph(
                glyph_hsv,
                float(classified[2] or 0.0),
                config,
            ):
                return classified
            fixed_glyph = NoiseformDetector._find_fixed_glyph_task_cue(
                frame,
                config,
            )
            if fixed_glyph is not None:
                return fixed_glyph
        fixed_glyph = NoiseformDetector._find_fixed_glyph_task_cue(frame, config)
        if fixed_glyph is not None:
            return fixed_glyph
        return NoiseformDetector._find_fixed_black_task_cue(
            crop,
            previous,
            width,
            height,
            cue_rect,
            config,
        )

    @staticmethod
    def _find_trained_task_cue_color(
        frame: np.ndarray,
        trained: TrainedVisionFrame | None,
        config: dict,
    ) -> tuple[str, float | None, float | None] | None:
        """Validate model cue geometry, then read its color from live pixels."""
        if trained is None:
            return None
        height, width = frame.shape[:2]
        candidates = sorted(
            trained.detections("cue", trusted=True),
            key=lambda item: item.confidence,
            reverse=True,
        )
        for item in candidates:
            width_ratio = item.width / max(1.0, width)
            height_ratio = item.height / max(1.0, height)
            aspect = item.width / max(1.0, item.height)
            if not (
                float(config.get("task_cue_shape_width_minimum", 0.09))
                <= width_ratio
                <= float(config.get("task_cue_shape_width_maximum", 0.32))
                and float(config.get("task_cue_shape_height_minimum", 0.14))
                <= height_ratio
                <= float(config.get("task_cue_shape_height_maximum", 0.48))
                and float(config.get("task_cue_shape_aspect_minimum", 0.62))
                <= aspect
                <= float(config.get("task_cue_shape_aspect_maximum", 1.45))
                and width * float(config.get("task_cue_center_x_minimum", 0.39))
                <= item.center_x
                <= width * float(config.get("task_cue_center_x_maximum", 0.61))
                and height * float(config.get("task_cue_center_y_minimum", 0.30))
                <= item.center_y
                <= height * float(config.get("task_cue_center_y_maximum", 0.70))
            ):
                continue
            shape = frame[item.top : item.bottom, item.left : item.right]
            if shape.size == 0:
                continue
            inset_x = max(2, round(shape.shape[1] * 0.18))
            inset_y = max(2, round(shape.shape[0] * 0.18))
            if shape.shape[1] > inset_x * 2 and shape.shape[0] > inset_y * 2:
                shape = shape[inset_y:-inset_y, inset_x:-inset_x]
            cue = NoiseformDetector._classify_task_cue_shape(shape, config)
            if cue is not None:
                return cue
        return None

    @staticmethod
    def _classify_task_cue_shape(
        shape: np.ndarray,
        config: dict,
        sample_mask: np.ndarray | None = None,
    ) -> tuple[str, float | None, float | None] | None:
        if shape.size == 0:
            return None
        shape_hsv = cv2.cvtColor(shape, cv2.COLOR_BGR2HSV)
        samples = shape_hsv.reshape(-1, 3)
        if sample_mask is not None and sample_mask.shape == shape_hsv.shape[:2]:
            temporal_samples = samples[sample_mask.reshape(-1).astype(bool)]
            minimum_temporal_samples = max(64, round(samples.shape[0] * 0.02))
            if temporal_samples.shape[0] >= minimum_temporal_samples:
                samples = temporal_samples
        colored_mask = (
            (samples[:, 1] >= int(config.get("task_cue_saturation_minimum", 100)))
            & (
                samples[:, 2]
                >= int(config.get("task_cue_colored_value_minimum", 90))
            )
        )
        hue_minimum = float(config.get("task_cue_colored_hue_minimum", 45))
        hue_maximum = float(config.get("task_cue_colored_hue_maximum", 95))
        green_mask = colored_mask & (
            (samples[:, 0] >= hue_minimum) & (samples[:, 0] <= hue_maximum)
        )
        green_fraction = float(np.count_nonzero(green_mask) / len(samples))
        if green_fraction >= float(
            config.get("task_cue_colored_minimum_fraction", 0.15)
        ):
            hue = float(np.median(samples[:, 0][green_mask]))
            value = float(np.median(samples[:, 2][green_mask]))
            return "hue", hue, value
        colored_fraction = float(np.count_nonzero(colored_mask) / len(samples))
        if colored_fraction >= float(
            config.get("task_cue_colored_minimum_fraction", 0.15)
        ):
            # A large bright purple/orange player effect is not an achromatic
            # cue merely because the night background inside its box is dark.
            return None

        achromatic_mask = samples[:, 1] <= int(
            config.get("task_cue_achromatic_saturation_maximum", 72)
        )
        achromatic_fraction = float(np.count_nonzero(achromatic_mask) / len(samples))
        if achromatic_fraction >= float(
            config.get("task_cue_achromatic_minimum_fraction", 0.30)
        ):
            value = NoiseformDetector._task_cue_fill_value(samples, config)
            return "achromatic", None, value

        dark_mask = samples[:, 2] <= int(config.get("task_cue_black_value_maximum", 35))
        dark_fraction = float(np.count_nonzero(dark_mask) / len(samples))
        if dark_fraction >= float(config.get("task_cue_black_minimum_fraction", 0.45)):
            return "achromatic", None, float(np.median(samples[:, 2][dark_mask]))
        return None

    @staticmethod
    def _has_task_cue_glyph(
        component_hsv: np.ndarray,
        fill_value: float,
        config: dict,
    ) -> bool:
        """Require the centered high-contrast exclamation carried by real cues."""
        component_height, component_width = component_hsv.shape[:2]
        if component_height < 8 or component_width < 8:
            return False
        contrast = np.abs(component_hsv[:, :, 2].astype(np.float32) - fill_value)
        contrast_mask = (
            contrast
            >= float(config.get("task_cue_glyph_contrast_minimum", 45))
        ).astype(np.uint8)
        count, _, stats, centroids = cv2.connectedComponentsWithStats(
            contrast_mask,
            8,
        )
        line_candidates: list[tuple[int, float, int]] = []
        center_x_tolerance = component_width * float(
            config.get("task_cue_glyph_center_x_tolerance", 0.13)
        )
        minimum_area = component_width * component_height * float(
            config.get("task_cue_glyph_line_minimum_area_fraction", 0.004)
        )
        for index in range(1, count):
            top = int(stats[index, cv2.CC_STAT_TOP])
            glyph_width = int(stats[index, cv2.CC_STAT_WIDTH])
            glyph_height = int(stats[index, cv2.CC_STAT_HEIGHT])
            area = int(stats[index, cv2.CC_STAT_AREA])
            center_x = float(centroids[index, 0])
            center_y = float(centroids[index, 1])
            if (
                abs(center_x - component_width / 2.0) <= center_x_tolerance
                and component_height * 0.18 <= glyph_height <= component_height * 0.58
                and glyph_width <= component_width * 0.18
                and glyph_height >= glyph_width * 1.8
                and component_height * 0.20 <= center_y <= component_height * 0.58
                and area >= minimum_area
            ):
                line_candidates.append((index, center_x, top + glyph_height))
        if not line_candidates:
            return False
        dot_minimum_area = component_width * component_height * float(
            config.get("task_cue_glyph_dot_minimum_area_fraction", 0.001)
        )
        for line_index, line_center_x, line_bottom in line_candidates:
            for index in range(1, count):
                if index == line_index:
                    continue
                glyph_width = int(stats[index, cv2.CC_STAT_WIDTH])
                glyph_height = int(stats[index, cv2.CC_STAT_HEIGHT])
                area = int(stats[index, cv2.CC_STAT_AREA])
                center_x = float(centroids[index, 0])
                center_y = float(centroids[index, 1])
                if (
                    abs(center_x - line_center_x) <= center_x_tolerance
                    and line_bottom <= center_y <= component_height * 0.84
                    and component_width * 0.015 <= glyph_width <= component_width * 0.22
                    and component_height * 0.015 <= glyph_height <= component_height * 0.22
                    and area >= dot_minimum_area
                ):
                    return True
        return False

    @staticmethod
    def _find_fixed_glyph_task_cue(
        frame: np.ndarray,
        config: dict,
    ) -> tuple[str, float | None, float | None] | None:
        """Read cue color from the fixed prompt center after glyph validation."""
        height, width = frame.shape[:2]
        center_x = round(width * float(config.get("task_cue_fixed_center_x", 0.50)))
        center_y = round(height * float(config.get("task_cue_fixed_center_y", 0.50)))
        radius = max(
            40,
            round(
                min(width, height)
                * float(config.get("task_cue_fixed_radius_normalized", 0.135))
            ),
        )
        left = max(0, center_x - radius)
        right = min(width, center_x + radius + 1)
        top = max(0, center_y - radius)
        bottom = min(height, center_y + radius + 1)
        crop = frame[top:bottom, left:right]
        if crop.size == 0:
            return None
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        crop_height, crop_width = hsv.shape[:2]
        yy, xx = np.ogrid[:crop_height, :crop_width]
        normalized_x = (xx - (crop_width - 1) / 2.0) / max(1.0, crop_width / 2.0)
        normalized_y = (yy - (crop_height - 1) / 2.0) / max(1.0, crop_height / 2.0)
        normalized_radius = np.sqrt(
            normalized_x * normalized_x + normalized_y * normalized_y
        )
        sample_mask = (
            normalized_radius
            >= float(config.get("task_cue_sample_inner_radius", 0.28))
        ) & (
            normalized_radius
            <= float(config.get("task_cue_sample_outer_radius", 0.70))
        )
        samples = hsv[sample_mask]
        if samples.size == 0:
            return None
        sample_saturation = float(np.median(samples[:, 1]))
        sample_value = NoiseformDetector._task_cue_fill_value(samples, config)
        if not NoiseformDetector._has_task_cue_glyph(hsv, sample_value, config):
            return None
        if sample_saturation <= float(
            config.get("task_cue_achromatic_saturation_maximum", 72)
        ):
            return "achromatic", None, sample_value
        saturated_samples = samples[
            samples[:, 1] >= int(config.get("task_cue_saturation_minimum", 100))
        ]
        if saturated_samples.size == 0:
            return None
        return "hue", float(np.median(saturated_samples[:, 0])), sample_value

    @staticmethod
    def _find_fixed_black_task_cue(
        crop: np.ndarray,
        previous_gray: np.ndarray,
        frame_width: int,
        frame_height: int,
        cue_rect: PixelRect,
        config: dict,
    ) -> tuple[str, float | None, float | None] | None:
        """Recover a black cue from its dark onset when its outline merges."""
        center_x = round(
            frame_width * float(config.get("task_cue_fixed_center_x", 0.50))
        ) - cue_rect.left
        center_y = round(
            frame_height * float(config.get("task_cue_fixed_center_y", 0.50))
        ) - cue_rect.top
        radius = max(
            24,
            round(
                min(frame_width, frame_height)
                * float(config.get("task_cue_fixed_radius_normalized", 0.135))
            ),
        )
        left = max(0, center_x - radius)
        right = min(crop.shape[1], center_x + radius + 1)
        top = max(0, center_y - radius)
        bottom = min(crop.shape[0], center_y + radius + 1)
        patch = crop[top:bottom, left:right]
        old = previous_gray[top:bottom, left:right]
        if patch.size == 0 or old.size == 0:
            return None
        current_gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        changed = cv2.absdiff(current_gray, old) >= int(
            config.get("task_cue_shape_difference_minimum", 18)
        )
        changed_count = int(np.count_nonzero(changed))
        if changed_count / max(1, changed.size) < float(
            config.get("task_cue_fixed_black_minimum_changed_fraction", 0.06)
        ):
            return None
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        dark = changed & (
            hsv[:, :, 2]
            <= int(config.get("task_cue_black_support_value_ceiling", 75))
        )
        if np.count_nonzero(dark) / max(1, changed_count) < float(
            config.get("task_cue_fixed_black_minimum_dark_fraction", 0.35)
        ):
            return None
        fill_value = float(np.median(hsv[:, :, 2][dark]))
        if bool(
            config.get("task_cue_require_contrast_glyph", False)
        ) and not NoiseformDetector._has_task_cue_glyph(hsv, fill_value, config):
            return None
        return "achromatic", None, fill_value

    @staticmethod
    def _task_cue_fill_value(samples: np.ndarray, config: dict) -> float:
        """Estimate the remembered black/gray/white cue body value."""
        values = samples[:, 2].astype(np.float32)
        saturation = samples[:, 1]
        achromatic = saturation <= int(
            config.get("task_cue_achromatic_saturation_maximum", 72)
        )
        achromatic_values = values[achromatic]
        if achromatic_values.size == 0:
            return float(np.median(values))

        achromatic_fraction = achromatic_values.size / max(1, values.size)
        if achromatic_fraction < float(
            config.get("task_cue_black_minimum_achromatic_fraction", 0.65)
        ):
            return float(np.median(values))

        dark_ceiling = float(config.get("task_cue_black_support_value_ceiling", 75))
        dark_values = achromatic_values[achromatic_values <= dark_ceiling]
        dark_fraction = dark_values.size / max(1, achromatic_values.size)
        if dark_fraction >= float(
            config.get("task_cue_black_minimum_dark_fraction", 0.20)
        ):
            return float(np.median(dark_values))
        return float(np.median(achromatic_values))

    @staticmethod
    def _find_task_tile_x(
        frame: np.ndarray,
        cue_mode: str,
        cue_hue: float | None,
        cue_value: float | None,
        config: dict,
        latched_target_x: float | None,
        trained: TrainedVisionFrame | None = None,
    ) -> tuple[float, float] | None:
        height, width = frame.shape[:2]
        task_rect = NormalizedRect(
            *map(float, config.get("task_rail_rect", [0.285, 0.78, 0.715, 0.845]))
        ).pixels(width, height)
        crop = frame[task_rect.top : task_rect.bottom, task_rect.left : task_rect.right]
        if crop.size == 0:
            return None
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        white_mask = (
            (hsv[:, :, 2] >= int(config.get("task_glyph_value_minimum", 145)))
            & (hsv[:, :, 1] <= int(config.get("task_glyph_saturation_maximum", 75)))
        ).astype(np.uint8)
        count, _, stats, centroids = cv2.connectedComponentsWithStats(white_mask, 8)
        components: list[tuple[float, float, int, int]] = []
        minimum_width = max(
            4,
            round(width * float(config.get("task_glyph_minimum_width_ratio", 0.004))),
        )
        maximum_width = max(
            minimum_width,
            round(width * float(config.get("task_glyph_maximum_width_ratio", 0.030))),
        )
        minimum_height = max(
            6,
            round(height * float(config.get("task_glyph_minimum_height_ratio", 0.008))),
        )
        maximum_height = max(
            minimum_height,
            round(height * float(config.get("task_glyph_maximum_height_ratio", 0.045))),
        )
        minimum_area = max(
            30,
            round(width * height * float(config.get("task_glyph_minimum_area_fraction", 0.000035))),
        )
        for index in range(1, count):
            component_width = int(stats[index, cv2.CC_STAT_WIDTH])
            component_height = int(stats[index, cv2.CC_STAT_HEIGHT])
            area = int(stats[index, cv2.CC_STAT_AREA])
            if not (
                minimum_width <= component_width <= maximum_width
                and minimum_height <= component_height <= maximum_height
                and area >= minimum_area
            ):
                continue
            center_x = task_rect.left + float(centroids[index, 0])
            center_y = task_rect.top + float(centroids[index, 1])
            if not (
                height * float(config.get("task_glyph_center_y_minimum", 0.77))
                <= center_y
                <= height * float(config.get("task_glyph_center_y_maximum", 0.86))
            ):
                continue
            component_left = task_rect.left + int(stats[index, cv2.CC_STAT_LEFT])
            component_right = component_left + component_width
            components.append(
                (center_x, center_y, component_left, component_right)
            )

        trained_candidates = NoiseformDetector._trained_task_tile_candidates(
            frame,
            trained,
            task_rect,
            config,
        )
        if bool(config.get("task_require_trained_tile_anchor", True)) and not trained_candidates:
            return None
        if (
            len(components) < int(config.get("task_minimum_glyph_count", 2))
            and not trained_candidates
        ):
            return None

        # The white task symbols can split into several components when the
        # diagonal tile pattern passes through them.  First isolate the row
        # containing the most components; this rejects the instruction text
        # and transient arrows above the rail without assuming one fixed Y.
        row_tolerance = max(
            6.0,
            height * float(config.get("task_glyph_row_tolerance_ratio", 0.020)),
        )
        if components:
            row_center = max(
                (component[1] for component in components),
                key=lambda center_y: (
                    sum(
                        abs(other[1] - center_y) <= row_tolerance
                        for other in components
                    ),
                    center_y,
                ),
            )
            row_components = [
                component
                for component in components
                if abs(component[1] - row_center) <= row_tolerance
            ]
        else:
            row_components = []

        # Reassemble split symbols into one candidate tile.  The merge gap is
        # smaller than the normal distance between adjacent task blocks, but
        # wide enough to join the separated bright pieces of the striped !.
        merge_gap = max(
            12,
            round(
                width
                * float(config.get("task_glyph_component_merge_gap_ratio", 0.045))
            ),
        )
        component_groups: list[list[tuple[float, float, int, int]]] = []
        for component in sorted(row_components, key=lambda item: item[2]):
            if (
                not component_groups
                or component[2] - max(item[3] for item in component_groups[-1])
                > merge_gap
            ):
                component_groups.append([component])
            else:
                component_groups[-1].append(component)

        require_trained_anchor = bool(
            config.get("task_require_trained_tile_anchor", True)
        )
        candidates: list[tuple[float, float, bool]] = [
            (center_x, center_y, True)
            for center_x, center_y in trained_candidates
        ]
        # A trained tile is the row-level proof that this is a real Noiseform
        # task, not ordinary rail color.  Once that proof exists, include every
        # classically validated white-glyph group so a matching tile omitted by
        # the model's imperfect per-tile recall can still be selected.
        if trained_candidates or not require_trained_anchor:
            for group in component_groups:
                if len(group) == 1:
                    center_x, center_y, _, _ = group[0]
                else:
                    center_x = (
                        min(item[2] for item in group)
                        + max(item[3] for item in group)
                    ) / 2.0
                    center_y = float(np.median([item[1] for item in group]))
                candidates.append((center_x, center_y, False))

        # A model proposal is geometry only.  De-duplicate it with the
        # classical white-symbol groups, then keep all the color/fill checks
        # below as the authority for which tile matches the remembered cue.
        deduplicated: list[tuple[float, float, bool]] = []
        duplicate_tolerance = width * 0.025
        for candidate in candidates:
            if not any(
                abs(candidate[0] - existing[0]) <= duplicate_tolerance
                for existing in deduplicated
            ):
                deduplicated.append(candidate)
        candidates = deduplicated

        minimum_candidate_count = int(
            config.get(
                "task_minimum_trained_tile_count" if require_trained_anchor else "task_minimum_tile_count",
                1 if require_trained_anchor else 2,
            )
        )
        if len(candidates) < minimum_candidate_count:
            return None

        half_width = max(12, round(width * float(config.get("task_tile_half_width", 0.024))))
        half_height = max(
            8,
            round(height * float(config.get("task_tile_half_height", 0.022))),
        )
        hue_tolerance = float(config.get("task_tile_hue_tolerance", 12))
        scored_candidates: list[tuple[float, float, float, bool]] = []
        for center_x, center_y, is_trained_candidate in candidates:
            left = max(0, round(center_x) - half_width)
            right = min(width, round(center_x) + half_width + 1)
            top = max(0, round(center_y) - half_height)
            bottom = min(height, round(center_y) + half_height + 1)
            tile = cv2.cvtColor(frame[top:bottom, left:right], cv2.COLOR_BGR2HSV)
            if tile.size == 0:
                continue
            inset_x = max(
                1,
                round(
                    tile.shape[1]
                    * float(config.get("task_tile_sample_inset_ratio", 0.07))
                ),
            )
            inset_y = max(
                1,
                round(
                    tile.shape[0]
                    * float(config.get("task_tile_sample_inset_ratio", 0.07))
                ),
            )
            if tile.shape[1] > inset_x * 2 and tile.shape[0] > inset_y * 2:
                tile = tile[inset_y:-inset_y, inset_x:-inset_x]

            # The white symbol is an identity marker, not part of the tile's
            # color.  Excluding it prevents every block from looking gray and
            # lets the patterned fill determine which prompt it matches.
            glyph_mask = (
                (tile[:, :, 2] >= int(config.get("task_glyph_value_minimum", 145)))
                & (
                    tile[:, :, 1]
                    <= int(config.get("task_glyph_saturation_maximum", 75))
                )
            )
            fill_mask = ~glyph_mask
            fill_count = int(np.count_nonzero(fill_mask))
            if fill_count == 0:
                continue
            if cue_mode == "achromatic":
                assert cue_value is not None
                if cue_value <= float(
                    config.get("task_tile_black_cue_value_maximum", 70)
                ):
                    # Green rail glow can make a genuinely black task block
                    # highly saturated.  Its low-value fill pixels remain dark,
                    # while gray/white and green blocks do not.  Use that
                    # illumination-resistant darkness signal before requiring
                    # achromatic fill.
                    fill_values = tile[:, :, 2][fill_mask]
                    tile_value = float(
                        np.percentile(
                            fill_values,
                            float(config.get("task_tile_black_value_percentile", 20)),
                        )
                    )
                    maximum_value = float(
                        config.get("task_tile_black_maximum_value", 50)
                    )
                    if tile_value > maximum_value:
                        continue
                    score_span = float(
                        config.get("task_tile_black_score_span", 90)
                    )
                    fraction = max(
                        0.0, 1.0 - tile_value / max(1.0, score_span)
                    )
                    minimum_fraction = float(
                        config.get("task_tile_minimum_black_score", 0.40)
                    )
                else:
                    achromatic_mask = fill_mask & (
                        tile[:, :, 1]
                        <= int(config.get("task_tile_achromatic_saturation_maximum", 78))
                    )
                    achromatic_fraction = float(
                        np.count_nonzero(achromatic_mask) / fill_count
                    )
                    if achromatic_fraction < float(
                        config.get(
                            "task_tile_minimum_achromatic_fill_fraction", 0.18
                        )
                    ):
                        continue
                    tile_values = tile[:, :, 2][achromatic_mask]
                    if tile_values.size == 0:
                        continue
                    # Gray/white task tiles use alternating dark and bright
                    # diagonal bands.  Their median often lands on the black
                    # stripe even when the prompt is very bright.  The white
                    # symbol has already been removed, so an upper fill
                    # percentile represents the tile color without confusing
                    # the symbol itself for the answer.
                    tile_value = float(
                        np.percentile(
                            tile_values,
                            float(config.get("task_tile_gray_value_percentile", 70)),
                        )
                    )
                    gray_floor = float(config.get("task_tile_gray_value_floor", 30))
                    gray_span = float(config.get("task_tile_gray_value_span", 140))
                    value_quality = max(
                        0.0,
                        min(1.0, (tile_value - gray_floor) / max(1.0, gray_span)),
                    )
                    fraction = achromatic_fraction * 0.30 + value_quality * 0.70
                    minimum_fraction = float(
                        config.get("task_tile_minimum_achromatic_score", 0.55)
                    )
            else:
                assert cue_hue is not None
                colored_fill = (
                    fill_mask
                    & (tile[:, :, 1] >= int(config.get("task_tile_saturation_minimum", 80)))
                    & (tile[:, :, 2] >= int(config.get("task_tile_value_minimum", 60)))
                )
                if bool(config.get("task_tile_match_green_family", False)):
                    hue_minimum = float(
                        config.get("task_tile_green_hue_minimum", 45)
                    )
                    hue_maximum = float(
                        config.get("task_tile_green_hue_maximum", 100)
                    )
                    matching = (
                        colored_fill
                        & (tile[:, :, 0] >= hue_minimum)
                        & (tile[:, :, 0] <= hue_maximum)
                    )
                else:
                    hue_delta = np.abs(tile[:, :, 0].astype(np.float32) - cue_hue)
                    hue_delta = np.minimum(hue_delta, 180.0 - hue_delta)
                    matching = colored_fill & (hue_delta <= hue_tolerance)
                minimum_fraction = float(
                    config.get("task_tile_minimum_color_fraction", 0.45)
                )
                fraction = float(np.count_nonzero(matching) / fill_count)
            if fraction < minimum_fraction:
                continue
            if latched_target_x is not None:
                reacquire_tolerance = width * float(
                    config.get("task_target_reacquire_tolerance", 0.025)
                )
                distance = abs(center_x - latched_target_x)
                if distance > reacquire_tolerance:
                    continue
                selection_score = fraction - distance / max(1.0, width) * 2.0
            else:
                selection_score = fraction
            scored_candidates.append(
                (selection_score, fraction, center_x, is_trained_candidate)
            )
        if not scored_candidates:
            return None
        trained_scored = [item for item in scored_candidates if item[3]]
        selection_pool = trained_scored or scored_candidates
        _, best_fraction, best_x, _ = max(
            selection_pool,
            key=lambda item: item[0],
        )
        confidence = min(1.0, 0.55 + best_fraction * 0.5)
        return best_x, confidence

    @staticmethod
    def _trained_task_tile_candidates(
        frame: np.ndarray,
        trained: TrainedVisionFrame | None,
        task_rect: PixelRect,
        config: dict,
    ) -> list[tuple[float, float]]:
        """Return model tile geometry only after validating its white symbol."""
        if trained is None:
            return []
        height, width = frame.shape[:2]
        candidates: list[tuple[float, float]] = []
        for item in trained.detections("tile", trusted=True):
            if not (
                width * 0.030 <= item.width <= width * 0.085
                and height * 0.012 <= item.height <= height * 0.060
                and task_rect.left <= item.center_x <= task_rect.right
                and task_rect.top <= item.center_y <= task_rect.bottom
            ):
                continue
            tile = frame[item.top : item.bottom, item.left : item.right]
            if tile.size == 0:
                continue
            hsv = cv2.cvtColor(tile, cv2.COLOR_BGR2HSV)
            white = (
                (hsv[:, :, 2] >= int(config.get("task_glyph_value_minimum", 145)))
                & (
                    hsv[:, :, 1]
                    <= int(config.get("task_glyph_saturation_maximum", 75))
                )
            ).astype(np.uint8)
            count, _, stats, centroids = cv2.connectedComponentsWithStats(white, 8)
            for index in range(1, count):
                glyph_width = int(stats[index, cv2.CC_STAT_WIDTH])
                glyph_height = int(stats[index, cv2.CC_STAT_HEIGHT])
                area = int(stats[index, cv2.CC_STAT_AREA])
                center_x = float(centroids[index, 0])
                center_y = float(centroids[index, 1])
                if (
                    area >= 12
                    and glyph_width <= item.width * 0.45
                    and glyph_height <= item.height * 0.90
                    and abs(center_x - item.width / 2.0) <= item.width * 0.30
                    and abs(center_y - item.height / 2.0) <= item.height * 0.35
                ):
                    candidates.append((item.center_x, item.center_y))
                    break
        return candidates

    def _track_bar(
        self,
        strength: np.ndarray,
        luma: np.ndarray,
        rail_crop: np.ndarray,
        rail: PixelRect,
        bar_width: int,
        timestamp_ms: float,
        config: dict,
        stick_hint_x: float | None = None,
    ) -> tuple[float | None, float, str]:
        assert self.background_strength is not None
        assert self.background_luma is not None
        self.last_raw_bar_center = None
        self.last_raw_bar_width = None
        self.last_raw_bar_source = ""
        self.last_raw_bar_confidence = 0.0
        self.last_bar_trust_state = "missing"
        self.last_bar_trust_reason = "no_candidate"
        self.last_bar_history_updated = False
        self.last_previous_trusted_bar_center = self.previous_bar_center
        self.last_structure_body_coverage = 0.0
        self.last_structure_left_boundary = 0.0
        self.last_structure_right_boundary = 0.0
        self.last_structure_top_coverage = 0.0
        self.last_structure_bottom_coverage = 0.0
        self.last_structure_green_range = 0.0
        previous_center = self.previous_bar_center or rail.center_x
        elapsed = (
            0.0
            if self.previous_timestamp_ms is None
            else max(0.0, timestamp_ms - self.previous_timestamp_ms)
        )
        predicted_center = previous_center + self.bar_velocity_px_per_ms * elapsed
        predicted_center = max(
            rail.left + bar_width / 2.0,
            min(rail.right - bar_width / 2.0, predicted_center),
        )
        # Broad residuals and contrast edges are weak evidence because the five
        # animated arches can produce both. They may follow established motion
        # or expand their search after a real gap, but cannot teleport the bar
        # across the rail on an ordinary frame. Strong paired outline/arrow
        # geometry remains globally reacquirable below.
        maximum_speed = float(config.get("maximum_speed_px_per_ms", 0.85))
        weak_search_radius = (
            bar_width
            * float(config.get("bar_weak_search_radius_ratio", 0.48))
            + maximum_speed
            * min(elapsed, float(config.get("bar_weak_search_growth_cap_ms", 420)))
        )

        residual = np.maximum(
            0.0,
            strength
            - self.background_strength
            - float(config.get("background_residual_floor", 4.0)),
        )
        luma_residual = np.maximum(
            0.0,
            luma
            - self.background_luma
            - float(config.get("background_luma_residual_floor", 3.0)),
        )
        # Narrow animated arches can be much brighter than the control body.
        # Cap their per-pixel contribution so a broad faint body wins by
        # coverage rather than losing to a few saturated columns.
        combined_residual = np.minimum(
            residual,
            float(config.get("bar_green_residual_cap", 45.0)),
        ) + float(config.get("bar_luma_residual_weight", 0.55)) * np.minimum(
            luma_residual,
            float(config.get("bar_luma_residual_cap", 32.0)),
        )
        column_score = np.mean(combined_residual, axis=0)
        body_threshold = float(config.get("bar_body_column_threshold", 3.0))
        body_active = (column_score >= body_threshold).astype(np.float32)
        kernel = np.ones(bar_width, dtype=np.float32)
        if rail.width >= bar_width:
            coverage_scores = np.convolve(body_active, kernel, mode="valid") / bar_width
            capped_columns = np.minimum(
                column_score,
                float(config.get("bar_body_column_score_cap", 24.0)),
            )
            mean_scores = np.convolve(capped_columns, kernel, mode="valid") / bar_width
        else:
            coverage_scores = np.array([float(np.mean(body_active))])
            mean_scores = np.array([float(np.mean(column_score))])

        if self.short_bar_recovery_enabled:
            minimum_coverage = float(
                config.get("bar_short_body_minimum_coverage", 0.45)
            )
            minimum_mean = float(
                config.get("bar_short_body_minimum_mean_score", 2.2)
            )
        else:
            minimum_coverage = float(config.get("bar_body_minimum_coverage", 0.68))
            minimum_mean = float(config.get("bar_body_minimum_mean_score", 3.0))
        valid_indexes = np.flatnonzero(
            (coverage_scores >= minimum_coverage) & (mean_scores >= minimum_mean)
        )
        self.last_body_candidate_count = int(valid_indexes.size)
        body_center: float | None = None
        body_score = 0.0
        body_coverage = 0.0
        best_selection = float("-inf")
        global_body_center: float | None = None
        global_body_score = 0.0
        global_body_coverage = 0.0
        global_best_selection = float("-inf")
        rejected_active = (
            self.rejected_bar_center is not None
            and self.rejected_bar_until_ms is not None
            and timestamp_ms <= self.rejected_bar_until_ms
        )
        for raw_index in valid_indexes:
            index = int(raw_index)
            candidate_center = rail.left + index + bar_width / 2.0
            if (
                rejected_active
                and abs(candidate_center - float(self.rejected_bar_center))
                <= bar_width * float(config.get("bar_rejected_center_radius_ratio", 0.42))
            ):
                continue
            coverage = float(coverage_scores[index])
            mean_score = float(mean_scores[index])
            continuity_penalty = (
                abs(candidate_center - predicted_center)
                / max(1.0, rail.width)
                * float(config.get("bar_body_continuity_penalty", 8.0))
            )
            selection = coverage * 60.0 + mean_score - continuity_penalty
            if selection > global_best_selection:
                global_best_selection = selection
                global_body_center = candidate_center
                global_body_score = mean_score
                global_body_coverage = coverage
            if abs(candidate_center - predicted_center) > weak_search_radius:
                continue
            if selection > best_selection:
                best_selection = selection
                body_center = candidate_center
                body_score = mean_score
                body_coverage = coverage
        self.last_body_coverage = body_coverage

        # A remembered cue by itself does not cover the rail.  Keep the
        # outline/arrow geometry fallback available while waiting for tiles;
        # otherwise the blended bar becomes prediction-only for several
        # seconds after every color flash.
        arrow_center, arrow_confidence, arrow_source = (
            self._find_bar_arrow_center(
                rail_crop,
                rail,
                bar_width,
                predicted_center,
                config,
            )
        )
        contrast_center, contrast_confidence = self._find_bar_contrast_center(
            rail_crop,
            rail,
            bar_width,
            predicted_center,
            config,
            stick_hint_x=stick_hint_x,
        )
        if arrow_center is not None:
            center = arrow_center
            best_score = 28.0 + arrow_confidence * 20.0
            geometry_source = arrow_source
        elif (
            contrast_center is not None
            and abs(contrast_center - predicted_center) > weak_search_radius
        ):
            contrast_center = None
            contrast_confidence = 0.0
        # A far broad body is allowed to reacquire immediately only when a
        # local pair of bar-width contrast edges independently supports it.
        # This preserves detached-bar recovery without letting a bright arch
        # teleport the established track.
        if body_center is None and global_body_center is not None:
            global_contrast_center, global_contrast_confidence = (
                self._find_bar_contrast_center(
                    rail_crop,
                    rail,
                    bar_width,
                    global_body_center,
                    config,
                    maximum_distance=bar_width
                    * float(config.get("bar_global_body_edge_radius_ratio", 0.34)),
                    stick_hint_x=stick_hint_x,
                )
            )
            if (
                global_contrast_center is not None
                and abs(global_contrast_center - global_body_center)
                <= bar_width
                * float(config.get("bar_geometry_body_agreement_ratio", 0.30))
            ):
                body_center = global_body_center
                body_score = global_body_score
                body_coverage = global_body_coverage
                contrast_center = global_contrast_center
                contrast_confidence = global_contrast_confidence
                self.last_body_coverage = body_coverage

        if arrow_center is not None:
            visual_centers = [
                candidate
                for candidate in (body_center, contrast_center)
                if candidate is not None
            ]
            if visual_centers:
                # Outline/arrow pairs are useful when the detached body is
                # faint, but falling task arrows create the same geometry.
                # They may refine coherent body evidence, never replace it
                # with a distant location.
                agreement = min(
                    abs(arrow_center - candidate) for candidate in visual_centers
                )
                arrow_covers_real_stick = (
                    stick_hint_x is not None
                    and abs(arrow_center - stick_hint_x) <= bar_width * 0.52
                )
                visual_covers_real_stick = (
                    stick_hint_x is not None
                    and any(
                        abs(candidate - stick_hint_x) <= bar_width * 0.52
                        for candidate in visual_centers
                    )
                )
                trusted_outline = arrow_source == "noiseform_black_outline_pair"
                if (
                    agreement
                    > bar_width * float(
                        config.get("bar_arrow_visual_agreement_ratio", 0.35)
                    )
                    and not trusted_outline
                    and not (
                        arrow_covers_real_stick and not visual_covers_real_stick
                    )
                ):
                    arrow_center = None
                    arrow_confidence = 0.0
                    arrow_source = ""
            elif abs(arrow_center - predicted_center) > bar_width * float(
                config.get("bar_arrow_maximum_prediction_distance_ratio", 0.75)
            ):
                arrow_center = None
                arrow_confidence = 0.0
                arrow_source = ""

        raw_center: float | None
        raw_confidence = 0.0
        if arrow_center is not None:
            raw_center = arrow_center
            best_score = 28.0 + arrow_confidence * 20.0
            geometry_source = arrow_source
            raw_confidence = arrow_confidence
        elif (
            contrast_center is not None
            and body_center is not None
            and abs(contrast_center - body_center)
            <= bar_width * float(config.get("bar_geometry_body_agreement_ratio", 0.30))
        ):
            raw_center = contrast_center
            best_score = 24.0 + contrast_confidence * 22.0
            geometry_source = "noiseform_contrast_edge_pair"
            raw_confidence = contrast_confidence
        elif contrast_center is not None:
            raw_center = contrast_center
            best_score = 24.0 + contrast_confidence * 22.0
            geometry_source = "noiseform_contrast_edge_pair"
            raw_confidence = contrast_confidence
        elif body_center is not None:
            raw_center = body_center
            best_score = body_score + body_coverage * 30.0
            geometry_source = "noiseform_broad_body"
            raw_confidence = min(1.0, body_coverage)
        elif (
            self.bootstrap_started_timestamp_ms is not None
            and timestamp_ms - self.bootstrap_started_timestamp_ms
            <= float(config.get("bar_bootstrap_hold_ms", 220))
        ):
            raw_center = predicted_center
            best_score = 8.0
            geometry_source = "noiseform_bootstrap"
            raw_confidence = 0.10
        else:
            raw_center = None
            best_score = 0.0
            geometry_source = "noiseform_bar_unconfirmed"

        self.last_raw_bar_center = raw_center
        self.last_raw_bar_width = float(bar_width) if raw_center is not None else None
        self.last_raw_bar_source = geometry_source
        self.last_raw_bar_confidence = raw_confidence

        center: float | None = raw_center
        if geometry_source == "noiseform_broad_body":
            # Broad residual activity is useful supporting evidence only.  The
            # x≈653 failures were decorative arches with excellent area scores;
            # they must never become playable geometry or trusted history.
            center = None
            self.last_bar_trust_state = "partial"
            self.last_bar_trust_reason = "broad_body_support_only"
        elif geometry_source == "noiseform_bootstrap":
            center = None
            self.last_bar_trust_state = "partial"
            self.last_bar_trust_reason = "bootstrap_hint_only"
        elif raw_center is not None:
            structure = self._bar_structure_metrics(
                rail_crop,
                rail,
                raw_center,
                bar_width,
                config,
            )
            structure_ok, structure_reason = self._bar_structure_is_trusted(
                structure,
                config,
            )
            trusted_step_limit = (
                bar_width
                * float(config.get("bar_trusted_maximum_center_step_ratio", 0.16))
                + maximum_speed
                * min(
                    elapsed,
                    float(config.get("bar_trusted_center_step_growth_cap_ms", 360)),
                )
            )
            continuity_ok = (
                self.previous_bar_center is None
                and abs(raw_center - rail.center_x)
                <= bar_width
                * float(config.get("bar_bootstrap_maximum_center_distance_ratio", 0.75))
            ) or (
                self.previous_bar_center is not None
                and abs(raw_center - predicted_center) <= trusted_step_limit
            )
            if not structure_ok:
                center = None
                self.last_bar_trust_state = "partial"
                self.last_bar_trust_reason = structure_reason
            elif not continuity_ok:
                center = None
                self.last_bar_trust_state = "partial"
                self.last_bar_trust_reason = "trusted_center_continuity_failed"
            else:
                self.last_bar_trust_state = "trusted"
                self.last_bar_trust_reason = structure_reason

        if center is not None and rejected_active:
            if abs(center - float(self.rejected_bar_center)) <= bar_width * float(
                config.get("bar_rejected_center_radius_ratio", 0.42)
            ):
                center = None
                best_score = 0.0
                geometry_source = "noiseform_rejected_frozen_center"
                self.last_bar_trust_state = "partial"
                self.last_bar_trust_reason = "temporarily_rejected_center"

        if center is not None and self._bar_response_rejects(
            center,
            rail,
            bar_width,
            timestamp_ms,
            config,
            geometry_source,
        ):
            # A stationary decorative candidate must never keep issuing a
            # directional command. Reject it briefly so recovery can perform a
            # global outline search instead of inheriting the false center.
            self.rejected_bar_center = center
            self.rejected_bar_until_ms = timestamp_ms + float(
                config.get("bar_rejected_center_hold_ms", 260)
            )
            center = None
            best_score = 0.0
            geometry_source = "noiseform_frozen_weak_geometry"
            self.last_bar_trust_state = "partial"
            self.last_bar_trust_reason = "response_watchdog_rejected"

        if center is not None:
            self._update_trusted_bar_history(center, timestamp_ms, config)
            self.last_reliable_bar_center = center
            self.last_reliable_bar_timestamp_ms = timestamp_ms
            if (
                self.rejected_bar_until_ms is not None
                and timestamp_ms > self.rejected_bar_until_ms
            ):
                self.rejected_bar_center = None
                self.rejected_bar_until_ms = None
        elif self.last_bar_trust_state == "missing":
            self.last_bar_trust_reason = "no_trusted_bar_observation"
        self.background_strength = np.minimum(self.background_strength, strength)
        self.background_luma = np.minimum(self.background_luma, luma)
        return center, best_score, geometry_source

    def _bar_response_rejects(
        self,
        center: float,
        rail: PixelRect,
        bar_width: int,
        timestamp_ms: float,
        config: dict,
        geometry_source: str,
    ) -> bool:
        if not bool(config.get("bar_response_validation_enabled", True)):
            return False
        allowed_sources = config.get("bar_response_validation_sources")
        if allowed_sources and geometry_source not in allowed_sources:
            return False
        if (
            self.last_command_action == CommandAction.NEUTRAL
            or self.command_anchor_timestamp_ms is None
            or self.command_anchor_bar_center is None
            or self.last_command_error_px is None
        ):
            return False
        response_window_ms = float(config.get("bar_response_window_ms", 240))
        if timestamp_ms - self.command_anchor_timestamp_ms < response_window_ms:
            return False
        if abs(self.last_command_error_px) < bar_width * float(
            config.get("bar_response_minimum_error_ratio", 0.25)
        ):
            self.command_anchor_timestamp_ms = timestamp_ms
            self.command_anchor_bar_center = center
            return False
        boundary_margin = bar_width * 0.08
        if (
            center <= rail.left + bar_width / 2.0 + boundary_margin
            or center >= rail.right - bar_width / 2.0 - boundary_margin
        ):
            self.command_anchor_timestamp_ms = timestamp_ms
            self.command_anchor_bar_center = center
            return False
        # The physical bar may continue coasting opposite the newly requested
        # direction while its momentum is being braked.  That is still real
        # motion and must not invalidate the track.  The failure this watchdog
        # is intended to catch is a genuinely frozen decorative/body candidate:
        # a large sustained command with essentially no visual displacement.
        response = abs(center - self.command_anchor_bar_center)
        minimum_response = bar_width * float(
            config.get("bar_response_minimum_motion_ratio", 0.025)
        )
        self.command_anchor_timestamp_ms = timestamp_ms
        self.command_anchor_bar_center = center
        return response < minimum_response

    @staticmethod
    def _find_bar_contrast_center(
        rail_crop: np.ndarray,
        rail: PixelRect,
        bar_width: int,
        predicted_center: float,
        config: dict,
        maximum_distance: float | None = None,
        stick_hint_x: float | None = None,
    ) -> tuple[float | None, float]:
        gray = cv2.cvtColor(rail_crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
        margin = max(1, round(gray.shape[0] * 0.18))
        middle = gray[margin : max(margin + 1, gray.shape[0] - margin), :]
        columns = np.median(middle, axis=0)
        strip = max(3, round(bar_width * 0.026))
        minimum_contrast = float(config.get("bar_edge_minimum_contrast", 7.0))
        if maximum_distance is None:
            maximum_distance = bar_width * float(
                config.get("bar_edge_maximum_prediction_distance_ratio", 1.8)
            )
        first = strip * 4
        stop = rail.width - bar_width - strip * 4
        if stop <= first:
            return None, 0.0
        left = np.arange(first, stop, dtype=np.int32)
        right = left + bar_width
        prefix = np.concatenate(
            (np.zeros(1, dtype=np.float32), np.cumsum(columns, dtype=np.float32))
        )

        def interval_mean(starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
            return (prefix[ends] - prefix[starts]) / np.maximum(1, ends - starts)

        outside_left = interval_mean(left - strip * 4, left - strip)
        inside_left = interval_mean(left + strip, left + strip * 4)
        inside_right = interval_mean(right - strip * 4, right - strip)
        outside_right = interval_mean(right + strip, right + strip * 4)
        edge_contrast = np.minimum(
            inside_left - outside_left,
            inside_right - outside_right,
        )
        centers = rail.left + left.astype(np.float32) + bar_width / 2.0
        distance = np.abs(centers - predicted_center)
        # A valid Noiseform bar has two complete dark outer edges.  Internal
        # arrow edges and decorative arches can create a convincing luma pair,
        # but they do not produce both full-height boundaries at bar width.
        dark_fraction = np.mean(
            middle <= int(config.get("bar_structure_boundary_dark_max", 75)),
            axis=0,
        ).astype(np.float32)
        boundary_half = max(2, round(middle.shape[0] * 0.07))
        boundary_kernel = np.ones(boundary_half * 2 + 1, dtype=np.float32)
        boundary_kernel /= float(boundary_kernel.size)
        boundary_coverage = np.convolve(dark_fraction, boundary_kernel, mode="same")
        minimum_boundary = float(
            config.get("bar_structure_minimum_boundary_coverage", 0.06)
        )
        bright_columns = (
            np.median(middle, axis=0)
            >= float(config.get("bar_structure_body_luma_minimum", 70.0))
        ).astype(np.float32)
        body_prefix = np.concatenate(
            (np.zeros(1, dtype=np.float32), np.cumsum(bright_columns))
        )
        body_coverage = (body_prefix[right] - body_prefix[left]) / max(1, bar_width)
        stick_distance = (
            np.full_like(centers, np.inf)
            if stick_hint_x is None
            else np.abs(centers - float(stick_hint_x))
        )
        stick_centered = stick_distance <= bar_width * float(
            config.get("bar_edge_stick_centered_ratio", 0.22)
        )
        contrast_valid = (edge_contrast >= minimum_contrast) | (
            stick_centered
            & (
                edge_contrast
                >= float(config.get("bar_edge_stick_centered_minimum_contrast", 0.0))
            )
        )
        valid = (
            contrast_valid
            & (distance <= maximum_distance)
            & (boundary_coverage[left] >= minimum_boundary)
            & (boundary_coverage[right] >= minimum_boundary)
            & (
                body_coverage
                >= float(config.get("bar_structure_minimum_body_coverage", 0.90))
            )
        )
        if not np.any(valid):
            return None, 0.0
        selection = edge_contrast - distance / max(1.0, bar_width) * float(
            config.get("bar_edge_continuity_penalty", 1.5)
        )
        if stick_hint_x is not None:
            centered_quality = np.maximum(
                0.0,
                1.0 - stick_distance / max(1.0, bar_width * 0.5),
            )
            selection += centered_quality * float(
                config.get("bar_edge_stick_centered_bonus", 18.0)
            )
        selection[~valid] = -np.inf
        best_index = int(np.argmax(selection))
        best_score = float(selection[best_index])
        best_center = float(centers[best_index])
        confidence = min(1.0, 0.45 + best_score / 45.0)
        return best_center, confidence

    @staticmethod
    def _find_bar_arrow_center(
        rail_crop: np.ndarray,
        rail: PixelRect,
        bar_width: int,
        predicted_center: float,
        config: dict,
    ) -> tuple[float | None, float, str]:
        gray = cv2.cvtColor(rail_crop, cv2.COLOR_BGR2GRAY)
        vertical_margin = max(
            1,
            round(gray.shape[0] * float(config.get("bar_geometry_vertical_margin", 0.16))),
        )
        vertical_gray = gray[
            vertical_margin : max(vertical_margin + 1, gray.shape[0] - vertical_margin),
            :,
        ]
        dark_counts = np.sum(
            vertical_gray <= int(config.get("bar_arrow_dark_max", 55)),
            axis=0,
        )
        minimum_width = int(config.get("bar_arrow_minimum_width_px_at_1080p", 1))
        maximum_width = int(config.get("bar_arrow_maximum_width_px_at_1080p", 14))

        def collect_stems(minimum_height_ratio: float) -> list[tuple[float, int, float]]:
            minimum_rows = max(
                6,
                round(vertical_gray.shape[0] * minimum_height_ratio),
            )
            result: list[tuple[float, int, float]] = []
            for start, end in _runs(dark_counts >= minimum_rows):
                run_width = end - start
                if not minimum_width <= run_width <= maximum_width:
                    continue
                center = (start + end) / 2.0
                patch_half_width = max(10, round(rail.height * 0.52))
                patch_left = max(0, round(center) - patch_half_width)
                patch_right = min(rail.width, round(center) + patch_half_width + 1)
                patch_fraction = float(
                    np.mean(
                        gray[:, patch_left:patch_right]
                        <= int(config.get("bar_arrow_dark_max", 55))
                    )
                )
                result.append(
                    (center, int(np.max(dark_counts[start:end])), patch_fraction)
                )
            return result

        outline_stems = collect_stems(
            float(config.get("bar_outline_minimum_height_ratio", 0.48))
        )
        arrow_stems = collect_stems(
            float(config.get("bar_arrow_minimum_height_ratio", 0.58))
        )

        def pair_candidates(
            stems: list[tuple[float, int, float]],
            separation_ratio: float,
            tolerance_ratio: float,
            minimum_patch_fraction: float,
            source: str,
            maximum_prediction_distance: float | None = None,
        ) -> list[tuple[float, float, float, str]]:
            expected_separation = bar_width * separation_ratio
            tolerance = bar_width * tolerance_ratio
            result: list[tuple[float, float, float, str]] = []
            for index, (left_x, left_rows, left_patch) in enumerate(stems):
                for right_x, right_rows, right_patch in stems[index + 1 :]:
                    if min(left_patch, right_patch) < minimum_patch_fraction:
                        continue
                    separation_error = abs((right_x - left_x) - expected_separation)
                    if separation_error > tolerance:
                        continue
                    center_x = rail.left + (left_x + right_x) / 2.0
                    if not (
                        rail.left + bar_width / 2.0
                        <= center_x
                        <= rail.right - bar_width / 2.0
                    ):
                        continue
                    if (
                        maximum_prediction_distance is not None
                        and abs(center_x - predicted_center)
                        > maximum_prediction_distance
                    ):
                        continue
                    darkness = min(left_rows, right_rows) / max(
                        1.0,
                        vertical_gray.shape[0],
                    )
                    separation_quality = 1.0 - separation_error / max(1.0, tolerance)
                    continuity_penalty = (
                        abs(center_x - predicted_center)
                        / max(1.0, rail.width)
                        * float(config.get("bar_arrow_continuity_penalty", 0.35))
                    )
                    score = darkness + separation_quality - continuity_penalty
                    confidence = min(
                        1.0,
                        0.45 + darkness * 0.35 + separation_quality * 0.25,
                    )
                    result.append((score, center_x, confidence, source))
            return result

        outline_candidates = pair_candidates(
            outline_stems,
            float(config.get("bar_outline_separation_ratio", 1.0)),
            float(config.get("bar_outline_separation_tolerance_ratio", 0.055)),
            0.0,
            "noiseform_black_outline_pair",
            bar_width
            * float(config.get("bar_outline_maximum_prediction_distance_ratio", 0.42)),
        )
        candidates = outline_candidates or pair_candidates(
            arrow_stems,
            float(config.get("bar_arrow_separation_ratio", 0.763)),
            float(config.get("bar_arrow_separation_tolerance_ratio", 0.09)),
            float(config.get("bar_arrow_minimum_patch_fraction", 0.28)),
            "noiseform_black_arrow_pair",
        )
        if not candidates:
            return None, 0.0, ""
        _, center_x, confidence, source = max(candidates, key=lambda item: item[0])
        return center_x, confidence, source

    def _find_stick(
        self,
        frame: np.ndarray,
        strength: np.ndarray,
        rail: PixelRect,
        bar: PixelRect,
        timestamp_ms: float,
        config: dict,
    ) -> tuple[float | None, float, str]:
        self.last_stick_jump_rejections = 0
        gray = cv2.cvtColor(
            frame[rail.top : rail.bottom, rail.left : rail.right],
            cv2.COLOR_BGR2GRAY,
        )
        dark_counts = np.sum(
            gray <= int(config.get("stick_dark_max", 48)), axis=0
        )
        minimum_dark_rows = max(
            8,
            round(rail.height * float(config.get("stick_minimum_height_ratio", 0.48))),
        )
        candidates: list[tuple[float, float, int, str]] = []
        elapsed_since_stick = (
            0.0
            if self.last_stick_timestamp_ms is None
            else max(0.0, timestamp_ms - self.last_stick_timestamp_ms)
        )
        maximum_stick_jump = (
            rail.width * float(config.get("stick_maximum_jump_base_ratio", 0.025))
            + float(config.get("stick_maximum_speed_px_per_ms", 1.40))
            * min(
                elapsed_since_stick,
                float(config.get("stick_jump_growth_cap_ms", 180)),
            )
        )
        for start, end in _runs(dark_counts >= minimum_dark_rows):
            run_width = end - start
            if not (
                int(config.get("stick_minimum_width_px_at_1080p", 4))
                <= run_width
                <= int(config.get("stick_maximum_width_px_at_1080p", 10))
            ):
                continue
            center_local = (start + end) / 2.0
            center_x = rail.left + center_local
            if (
                self.previous_stick_x is not None
                and abs(center_x - self.previous_stick_x) > maximum_stick_jump
            ):
                self.last_stick_jump_rejections += 1
                continue
            dark_row_presence = np.any(
                gray[:, start:end] <= int(config.get("stick_dark_max", 48)),
                axis=1,
            )
            contiguous_runs = _runs(dark_row_presence)
            contiguous_height = max(
                (run_end - run_start for run_start, run_end in contiguous_runs),
                default=0,
            )
            if contiguous_height < rail.height * float(
                config.get("stick_minimum_contiguous_height_ratio", 0.42)
            ):
                continue
            halo_width = max(3, round(rail.height * 0.12))
            left_slice = strength[:, max(0, start - halo_width) : start]
            right_slice = strength[:, end : min(rail.width, end + halo_width)]
            left_green = float(np.mean(left_slice)) if left_slice.size else 0.0
            right_green = float(np.mean(right_slice)) if right_slice.size else 0.0
            halo_slice = strength[
                :,
                max(0, start - halo_width) : min(rail.width, end + halo_width),
            ]
            halo_percentile = (
                float(np.percentile(halo_slice, 90)) if halo_slice.size else 0.0
            )
            halo_visible = (
                min(left_green, right_green)
                >= float(config.get("stick_halo_mean_minimum", 4.0))
                and halo_percentile
                >= float(config.get("stick_halo_percentile_minimum", 8.0))
            )
            if self.previous_stick_x is None:
                # Every recorded catch introduces the real stick at the rail's
                # center. Decorative arches can also contain a tall dark seam;
                # without a bootstrap prior that seam becomes the remembered
                # stick for the entire catch.
                continuity_penalty = abs(center_x - rail.center_x) * float(
                    config.get("stick_bootstrap_center_penalty_per_pixel", 0.08)
                )
            else:
                continuity_penalty = abs(center_x - self.previous_stick_x) * float(
                    config.get("stick_continuity_penalty_per_pixel", 0.03)
                )
            dark_rows = min(
                int(np.max(dark_counts[start:end])),
                contiguous_height,
            )
            if halo_visible:
                score = (
                    float(dark_rows)
                    + min(left_green, right_green) * 0.08
                    + halo_percentile * 0.04
                    - continuity_penalty
                )
                source = "noiseform_green_halo_stick"
            else:
                # Once the bar separates, the black stick no longer has green
                # on both sides.  Its narrow, tall geometry and continuity are
                # still sufficient; requiring the halo here made detection
                # depend on the bar already overlapping the stick.
                if self.previous_stick_x is None:
                    continue
                maximum_distance = rail.width * float(
                    config.get("stick_no_halo_reacquire_radius_ratio", 0.12)
                )
                if (
                    abs(center_x - self.previous_stick_x) > maximum_distance
                    or dark_rows
                    < rail.height
                    * float(config.get("stick_no_halo_minimum_height_ratio", 0.62))
                ):
                    continue
                score = (
                    float(dark_rows)
                    - continuity_penalty
                    - float(config.get("stick_no_halo_score_penalty", 8.0))
                )
                source = "noiseform_no_halo_stick"
            candidates.append(
                (score, center_x, dark_rows, source)
            )

        if candidates:
            _, stick_x, dark_rows, source = max(candidates, key=lambda item: item[0])
            self.previous_stick_x = stick_x
            self.last_stick_timestamp_ms = timestamp_ms
            confidence = min(1.0, 0.45 + dark_rows / max(1.0, rail.height))
            if source == "noiseform_no_halo_stick":
                confidence = min(confidence, 0.68)
            return stick_x, confidence, source

        # Outside the green control body, both the stick and rail are dark and
        # merge into one wide threshold run.  Recover the stick from its paired
        # luma edges near the last confirmed position instead.
        edge_stick = self._find_no_halo_stick_edges(gray, rail, config)
        if edge_stick is not None:
            self.previous_stick_x = edge_stick
            self.last_stick_timestamp_ms = timestamp_ms
            return edge_stick, 0.58, "noiseform_no_halo_stick"

        hold_ms = float(config.get("stick_prediction_hold_ms", 320))
        if (
            self.previous_stick_x is not None
            and self.last_stick_timestamp_ms is not None
            and timestamp_ms - self.last_stick_timestamp_ms <= hold_ms
        ):
            return self.previous_stick_x, 0.35, "noiseform_temporary_prediction"
        return None, 0.0, "noiseform_stick_missing"

    def _find_no_halo_stick_edges(
        self,
        gray: np.ndarray,
        rail: PixelRect,
        config: dict,
    ) -> float | None:
        if self.previous_stick_x is None or gray.size == 0:
            return None
        previous_local = self.previous_stick_x - rail.left
        search_radius = max(
            5,
            round(
                rail.width
                * float(config.get("stick_no_halo_edge_search_radius_ratio", 0.020))
            ),
        )
        minimum_width = int(config.get("stick_minimum_width_px_at_1080p", 4))
        maximum_width = int(config.get("stick_maximum_width_px_at_1080p", 10))
        outside_width = max(2, round(rail.height * 0.07))
        pixel_contrast_minimum = float(
            config.get("stick_no_halo_pixel_contrast_minimum", 5.0)
        )
        mean_contrast_minimum = float(
            config.get("stick_no_halo_mean_contrast_minimum", 7.0)
        )
        row_fraction_minimum = float(
            config.get("stick_no_halo_contrast_row_fraction", 0.52)
        )
        best: tuple[float, float] | None = None
        start_center = max(maximum_width, round(previous_local) - search_radius)
        end_center = min(
            rail.width - maximum_width,
            round(previous_local) + search_radius,
        )
        gray_float = gray.astype(np.float32)
        for center_local in range(start_center, end_center + 1):
            for stick_width in range(minimum_width, maximum_width + 1):
                left = center_local - stick_width // 2
                right = left + stick_width
                outside_left = gray_float[:, left - outside_width : left]
                outside_right = gray_float[:, right : right + outside_width]
                inside = gray_float[:, left:right]
                if (
                    inside.size == 0
                    or outside_left.size == 0
                    or outside_right.size == 0
                ):
                    continue
                inside_rows = np.mean(inside, axis=1)
                outside_rows = np.minimum(
                    np.mean(outside_left, axis=1),
                    np.mean(outside_right, axis=1),
                )
                row_contrast = outside_rows - inside_rows
                row_fraction = float(
                    np.mean(row_contrast >= pixel_contrast_minimum)
                )
                mean_contrast = float(np.mean(np.maximum(0.0, row_contrast)))
                if (
                    row_fraction < row_fraction_minimum
                    or mean_contrast < mean_contrast_minimum
                ):
                    continue
                distance = abs(center_local - previous_local)
                score = (
                    mean_contrast
                    + row_fraction * 12.0
                    - distance
                    * float(config.get("stick_continuity_penalty_per_pixel", 0.03))
                )
                if best is None or score > best[0]:
                    best = (score, rail.left + float(center_local))
        return None if best is None else best[1]
