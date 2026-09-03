from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from app.core.models import DetectionSnapshot, FramePacket, PixelRect
from app.detection.base import Detector
from app.detection.special_standard import SpecialStandardDetector
from app.detection.trained_rod_vision import (
    TrainedRodVisionAssist,
    TrainedVisionFrame,
)


@dataclass
class _FallingNoteTrack:
    identifier: int
    x: float
    y: float
    timestamp_ms: float
    velocity_y_px_per_ms: float = 0.0
    confirmations: int = 1
    missing_frames: int = 0


@dataclass(frozen=True)
class _NoteTargetSelection:
    track: _FallingNoteTrack
    eta_ms: float
    lead_time_ms: float
    horizontal_travel_px: float


class PinionsAriaDetector(Detector):
    """Standard tracking plus moving musical-note targets for Pinion's Aria."""

    def __init__(self, profile: dict):
        self.profile = profile
        self.standard = SpecialStandardDetector(profile)
        self.trained_vision = TrainedRodVisionAssist(profile, "pinions_aria")
        self.tracks: list[_FallingNoteTrack] = []
        self.committed_control_targets: dict[int, float] = {}
        self.active_target_identifier: int | None = None
        self.next_identifier = 1
        self.trained_geometry_armed = False
        self.last_classic_lock_timestamp_ms: float | None = None
        self.last_bar_visible_timestamp_ms: float | None = None
        self.red_flash_width: int | None = None
        self.red_flash_until_timestamp_ms: float | None = None
        self.red_flash_was_present = False
        self.miss_recovery_until_timestamp_ms: float | None = None
        self.classic_missing_frames = 0
        self.last_note_scan_timestamp_ms: float | None = None
        self.completed_note_identifiers: set[int] = set()
        self.pending_note_successes: dict[int, float] = {}
        self.successful_note_count = 0
        self.note_only_mode = False
        self.minigame_acquired = False

    def reset(self) -> None:
        self.standard.reset()
        self.trained_vision.reset()
        self.tracks.clear()
        self.committed_control_targets.clear()
        self.active_target_identifier = None
        self.next_identifier = 1
        self.trained_geometry_armed = False
        self.last_classic_lock_timestamp_ms = None
        self.last_bar_visible_timestamp_ms = None
        self.red_flash_width = None
        self.red_flash_until_timestamp_ms = None
        self.red_flash_was_present = False
        self.miss_recovery_until_timestamp_ms = None
        self.classic_missing_frames = 0
        self.last_note_scan_timestamp_ms = None
        self.completed_note_identifiers.clear()
        self.pending_note_successes.clear()
        self.successful_note_count = 0
        self.note_only_mode = False
        self.minigame_acquired = False

    def detect(self, packet: FramePacket) -> DetectionSnapshot:
        snapshot = self.standard.detect(packet)
        config = self.profile["detection"]["pinions_aria"]
        red_flash_bar = self._find_red_flash_bar(packet.frame_bgr, config)
        red_flash_tint_present = self._has_red_flash_tint(packet.frame_bgr, config)
        if red_flash_bar is not None:
            snapshot.bar = red_flash_bar
            snapshot.bar_confidence = max(0.88, snapshot.bar_confidence)
            snapshot.extra["bar_geometry_source"] = "pinions_red_flash_core"
            snapshot.extra["raw_bar_left"] = red_flash_bar.left
            snapshot.extra["raw_bar_right"] = red_flash_bar.right
            snapshot.extra["raw_bar_width"] = red_flash_bar.width
            snapshot.extra["live_bar_width"] = red_flash_bar.width
            snapshot.extra["live_width_state"] = "pinions_red_flash_core"
            snapshot.extra["pinions_red_flash_core_width"] = red_flash_bar.width
            self.red_flash_width = red_flash_bar.width
            self.red_flash_until_timestamp_ms = packet.timestamp_ms + float(
                config.get("red_flash_width_hold_ms", 1150)
            )
            self.standard.previous_bar = red_flash_bar
        elif (
            snapshot.bar is not None
            and self.red_flash_width is not None
            and self.red_flash_until_timestamp_ms is not None
            and packet.timestamp_ms <= self.red_flash_until_timestamp_ms
            and red_flash_tint_present
        ):
            center_x = snapshot.bar.center_x
            held_left = round(center_x - (self.red_flash_width / 2.0))
            snapshot.bar = PixelRect(
                held_left,
                snapshot.bar.top,
                held_left + self.red_flash_width,
                snapshot.bar.bottom,
            ).clamped(packet.frame_bgr.shape[1], packet.frame_bgr.shape[0])
            snapshot.extra["bar_geometry_source"] = (
                "pinions_red_flash_fade_width_hold"
            )
            snapshot.extra["live_bar_width"] = snapshot.bar.width
            snapshot.extra["live_width_state"] = "pinions_red_flash_fade_width_hold"
            self.standard.previous_bar = snapshot.bar
        elif not red_flash_tint_present:
            self.red_flash_width = None
            self.red_flash_until_timestamp_ms = None
        snapshot.extra["pinions_red_flash_active"] = (
            self.red_flash_until_timestamp_ms is not None
            and packet.timestamp_ms <= self.red_flash_until_timestamp_ms
        )
        snapshot.extra["pinions_red_flash_core_detected"] = (
            red_flash_bar is not None
        )
        snapshot.extra["pinions_red_flash_tint_present"] = red_flash_tint_present
        if red_flash_tint_present:
            self.miss_recovery_until_timestamp_ms = packet.timestamp_ms + float(
                config.get("post_miss_real_fish_recovery_ms", 240)
            )
            if not self.red_flash_was_present:
                if self.active_target_identifier is not None:
                    self.completed_note_identifiers.add(
                        self.active_target_identifier
                    )
                self.active_target_identifier = None
                self.committed_control_targets.clear()
                self.completed_note_identifiers.update(
                    self.pending_note_successes.keys()
                )
                self.pending_note_successes.clear()
                self.successful_note_count = 0
                self.note_only_mode = False
        self.red_flash_was_present = red_flash_tint_present
        miss_recovery_active = (
            self.miss_recovery_until_timestamp_ms is not None
            and packet.timestamp_ms <= self.miss_recovery_until_timestamp_ms
        )
        snapshot.extra["pinions_miss_recovery_active"] = miss_recovery_active
        snapshot.minigame_visible = (
            snapshot.bar is not None
            and snapshot.rail is not None
            and snapshot.stick is not None
        )
        if snapshot.minigame_visible:
            snapshot.detector_state = "LOCKED"
            snapshot.rejection_reason = ""
        classic_locked = (
            snapshot.minigame_visible
            and snapshot.rail is not None
            and snapshot.bar is not None
            and snapshot.stick is not None
        )
        if classic_locked:
            self.classic_missing_frames = 0
            self.trained_geometry_armed = True
            self.last_classic_lock_timestamp_ms = packet.timestamp_ms
        else:
            self.classic_missing_frames += 1
        trained_mode = str(
            self.profile["detection"].get("trained_vision", {}).get(
                "mode", "continuous"
            )
        )
        recovery_after_missing = int(
            self.profile["detection"].get("trained_vision", {}).get(
                "recovery_after_missing_frames", 1
            )
        )
        trained_should_run = (
            trained_mode != "recovery_only"
            or self.classic_missing_frames >= recovery_after_missing
        )
        if trained_should_run:
            trained = self.trained_vision.detect(packet.frame_bgr, packet.timestamp_ms)
        else:
            trained = TrainedVisionFrame(
                self.trained_vision.available,
                False,
                (),
                (),
                0.0,
            )
        snapshot.extra["pinions_trained_vision_mode"] = trained_mode
        snapshot.extra["pinions_classic_missing_frames"] = self.classic_missing_frames
        snapshot.extra["pinions_trained_vision_skipped_healthy"] = (
            not trained_should_run
        )
        self._record_trained_telemetry(snapshot, trained)
        assist_allowed, assist_reason = self._trained_geometry_assist_allowed(
            snapshot,
            packet.timestamp_ms,
            config,
        )
        assist_applied = self._apply_trained_geometry(
            snapshot,
            trained,
            assist_allowed=assist_allowed,
        )
        snapshot.extra["pinions_trained_geometry_armed"] = self.trained_geometry_armed
        snapshot.extra["pinions_trained_geometry_allowed"] = assist_allowed
        snapshot.extra["pinions_trained_geometry_applied"] = assist_applied
        snapshot.extra["pinions_trained_geometry_gate"] = assist_reason
        snapshot.extra["pinions_note_target_active"] = False
        snapshot.extra["pinions_note_candidates"] = 0
        snapshot.extra["pinions_note_tracks"] = len(self.tracks)
        snapshot.extra["pinions_note_target_x"] = ""
        snapshot.extra["pinions_note_target_y"] = ""
        snapshot.extra["pinions_control_target_x"] = ""
        snapshot.extra["pinions_note_target_identifier"] = ""
        snapshot.extra["pinions_note_eta_ms"] = ""
        snapshot.extra["pinions_note_lead_time_ms"] = ""
        snapshot.extra["pinions_note_horizontal_travel_px"] = ""
        snapshot.extra["pinions_note_new_tracks"] = 0
        snapshot.extra["pinions_note_late_spawn_rejections"] = 0
        snapshot.extra["pinions_note_success_count"] = self.successful_note_count
        snapshot.extra["pinions_note_only_mode"] = self.note_only_mode
        snapshot.extra["pinions_note_target_released"] = False
        current_rail_visible, rail_run_px, rail_coverage = (
            self._detect_current_rail_body(
                packet.frame_bgr,
                snapshot.rail,
                config,
            )
        )
        snapshot.extra["pinions_base_rail_geometry_source"] = snapshot.extra.get(
            "rail_geometry_source", ""
        )
        snapshot.extra["pinions_current_rail_visible"] = current_rail_visible
        snapshot.extra["pinions_current_rail_longest_run_px"] = rail_run_px
        snapshot.extra["pinions_current_rail_coverage"] = round(rail_coverage, 4)
        snapshot.extra["pinions_current_rail_source"] = (
            "palette_body" if current_rail_visible else "absent"
        )
        if snapshot.minigame_visible and current_rail_visible:
            self.minigame_acquired = True
        snapshot.extra["pinions_minigame_acquired"] = self.minigame_acquired

        # The normalized rail and paired-edge stick fallbacks can both outlive
        # the real Pinions UI. Once a genuine current-frame rail has acquired
        # this minigame, stick-only evidence without either a live bar or live
        # lavender rail is not playable evidence. Clear only the observation;
        # the lifecycle's existing 650 ms release timer still absorbs brief
        # visual dropouts and the detector remains able to reacquire the UI.
        if (
            self.minigame_acquired
            and snapshot.bar is None
            and not current_rail_visible
        ):
            snapshot.rail = None
            snapshot.stick = None
            snapshot.stick_confidence = 0.0
            snapshot.minigame_visible = False
            snapshot.detector_state = "SEARCHING"
            snapshot.rejection_reason = "pinions_current_rail_absent"
            snapshot.extra["pinions_lifecycle_evidence"] = (
                "fallback_components_suppressed"
            )
        elif snapshot.bar is None and current_rail_visible:
            snapshot.extra["pinions_lifecycle_evidence"] = "current_rail_partial"
        elif snapshot.minigame_visible:
            snapshot.extra["pinions_lifecycle_evidence"] = "complete_components"
        else:
            snapshot.extra["pinions_lifecycle_evidence"] = "not_acquired"

        if snapshot.bar is not None:
            self.last_bar_visible_timestamp_ms = packet.timestamp_ms
        bar_missing_age_ms = (
            None
            if self.last_bar_visible_timestamp_ms is None
            else max(0.0, packet.timestamp_ms - self.last_bar_visible_timestamp_ms)
        )
        bar_geometry_grace_ms = float(
            config.get("bar_geometry_missing_grace_ms", 0)
        )
        bar_geometry_grace_active = (
            snapshot.bar is None
            and bar_missing_age_ms is not None
            and bar_missing_age_ms <= bar_geometry_grace_ms
        )
        snapshot.extra["pinions_bar_geometry_missing_age_ms"] = (
            "" if bar_missing_age_ms is None else round(bar_missing_age_ms, 3)
        )
        snapshot.extra["pinions_bar_geometry_grace_active"] = (
            bar_geometry_grace_active
        )

        # Note mechanics exist only while the standard rail/bar minigame is
        # present. Catch/result screens must not turn unrelated bright effects
        # into persistent targets.
        if snapshot.bar is None:
            # A missed Pinions note covers the real body in red for roughly one
            # second.  Preserve identity and the committed intercept throughout
            # that short visual effect; clearing here made the same falling note
            # return with a new ID and sent the bar toward a different goal.
            if not bar_geometry_grace_active:
                self.tracks.clear()
                self.committed_control_targets.clear()
                self.active_target_identifier = None
                self.completed_note_identifiers.clear()
                self.pending_note_successes.clear()
                self.successful_note_count = 0
                self.note_only_mode = False
            return snapshot

        note_scan_interval_ms = float(config.get("note_scan_interval_ms", 0))
        note_scan_due = (
            self.last_note_scan_timestamp_ms is None
            or packet.timestamp_ms - self.last_note_scan_timestamp_ms
            >= note_scan_interval_ms
        )
        if note_scan_due:
            classic_candidates = self._find_note_candidates(
                packet.frame_bgr, config
            )
            self.last_note_scan_timestamp_ms = packet.timestamp_ms
        else:
            classic_candidates = []
        trained_candidates = (
            self._trained_note_candidates(
                trained,
                packet.frame_bgr.shape[1],
                packet.frame_bgr.shape[0],
                config,
            )
            if classic_locked or assist_applied
            else []
        )
        candidates = self._merge_candidate_sources(
            classic_candidates,
            trained_candidates,
            packet.frame_bgr.shape[1],
            packet.frame_bgr.shape[0],
        )
        if note_scan_due or trained_candidates:
            new_tracks, late_spawn_rejections = self._update_tracks(
                candidates,
                packet.timestamp_ms,
                packet.frame_bgr.shape[1],
                packet.frame_bgr.shape[0],
                config,
            )
        else:
            new_tracks, late_spawn_rejections = 0, 0
        snapshot.extra["pinions_note_candidates"] = len(candidates)
        snapshot.extra["pinions_classic_note_candidates"] = len(classic_candidates)
        snapshot.extra["pinions_trained_note_candidates"] = len(trained_candidates)
        snapshot.extra["pinions_note_scan_ran"] = note_scan_due
        snapshot.extra["pinions_note_new_tracks"] = new_tracks
        snapshot.extra["pinions_note_late_spawn_rejections"] = (
            late_spawn_rejections
        )
        released_target = False
        if not miss_recovery_active:
            released_target = self._update_note_progress(
                config,
                packet.frame_bgr.shape[0],
                packet.timestamp_ms,
            )
        snapshot.extra["pinions_note_success_count"] = self.successful_note_count
        snapshot.extra["pinions_note_only_mode"] = self.note_only_mode
        snapshot.extra["pinions_note_target_released"] = released_target
        snapshot.extra["pinions_note_tracks"] = len(self.tracks)
        snapshot.extra["pinions_note_positions"] = ";".join(
            f"{track.identifier}:{track.x:.1f},{track.y:.1f},{track.confirmations}"
            for track in sorted(self.tracks, key=lambda item: item.y, reverse=True)
        )
        active_track_ids = {track.identifier for track in self.tracks}
        self.committed_control_targets = {
            identifier: target_x
            for identifier, target_x in self.committed_control_targets.items()
            if identifier in active_track_ids
        }
        if self.active_target_identifier not in active_track_ids:
            self.active_target_identifier = None

        if miss_recovery_active:
            snapshot.extra["pinions_note_target_state"] = (
                "miss_recovery_real_fish"
            )
            return snapshot

        bar = snapshot.bar
        assert bar is not None
        selection = self._select_target(
            config,
            bar,
            packet.frame_bgr.shape[0],
            packet.timestamp_ms,
        )
        if selection is None:
            if self.tracks:
                snapshot.extra["pinions_note_target_state"] = (
                    "unreachable_note_ignored"
                )
            if self.note_only_mode:
                self._center_virtual_stick_on_bar(snapshot, bar)
                snapshot.extra["pinions_note_target_state"] = (
                    "note_only_waiting"
                )
            return snapshot

        target = selection.track
        actual_stick = snapshot.stick
        inset = self._catch_inset(bar, packet.frame_bgr.shape[0], config)
        safe_left = bar.left + inset
        safe_right = bar.right - inset
        snapshot.extra["pinions_note_target_x"] = round(target.x, 2)
        snapshot.extra["pinions_note_target_y"] = round(target.y, 2)
        snapshot.extra["pinions_note_target_identifier"] = target.identifier
        snapshot.extra["pinions_note_eta_ms"] = round(selection.eta_ms, 1)
        snapshot.extra["pinions_note_lead_time_ms"] = round(
            selection.lead_time_ms, 1
        )
        snapshot.extra["pinions_note_horizontal_travel_px"] = round(
            selection.horizontal_travel_px, 1
        )
        snapshot.extra["pinions_original_stick_x"] = (
            "" if actual_stick is None else round(actual_stick.center_x, 2)
        )

        committed_target_x = self.committed_control_targets.get(target.identifier)
        if committed_target_x is None and safe_left <= target.x <= safe_right:
            snapshot.extra["pinions_note_target_state"] = "covered_by_bar"
            if self.note_only_mode:
                self._center_virtual_stick_on_bar(snapshot, bar)
            return snapshot

        snapshot.extra["pinions_note_target_active"] = True
        self.active_target_identifier = target.identifier

        if committed_target_x is not None:
            control_target_x = committed_target_x
            target_state = "hold_note_intercept"
        elif target.x < safe_left:
            control_target_x = target.x + bar.width / 2.0 - inset
            target_state = "catch_with_left_edge"
        else:
            control_target_x = target.x - bar.width / 2.0 + inset
            target_state = "catch_with_right_edge"

        rail = snapshot.rail
        if rail is not None:
            half_width = bar.width / 2.0
            control_target_x = min(
                rail.right - half_width,
                max(rail.left + half_width, control_target_x),
            )
        if committed_target_x is None:
            # Once steering begins, keep one stable horizontal intercept until
            # the falling note is caught or leaves the scan. Returning to the
            # fish stick as soon as the note first enters the bar caused a
            # second, late emergency swing for the same note.
            self.committed_control_targets[target.identifier] = control_target_x
        stick_half_width = 5
        if actual_stick is not None:
            stick_half_width = max(2, round(actual_stick.width / 2.0))
            top, bottom = actual_stick.top, actual_stick.bottom
        else:
            top, bottom = bar.top, bar.bottom
        center = round(control_target_x)
        snapshot.stick = PixelRect(
            center - stick_half_width,
            top,
            center + stick_half_width,
            bottom,
        )
        snapshot.stick_confidence = min(
            0.92,
            max(0.72, snapshot.stick_confidence),
        )
        snapshot.minigame_visible = snapshot.rail is not None
        snapshot.detector_state = "PINIONS_NOTE_TARGET"
        snapshot.rejection_reason = ""
        snapshot.extra["pinions_note_target_state"] = target_state
        snapshot.extra["pinions_control_target_x"] = round(control_target_x, 2)
        return snapshot

    def _update_note_progress(
        self,
        config: dict,
        frame_height: int,
        timestamp_ms: float,
    ) -> bool:
        """Confirm completed notes and retire their stale steering intercepts."""
        intercept_y = frame_height * float(
            config.get("note_intercept_y", config.get("note_track_bottom", 0.79))
        )
        minimum_confirmations = int(config.get("note_confirmation_frames", 2))
        minimum_velocity = float(
            config.get("note_minimum_downward_velocity_px_per_ms", 0.04)
        )
        success_confirmation_ms = float(
            config.get("note_success_confirmation_ms", 120)
        )
        release_after_ms = float(
            config.get("note_post_intercept_release_ms", 120)
        )

        expired_identifiers: set[int] = set()
        for track in self.tracks:
            if (
                track.confirmations < minimum_confirmations
                or track.velocity_y_px_per_ms < minimum_velocity
            ):
                continue
            elapsed = max(0.0, timestamp_ms - track.timestamp_ms)
            projected_y = track.y + track.velocity_y_px_per_ms * elapsed
            if projected_y < intercept_y:
                continue
            time_past_intercept_ms = (
                projected_y - intercept_y
            ) / track.velocity_y_px_per_ms
            crossing_timestamp_ms = timestamp_ms - time_past_intercept_ms
            if (
                track.identifier not in self.completed_note_identifiers
                and track.identifier not in self.pending_note_successes
            ):
                self.pending_note_successes[track.identifier] = (
                    crossing_timestamp_ms
                )
            if time_past_intercept_ms >= release_after_ms:
                expired_identifiers.add(track.identifier)

        for identifier, crossing_timestamp_ms in list(
            self.pending_note_successes.items()
        ):
            if timestamp_ms - crossing_timestamp_ms < success_confirmation_ms:
                continue
            self.pending_note_successes.pop(identifier, None)
            self.completed_note_identifiers.add(identifier)
            self.successful_note_count += 1

        activation_count = int(config.get("note_only_activation_count", 7))
        if self.successful_note_count >= activation_count:
            self.note_only_mode = True

        if not expired_identifiers:
            return False
        self.tracks = [
            track
            for track in self.tracks
            if track.identifier not in expired_identifiers
        ]
        for identifier in expired_identifiers:
            self.committed_control_targets.pop(identifier, None)
        if self.active_target_identifier in expired_identifiers:
            self.active_target_identifier = None
        return True

    @staticmethod
    def _center_virtual_stick_on_bar(
        snapshot: DetectionSnapshot,
        bar: PixelRect,
    ) -> None:
        actual_stick = snapshot.stick
        if actual_stick is None:
            half_width = 5
            top, bottom = bar.top, bar.bottom
        else:
            half_width = max(2, round(actual_stick.width / 2.0))
            top, bottom = actual_stick.top, actual_stick.bottom
        center = round(bar.center_x)
        snapshot.stick = PixelRect(
            center - half_width,
            top,
            center + half_width,
            bottom,
        )
        snapshot.stick_confidence = min(
            0.92,
            max(0.72, snapshot.stick_confidence),
        )
        snapshot.minigame_visible = snapshot.rail is not None
        snapshot.detector_state = "PINIONS_NOTE_ONLY"
        snapshot.rejection_reason = ""

    @staticmethod
    def _detect_current_rail_body(
        frame: np.ndarray,
        rail: PixelRect | None,
        config: dict,
    ) -> tuple[bool, int, float]:
        """Validate Pinions' long lavender rail in the current frame."""
        if rail is None or rail.width <= 0 or rail.height <= 0:
            return False, 0, 0.0

        height, width = frame.shape[:2]
        safe_rail = rail.clamped(width, height)
        crop = frame[
            safe_rail.top : safe_rail.bottom,
            safe_rail.left : safe_rail.right,
        ]
        if crop.size == 0:
            return False, 0, 0.0

        evidence = config.get("current_rail_evidence", {})
        blue, green, red = (
            channel.astype(np.int16) for channel in cv2.split(crop)
        )
        mask = (
            (blue >= int(evidence.get("blue_minimum", 140)))
            & (green >= int(evidence.get("green_minimum", 100)))
            & (red >= int(evidence.get("red_minimum", 95)))
            & (
                blue - red
                >= int(evidence.get("blue_minus_red_minimum", 0))
            )
            & (
                blue - green
                <= int(evidence.get("blue_minus_green_maximum", 110))
            )
        )
        column_occupancy = np.mean(mask, axis=0) >= float(
            evidence.get("minimum_column_occupancy", 0.18)
        )
        longest_run = 0
        current_run = 0
        for present in column_occupancy:
            if present:
                current_run += 1
                longest_run = max(longest_run, current_run)
            else:
                current_run = 0

        coverage = float(np.mean(mask))
        minimum_run = safe_rail.width * float(
            evidence.get("minimum_longest_run_ratio", 0.30)
        )
        minimum_coverage = float(evidence.get("minimum_coverage", 0.20))
        return (
            longest_run >= minimum_run and coverage >= minimum_coverage,
            longest_run,
            coverage,
        )

    @staticmethod
    def _find_red_flash_bar(frame: np.ndarray, config: dict) -> PixelRect | None:
        """Return the solid red playable body, excluding its glow/progress line."""
        height, width = frame.shape[:2]
        left = round(width * float(config.get("red_flash_scan_left", 0.25)))
        right = round(width * float(config.get("red_flash_scan_right", 0.75)))
        top = round(height * float(config.get("red_flash_scan_top", 0.82)))
        bottom = round(height * float(config.get("red_flash_scan_bottom", 0.885)))
        crop = frame[top:bottom, left:right]
        if crop.size == 0:
            return None
        blue, green, red = (
            channel.astype(np.int16) for channel in cv2.split(crop)
        )
        match = (
            (blue >= int(config.get("red_flash_blue_minimum", 35)))
            & (blue <= int(config.get("red_flash_blue_maximum", 115)))
            & (green >= int(config.get("red_flash_green_minimum", 35)))
            & (green <= int(config.get("red_flash_green_maximum", 115)))
            & (red >= int(config.get("red_flash_red_minimum", 175)))
            & (red <= int(config.get("red_flash_red_maximum", 240)))
            & (
                red - green
                >= int(config.get("red_flash_red_minus_green_minimum", 60))
            )
            & (
                red - blue
                >= int(config.get("red_flash_red_minus_blue_minimum", 60))
            )
        )
        mask = np.zeros(crop.shape[:2], dtype=np.uint8)
        mask[match] = 255
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            np.ones((3, 9), np.uint8),
        )
        count, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        candidates: list[tuple[float, PixelRect]] = []
        for index in range(1, count):
            x, y, candidate_width, candidate_height, area = stats[index]
            width_ratio = candidate_width / max(1, width)
            height_ratio = candidate_height / max(1, height)
            if (
                0.05 <= width_ratio <= 0.35
                and 0.012 <= height_ratio <= 0.075
                and candidate_width / max(1, candidate_height) >= 3.0
                and area >= 80
            ):
                rect = PixelRect(
                    left + int(x),
                    top + int(y),
                    left + int(x + candidate_width),
                    top + int(y + candidate_height),
                )
                candidates.append((float(area + candidate_width * 2), rect))
        if not candidates:
            return None
        return max(candidates, key=lambda item: item[0])[1]

    @staticmethod
    def _has_red_flash_tint(frame: np.ndarray, config: dict) -> bool:
        """Keep the width hold only while the red miss overlay is visibly fading."""
        height, width = frame.shape[:2]
        left = round(width * float(config.get("red_flash_scan_left", 0.25)))
        right = round(width * float(config.get("red_flash_scan_right", 0.75)))
        top = round(height * float(config.get("red_flash_scan_top", 0.82)))
        bottom = round(height * float(config.get("red_flash_scan_bottom", 0.885)))
        crop = frame[top:bottom, left:right]
        if crop.size == 0:
            return False

        blue, green, red = [channel.astype(np.int16) for channel in cv2.split(crop)]
        match = (
            (blue >= int(config.get("red_flash_fade_blue_minimum", 35)))
            & (blue <= int(config.get("red_flash_fade_blue_maximum", 235)))
            & (green >= int(config.get("red_flash_fade_green_minimum", 35)))
            & (green <= int(config.get("red_flash_fade_green_maximum", 215)))
            & (red >= int(config.get("red_flash_fade_red_minimum", 145)))
            & (red <= int(config.get("red_flash_fade_red_maximum", 245)))
            & (
                red - green
                >= int(config.get("red_flash_fade_red_minus_green_minimum", 12))
            )
            & (
                red - blue
                >= int(config.get("red_flash_fade_red_minus_blue_minimum", 12))
            )
        )
        return int(np.count_nonzero(match)) >= int(
            config.get("red_flash_fade_minimum_pixels_1080p", 120)
        )

    @staticmethod
    def _record_trained_telemetry(
        snapshot: DetectionSnapshot,
        trained: TrainedVisionFrame,
    ) -> None:
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

    def _trained_geometry_assist_allowed(
        self,
        snapshot: DetectionSnapshot,
        timestamp_ms: float,
        config: dict,
    ) -> tuple[bool, str]:
        if not self.trained_geometry_armed or self.last_classic_lock_timestamp_ms is None:
            return False, "not_armed_by_classic_lock"
        grace_ms = float(config.get("trained_geometry_grace_ms", 320.0))
        elapsed_ms = timestamp_ms - self.last_classic_lock_timestamp_ms
        if elapsed_ms < 0 or elapsed_ms > grace_ms:
            self.trained_geometry_armed = False
            return False, "classic_lock_expired"
        if snapshot.rail is None:
            self.trained_geometry_armed = False
            return False, "rail_missing"
        current_classic_evidence = (
            int(snapshot.extra.get("bar_candidate_count", 0)) > 0
            or int(snapshot.extra.get("stick_candidate_count", 0)) > 0
            or int(snapshot.extra.get("bar_fallback_arrow_count", 0)) > 0
        )
        if not current_classic_evidence:
            self.trained_geometry_armed = False
            return False, "no_current_classic_evidence"
        return True, "recent_classic_lock_with_current_evidence"

    @staticmethod
    def _apply_trained_geometry(
        snapshot: DetectionSnapshot,
        trained: TrainedVisionFrame,
        *,
        assist_allowed: bool,
    ) -> bool:
        if not assist_allowed:
            return False
        rail = snapshot.rail
        if rail is None:
            return False
        applied = False
        width = snapshot.frame_width
        height = snapshot.frame_height
        rail_center_y = (rail.top + rail.bottom) / 2.0
        bar_candidates = trained.detections("bar", trusted=True)
        model_bar = max(bar_candidates, key=lambda item: item.confidence, default=None)
        bar_valid = (
            model_bar is not None
            and width * 0.07 <= model_bar.width <= width * 0.38
            and height * 0.012 <= model_bar.height <= height * 0.11
            and rail.left <= model_bar.center_x <= rail.right
            and abs(model_bar.center_y - rail_center_y) <= height * 0.06
        )
        if snapshot.bar is None and bar_valid:
            assert model_bar is not None
            left = max(rail.left, min(rail.right - model_bar.width, model_bar.left))
            snapshot.bar = PixelRect(left, rail.top, left + model_bar.width, rail.bottom)
            snapshot.bar_confidence = min(0.88, model_bar.confidence)
            snapshot.extra["bar_geometry_source"] = "pinions_trained_fallback"
            applied = True

        stick_candidates = trained.detections("stick", trusted=True)
        model_stick = max(stick_candidates, key=lambda item: item.confidence, default=None)
        stick_valid = (
            model_stick is not None
            and 1 <= model_stick.width <= width * 0.025
            and height * 0.018 <= model_stick.height <= height * 0.15
            and rail.left <= model_stick.center_x <= rail.right
            and abs(model_stick.center_y - rail_center_y) <= height * 0.06
        )
        if snapshot.stick is None and snapshot.bar is not None and stick_valid:
            assert model_stick is not None
            half_width = max(2, round(model_stick.width / 2.0))
            center = round(model_stick.center_x)
            snapshot.stick = PixelRect(
                center - half_width,
                min(rail.top, model_stick.top),
                center + half_width,
                max(rail.bottom, model_stick.bottom),
            )
            snapshot.stick_confidence = min(0.88, model_stick.confidence)
            snapshot.extra["stick_geometry_source"] = "pinions_trained_fallback"
            applied = True

        if snapshot.bar is not None and snapshot.stick is not None:
            snapshot.minigame_visible = True
            if snapshot.detector_state in {"SEARCHING", "PARTIAL"}:
                snapshot.detector_state = "LOCKED"
                snapshot.rejection_reason = ""
        return applied

    @staticmethod
    def _trained_note_candidates(
        trained: TrainedVisionFrame,
        frame_width: int,
        frame_height: int,
        config: dict,
    ) -> list[tuple[float, float]]:
        left = frame_width * float(config.get("note_scan_left", 0.09))
        right = frame_width * float(config.get("note_scan_right", 0.91))
        top = frame_height * float(config.get("note_scan_top", 0.018))
        bottom = frame_height * float(config.get("note_scan_bottom", 0.82))
        candidates: list[tuple[float, float]] = []
        for item in trained.detections("note"):
            if not (
                left <= item.center_x <= right
                and top <= item.center_y <= bottom
                and frame_width * 0.006 <= item.width <= frame_width * 0.075
                and frame_height * 0.012 <= item.height <= frame_height * 0.13
            ):
                continue
            candidates.append((item.center_x, item.center_y))
        return candidates

    @staticmethod
    def _merge_candidate_sources(
        classic: list[tuple[float, float]],
        trained: list[tuple[float, float]],
        frame_width: int,
        frame_height: int,
    ) -> list[tuple[float, float]]:
        merged = list(classic)
        x_tolerance = frame_width * 0.025
        y_tolerance = frame_height * 0.035
        for candidate in trained:
            if any(
                abs(candidate[0] - current[0]) <= x_tolerance
                and abs(candidate[1] - current[1]) <= y_tolerance
                for current in merged
            ):
                continue
            merged.append(candidate)
        return merged

    def _select_target(
        self,
        config: dict,
        bar: PixelRect,
        frame_height: int,
        timestamp_ms: float,
    ) -> _NoteTargetSelection | None:
        minimum_confirmations = int(config.get("note_confirmation_frames", 2))
        committed_missing_grace = int(
            config.get("note_commit_missing_frame_grace", 3)
        )
        eligible = [
            track
            for track in self.tracks
            if track.confirmations >= minimum_confirmations
            and (
                track.missing_frames == 0
                or (
                    track.identifier in self.committed_control_targets
                    and track.missing_frames <= committed_missing_grace
                )
            )
            and track.velocity_y_px_per_ms
            >= float(config.get("note_minimum_downward_velocity_px_per_ms", 0.04))
        ]
        if not eligible:
            return None

        intercept_y = frame_height * float(
            config.get("note_intercept_y", config.get("note_track_bottom", 0.79))
        )
        inset = self._catch_inset(bar, frame_height, config)
        safe_left = bar.left + inset
        safe_right = bar.right - inset
        horizontal_speed = max(
            0.05,
            float(config.get("note_reachable_horizontal_speed_px_per_ms", 0.42)),
        )
        braking_reserve_ms = float(
            config.get("note_reachability_braking_reserve_ms", 260)
        )
        feasibility_grace_ms = float(
            config.get("note_reachability_grace_ms", 90)
        )
        selections: list[_NoteTargetSelection] = []
        for track in eligible:
            elapsed = max(0.0, timestamp_ms - track.timestamp_ms)
            projected_y = track.y + track.velocity_y_px_per_ms * elapsed
            eta_ms = max(
                0.0,
                (intercept_y - projected_y) / track.velocity_y_px_per_ms,
            )
            if track.x < safe_left:
                horizontal_travel = safe_left - track.x
            elif track.x > safe_right:
                horizontal_travel = track.x - safe_right
            else:
                horizontal_travel = 0.0
            lead_time_ms = (
                horizontal_travel / horizontal_speed + braking_reserve_ms
            )
            already_committed = (
                track.identifier in self.committed_control_targets
            )
            if not already_committed and eta_ms + feasibility_grace_ms < lead_time_ms:
                continue
            selections.append(
                _NoteTargetSelection(
                    track,
                    eta_ms,
                    lead_time_ms,
                    horizontal_travel,
                )
            )
        if not selections:
            return None

        best = min(
            selections,
            key=lambda item: (item.eta_ms, -item.track.y, item.track.x),
        )
        current = next(
            (
                item
                for item in selections
                if item.track.identifier == self.active_target_identifier
            ),
            None,
        )
        switch_margin_ms = float(config.get("note_priority_switch_margin_ms", 220))
        if current is not None and current.eta_ms <= best.eta_ms + switch_margin_ms:
            return current
        return best

    @staticmethod
    def _catch_inset(bar: PixelRect, frame_height: int, config: dict) -> float:
        scaled_minimum = float(
            config.get("note_minimum_catch_inset_1080p", 24.0)
        ) * frame_height / 1080.0
        requested = max(
            4.0,
            scaled_minimum,
            bar.width * float(config.get("note_catch_inset_bar_ratio", 0.10)),
        )
        return min(requested, max(2.0, bar.width / 2.0 - 2.0))

    def _update_tracks(
        self,
        candidates: list[tuple[float, float]],
        timestamp_ms: float,
        frame_width: int,
        frame_height: int,
        config: dict,
    ) -> tuple[int, int]:
        unmatched = set(range(len(candidates)))
        x_tolerance = frame_width * float(config.get("note_track_x_tolerance", 0.035))
        y_tolerance = frame_height * float(config.get("note_track_y_tolerance", 0.10))
        minimum_velocity = float(
            config.get("note_minimum_downward_velocity_px_per_ms", 0.04)
        )
        maximum_velocity = float(
            config.get("note_maximum_downward_velocity_px_per_ms", 1.6)
        )

        for track in sorted(self.tracks, key=lambda item: item.y, reverse=True):
            elapsed = max(1.0, timestamp_ms - track.timestamp_ms)
            predicted_y = track.y + max(0.0, track.velocity_y_px_per_ms) * elapsed
            feasible = []
            for index in unmatched:
                new_x, new_y = candidates[index]
                measured_velocity = (new_y - track.y) / elapsed
                if (
                    abs(new_x - track.x) <= x_tolerance
                    and abs(new_y - predicted_y) <= y_tolerance
                    and minimum_velocity <= measured_velocity <= maximum_velocity
                ):
                    feasible.append(index)
            candidate = min(
                feasible,
                key=lambda index: (
                    abs(candidates[index][0] - track.x)
                    + abs(candidates[index][1] - predicted_y) * 0.35
                ),
                default=None,
            )
            if candidate is None:
                track.missing_frames += 1
                continue
            new_x, new_y = candidates[candidate]
            measured_velocity = (new_y - track.y) / elapsed
            unmatched.remove(candidate)
            smoothing = float(config.get("note_velocity_smoothing", 0.45))
            if track.velocity_y_px_per_ms <= 0:
                track.velocity_y_px_per_ms = measured_velocity
            else:
                track.velocity_y_px_per_ms += smoothing * (
                    measured_velocity - track.velocity_y_px_per_ms
                )
            track.confirmations += 1
            track.x += 0.25 * (new_x - track.x)
            track.y = new_y
            track.timestamp_ms = timestamp_ms
            track.missing_frames = 0

        new_track_limit = frame_height * float(
            config.get("note_new_track_max_y", 0.68)
        )
        new_tracks = 0
        late_spawn_rejections = 0
        for index in unmatched:
            x, y = candidates[index]
            if y > new_track_limit:
                late_spawn_rejections += 1
                continue
            self.tracks.append(
                _FallingNoteTrack(
                    identifier=self.next_identifier,
                    x=x,
                    y=y,
                    timestamp_ms=timestamp_ms,
                )
            )
            self.next_identifier += 1
            new_tracks += 1

        maximum_missing = int(config.get("note_track_maximum_missing_frames", 3))
        bottom_limit = frame_height * float(
            config.get(
                "note_track_bottom",
                config.get("note_scan_bottom", 0.82),
            )
        )
        self.tracks = [
            track
            for track in self.tracks
            if track.missing_frames <= maximum_missing and track.y <= bottom_limit
        ]
        return new_tracks, late_spawn_rejections

    @staticmethod
    def _find_note_candidates(frame: np.ndarray, config: dict) -> list[tuple[float, float]]:
        height, width = frame.shape[:2]
        scale = min(width / 1920.0, height / 1080.0)
        left = round(width * float(config.get("note_scan_left", 0.09)))
        right = round(width * float(config.get("note_scan_right", 0.91)))
        top = round(height * float(config.get("note_scan_top", 0.018)))
        bottom = round(height * float(config.get("note_scan_bottom", 0.82)))
        crop = frame[top:bottom, left:right]
        blue, green, red = (
            channel.astype(np.int16) for channel in cv2.split(crop)
        )
        mask = (
            (blue >= int(config.get("note_blue_minimum", 170)))
            & (green >= int(config.get("note_green_minimum", 150)))
            & (red >= int(config.get("note_red_minimum", 135)))
            & (
                blue - red
                >= int(config.get("note_blue_minus_red_minimum", 12))
            )
        ).astype(np.uint8)
        count, _, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
        minimum_area = max(30, round(float(config.get("note_minimum_area_1080p", 80)) * scale * scale))
        maximum_area = max(minimum_area + 1, round(float(config.get("note_maximum_area_1080p", 5000)) * scale * scale))
        minimum_width = max(8, round(float(config.get("note_minimum_width_1080p", 18)) * scale))
        maximum_width = max(minimum_width + 1, round(float(config.get("note_maximum_width_1080p", 100)) * scale))
        minimum_height = max(10, round(float(config.get("note_minimum_height_1080p", 22)) * scale))
        maximum_height = max(minimum_height + 1, round(float(config.get("note_maximum_height_1080p", 110)) * scale))
        candidates: list[tuple[float, float]] = []
        for index in range(1, count):
            _, _, component_width, component_height, area = stats[index]
            if not (
                minimum_area <= area <= maximum_area
                and minimum_width <= component_width <= maximum_width
                and minimum_height <= component_height <= maximum_height
            ):
                continue
            candidates.append(
                (
                    left + float(centroids[index][0]),
                    top + float(centroids[index][1]),
                )
            )
        return PinionsAriaDetector._merge_spawn_fragments(candidates, width, height, config)

    @staticmethod
    def _merge_spawn_fragments(
        candidates: list[tuple[float, float]],
        width: int,
        height: int,
        config: dict,
    ) -> list[tuple[float, float]]:
        x_tolerance = width * float(config.get("note_fragment_merge_x", 0.055))
        y_tolerance = height * float(config.get("note_fragment_merge_y", 0.05))
        remaining = list(candidates)
        merged: list[tuple[float, float]] = []
        while remaining:
            seed_x, seed_y = remaining.pop(0)
            group = [(seed_x, seed_y)]
            kept: list[tuple[float, float]] = []
            for x, y in remaining:
                if abs(x - seed_x) <= x_tolerance and abs(y - seed_y) <= y_tolerance:
                    group.append((x, y))
                else:
                    kept.append((x, y))
            remaining = kept
            merged.append(
                (
                    sum(item[0] for item in group) / len(group),
                    sum(item[1] for item in group) / len(group),
                )
            )
        return merged
