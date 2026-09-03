from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from app.core.models import DetectionSnapshot, FramePacket, NormalizedRect
from app.detection.base import Detector
from app.detection.standard import StandardDetector


@dataclass
class _NoteTrack:
    identifier: int
    y: float
    timestamp_ms: float
    velocity_px_per_ms: float
    confirmations: int = 1
    missing_frames: int = 0
    scheduled: bool = False


class TranquilityDetector(Detector):
    """Four-lane circle tracker for Tranquility's rhythm minigame."""

    def __init__(self, profile: dict):
        self.profile = profile
        self.standard = StandardDetector(profile)
        self.tracks: dict[str, list[_NoteTrack]] = {
            key: [] for key in ("D", "F", "J", "K")
        }
        self.pending_taps: list[tuple[float, str, int]] = []
        self.last_scheduled_due_ms: dict[str, float] = {}
        self.next_track_identifier = 1
        self.rhythm_active = False

    def reset(self) -> None:
        self.standard.reset()
        self._reset_rhythm()

    def _reset_rhythm(self) -> None:
        for tracks in self.tracks.values():
            tracks.clear()
        self.pending_taps.clear()
        self.last_scheduled_due_ms.clear()
        self.next_track_identifier = 1
        self.rhythm_active = False

    def detect(self, packet: FramePacket) -> DetectionSnapshot:
        frame = packet.frame_bgr
        height, width = frame.shape[:2]
        config = self.profile["detection"]["tranquility"]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        visible_lanes = self._visible_receptor_count(hsv, config)
        required_lanes = int(config.get("signature_minimum_lane_count", 3))
        if visible_lanes < required_lanes:
            if self.rhythm_active:
                self._reset_rhythm()
            snapshot = self.standard.detect(packet)
            snapshot.extra["tranquility_visible_receptors"] = visible_lanes
            return snapshot

        self.rhythm_active = True
        roi = NormalizedRect(
            *self.profile["detection"]["minigame_roi"]
        ).pixels(width, height)
        snapshot = DetectionSnapshot(
            timestamp_ms=packet.timestamp_ms,
            source_name=packet.source_name,
            frame_width=width,
            frame_height=height,
            minigame_roi=roi,
            minigame_visible=True,
            detector_state="RHYTHM_LOCKED",
            extra={"tranquility_visible_receptors": visible_lanes},
        )

        detections_by_lane: dict[str, list[float]] = {}
        for lane in config["lanes"]:
            key = str(lane["key"]).upper()
            detections = self._find_lane_notes(hsv, lane, config)
            detections_by_lane[key] = detections
            self._update_lane_tracks(
                key,
                detections,
                packet.timestamp_ms,
                height,
                config,
            )

        lookahead_ms = float(config.get("tap_dispatch_lookahead_ms", 3.0))
        due_taps: list[str] = []
        remaining: list[tuple[float, str, int]] = []
        for due_ms, key, identifier in sorted(self.pending_taps):
            if due_ms <= packet.timestamp_ms + lookahead_ms:
                due_taps.append(key)
            else:
                remaining.append((due_ms, key, identifier))
        self.pending_taps = remaining
        snapshot.extra["rhythm_taps"] = due_taps
        snapshot.extra["tranquility_pending_taps"] = len(self.pending_taps)
        snapshot.extra["tranquility_note_detections"] = sum(
            len(items) for items in detections_by_lane.values()
        )
        snapshot.extra["tranquility_tracks"] = sum(
            len(items) for items in self.tracks.values()
        )
        snapshot.extra["tranquility_lane_notes"] = ";".join(
            f"{key}:{','.join(f'{value:.1f}' for value in values)}"
            for key, values in detections_by_lane.items()
            if values
        )
        return snapshot

    def _visible_receptor_count(self, hsv: np.ndarray, config: dict) -> int:
        height, width = hsv.shape[:2]
        hit_y = round(height * float(config.get("hit_y_normalized", 0.768)))
        half_width = max(20, round(width * 0.037))
        half_height = max(20, round(height * 0.065))
        minimum_pixels = max(
            80,
            round(
                width
                * height
                * float(config.get("signature_minimum_color_fraction", 0.0007))
            ),
        )
        visible = 0
        for lane in config["lanes"]:
            center_x = round(width * float(lane["x_normalized"]))
            crop = hsv[
                max(0, hit_y - half_height) : min(height, hit_y + half_height + 1),
                max(0, center_x - half_width) : min(width, center_x + half_width + 1),
            ]
            if crop.size == 0:
                continue
            mask = self._lane_color_mask(crop, lane, config)
            if int(np.count_nonzero(mask)) >= minimum_pixels:
                visible += 1
        return visible

    def _find_lane_notes(
        self,
        hsv: np.ndarray,
        lane: dict,
        config: dict,
    ) -> list[float]:
        height, width = hsv.shape[:2]
        center_x = round(width * float(lane["x_normalized"]))
        lane_half_width = max(
            35,
            round(width * float(config.get("note_lane_half_width_normalized", 0.038))),
        )
        top = round(height * float(config.get("note_scan_top_normalized", 0.145)))
        bottom = round(height * float(config.get("note_scan_bottom_normalized", 0.725)))
        left = max(0, center_x - lane_half_width)
        right = min(width, center_x + lane_half_width + 1)
        crop = hsv[max(0, top) : min(height, bottom), left:right]
        if crop.size == 0:
            return []
        mask = self._lane_color_mask(crop, lane, config).astype(np.uint8) * 255
        scale = min(width / 1920.0, height / 1080.0)
        # A thin colored rail artifact sits at the lower edge of every lane.
        # Remove it and skip circle search entirely when no actual note color
        # remains. Searching at half scale preserves the circular silhouette
        # while keeping the live detector comfortably real-time.
        artifact_rows = max(6, round(12 * scale))
        mask[-artifact_rows:, :] = 0
        if np.count_nonzero(mask) < max(80, round(150 * scale * scale)):
            return []
        search_scale = float(config.get("note_search_scale", 0.5))
        search = cv2.resize(
            mask,
            None,
            fx=search_scale,
            fy=search_scale,
            interpolation=cv2.INTER_AREA,
        )
        search = cv2.GaussianBlur(search, (5, 5), 1)
        circles = cv2.HoughCircles(
            search,
            cv2.HOUGH_GRADIENT,
            dp=float(config.get("note_hough_dp", 1.2)),
            minDist=max(10, round(35 * scale * search_scale)),
            param1=float(config.get("note_hough_edge_threshold", 60)),
            param2=float(config.get("note_hough_score_threshold", 7)),
            minRadius=max(5, round(16 * scale * search_scale)),
            maxRadius=max(10, round(48 * scale * search_scale)),
        )
        if circles is None:
            return []
        local_center_x = center_x - left
        x_tolerance = width * float(config.get("note_center_x_tolerance_normalized", 0.018))
        result = [
            top + float(circle_y) / search_scale
            for circle_x, circle_y, _ in circles[0]
            if abs(float(circle_x) / search_scale - local_center_x) <= x_tolerance
        ]
        return sorted(result)

    @staticmethod
    def _lane_color_mask(hsv: np.ndarray, lane: dict, config: dict) -> np.ndarray:
        hue = float(lane["hue"])
        tolerance = float(lane.get("hue_tolerance", 12))
        delta = np.abs(hsv[:, :, 0].astype(np.float32) - hue)
        delta = np.minimum(delta, 180.0 - delta)
        return (
            (delta <= tolerance)
            & (hsv[:, :, 1] >= int(config.get("note_saturation_minimum", 65)))
            & (hsv[:, :, 2] >= int(config.get("note_value_minimum", 150)))
        )

    def _update_lane_tracks(
        self,
        key: str,
        detections: list[float],
        timestamp_ms: float,
        frame_height: int,
        config: dict,
    ) -> None:
        tracks = self.tracks[key]
        unmatched = set(range(len(detections)))
        maximum_missing = int(config.get("note_track_maximum_missing_frames", 7))
        expected_velocity = frame_height * float(
            config.get("note_expected_speed_height_per_ms", 0.00068)
        )

        for track in sorted(tracks, key=lambda item: item.y, reverse=True):
            elapsed = max(1.0, timestamp_ms - track.timestamp_ms)
            predicted_y = track.y + track.velocity_px_per_ms * elapsed
            maximum_distance = max(
                frame_height * float(config.get("note_match_distance_normalized", 0.035)),
                expected_velocity * elapsed + 12.0,
            )
            candidate = min(
                unmatched,
                key=lambda index: abs(detections[index] - predicted_y),
                default=None,
            )
            if (
                candidate is None
                or abs(detections[candidate] - predicted_y) > maximum_distance
            ):
                track.missing_frames += 1
                continue
            new_y = detections[candidate]
            unmatched.remove(candidate)
            measured_velocity = (new_y - track.y) / elapsed
            minimum_velocity = frame_height * float(
                config.get("note_minimum_speed_height_per_ms", 0.00035)
            )
            maximum_velocity = frame_height * float(
                config.get("note_maximum_speed_height_per_ms", 0.00115)
            )
            if minimum_velocity <= measured_velocity <= maximum_velocity:
                smoothing = float(config.get("note_velocity_smoothing", 0.35))
                track.velocity_px_per_ms += smoothing * (
                    measured_velocity - track.velocity_px_per_ms
                )
                track.confirmations += 1
            track.y = new_y
            track.timestamp_ms = timestamp_ms
            track.missing_frames = 0

        for index in unmatched:
            tracks.append(
                _NoteTrack(
                    identifier=self.next_track_identifier,
                    y=detections[index],
                    timestamp_ms=timestamp_ms,
                    velocity_px_per_ms=expected_velocity,
                )
            )
            self.next_track_identifier += 1

        hit_y = frame_height * float(config.get("hit_y_normalized", 0.768))
        trigger_y = frame_height * float(config.get("note_trigger_y_normalized", 0.61))
        confirmation_frames = int(config.get("note_confirmation_frames", 3))
        minimum_velocity = frame_height * float(
            config.get("note_minimum_speed_height_per_ms", 0.00035)
        )
        maximum_velocity = frame_height * float(
            config.get("note_maximum_speed_height_per_ms", 0.00115)
        )
        lead_ms = float(config.get("tap_input_lead_ms", 10.0))
        same_lane_interval_ms = float(config.get("minimum_same_lane_tap_interval_ms", 100))
        for track in tracks:
            if (
                track.scheduled
                or track.confirmations < confirmation_frames
                or track.y < trigger_y
                or not minimum_velocity <= track.velocity_px_per_ms <= maximum_velocity
            ):
                continue
            arrival_ms = timestamp_ms + max(
                0.0,
                (hit_y - track.y) / max(0.001, track.velocity_px_per_ms),
            )
            due_ms = arrival_ms - lead_ms
            previous_due = self.last_scheduled_due_ms.get(key)
            if previous_due is not None and due_ms - previous_due < same_lane_interval_ms:
                track.scheduled = True
                continue
            self.pending_taps.append((due_ms, key, track.identifier))
            self.last_scheduled_due_ms[key] = due_ms
            track.scheduled = True

        self.tracks[key] = [
            track
            for track in tracks
            if track.missing_frames <= maximum_missing
            and track.y <= hit_y + frame_height * 0.06
        ]
