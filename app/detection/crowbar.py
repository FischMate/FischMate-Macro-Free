from __future__ import annotations

from copy import deepcopy

import cv2
import numpy as np

from app.core.models import FramePacket, NormalizedRect, PixelRect
from app.detection.special_standard import _relationship_mask
from app.detection.special_standard import SpecialStandardDetector


class CrowbarDetector(SpecialStandardDetector):
    """Crowbar-only grayscale bar and outlined-stick detector.

    Crowbar's playable body is a persistent horizontal grayscale gradient, so
    the ordinary white/green palette cannot identify it.  Keep the grayscale
    relationship and edge-stick settings inside this detector instead of
    exposing them as shared standard-detector/controller feature flags.
    """

    def __init__(self, profile: dict):
        detector_profile = deepcopy(profile)
        detection = detector_profile["detection"]
        crowbar = detection.get("crowbar", {})

        detection["bar_color_relationships"] = [
            {
                "name": "crowbar_neutral_gray_gradient",
                "blue_min": int(crowbar.get("bar_luma_min", 28)),
                "blue_max": int(crowbar.get("bar_luma_max", 125)),
                "green_min": int(crowbar.get("bar_luma_min", 28)),
                "green_max": int(crowbar.get("bar_luma_max", 125)),
                "red_min": int(crowbar.get("bar_luma_min", 28)),
                "red_max": int(crowbar.get("bar_luma_max", 125)),
                "green_minus_blue_min": -int(crowbar.get("maximum_channel_delta", 5)),
                "green_minus_blue_max": int(crowbar.get("maximum_channel_delta", 5)),
                "green_minus_red_min": -int(crowbar.get("maximum_channel_delta", 5)),
                "green_minus_red_max": int(crowbar.get("maximum_channel_delta", 5)),
                "blue_minus_red_min": -int(crowbar.get("maximum_channel_delta", 5)),
                "red_minus_blue_min": -int(crowbar.get("maximum_channel_delta", 5)),
            }
        ]
        # Fixed-color matching is intentionally disabled in the detector copy.
        # A single gray sample would also occur in the water/background; the
        # neutral-gradient relationship is only admitted inside a validated
        # outlined playable body below.
        detection["colors"]["bar"] = "0xFF00FF"
        detection["colors"]["bar_secondary"] = "0xFF00FF"
        detection["tolerance"]["bar"] = 0
        detection["bar_candidate_center_y"] = {
            "minimum_normalized": float(crowbar.get("bar_center_y_min", 0.80)),
            "maximum_normalized": float(crowbar.get("bar_center_y_max", 0.90)),
        }
        detection["bar_candidate_maximum_vertical_shift_ratio"] = float(
            crowbar.get("maximum_vertical_shift_ratio", 0.025)
        )
        detection["stick_detection"] = {
            "mode": "paired_vertical_edges",
            "canny_low": int(crowbar.get("stick_canny_low", 18)),
            "canny_high": int(crowbar.get("stick_canny_high", 58)),
            "minimum_width_px": int(crowbar.get("stick_minimum_width_px", 5)),
            "maximum_width_px": int(crowbar.get("stick_maximum_width_px", 15)),
            "minimum_edge_height_ratio": float(
                crowbar.get("stick_minimum_edge_height_ratio", 0.65)
            ),
            "bar_edge_rejection_margin_px": int(
                crowbar.get("bar_edge_rejection_margin_px", 10)
            ),
            "bar_edge_override_height_ratio": float(
                crowbar.get("bar_edge_override_height_ratio", 1.18)
            ),
        }
        detection["bar_fallback_arrow"] = {"enabled": False}
        detection["bar_width"] = {
            **detection.get("bar_width", {}),
            "learning_requires_lock": True,
        }

        super().__init__(detector_profile)
        self.previous_crowbar_stick: PixelRect | None = None
        self.previous_crowbar_stick_timestamp_ms: float | None = None

    def reset(self) -> None:
        super().reset()
        self.previous_crowbar_stick = None
        self.previous_crowbar_stick_timestamp_ms = None

    def _find_outlined_bar(self, frame: np.ndarray) -> PixelRect | None:
        height, width = frame.shape[:2]
        roi = NormalizedRect(
            *map(float, self.profile["detection"]["minigame_roi"])
        ).pixels(width, height)
        crop = frame[roi.top : roi.bottom, roi.left : roi.right]
        if crop.size == 0:
            return None

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 18, 65)
        edges = cv2.morphologyEx(
            edges,
            cv2.MORPH_CLOSE,
            np.ones((3, 7), np.uint8),
        )
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        crowbar = self.profile["detection"].get("crowbar", {})
        center_min = float(crowbar.get("bar_center_y_min", 0.80))
        center_max = float(crowbar.get("bar_center_y_max", 0.90))
        minimum_width_ratio = float(crowbar.get("bar_minimum_width_ratio", 0.20))
        maximum_width_ratio = float(crowbar.get("bar_maximum_width_ratio", 0.42))
        body_segments: list[PixelRect] = []
        for contour in contours:
            x, y, candidate_width, candidate_height = cv2.boundingRect(contour)
            rect = PixelRect(
                roi.left + x,
                roi.top + y,
                roi.left + x + candidate_width,
                roi.top + y + candidate_height,
            )
            height_ratio = rect.height / max(1, height)
            center_y = (rect.top + rect.bottom) / 2.0 / max(1, height)
            if not (
                rect.width >= 8
                and 0.025 <= height_ratio <= 0.075
                and center_min <= center_y <= center_max
            ):
                continue
            inset_x = max(1, min(3, rect.width // 5))
            inset_y = max(2, round(rect.height * 0.12))
            inner = frame[
                rect.top + inset_y : rect.bottom - inset_y,
                rect.left + inset_x : rect.right - inset_x,
            ]
            if inner.size == 0:
                continue
            inner_i16 = inner.astype(np.int16)
            channel_spread = inner_i16.max(axis=2) - inner_i16.min(axis=2)
            inner_luma = cv2.cvtColor(inner, cv2.COLOR_BGR2GRAY)
            neutral_body = (
                (channel_spread <= int(crowbar.get("maximum_channel_delta", 5)))
                & (inner_luma >= int(crowbar.get("bar_luma_min", 28)))
                & (inner_luma <= int(crowbar.get("bar_body_luma_max", 155)))
            )
            if float(np.mean(neutral_body)) >= 0.45:
                body_segments.append(rect)

        # The tall stick can cut the black outline into two contours.  Merge
        # only gray-body segments separated by a stick-sized gap; dark rail
        # segments never enter this set and therefore cannot enlarge the bar.
        candidate_rects = list(body_segments)
        ordered_segments = sorted(body_segments, key=lambda item: item.left)
        for index, left in enumerate(ordered_segments):
            for right in ordered_segments[index + 1 :]:
                gap = right.left - left.right
                if gap < 0 or gap > 20:
                    continue
                overlap = max(
                    0,
                    min(left.bottom, right.bottom) - max(left.top, right.top),
                )
                if overlap < min(left.height, right.height) * 0.70:
                    continue
                candidate_rects.append(
                    PixelRect(
                        left.left,
                        min(left.top, right.top),
                        right.right,
                        max(left.bottom, right.bottom),
                    )
                )

        candidates: list[tuple[float, PixelRect]] = []
        for rect in candidate_rects:
            width_ratio = rect.width / max(1, roi.width)
            height_ratio = rect.height / max(1, height)
            center_y = (rect.top + rect.bottom) / 2.0 / max(1, height)
            aspect = rect.width / max(1, rect.height)
            if not (
                minimum_width_ratio <= width_ratio <= maximum_width_ratio
                and 0.025 <= height_ratio <= 0.075
                and aspect >= 4.0
                and center_min <= center_y <= center_max
            ):
                continue

            inset_x = max(3, round(rect.width * 0.025))
            inset_y = max(3, round(rect.height * 0.15))
            inner = frame[
                rect.top + inset_y : rect.bottom - inset_y,
                rect.left + inset_x : rect.right - inset_x,
            ]
            if inner.size == 0:
                continue
            inner_i16 = inner.astype(np.int16)
            channel_spread = inner_i16.max(axis=2) - inner_i16.min(axis=2)
            inner_luma = cv2.cvtColor(inner, cv2.COLOR_BGR2GRAY)
            neutral_body = (
                (channel_spread <= int(crowbar.get("maximum_channel_delta", 5)))
                & (inner_luma >= int(crowbar.get("bar_luma_min", 28)))
                & (inner_luma <= int(crowbar.get("bar_body_luma_max", 155)))
            )
            neutral_coverage = float(np.mean(neutral_body))
            if neutral_coverage < 0.55:
                continue

            column_luma = np.mean(inner_luma, axis=0)
            quarter = max(1, len(column_luma) // 4)
            center_start = max(0, len(column_luma) // 2 - quarter // 2)
            center_end = min(len(column_luma), center_start + quarter)
            center_mean = float(np.mean(column_luma[center_start:center_end]))
            outer_mean = float(
                np.mean(np.concatenate((column_luma[:quarter], column_luma[-quarter:])))
            )
            gradient_gain = center_mean - outer_mean
            if gradient_gain < float(crowbar.get("minimum_center_gradient_gain", 10)):
                continue

            score = (
                rect.width * rect.height
                + neutral_coverage * 5000.0
                + gradient_gain * 100.0
            )
            if self.previous_bar is not None:
                score -= abs(rect.center_x - self.previous_bar.center_x) * 0.35
            candidates.append((score, rect))

        if not candidates:
            return None
        return max(candidates, key=lambda item: item[0])[1]

    @staticmethod
    def _true_runs(columns: np.ndarray) -> list[tuple[int, int]]:
        indexes = np.flatnonzero(columns)
        if indexes.size == 0:
            return []
        runs: list[tuple[int, int]] = []
        start = int(indexes[0])
        previous = start
        for value in indexes[1:]:
            current = int(value)
            if current != previous + 1:
                runs.append((start, previous + 1))
                start = current
            previous = current
        runs.append((start, previous + 1))
        return runs

    def _find_rail_around_bar(
        self,
        frame: np.ndarray,
        bar: PixelRect,
    ) -> PixelRect | None:
        height, width = frame.shape[:2]
        inset = max(2, round(bar.height * 0.16))
        band_top = max(0, bar.top + inset)
        band_bottom = min(height, bar.bottom - inset)
        if band_bottom <= band_top:
            return None
        gray = cv2.cvtColor(frame[band_top:band_bottom], cv2.COLOR_BGR2GRAY)
        dark_columns = np.mean(gray <= 27, axis=0) >= 0.68
        raw_runs = self._true_runs(dark_columns)
        # A detached vertical stick cuts one side of the dark rail into two
        # runs. Bridge only stick-sized gaps; the much wider playable bar gap
        # remains separate and still anchors the two rail sides.
        merged_runs: list[tuple[int, int]] = []
        for run in raw_runs:
            if merged_runs and run[0] - merged_runs[-1][1] <= 20:
                merged_runs[-1] = (merged_runs[-1][0], run[1])
            else:
                merged_runs.append(run)
        runs = [run for run in merged_runs if run[1] - run[0] >= 16]
        maximum_gap = width * 0.06
        left_runs = [
            run
            for run in runs
            if run[1] <= bar.left + 12 and bar.left - run[1] <= maximum_gap
        ]
        right_runs = [
            run
            for run in runs
            if run[0] >= bar.right - 12 and run[0] - bar.right <= maximum_gap
        ]
        if not left_runs or not right_runs:
            return None
        left_run = max(left_runs, key=lambda run: run[1])
        right_run = min(right_runs, key=lambda run: run[0])
        if right_run[1] - left_run[0] < width * 0.28:
            return None
        return PixelRect(left_run[0], band_top, right_run[1], band_bottom)

    def _find_outlined_stick(
        self,
        frame: np.ndarray,
        bar: PixelRect,
        rail: PixelRect,
        timestamp_ms: float,
    ) -> tuple[PixelRect | None, int, str]:
        height, _ = frame.shape[:2]
        crowbar = self.profile["detection"].get("crowbar", {})
        vertical_margin = max(12, round(bar.height * 0.45))
        region = PixelRect(
            rail.left,
            max(0, bar.top - vertical_margin),
            rail.right,
            min(height, bar.bottom + vertical_margin),
        )
        crop = frame[region.top : region.bottom, region.left : region.right]
        if crop.size == 0:
            return None, 0, "empty_stick_region"

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(
            gray,
            int(crowbar.get("stick_canny_low", 18)),
            int(crowbar.get("stick_canny_high", 58)),
        )
        counts = np.sum(edges > 0, axis=0)
        minimum_gap = int(crowbar.get("stick_minimum_width_px", 5))
        maximum_gap = int(crowbar.get("stick_maximum_width_px", 15))
        minimum_overlap = max(
            12,
            round(bar.height * float(crowbar.get("stick_minimum_edge_height_ratio", 0.65))),
        )
        edge_margin = float(crowbar.get("bar_edge_rejection_margin_px", 10))
        edge_override = float(crowbar.get("bar_edge_override_height_ratio", 1.18))
        candidates: list[tuple[float, PixelRect]] = []
        previous = self.previous_crowbar_stick
        previous_timestamp = self.previous_crowbar_stick_timestamp_ms
        elapsed_ms = (
            0.0
            if previous_timestamp is None
            else max(0.0, timestamp_ms - previous_timestamp)
        )
        maximum_jump = float(crowbar.get("stick_continuity_base_jump_px", 80)) + (
            elapsed_ms * float(crowbar.get("stick_continuity_velocity_px_per_ms", 1.8))
        )

        for left_index in range(len(counts)):
            if counts[left_index] < minimum_overlap:
                continue
            for gap in range(minimum_gap, maximum_gap + 1):
                right_index = left_index + gap
                if right_index >= len(counts) or counts[right_index] < minimum_overlap:
                    continue
                shared_rows = np.flatnonzero(
                    (edges[:, left_index] > 0) & (edges[:, right_index] > 0)
                )
                if shared_rows.size < minimum_overlap:
                    continue
                pair_rows = np.flatnonzero(
                    (edges[:, left_index] > 0) | (edges[:, right_index] > 0)
                )
                if pair_rows.size == 0:
                    continue
                rect = PixelRect(
                    region.left + left_index,
                    region.top + int(pair_rows.min()),
                    region.left + right_index + 1,
                    region.top + int(pair_rows.max()) + 1,
                )
                near_bar_edge = min(
                    abs(rect.center_x - bar.left),
                    abs(rect.center_x - bar.right),
                ) <= edge_margin
                if near_bar_edge and rect.height < bar.height * edge_override:
                    continue
                if rect.height < bar.height * 0.95:
                    continue
                jump = 0.0 if previous is None else abs(rect.center_x - previous.center_x)
                if previous is not None and jump > maximum_jump:
                    continue
                score = (
                    float(shared_rows.size) * 5.0
                    + rect.height * 1.5
                    + float(counts[left_index] + counts[right_index])
                    - jump * 0.45
                )
                candidates.append((score, rect))

        if not candidates:
            reason = "continuity_jump_or_no_edge_pair" if previous is not None else "no_edge_pair"
            return None, 0, reason
        _, selected = max(candidates, key=lambda item: item[0])
        return selected, len(candidates), "outlined_edge_pair"

    def detect(self, packet: FramePacket):
        outline = self._find_outlined_bar(packet.frame_bgr)
        if outline is None:
            frame_height, frame_width = packet.frame_bgr.shape[:2]
            roi = NormalizedRect(
                *map(float, self.profile["detection"]["minigame_roi"])
            ).pixels(frame_width, frame_height)
            blanked = packet.frame_bgr.copy()
            blanked[roi.top : roi.bottom, roi.left : roi.right] = (0, 255, 0)
            snapshot = super().detect(
                FramePacket(
                    blanked,
                    packet.timestamp_ms,
                    packet.source_name,
                    packet.sequence,
                    packet.window_rect,
                )
            )
            snapshot.bar = None
            snapshot.stick = None
            snapshot.bar_confidence = 0.0
            snapshot.stick_confidence = 0.0
            snapshot.minigame_visible = False
            snapshot.detector_state = "SEARCHING"
            snapshot.rejection_reason = "crowbar_outline_missing"
            snapshot.extra["crowbar_outline_found"] = False
            return snapshot

        filtered = packet.frame_bgr.copy()
        frame_height, frame_width = filtered.shape[:2]
        roi = NormalizedRect(
            *map(float, self.profile["detection"]["minigame_roi"])
        ).pixels(frame_width, frame_height)
        filtered_roi = filtered[roi.top : roi.bottom, roi.left : roi.right]
        relationship = _relationship_mask(
            filtered_roi,
            self.profile["detection"]["bar_color_relationships"],
        )
        admitted = np.zeros(relationship.shape, dtype=bool)
        admitted[
            outline.top - roi.top : outline.bottom - roi.top,
            outline.left - roi.left : outline.right - roi.left,
        ] = True
        suppress = (relationship > 0) & ~admitted
        # Saturated green is outside the neutral-gray relationship and stays
        # bright in luma, so suppression cannot manufacture false dark rail
        # columns for the downstream geometry pass.
        filtered_roi[suppress] = (0, 255, 0)
        crowbar_rail = self._find_rail_around_bar(packet.frame_bgr, outline)
        if crowbar_rail is not None:
            scan_margin = round(packet.frame_bgr.shape[0] * 0.015)
            scan_top = max(0, outline.top - scan_margin)
            scan_bottom = min(packet.frame_bgr.shape[0], outline.bottom + scan_margin)
            filtered[scan_top:scan_bottom, : crowbar_rail.left] = (0, 255, 0)
            filtered[scan_top:scan_bottom, crowbar_rail.right :] = (0, 255, 0)

        snapshot = super().detect(
            FramePacket(
                filtered,
                packet.timestamp_ms,
                packet.source_name,
                packet.sequence,
                packet.window_rect,
            )
        )
        snapshot.extra["crowbar_outline_found"] = True
        snapshot.extra["crowbar_outline_left"] = outline.left
        snapshot.extra["crowbar_outline_right"] = outline.right
        snapshot.extra["crowbar_outline_top"] = outline.top
        snapshot.extra["crowbar_outline_bottom"] = outline.bottom
        if crowbar_rail is not None:
            snapshot.rail = crowbar_rail
            snapshot.extra["rail_geometry_source"] = "crowbar_adjacent_dark_runs"
            snapshot.extra["crowbar_rail_left"] = crowbar_rail.left
            snapshot.extra["crowbar_rail_right"] = crowbar_rail.right
            crowbar_stick, candidate_count, stick_source = self._find_outlined_stick(
                packet.frame_bgr,
                outline,
                crowbar_rail,
                packet.timestamp_ms,
            )
            snapshot.stick = crowbar_stick
            snapshot.extra["crowbar_stick_candidate_count"] = candidate_count
            snapshot.extra["crowbar_stick_source"] = stick_source
            if crowbar_stick is not None:
                snapshot.stick_confidence = min(1.0, 0.65 + candidate_count * 0.03)
                self.previous_crowbar_stick = crowbar_stick
                self.previous_crowbar_stick_timestamp_ms = packet.timestamp_ms
            else:
                snapshot.stick_confidence = 0.0

        snapshot.minigame_visible = (
            snapshot.bar is not None
            and snapshot.rail is not None
            and snapshot.stick is not None
        )
        snapshot.detector_state = "LOCKED" if snapshot.minigame_visible else "PARTIAL"
        if snapshot.bar is None:
            snapshot.rejection_reason = "bar_missing"
        elif snapshot.rail is None:
            snapshot.rejection_reason = "rail_missing"
        elif snapshot.stick is None:
            snapshot.rejection_reason = "stick_missing"
        else:
            snapshot.rejection_reason = ""
        return snapshot
