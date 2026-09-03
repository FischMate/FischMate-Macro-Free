from __future__ import annotations

from collections import deque
from statistics import median
from typing import Iterable

import cv2
import numpy as np

from app.core.models import DetectionSnapshot, FramePacket, NormalizedRect, PixelRect
from app.detection.base import Detector


def _ahk_bgr(value: str) -> np.ndarray:
    number = int(value, 16)
    return np.array([(number >> 16) & 0xFF, (number >> 8) & 0xFF, number & 0xFF], dtype=np.int16)


def _color_mask(image: np.ndarray, colors: Iterable[str], tolerance: int) -> np.ndarray:
    pixels = image.astype(np.int16)
    combined = np.zeros(image.shape[:2], dtype=np.uint8)
    for color in colors:
        target = _ahk_bgr(color)
        match = np.max(np.abs(pixels - target), axis=2) <= tolerance
        combined[match] = 255
    return combined


def _relationship_mask(image: np.ndarray, rules: Iterable[dict]) -> np.ndarray:
    """Match translucent colors by channel relationships configured per rod."""
    blue, green, red = (channel.astype(np.int16) for channel in cv2.split(image))
    combined = np.zeros(image.shape[:2], dtype=np.uint8)
    for rule in rules:
        match = (
            (blue >= int(rule.get("blue_min", 0)))
            & (blue <= int(rule.get("blue_max", 255)))
            & (green >= int(rule.get("green_min", 0)))
            & (green <= int(rule.get("green_max", 255)))
            & (red >= int(rule.get("red_min", 0)))
            & (red <= int(rule.get("red_max", 255)))
            & ((green - blue) >= int(rule.get("green_minus_blue_min", -255)))
            & ((green - blue) <= int(rule.get("green_minus_blue_max", 255)))
            & ((blue - red) >= int(rule.get("blue_minus_red_min", -255)))
            & ((red - blue) >= int(rule.get("red_minus_blue_min", -255)))
            & ((red - blue) <= int(rule.get("red_minus_blue_max", 255)))
        )
        combined[match] = 255
    return combined


def _components(mask: np.ndarray, offset_x: int, offset_y: int) -> list[tuple[PixelRect, int]]:
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    result: list[tuple[PixelRect, int]] = []
    for index in range(1, count):
        x, y, width, height, area = stats[index]
        result.append((PixelRect(offset_x + x, offset_y + y, offset_x + x + width, offset_y + y + height), int(area)))
    return result


def _merge_split_bar_components(
    components: list[tuple[PixelRect, int]],
) -> list[tuple[PixelRect, int]]:
    """Add bar candidates formed by two halves split by the vertical stick."""
    merged = list(components)
    ordered = sorted(components, key=lambda item: item[0].left)
    for index, (left_rect, left_area) in enumerate(ordered):
        for right_rect, right_area in ordered[index + 1 :]:
            gap = right_rect.left - left_rect.right
            if gap < 0 or gap > max(24, round(max(left_rect.height, right_rect.height) * 0.7)):
                continue
            vertical_overlap = max(
                0,
                min(left_rect.bottom, right_rect.bottom)
                - max(left_rect.top, right_rect.top),
            )
            if vertical_overlap < min(left_rect.height, right_rect.height) * 0.75:
                continue
            if min(left_rect.width, right_rect.width) < max(left_rect.width, right_rect.width) * 0.45:
                continue
            merged.append(
                (
                    PixelRect(
                        left_rect.left,
                        min(left_rect.top, right_rect.top),
                        right_rect.right,
                        max(left_rect.bottom, right_rect.bottom),
                    ),
                    left_area + right_area,
                )
            )
    return merged


class RuinousLegacyDetector(Detector):
    """Initial color/geometry detector shared by replay and live capture.

    This is deliberately a baseline detector, not the final tracker. It emits
    independent bar and stick facts and never decides mouse input.
    """

    def __init__(self, profile: dict):
        self.profile = profile
        self.previous_bar: PixelRect | None = None
        self.previous_stick: PixelRect | None = None
        self.previous_stick_timestamp_ms: float | None = None
        self.previous_rail: PixelRect | None = None
        self.clean_bar_widths: deque[int] = deque(maxlen=9)

    def reset(self) -> None:
        self.previous_bar = None
        self.previous_stick = None
        self.previous_stick_timestamp_ms = None
        self.previous_rail = None
        self.clean_bar_widths.clear()

    @property
    def learned_bar_width(self) -> int | None:
        if len(self.clean_bar_widths) < 3:
            return None
        return round(median(self.clean_bar_widths))

    def _restore_partial_bar(self, rect: PixelRect) -> tuple[PixelRect, str]:
        """Restore the full body when a color-state transition hides one side.

        White and green are both legitimate configured bar colors. During their
        transition, only one contiguous portion can pass the strict color mask.
        The continuously observed physical edge identifies which side survived;
        the clean width learned in this minigame supplies the missing extent.
        """
        learned_width = self.learned_bar_width
        previous = self.previous_bar
        if learned_width is None or previous is None:
            return rect, "raw"
        expected_width = learned_width
        width_config = self.profile["detection"].get("bar_width", {})
        maximum_shrink_ratio = width_config.get("maximum_shrink_per_frame_ratio")
        dynamic_continuity = (
            maximum_shrink_ratio is not None
            and previous.width >= learned_width * 1.12
        )
        if dynamic_continuity:
            shrink_ratio = min(0.25, max(0.01, float(maximum_shrink_ratio)))
            maximum_growth_ratio = width_config.get("maximum_growth_per_frame_ratio")
            retrigger_growth_ratio = float(
                width_config.get("retrigger_growth_ratio", 1.35)
            )
            if maximum_growth_ratio is not None:
                growth_ratio = min(0.25, max(0.0, float(maximum_growth_ratio)))
                growth_ceiling = round(previous.width * (1.0 + growth_ratio))
                if (
                    rect.width > growth_ceiling
                    and rect.width < previous.width * retrigger_growth_ratio
                ):
                    left_motion = abs(rect.left - previous.left)
                    right_motion = abs(rect.right - previous.right)
                    if right_motion < left_motion:
                        return (
                            PixelRect(
                                rect.right - growth_ceiling,
                                rect.top,
                                rect.right,
                                rect.bottom,
                            ),
                            "dynamic_right_edge_limited",
                        )
                    return (
                        PixelRect(
                            rect.left,
                            rect.top,
                            rect.left + growth_ceiling,
                            rect.bottom,
                        ),
                        "dynamic_left_edge_limited",
                    )
            continuity_floor = round(previous.width * (1.0 - shrink_ratio))
            if rect.width >= continuity_floor:
                return rect, "dynamic_width_body"
            if rect.width >= learned_width * 0.30:
                # A truly shrinking effect changes smoothly. A sudden much
                # smaller component is a partially matched color band, so use
                # the most it could plausibly have shrunk in one scan.
                expected_width = max(learned_width, continuity_floor)
        # A state-color fade can expose progressively larger fractions of the
        # same fixed body (the failed live run grew 180 -> 267 for a 272px bar).
        # Keep restoring until the observed body is effectively full width.
        if rect.width >= expected_width * 0.96:
            return rect, "full_color_body"
        if rect.width < learned_width * 0.30:
            return rect, "fragment_too_small"

        left_motion = abs(rect.left - previous.left)
        right_motion = abs(rect.right - previous.right)
        maximum_edge_motion = max(18, round(expected_width * 0.12))
        if right_motion < left_motion and right_motion <= maximum_edge_motion:
            return (
                PixelRect(rect.right - expected_width, rect.top, rect.right, rect.bottom),
                "right_edge_restored",
            )
        if left_motion <= maximum_edge_motion:
            return (
                PixelRect(rect.left, rect.top, rect.left + expected_width, rect.bottom),
                "left_edge_restored",
            )
        return rect, "fragment_unanchored"

    def _slash_repaired_bar_candidates(
        self,
        mask: np.ndarray,
        roi: PixelRect,
        frame_width: int,
    ) -> list[tuple[float, PixelRect, int, float]]:
        """Reconnect a bar body divided by Ruinous Oath's diagonal slash.

        This pass is deliberately history-gated.  It cannot acquire a bar and
        it cannot widen the normal color tolerance; it only joins already
        matching bar pixels when the result remains continuous with the last
        confirmed Ruinous bar.
        """
        previous = self.previous_bar
        config = self.profile["detection"].get("slash_occlusion", {})
        if previous is None or not bool(config.get("enabled", False)):
            return []

        gap_px = max(
            9,
            round(frame_width * float(config.get("bar_max_gap_normalized", 0.032))),
        )
        if gap_px % 2 == 0:
            gap_px += 1
        repaired_mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            np.ones((3, gap_px), np.uint8),
        )
        maximum_center_step = max(
            24.0,
            frame_width * float(config.get("bar_maximum_center_step_normalized", 0.065)),
        )
        minimum_width_ratio = float(config.get("bar_minimum_previous_width_ratio", 0.30))
        maximum_width_ratio = float(config.get("bar_maximum_previous_width_ratio", 1.30))
        minimum_evidence_coverage = float(
            config.get("bar_minimum_evidence_coverage", 0.42)
        )
        candidates: list[tuple[float, PixelRect, int, float]] = []
        for rect, _ in _components(repaired_mask, roi.left, roi.top):
            if abs(rect.center_x - previous.center_x) > maximum_center_step:
                continue
            if not (
                previous.width * minimum_width_ratio
                <= rect.width
                <= previous.width * maximum_width_ratio
            ):
                continue
            if not (
                previous.height * 0.60
                <= rect.height
                <= max(previous.height * 1.50, previous.height + 8)
            ):
                continue
            vertical_overlap = max(
                0,
                min(rect.bottom, previous.bottom) - max(rect.top, previous.top),
            )
            if vertical_overlap < min(rect.height, previous.height) * 0.55:
                continue
            local = mask[
                rect.top - roi.top : rect.bottom - roi.top,
                rect.left - roi.left : rect.right - roi.left,
            ]
            evidence_area = int(np.count_nonzero(local))
            evidence_coverage = evidence_area / max(1, rect.width * rect.height)
            if evidence_area < 80 or evidence_coverage < minimum_evidence_coverage:
                continue
            aspect = rect.width / max(1, rect.height)
            if aspect < 3.0:
                continue
            score = (
                evidence_area
                + rect.width * 2
                + aspect * 8
                - abs(rect.center_x - previous.center_x) * 0.8
            )
            candidates.append((score, rect, evidence_area, evidence_coverage))
        return candidates

    def _slash_repaired_stick_candidates(
        self,
        mask: np.ndarray,
        stick_region: PixelRect,
        reference_bar: PixelRect,
        frame_height: int,
    ) -> list[tuple[float, PixelRect, int, float]]:
        """Reconnect collinear stick fragments above and below the slash."""
        previous = self.previous_stick
        config = self.profile["detection"].get("slash_occlusion", {})
        if previous is None or not bool(config.get("enabled", False)):
            return []

        gap_px = max(
            7,
            round(frame_height * float(config.get("stick_max_gap_normalized", 0.040))),
        )
        if gap_px % 2 == 0:
            gap_px += 1
        repaired_mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            np.ones((gap_px, 3), np.uint8),
        )
        maximum_center_step = max(
            18.0,
            reference_bar.width
            * float(config.get("stick_maximum_center_step_bar_ratio", 0.22)),
        )
        minimum_evidence_coverage = float(
            config.get("stick_minimum_evidence_coverage", 0.28)
        )
        candidates: list[tuple[float, PixelRect, int, float]] = []
        for rect, _ in _components(
            repaired_mask,
            stick_region.left,
            stick_region.top,
        ):
            if abs(rect.center_x - previous.center_x) > maximum_center_step:
                continue
            vertical_aspect = rect.height / max(1, rect.width)
            if vertical_aspect < 1.8:
                continue
            if rect.width > max(18, reference_bar.width * 0.12):
                continue
            if rect.height < reference_bar.height * 0.55:
                continue
            # The playable stick spans the bar band.  Same-colored vertical UI
            # fragments that live wholly above or below the bar cannot qualify.
            if (
                rect.top > reference_bar.top + reference_bar.height * 0.30
                or rect.bottom < reference_bar.bottom - reference_bar.height * 0.30
            ):
                continue
            local = mask[
                rect.top - stick_region.top : rect.bottom - stick_region.top,
                rect.left - stick_region.left : rect.right - stick_region.left,
            ]
            evidence_area = int(np.count_nonzero(local))
            evidence_coverage = evidence_area / max(1, rect.width * rect.height)
            if evidence_area < 10 or evidence_coverage < minimum_evidence_coverage:
                continue
            score = (
                evidence_area
                + vertical_aspect * 8
                - abs(rect.center_x - previous.center_x) * 0.25
            )
            candidates.append((score, rect, evidence_area, evidence_coverage))
        return candidates

    def _slash_covers_previous_stick(
        self,
        frame: np.ndarray,
        stick_region: PixelRect,
        reference_bar: PixelRect,
        timestamp_ms: float,
    ) -> tuple[bool, float, float]:
        """Recognize the sampled black/red slash over the last trusted stick.

        The slash can cover the entire stick, leaving no same-color fragments
        to reconnect.  In that case the safest usable position is the last
        confirmed stick, but only while actual slash pixels occupy that exact
        rail-local location and only for a short bounded interval.
        """
        previous = self.previous_stick
        previous_timestamp = self.previous_stick_timestamp_ms
        config = self.profile["detection"].get("slash_occlusion", {})
        if (
            previous is None
            or previous_timestamp is None
            or not bool(config.get("enabled", False))
            or timestamp_ms - previous_timestamp
            > float(config.get("stick_carry_max_ms", 650))
        ):
            return False, 0.0, 0.0
        if not (
            reference_bar.left - reference_bar.width * 0.15
            <= previous.center_x
            <= reference_bar.right + reference_bar.width * 0.15
        ):
            return False, 0.0, 0.0

        half_width = max(
            7,
            round(
                frame.shape[1]
                * float(config.get("stick_occlusion_half_width_normalized", 0.007))
            ),
        )
        center_x = round(previous.center_x)
        left = max(stick_region.left, center_x - half_width)
        right = min(stick_region.right, center_x + half_width + 1)
        local = frame[stick_region.top : stick_region.bottom, left:right]
        if local.size == 0:
            return False, 0.0, 0.0
        gray = cv2.cvtColor(local, cv2.COLOR_BGR2GRAY)
        dark_fraction = float(
            np.mean(gray <= int(config.get("stick_occlusion_dark_max", 55)))
        )
        blue = local[:, :, 0].astype(np.int16)
        red = local[:, :, 2].astype(np.int16)
        red_fraction = float(
            np.mean(
                (red - blue)
                >= int(config.get("stick_occlusion_red_minus_blue_min", 30))
            )
        )
        is_slash = (
            dark_fraction
            >= float(config.get("stick_occlusion_minimum_dark_fraction", 0.22))
            and red_fraction
            >= float(config.get("stick_occlusion_minimum_red_fraction", 0.06))
        )
        return is_slash, dark_fraction, red_fraction

    def detect(self, packet: FramePacket) -> DetectionSnapshot:
        frame = packet.frame_bgr
        height, width = frame.shape[:2]
        roi_values = self.profile["detection"]["minigame_roi"]
        roi = NormalizedRect(*map(float, roi_values)).pixels(width, height)
        crop = frame[roi.top : roi.bottom, roi.left : roi.right]
        snapshot = DetectionSnapshot(
            timestamp_ms=packet.timestamp_ms,
            source_name=packet.source_name,
            frame_width=width,
            frame_height=height,
            minigame_roi=roi,
        )
        if crop.size == 0:
            snapshot.rejection_reason = "empty_roi"
            return snapshot

        shake_config = self.profile["detection"].get("shake_button", {})
        shake_roi = NormalizedRect(
            *map(float, shake_config.get("scan_roi", [0.02, 0.08, 0.98, 0.82]))
        ).pixels(width, height)
        shake_crop = frame[
            shake_roi.top : shake_roi.bottom,
            shake_roi.left : shake_roi.right,
        ]
        shake_candidates: list[tuple[float, PixelRect]] = []
        if shake_crop.size:
            blue, green, red = cv2.split(shake_crop)
            blue_red_gap = blue.astype(np.int16) - red.astype(np.int16)
            shake_mask = (
                (blue >= int(shake_config.get("ring_b_min", 170)))
                & (green >= int(shake_config.get("ring_g_min", 100)))
                & (red <= int(shake_config.get("ring_r_max", 170)))
                & (blue_red_gap >= int(shake_config.get("blue_red_gap_min", 45)))
            ).astype(np.uint8) * 255
            shake_mask = cv2.morphologyEx(
                shake_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)
            )
            min_width = width * float(shake_config.get("minimum_width_normalized", 0.08))
            max_width = width * float(shake_config.get("maximum_width_normalized", 0.22))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            for rect, area in _components(shake_mask, shake_roi.left, shake_roi.top):
                aspect = rect.width / max(1, rect.height)
                fill = area / max(1, rect.width * rect.height)
                if not (
                    min_width <= rect.width <= max_width
                    and height * 0.10 <= rect.height <= height * 0.34
                    and 0.72 <= aspect <= 1.28
                    and 0.04 <= fill <= 0.35
                ):
                    continue
                inset_x = max(1, round(rect.width * 0.25))
                inset_y = max(1, round(rect.height * 0.25))
                inner = gray[
                    rect.top + inset_y : rect.bottom - inset_y,
                    rect.left + inset_x : rect.right - inset_x,
                ]
                dark_fraction = float(np.mean(inner <= 95)) if inner.size else 0.0
                if dark_fraction < 0.32:
                    continue
                shake_candidates.append((area + dark_fraction * 5000.0, rect))
        if shake_candidates:
            _, shake_rect = max(shake_candidates, key=lambda item: item[0])
            snapshot.shake_visible = True
            snapshot.extra["shake_left"] = shake_rect.left
            snapshot.extra["shake_top"] = shake_rect.top
            snapshot.extra["shake_right"] = shake_rect.right
            snapshot.extra["shake_bottom"] = shake_rect.bottom
        snapshot.extra["shake_candidate_count"] = len(shake_candidates)

        colors = self.profile["detection"]["colors"]
        tolerance = self.profile["detection"]["tolerance"]
        bar_colors = colors.get("bar_states")
        if not isinstance(bar_colors, list) or not bar_colors:
            bar_colors = [colors["bar"], colors.get("bar_secondary", colors["bar"])]
        bar_mask = _color_mask(
            crop,
            bar_colors,
            int(tolerance["bar"]),
        )
        relationship_rules = self.profile["detection"].get(
            "bar_color_relationships", []
        )
        if relationship_rules:
            bar_mask = cv2.bitwise_or(
                bar_mask,
                _relationship_mask(crop, relationship_rules),
            )
        bar_mask = cv2.morphologyEx(bar_mask, cv2.MORPH_CLOSE, np.ones((3, 9), np.uint8))

        roi_width = max(1, roi.width)
        bar_candidates: list[tuple[float, PixelRect, int, str, float]] = []
        components = _merge_split_bar_components(_components(bar_mask, roi.left, roi.top))
        for rect, area in components:
            width_ratio = rect.width / roi_width
            aspect = rect.width / max(1, rect.height)
            height_ratio = rect.height / max(1, height)
            if (
                0.04 <= width_ratio <= 0.50
                and 0.012 <= height_ratio <= 0.075
                and aspect >= 3.0
                and area >= 80
            ):
                score = area + rect.width * 2 + aspect * 8
                if self.previous_bar is not None:
                    score -= abs(rect.center_x - self.previous_bar.center_x) * 0.8
                coverage = area / max(1, rect.width * rect.height)
                bar_candidates.append((score, rect, area, "normal", coverage))

        repaired_bar_candidates = self._slash_repaired_bar_candidates(
            bar_mask,
            roi,
            width,
        )
        normal_bar_width = max(
            (candidate[1].width for candidate in bar_candidates),
            default=0,
        )
        minimum_repair_gain = max(
            10,
            round((self.previous_bar.width if self.previous_bar is not None else 0) * 0.08),
        )
        for score, rect, area, coverage in repaired_bar_candidates:
            # Normal detection always wins unless the slash-repaired body
            # restores a materially missing span.  This keeps the second pass
            # dormant on ordinary, unobstructed Ruinous frames.
            if normal_bar_width and rect.width < normal_bar_width + minimum_repair_gain:
                continue
            bar_candidates.append((score, rect, area, "slash_reconstructed", coverage))

        if bar_candidates:
            _, raw_bar, bar_area, candidate_source, body_coverage = max(
                bar_candidates,
                key=lambda item: item[0],
            )
            raw_local = crop[
                raw_bar.top - roi.top : raw_bar.bottom - roi.top,
                raw_bar.left - roi.left : raw_bar.right - roi.left,
            ]
            shrinking_colors = colors.get("shrinking_bar_states", [])
            shrinking_area = 0
            shrinking_ratio = 0.0
            if isinstance(shrinking_colors, list) and shrinking_colors and raw_local.size:
                shrinking_mask = _color_mask(
                    raw_local,
                    shrinking_colors,
                    int(tolerance["bar"]),
                )
                shrinking_relationships = self.profile["detection"].get(
                    "shrinking_bar_relationships",
                    [],
                )
                if shrinking_relationships:
                    shrinking_mask = cv2.bitwise_or(
                        shrinking_mask,
                        _relationship_mask(raw_local, shrinking_relationships),
                    )
                shrinking_area = int(np.count_nonzero(shrinking_mask))
                matched_area = max(1, int(bar_area))
                shrinking_ratio = shrinking_area / matched_area
            learned_width = self.learned_bar_width
            red_width_floor = (
                0.0
                if learned_width is None
                else learned_width
                * float(
                    self.profile["detection"]
                    .get("red_live_width", {})
                    .get("minimum_learned_width_ratio", 0.28)
                )
            )
            red_live_config = self.profile["detection"].get("red_live_width", {})
            current_red_live_width = (
                bool(red_live_config.get("enabled", False))
                and shrinking_area >= int(red_live_config.get("minimum_pixels", 60))
                and shrinking_ratio
                >= float(red_live_config.get("minimum_state_ratio", 0.45))
                and raw_bar.width >= red_width_floor
            )
            if current_red_live_width:
                snapshot.bar = raw_bar
                bar_geometry_source = (
                    "slash_reconstructed_red_live_width"
                    if candidate_source == "slash_reconstructed"
                    else "current_red_live_width"
                )
            elif candidate_source == "slash_reconstructed":
                snapshot.bar = raw_bar
                bar_geometry_source = candidate_source
            else:
                snapshot.bar, bar_geometry_source = self._restore_partial_bar(raw_bar)
            snapshot.bar_confidence = min(1.0, 0.35 + bar_area / 500.0)
            snapshot.extra["raw_bar_left"] = raw_bar.left
            snapshot.extra["raw_bar_right"] = raw_bar.right
            snapshot.extra["raw_bar_width"] = raw_bar.width
            snapshot.extra["bar_geometry_source"] = bar_geometry_source
            snapshot.extra["bar_body_coverage"] = round(body_coverage, 4)
            snapshot.extra["bar_current_state"] = (
                "shrinking_red" if current_red_live_width else "normal_or_transition"
            )
            snapshot.extra["bar_shrinking_state_pixels"] = shrinking_area
            snapshot.extra["bar_shrinking_state_ratio"] = round(shrinking_ratio, 4)
            snapshot.extra["bar_slash_reconstructed"] = (
                candidate_source == "slash_reconstructed"
            )

            width_config = self.profile["detection"].get("bar_width", {})
            learning_requires_lock = bool(
                width_config.get("learning_requires_lock", False)
            )
            is_clean_width = (
                learned_width is None
                or abs(raw_bar.width - learned_width) <= max(5, learned_width * 0.035)
            )
            if is_clean_width and not learning_requires_lock:
                self.clean_bar_widths.append(raw_bar.width)

        # V13.4 fallback: the bar body can change to an unconfigured composite
        # color when it loses the stick, but its internal arrow remains stable.
        # Report that marker independently so the controller can keep moving in
        # the correct direction until normal body detection returns.
        arrow_colors = [colors["arrow"]]
        arrow_secondary = colors.get("arrow_secondary")
        if arrow_secondary and arrow_secondary not in arrow_colors:
            arrow_colors.append(arrow_secondary)
        arrow_mask = _color_mask(crop, arrow_colors, int(tolerance["arrow"]))
        arrow_candidates: list[tuple[int, PixelRect]] = []
        for rect, area in _components(arrow_mask, roi.left, roi.top):
            if area >= 3 and rect.width <= width * 0.08 and rect.height <= height * 0.07:
                arrow_candidates.append((area, rect))
        stick_candidates = []
        reference_bar = snapshot.bar or self.previous_bar
        if reference_bar is not None:
            # Establish the rail-local band from the independently found bar.
            # The dark rail may be interrupted by the bright bar, so use dark
            # column occupancy and take its outer evidence rather than requiring
            # one uninterrupted rectangle.
            band_top = max(roi.top, reference_bar.top - round(height * 0.012))
            band_bottom = min(roi.bottom, reference_bar.bottom + round(height * 0.012))
            band = frame[band_top:band_bottom, roi.left:roi.right]
            gray_band = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
            dark_columns = np.mean(gray_band <= 78, axis=0) >= 0.32
            dark_indexes = np.flatnonzero(dark_columns)
            if dark_indexes.size >= 2:
                rail_left = roi.left + int(dark_indexes.min())
                rail_right = roi.left + int(dark_indexes.max()) + 1
                if rail_right - rail_left >= width * 0.28:
                    snapshot.rail = PixelRect(rail_left, band_top, rail_right, band_bottom)
                    self.previous_rail = snapshot.rail
                    snapshot.extra["rail_geometry_source"] = "fresh_dark_edges"

            rail_fallback = self.profile["detection"].get("rail_fallback", {})
            if (
                snapshot.rail is None
                and rail_fallback.get("mode") == "normalized_horizontal"
            ):
                rail_left = round(
                    width * float(rail_fallback.get("left_normalized", roi.left / width))
                )
                rail_right = round(
                    width * float(rail_fallback.get("right_normalized", roi.right / width))
                )
                if rail_right - rail_left >= width * 0.28:
                    snapshot.rail = PixelRect(
                        rail_left,
                        band_top,
                        rail_right,
                        band_bottom,
                    ).clamped(width, height)
                    self.previous_rail = snapshot.rail
                    snapshot.extra["rail_geometry_source"] = "profile_normalized_fallback"

            enabled_mechanics = self.profile.get("mechanics", {}).get("enabled", [])
            if (
                snapshot.rail is None
                and self.previous_rail is not None
                and "temporary_occlusion" in enabled_mechanics
                and self.previous_rail.left <= reference_bar.center_x <= self.previous_rail.right
            ):
                # Some rod effects tint the entire rail for several seconds.
                # The rail is static within one minigame and reset between fish,
                # so its last confirmed geometry remains the strongest fact.
                snapshot.rail = self.previous_rail
                snapshot.extra["rail_geometry_source"] = "confirmed_session_rail"

            stick_region = snapshot.rail or PixelRect(roi.left, band_top, roi.right, band_bottom)
            stick_crop = frame[
                stick_region.top : stick_region.bottom,
                stick_region.left : stick_region.right,
            ]
            stick_tolerance = int(tolerance["stick"])
            stick_height_ratio = 0.55
            effect_stick_tolerance = self.profile["detection"].get(
                "effect_stick_tolerance"
            )
            effect_stick_height_ratio = self.profile["detection"].get(
                "effect_stick_height_ratio"
            )
            learned_width = self.learned_bar_width
            enlarged_effect = (
                learned_width is not None
                and reference_bar.width >= learned_width * 1.15
            )
            if enlarged_effect:
                if effect_stick_tolerance is not None:
                    stick_tolerance = int(effect_stick_tolerance)
                if effect_stick_height_ratio is not None:
                    stick_height_ratio = float(effect_stick_height_ratio)
                snapshot.extra["stick_tolerance_source"] = "enlarged_bar_effect"
            else:
                snapshot.extra["stick_tolerance_source"] = "normal"
            snapshot.extra["stick_tolerance"] = stick_tolerance
            snapshot.extra["stick_height_ratio"] = stick_height_ratio
            stick_detection = self.profile["detection"].get("stick_detection", {})
            if stick_detection.get("mode") == "paired_vertical_edges":
                gray_stick = cv2.cvtColor(stick_crop, cv2.COLOR_BGR2GRAY)
                edge_mask = cv2.Canny(
                    gray_stick,
                    int(stick_detection.get("canny_low", 25)),
                    int(stick_detection.get("canny_high", 80)),
                )
                edge_counts = np.sum(edge_mask > 0, axis=0)
                minimum_gap = int(stick_detection.get("minimum_width_px", 3))
                maximum_gap = int(stick_detection.get("maximum_width_px", 16))
                minimum_edge_count = max(
                    8,
                    round(
                        reference_bar.height
                        * float(stick_detection.get("minimum_edge_height_ratio", 0.45))
                    ),
                )
                for left_index in range(len(edge_counts)):
                    if edge_counts[left_index] < minimum_edge_count:
                        continue
                    for gap in range(minimum_gap, maximum_gap + 1):
                        right_index = left_index + gap
                        if (
                            right_index >= len(edge_counts)
                            or edge_counts[right_index] < minimum_edge_count
                        ):
                            continue
                        pair_edges = edge_mask[:, [left_index, right_index]] > 0
                        rows = np.flatnonzero(np.any(pair_edges, axis=1))
                        if rows.size < minimum_edge_count:
                            continue
                        rect = PixelRect(
                            stick_region.left + left_index,
                            stick_region.top + int(rows.min()),
                            stick_region.left + right_index + 1,
                            stick_region.top + int(rows.max()) + 1,
                        )
                        edge_rejection_margin = float(
                            stick_detection.get("bar_edge_rejection_margin_px", 0)
                        )
                        edge_override_height_ratio = float(
                            stick_detection.get("bar_edge_override_height_ratio", 1.15)
                        )
                        near_bar_outline = min(
                            abs(rect.center_x - reference_bar.left),
                            abs(rect.center_x - reference_bar.right),
                        ) <= edge_rejection_margin
                        if (
                            near_bar_outline
                            and rect.height
                            < reference_bar.height * edge_override_height_ratio
                        ):
                            continue
                        score = float(edge_counts[left_index] + edge_counts[right_index])
                        stick_candidates.append((score, rect, int(rows.size)))
                snapshot.extra["stick_detection_mode"] = "paired_vertical_edges"
            else:
                stick_mask = _color_mask(stick_crop, [colors["stick"]], stick_tolerance)
                stick_mask = cv2.morphologyEx(
                    stick_mask, cv2.MORPH_CLOSE, np.ones((5, 3), np.uint8)
                )
                def add_color_stick_candidates(mask: np.ndarray) -> None:
                    for rect, area in _components(
                        mask,
                        stick_region.left,
                        stick_region.top,
                    ):
                        vertical_aspect = rect.height / max(1, rect.width)
                        if (
                            area >= 10
                            and vertical_aspect >= 1.8
                            and rect.height >= reference_bar.height * stick_height_ratio
                            and rect.width <= max(18, reference_bar.width * 0.12)
                        ):
                            # The stick is allowed to be far outside the bar. Distance
                            # is only a weak tie-breaker, never an overlap assumption.
                            center_penalty = (
                                abs(rect.center_x - reference_bar.center_x) * 0.002
                            )
                            stick_candidates.append(
                                (
                                    area + vertical_aspect * 8 - center_penalty,
                                    rect,
                                    area,
                                )
                            )

                def add_vertical_core_candidate(mask: np.ndarray) -> bool:
                    core_config = self.profile["detection"].get(
                        "stick_vertical_core",
                        {},
                    )
                    if not bool(core_config.get("enabled", False)):
                        return False
                    minimum_column_height = max(
                        10,
                        round(
                            reference_bar.height
                            * float(
                                core_config.get(
                                    "minimum_column_height_ratio",
                                    1.05,
                                )
                            )
                        ),
                    )
                    qualifying = (
                        np.sum(mask > 0, axis=0) >= minimum_column_height
                    ).astype(np.uint8)
                    minimum_width = int(core_config.get("minimum_width_px", 2))
                    maximum_width = int(core_config.get("maximum_width_px", 18))
                    minimum_coverage = float(
                        core_config.get("minimum_fill_coverage", 0.68)
                    )
                    added = False
                    padded = np.pad(qualifying, (1, 1), constant_values=0)
                    boundaries = np.flatnonzero(np.diff(padded.astype(np.int8)))
                    for left_index, right_index in zip(
                        boundaries[0::2],
                        boundaries[1::2],
                    ):
                        run_width = int(right_index - left_index)
                        if not minimum_width <= run_width <= maximum_width:
                            continue
                        core = mask[:, left_index:right_index] > 0
                        rows = np.flatnonzero(np.any(core, axis=1))
                        if rows.size == 0:
                            continue
                        rect_height = int(rows.max() - rows.min() + 1)
                        area = int(np.count_nonzero(core))
                        coverage = area / max(1, run_width * rect_height)
                        if coverage < minimum_coverage:
                            continue
                        rect = PixelRect(
                            stick_region.left + int(left_index),
                            stick_region.top + int(rows.min()),
                            stick_region.left + int(right_index),
                            stick_region.top + int(rows.max()) + 1,
                        )
                        vertical_aspect = rect.height / max(1, rect.width)
                        center_penalty = (
                            abs(rect.center_x - reference_bar.center_x) * 0.002
                        )
                        stick_candidates.append(
                            (
                                area + vertical_aspect * 8 - center_penalty,
                                rect,
                                area,
                            )
                        )
                        added = True
                    return added

                add_color_stick_candidates(stick_mask)
                if stick_candidates:
                    snapshot.extra["stick_color_source"] = "exact_sample"
                elif add_vertical_core_candidate(stick_mask):
                    # Diagonal Ruinous background stripes can share the sampled
                    # stick color and merge sideways into its component. The
                    # real stick still contributes a dense, bar-height-spanning
                    # run of columns; the stripe fragments do not.
                    snapshot.extra["stick_color_source"] = "exact_vertical_core"
                else:
                    # The neutral Ruinous stick is translucent. Its sampled
                    # 0x5B4B43 core is stable on some backgrounds, but the same
                    # dark blue/brown body shifts outside a five-point absolute
                    # tolerance over the initial white bar. Match its channel
                    # relationship only as a fallback, while retaining the same
                    # tall/narrow rail geometry required above. Neutral gray
                    # arrows and the pink striped background fail this mask.
                    stick_relationships = self.profile["detection"].get(
                        "stick_color_relationships",
                        [],
                    )
                    if stick_relationships:
                        relationship_mask = _relationship_mask(
                            stick_crop,
                            stick_relationships,
                        )
                        relationship_mask = cv2.morphologyEx(
                            relationship_mask,
                            cv2.MORPH_CLOSE,
                            np.ones((5, 3), np.uint8),
                        )
                        add_color_stick_candidates(relationship_mask)
                        if stick_candidates or add_vertical_core_candidate(
                            relationship_mask
                        ):
                            stick_mask = relationship_mask
                            snapshot.extra["stick_color_source"] = (
                                "sampled_dark_relationship"
                            )
                if not stick_candidates:
                    repaired_sticks = self._slash_repaired_stick_candidates(
                        stick_mask,
                        stick_region,
                        reference_bar,
                        height,
                    )
                    if repaired_sticks:
                        score, rect, area, coverage = max(
                            repaired_sticks,
                            key=lambda item: item[0],
                        )
                        stick_candidates.append((score, rect, area))
                        snapshot.extra["stick_slash_reconstructed"] = True
                        snapshot.extra["stick_body_coverage"] = round(coverage, 4)
                if not stick_candidates and snapshot.bar is not None:
                    covered, dark_fraction, red_fraction = (
                        self._slash_covers_previous_stick(
                            frame,
                            stick_region,
                            reference_bar,
                            packet.timestamp_ms,
                        )
                    )
                    snapshot.extra["stick_slash_dark_fraction"] = round(
                        dark_fraction,
                        4,
                    )
                    snapshot.extra["stick_slash_red_fraction"] = round(
                        red_fraction,
                        4,
                    )
                    if covered and self.previous_stick is not None:
                        carried_area = max(
                            10,
                            round(
                                self.previous_stick.width
                                * self.previous_stick.height
                                * 0.45
                            ),
                        )
                        stick_candidates.append(
                            (float(carried_area), self.previous_stick, carried_area)
                        )
                        snapshot.extra["stick_slash_carried"] = True
            if stick_candidates:
                _, snapshot.stick, stick_area = max(stick_candidates, key=lambda item: item[0])
                snapshot.stick_confidence = min(1.0, 0.35 + stick_area / 100.0)
                snapshot.extra.setdefault("stick_color_source", "geometry")

        filtered_arrow_candidates = arrow_candidates
        arrow_filter = self.profile["detection"].get("arrow_fallback_filter", {})
        if bool(arrow_filter.get("enabled", False)):
            filtered_arrow_candidates = []
            arrow_reference_bar = snapshot.bar or self.previous_bar
            arrow_rail = snapshot.rail or self.previous_rail
            if arrow_reference_bar is not None and arrow_rail is not None:
                horizontal_margin = max(
                    float(arrow_filter.get("minimum_horizontal_margin_px", 48)),
                    arrow_reference_bar.width
                    * float(arrow_filter.get("bar_width_margin_ratio", 1.25)),
                )
                vertical_margin = max(
                    4.0,
                    arrow_reference_bar.height
                    * float(arrow_filter.get("bar_height_margin_ratio", 0.45)),
                )
                for area, rect in arrow_candidates:
                    center_y = (rect.top + rect.bottom) / 2.0
                    if not arrow_rail.left <= rect.center_x <= arrow_rail.right:
                        continue
                    if not (
                        arrow_reference_bar.top - vertical_margin
                        <= center_y
                        <= arrow_reference_bar.bottom + vertical_margin
                    ):
                        continue
                    if not (
                        arrow_reference_bar.left - horizontal_margin
                        <= rect.center_x
                        <= arrow_reference_bar.right + horizontal_margin
                    ):
                        continue
                    filtered_arrow_candidates.append((area, rect))
        if filtered_arrow_candidates:
            arrow_area, arrow_rect = max(
                filtered_arrow_candidates,
                key=lambda item: item[0],
            )
            snapshot.extra["bar_fallback_arrow_left"] = arrow_rect.left
            snapshot.extra["bar_fallback_arrow_right"] = arrow_rect.right
            snapshot.extra["bar_fallback_arrow_x"] = arrow_rect.center_x
            snapshot.extra["bar_fallback_arrow_area"] = arrow_area
        snapshot.extra["bar_fallback_arrow_filtered_count"] = len(
            filtered_arrow_candidates
        )
        snapshot.extra["bar_fallback_arrow_rejected_count"] = (
            len(arrow_candidates) - len(filtered_arrow_candidates)
        )

        # Lifecycle changes require the complete rail/bar/stick relationship.
        # A bar-like HUD rectangle without the stick is diagnostic PARTIAL only.
        snapshot.minigame_visible = (
            snapshot.bar is not None
            and snapshot.rail is not None
            and snapshot.stick is not None
        )
        width_config = self.profile["detection"].get("bar_width", {})
        if (
            snapshot.minigame_visible
            and bool(width_config.get("learning_requires_lock", False))
            and snapshot.extra.get("raw_bar_width") is not None
        ):
            raw_width = int(snapshot.extra["raw_bar_width"])
            learned_width = self.learned_bar_width
            is_clean_width = (
                learned_width is None
                or abs(raw_width - learned_width) <= max(5, learned_width * 0.035)
            )
            if is_clean_width:
                self.clean_bar_widths.append(raw_width)
        if snapshot.bar is not None:
            # Preserve bar identity through a temporary stick miss and accept
            # large legitimate movement across the rail.
            self.previous_bar = snapshot.bar
        if (
            snapshot.minigame_visible
            and snapshot.stick is not None
            and not snapshot.extra.get("stick_slash_carried", False)
        ):
            self.previous_stick = snapshot.stick
            self.previous_stick_timestamp_ms = packet.timestamp_ms
        snapshot.detector_state = (
            "LOCKED"
            if snapshot.minigame_visible
            else "PARTIAL"
            if snapshot.bar is not None or snapshot.stick is not None
            else "SEARCHING"
        )
        if snapshot.bar is None and snapshot.stick is None:
            snapshot.rejection_reason = "no_standard_components"
        elif snapshot.bar is None:
            snapshot.rejection_reason = "bar_missing"
        elif snapshot.stick is None:
            snapshot.rejection_reason = "stick_missing"
        snapshot.extra["bar_candidate_count"] = len(bar_candidates)
        snapshot.extra["bar_fallback_arrow_count"] = len(arrow_candidates)
        snapshot.extra["stick_candidate_count"] = len(stick_candidates)
        snapshot.extra.setdefault("bar_slash_reconstructed", False)
        snapshot.extra.setdefault("stick_slash_reconstructed", False)
        snapshot.extra.setdefault("stick_slash_carried", False)
        snapshot.extra.setdefault("stick_color_source", "none")
        snapshot.extra["learned_bar_width"] = self.learned_bar_width
        return snapshot
