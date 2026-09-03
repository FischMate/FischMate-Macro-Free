from __future__ import annotations

from collections import deque
from functools import lru_cache
from statistics import median
from typing import Iterable

import cv2
import numpy as np

from app.core.models import DetectionSnapshot, FramePacket, NormalizedRect, PixelRect
from app.detection.base import Detector
from app.detection.live_bar_width import LiveBarWidthTracker


def _ahk_bgr(value: str) -> np.ndarray:
    number = int(value, 16)
    return np.array([(number >> 16) & 0xFF, (number >> 8) & 0xFF, number & 0xFF], dtype=np.int16)


@lru_cache(maxsize=512)
def _color_bounds(
    value: str,
    tolerance: int,
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    target = _ahk_bgr(value)
    lower = np.maximum(target - tolerance, 0)
    upper = np.minimum(target + tolerance, 255)
    return (
        tuple(int(channel) for channel in lower),
        tuple(int(channel) for channel in upper),
    )


def _color_mask(image: np.ndarray, colors: Iterable[str], tolerance: int) -> np.ndarray:
    """Return the same AHK-style tolerance mask using native OpenCV loops."""
    combined = np.zeros(image.shape[:2], dtype=np.uint8)
    for color in colors:
        lower, upper = _color_bounds(color, tolerance)
        match = cv2.inRange(image, lower, upper)
        cv2.bitwise_or(combined, match, dst=combined)
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
            & ((green - red) >= int(rule.get("green_minus_red_min", -255)))
            & ((green - red) <= int(rule.get("green_minus_red_max", 255)))
            & ((blue - red) >= int(rule.get("blue_minus_red_min", -255)))
            & ((red - blue) >= int(rule.get("red_minus_blue_min", -255)))
            & ((red - green) >= int(rule.get("red_minus_green_min", -255)))
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
    *,
    maximum_gap_px: int | None = None,
    minimum_half_ratio: float = 0.45,
) -> list[tuple[PixelRect, int]]:
    """Add bar candidates formed by two halves split by the vertical stick."""
    merged = list(components)
    ordered = sorted(components, key=lambda item: item[0].left)
    for index, (left_rect, left_area) in enumerate(ordered):
        for right_rect, right_area in ordered[index + 1 :]:
            gap = right_rect.left - left_rect.right
            allowed_gap = (
                max(24, round(max(left_rect.height, right_rect.height) * 0.7))
                if maximum_gap_px is None
                else max(0, int(maximum_gap_px))
            )
            if gap < 0 or gap > allowed_gap:
                continue
            vertical_overlap = max(
                0,
                min(left_rect.bottom, right_rect.bottom)
                - max(left_rect.top, right_rect.top),
            )
            if vertical_overlap < min(left_rect.height, right_rect.height) * 0.75:
                continue
            if min(left_rect.width, right_rect.width) < (
                max(left_rect.width, right_rect.width)
                * max(0.0, float(minimum_half_ratio))
            ):
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


class SpecialStandardDetector(Detector):
    """Initial color/geometry detector shared by replay and live capture.

    This is deliberately a baseline detector, not the final tracker. It emits
    independent bar and stick facts and never decides mouse input.
    """

    def __init__(self, profile: dict):
        self.profile = profile
        self.previous_bar: PixelRect | None = None
        self.previous_rail: PixelRect | None = None
        self.previous_stick: PixelRect | None = None
        self.clean_bar_widths: deque[int] = deque(maxlen=9)
        self.live_bar_width = LiveBarWidthTracker(
            profile.get("detection", {}).get("fish_live_width")
        )

    def reset(self) -> None:
        self.previous_bar = None
        self.previous_rail = None
        self.previous_stick = None
        self.clean_bar_widths.clear()
        self.live_bar_width.reset()

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
        allow_below_learned_width = bool(
            width_config.get("allow_below_learned_width", False)
        )
        dynamic_continuity = (
            maximum_shrink_ratio is not None
            and (
                allow_below_learned_width
                or previous.width >= learned_width * 1.12
            )
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
        if (
            relationship_rules
            and bool(
                self.profile["detection"].get(
                    "bar_color_relationships_require_learned_width", False
                )
            )
            and self.learned_bar_width is None
        ):
            relationship_rules = [
                rule
                for rule in relationship_rules
                if not bool(rule.get("require_learned_width", True))
            ]
        if relationship_rules:
            relationship_mask = np.zeros(crop.shape[:2], dtype=np.uint8)
            for rule in relationship_rules:
                rule_mask = _relationship_mask(crop, [rule])
                vertical_minimum = rule.get("vertical_minimum_normalized")
                vertical_maximum = rule.get("vertical_maximum_normalized")
                if vertical_minimum is not None or vertical_maximum is not None:
                    fixed_top = max(
                        roi.top,
                        round(height * float(vertical_minimum or 0.0)),
                    )
                    fixed_bottom = min(
                        roi.bottom,
                        round(height * float(vertical_maximum or 1.0)),
                    )
                    fixed_band_mask = np.zeros(crop.shape[:2], dtype=np.uint8)
                    fixed_band_mask[
                        max(0, fixed_top - roi.top) : max(
                            0, fixed_bottom - roi.top
                        ),
                        :,
                    ] = 255
                    rule_mask = cv2.bitwise_and(rule_mask, fixed_band_mask)
                previous_band_margin = rule.get(
                    "previous_bar_vertical_margin_ratio"
                )
                if previous_band_margin is not None:
                    vertical_anchor = self.previous_rail or self.previous_bar
                    if vertical_anchor is None:
                        rule_mask.fill(0)
                    else:
                        margin = round(
                            height * max(0.0, float(previous_band_margin))
                        )
                        band_top = max(
                            roi.top,
                            vertical_anchor.top - margin,
                        )
                        band_bottom = min(
                            roi.bottom,
                            vertical_anchor.bottom + margin,
                        )
                        band_mask = np.zeros(crop.shape[:2], dtype=np.uint8)
                        band_mask[
                            max(0, band_top - roi.top) : max(
                                0, band_bottom - roi.top
                            ),
                            :,
                        ] = 255
                        rule_mask = cv2.bitwise_and(rule_mask, band_mask)
                cv2.bitwise_or(relationship_mask, rule_mask, dst=relationship_mask)
            bar_mask = cv2.bitwise_or(bar_mask, relationship_mask)
        bar_mask = cv2.morphologyEx(bar_mask, cv2.MORPH_CLOSE, np.ones((3, 9), np.uint8))

        roi_width = max(1, roi.width)
        bar_candidates = []
        split_config = self.profile["detection"].get("split_bar_components", {})
        components = _merge_split_bar_components(
            _components(bar_mask, roi.left, roi.top),
            maximum_gap_px=split_config.get("maximum_gap_px"),
            minimum_half_ratio=float(split_config.get("minimum_half_ratio", 0.45)),
        )
        candidate_y_config = self.profile["detection"].get("bar_candidate_center_y", {})
        minimum_candidate_center_y = candidate_y_config.get("minimum_normalized")
        maximum_candidate_center_y = candidate_y_config.get("maximum_normalized")
        minimum_candidate_height_ratio = float(
            self.profile["detection"].get(
                "bar_candidate_minimum_height_normalized", 0.012
            )
        )
        minimum_candidate_width_ratio = float(
            self.profile["detection"].get(
                "bar_candidate_minimum_width_normalized", 0.04
            )
        )
        minimum_candidate_aspect = float(
            self.profile["detection"].get("bar_candidate_minimum_aspect", 3.0)
        )
        minimum_candidate_area = int(
            self.profile["detection"].get("bar_candidate_minimum_area", 80)
        )
        maximum_candidate_center_step_ratio = self.profile["detection"].get(
            "bar_candidate_maximum_center_step_normalized"
        )
        candidate_continuity_penalty = float(
            self.profile["detection"].get("bar_candidate_continuity_penalty", 0.8)
        )
        maximum_candidate_height_ratio = float(
            self.profile["detection"].get(
                "bar_candidate_maximum_height_normalized", 0.075
            )
        )
        for rect, area in components:
            width_ratio = rect.width / roi_width
            aspect = rect.width / max(1, rect.height)
            height_ratio = rect.height / max(1, height)
            center_y_ratio = ((rect.top + rect.bottom) / 2.0) / max(1, height)
            outside_candidate_y_band = (
                (
                    minimum_candidate_center_y is not None
                    and center_y_ratio < float(minimum_candidate_center_y)
                )
                or (
                    maximum_candidate_center_y is not None
                    and center_y_ratio > float(maximum_candidate_center_y)
                )
            )
            maximum_vertical_shift_ratio = self.profile["detection"].get(
                "bar_candidate_maximum_vertical_shift_ratio"
            )
            exceeds_vertical_continuity = (
                self.previous_bar is not None
                and self.learned_bar_width is not None
                and maximum_vertical_shift_ratio is not None
                and abs(
                    (rect.top + rect.bottom) / 2.0
                    - (self.previous_bar.top + self.previous_bar.bottom) / 2.0
                )
                > height * float(maximum_vertical_shift_ratio)
            )
            exceeds_horizontal_continuity = (
                self.previous_bar is not None
                and self.learned_bar_width is not None
                and maximum_candidate_center_step_ratio is not None
                and abs(rect.center_x - self.previous_bar.center_x)
                > width * float(maximum_candidate_center_step_ratio)
            )
            if (
                minimum_candidate_width_ratio <= width_ratio <= 0.50
                and minimum_candidate_height_ratio
                <= height_ratio
                <= maximum_candidate_height_ratio
                and aspect >= minimum_candidate_aspect
                and area >= minimum_candidate_area
                and not outside_candidate_y_band
                and not exceeds_vertical_continuity
                and not exceeds_horizontal_continuity
            ):
                score = area + rect.width * 2 + aspect * 8
                if self.previous_bar is not None:
                    score -= (
                        abs(rect.center_x - self.previous_bar.center_x)
                        * candidate_continuity_penalty
                    )
                bar_candidates.append((score, rect, area))

        if bar_candidates:
            _, raw_bar, bar_area = max(bar_candidates, key=lambda item: item[0])
            restored_bar, bar_geometry_source = self._restore_partial_bar(raw_bar)
            live_width = self.live_bar_width.evaluate(
                raw_bar,
                restored_bar,
                bar_geometry_source,
                self.learned_bar_width,
            )
            snapshot.bar = live_width.bar
            bar_geometry_source = live_width.geometry_source
            width_config = self.profile["detection"].get("bar_width", {})
            if (
                bar_geometry_source == "fragment_too_small"
                and bool(width_config.get("reject_fragment_too_small", False))
            ):
                snapshot.bar = None
            snapshot.bar_confidence = min(1.0, 0.35 + bar_area / 500.0)
            snapshot.extra["raw_bar_left"] = raw_bar.left
            snapshot.extra["raw_bar_right"] = raw_bar.right
            snapshot.extra["raw_bar_width"] = raw_bar.width
            snapshot.extra["bar_geometry_source"] = bar_geometry_source
            snapshot.extra["nominal_bar_width"] = live_width.nominal_width
            snapshot.extra["live_bar_width"] = live_width.live_width
            snapshot.extra["live_width_state"] = live_width.state
            snapshot.extra["live_width_enabled"] = self.live_bar_width.enabled
            snapshot.extra["live_width_history_update_allowed"] = (
                live_width.allow_nominal_learning
            )

            learned_width = self.learned_bar_width
            learning_requires_lock = bool(
                width_config.get("learning_requires_lock", False)
            )
            minimum_learning_width_ratio = width_config.get(
                "nominal_learning_minimum_normalized"
            )
            large_enough_for_nominal = (
                minimum_learning_width_ratio is None
                or raw_bar.width
                >= roi_width * float(minimum_learning_width_ratio)
            )
            is_clean_width = (
                large_enough_for_nominal
                and (
                    learned_width is None
                    or abs(raw_bar.width - learned_width)
                    <= max(5, learned_width * 0.035)
                )
            )
            if (
                is_clean_width
                and not learning_requires_lock
                and live_width.allow_nominal_learning
            ):
                self.clean_bar_widths.append(raw_bar.width)

        # V13.4 fallback: the bar body can change to an unconfigured composite
        # color when it loses the stick, but its internal arrow remains stable.
        # Report that marker independently so the controller can keep moving in
        # the correct direction until normal body detection returns.
        arrow_candidates: list[tuple[int, PixelRect]] = []
        arrow_fallback_config = self.profile["detection"].get("bar_fallback_arrow", {})
        if bool(arrow_fallback_config.get("enabled", True)):
            arrow_colors = [colors["arrow"]]
            arrow_secondary = colors.get("arrow_secondary")
            if arrow_secondary and arrow_secondary not in arrow_colors:
                arrow_colors.append(arrow_secondary)
            arrow_mask = _color_mask(crop, arrow_colors, int(tolerance["arrow"]))
            for rect, area in _components(arrow_mask, roi.left, roi.top):
                if area >= 3 and rect.width <= width * 0.08 and rect.height <= height * 0.07:
                    arrow_candidates.append((area, rect))
        if arrow_candidates:
            arrow_area, arrow_rect = max(arrow_candidates, key=lambda item: item[0])
            snapshot.extra["bar_fallback_arrow_left"] = arrow_rect.left
            snapshot.extra["bar_fallback_arrow_right"] = arrow_rect.right
            snapshot.extra["bar_fallback_arrow_x"] = arrow_rect.center_x
            snapshot.extra["bar_fallback_arrow_area"] = arrow_area

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
                minimum_interior_color_fraction = float(
                    stick_detection.get("minimum_interior_color_fraction", 0.0)
                )
                interior_color_weight = float(
                    stick_detection.get("interior_color_weight", 0.0)
                )
                stick_color_mask = None
                if minimum_interior_color_fraction > 0 or interior_color_weight > 0:
                    stick_color_mask = _color_mask(
                        stick_crop,
                        [colors["stick"]],
                        stick_tolerance,
                    )
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
                        interior_color_fraction = 0.0
                        if stick_color_mask is not None:
                            interior = stick_color_mask[
                                int(rows.min()) : int(rows.max()) + 1,
                                left_index : right_index + 1,
                            ]
                            interior_color_fraction = (
                                float(np.mean(interior > 0))
                                if interior.size
                                else 0.0
                            )
                            if (
                                interior_color_fraction
                                < minimum_interior_color_fraction
                            ):
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
                        initial_bar_margin_ratio = stick_detection.get(
                            "initial_bar_margin_ratio"
                        )
                        if (
                            self.previous_stick is None
                            and initial_bar_margin_ratio is not None
                        ):
                            margin = reference_bar.width * max(
                                0.0, float(initial_bar_margin_ratio)
                            )
                            if not (
                                reference_bar.left - margin
                                <= rect.center_x
                                <= reference_bar.right + margin
                            ):
                                continue
                        maximum_center_jump = stick_detection.get(
                            "maximum_center_jump_normalized"
                        )
                        continuity_distance = 0.0
                        if self.previous_stick is not None:
                            continuity_distance = abs(
                                rect.center_x - self.previous_stick.center_x
                            )
                            if (
                                maximum_center_jump is not None
                                and continuity_distance
                                > width * max(0.0, float(maximum_center_jump))
                            ):
                                continue
                        score = float(edge_counts[left_index] + edge_counts[right_index])
                        score -= continuity_distance * float(
                            stick_detection.get("continuity_penalty", 0.0)
                        )
                        score += interior_color_fraction * interior_color_weight
                        stick_candidates.append((score, rect, int(rows.size)))
                snapshot.extra["stick_detection_mode"] = "paired_vertical_edges"
            else:
                stick_mask = _color_mask(stick_crop, [colors["stick"]], stick_tolerance)
                stick_mask = cv2.morphologyEx(
                    stick_mask, cv2.MORPH_CLOSE, np.ones((5, 3), np.uint8)
                )
                for rect, area in _components(stick_mask, stick_region.left, stick_region.top):
                    vertical_aspect = rect.height / max(1, rect.width)
                    if (
                        area >= 10
                        and vertical_aspect >= 1.8
                        and rect.height >= reference_bar.height * stick_height_ratio
                        and rect.width <= max(18, reference_bar.width * 0.12)
                    ):
                        # The stick is allowed to be far outside the bar. Distance
                        # is only a weak tie-breaker, never an overlap assumption.
                        center_penalty = abs(rect.center_x - reference_bar.center_x) * 0.002
                        stick_candidates.append((area + vertical_aspect * 8 - center_penalty, rect, area))
            if stick_candidates:
                _, snapshot.stick, stick_area = max(stick_candidates, key=lambda item: item[0])
                snapshot.stick_confidence = min(1.0, 0.35 + stick_area / 100.0)
                self.previous_stick = snapshot.stick

        # Lifecycle changes require the complete rail/bar/stick relationship.
        # A bar-like HUD rectangle without the stick is diagnostic PARTIAL only.
        snapshot.minigame_visible = (
            snapshot.bar is not None
            and snapshot.rail is not None
            and snapshot.stick is not None
        )
        width_config = self.profile["detection"].get("bar_width", {})
        live_width_history_allowed = bool(
            snapshot.extra.get("live_width_history_update_allowed", True)
        )
        if (
            snapshot.minigame_visible
            and bool(width_config.get("learning_requires_lock", False))
            and snapshot.extra.get("raw_bar_width") is not None
            and live_width_history_allowed
        ):
            raw_width = int(snapshot.extra["raw_bar_width"])
            learned_width = self.learned_bar_width
            minimum_learning_width_ratio = width_config.get(
                "nominal_learning_minimum_normalized"
            )
            large_enough_for_nominal = (
                minimum_learning_width_ratio is None
                or raw_width
                >= roi_width * float(minimum_learning_width_ratio)
            )
            is_clean_width = (
                large_enough_for_nominal
                and (
                    learned_width is None
                    or abs(raw_width - learned_width)
                    <= max(5, learned_width * 0.035)
                )
            )
            if is_clean_width:
                self.clean_bar_widths.append(raw_width)
        if snapshot.bar is not None:
            # Preserve bar identity through a temporary stick miss and accept
            # large legitimate movement across the rail.
            self.previous_bar = snapshot.bar
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
        snapshot.extra["learned_bar_width"] = self.learned_bar_width
        return snapshot
