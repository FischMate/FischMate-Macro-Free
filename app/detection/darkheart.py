from __future__ import annotations

import cv2
import numpy as np

from app.core.models import FramePacket, PixelRect
from app.detection.standard import StandardDetector, _color_mask, _components


class DarkheartDetector(StandardDetector):
    """Darkheart's black body can blend into the rail; recover from its arrows."""

    def __init__(self, profile: dict):
        super().__init__(profile)
        self.previous_darkheart_stick: PixelRect | None = None
        self.previous_darkheart_bar: PixelRect | None = None
        self.previous_darkheart_stick_timestamp_ms: float | None = None
        self._last_trusted_timestamp_ms: float | None = None
        self._flash_hold_disabled_until_reacquire = False
        self._flash_reference_region: PixelRect | None = None
        self._flash_reference_gray: np.ndarray | None = None
        self._flash_reference_saturation: np.ndarray | None = None
        self._last_dark_stick_candidate_count = 0
        self._last_dark_stick_rejection_reason = ""

    def reset(self) -> None:
        super().reset()
        self.previous_darkheart_stick = None
        self.previous_darkheart_bar = None
        self.previous_darkheart_stick_timestamp_ms = None
        self._last_trusted_timestamp_ms = None
        self._flash_hold_disabled_until_reacquire = False
        self._flash_reference_region = None
        self._flash_reference_gray = None
        self._flash_reference_saturation = None
        self._last_dark_stick_candidate_count = 0
        self._last_dark_stick_rejection_reason = ""

    def detect(self, packet: FramePacket):
        standard_history = self._capture_standard_history()
        snapshot = super().detect(packet)

        # Standard rods use a bright stick, so the shared detector can report
        # one of Darkheart's white arrows (or another white HUD fragment) as a
        # stick.  White is never a playable-stick color on Darkheart.  Discard
        # that observation unconditionally before doing any Darkheart-specific
        # work, including on frames where no bar was found.
        discarded_standard_stick = snapshot.stick is not None
        snapshot.stick = None
        snapshot.stick_confidence = 0.0
        snapshot.extra["darkheart_standard_stick_discarded"] = (
            discarded_standard_stick
        )

        frame = packet.frame_bgr
        height, width = frame.shape[:2]
        roi = snapshot.minigame_roi
        crop = frame[roi.top : roi.bottom, roi.left : roi.right]
        if crop.size == 0:
            self._restore_standard_history(standard_history)
            return snapshot
        self._last_dark_stick_candidate_count = 0
        self._last_dark_stick_rejection_reason = ""

        # The screenshots and live telemetry show Darkheart's playable stick is
        # a dark vertical body with a light rim. White is reserved for the
        # arrows/icons inside the bar, so never trust the shared white-stick
        # path for this profile.
        if snapshot.bar is not None:
            preferred_x = self._preferred_darkheart_x(snapshot)
            dark_stick = self._dark_stick_near_bar(
                frame,
                roi,
                snapshot.bar,
                preferred_x,
                packet.timestamp_ms,
            )
            snapshot.extra["darkheart_dark_stick_candidates"] = (
                self._last_dark_stick_candidate_count
            )
            snapshot.extra["darkheart_dark_stick_rejection"] = (
                self._last_dark_stick_rejection_reason
            )
            if dark_stick is not None:
                if self._lock_is_discontinuous(
                    snapshot.bar,
                    dark_stick,
                    width,
                    packet.timestamp_ms,
                ):
                    snapshot.stick = None
                    snapshot.minigame_visible = False
                    snapshot.detector_state = "PARTIAL"
                    snapshot.rejection_reason = "darkheart_lock_discontinuous"
                    return self._finish_untrusted_frame(
                        snapshot,
                        frame,
                        roi,
                        packet.timestamp_ms,
                        standard_history,
                    )
                self._accept_darkheart_lock(snapshot, frame, roi, snapshot.bar, dark_stick)
                self._remember_trusted_frame(frame, packet.timestamp_ms)
                snapshot.extra["darkheart_bar_source"] = snapshot.extra.get(
                    "bar_geometry_source", "standard_partial"
                )
                snapshot.extra["darkheart_stick_source"] = "dark_vertical_body"
                return snapshot
            # A standard lock with no Darkheart dark-stick verification is not
            # trusted; treat it as partial and try the arrow-pair acquisition
            # path below.
            snapshot.stick = None
            snapshot.minigame_visible = False
            snapshot.detector_state = "PARTIAL"
            snapshot.rejection_reason = "darkheart_stick_missing"

        colors = self.profile["detection"]["colors"]
        tolerance = int(self.profile["detection"]["tolerance"]["arrow"])
        white_mask = _color_mask(crop, [colors["arrow"]], tolerance)
        white_mask = cv2.morphologyEx(
            white_mask,
            cv2.MORPH_CLOSE,
            np.ones((3, 3), np.uint8),
        )

        arrow_candidates: list[tuple[int, PixelRect]] = []
        for rect, area in _components(white_mask, roi.left, roi.top):
            if area < 8:
                continue
            aspect = rect.width / max(1, rect.height)
            if (
                7 <= rect.width <= width * 0.045
                and 5 <= rect.height <= height * 0.045
                and 0.45 <= aspect <= 5.0
            ):
                arrow_candidates.append((area, rect))

        arrow_pair = self._choose_arrow_pair(arrow_candidates, width)
        if arrow_pair is None:
            snapshot.extra["darkheart_arrow_pair_count"] = 0
            return self._finish_untrusted_frame(
                snapshot,
                frame,
                roi,
                packet.timestamp_ms,
                standard_history,
            )

        left_arrow, right_arrow = arrow_pair
        snapshot.extra["darkheart_arrow_pair_count"] = len(arrow_candidates)
        span = right_arrow.center_x - left_arrow.center_x
        bar_width = round(span * float(self.profile["detection"].get("arrow_pair_bar_width_ratio", 1.92)))
        min_width = round(width * 0.08)
        max_width = round(width * 0.20)
        if not (min_width <= bar_width <= max_width):
            snapshot.extra["darkheart_arrow_pair_rejection"] = "width_outside_profile"
            return self._finish_untrusted_frame(
                snapshot,
                frame,
                roi,
                packet.timestamp_ms,
                standard_history,
            )

        center_x = (left_arrow.center_x + right_arrow.center_x) / 2.0
        arrow_top = min(left_arrow.top, right_arrow.top)
        arrow_bottom = max(left_arrow.bottom, right_arrow.bottom)
        vertical_pad = max(8, round(max(left_arrow.height, right_arrow.height) * 0.65))
        bar = PixelRect(
            round(center_x - bar_width / 2.0),
            arrow_top - vertical_pad,
            round(center_x + bar_width / 2.0),
            arrow_bottom + vertical_pad,
        ).clamped(width, height)

        dark_stick = self._dark_stick_near_bar(
            frame,
            roi,
            bar,
            center_x,
            packet.timestamp_ms,
        )
        snapshot.extra["darkheart_dark_stick_candidates"] = (
            self._last_dark_stick_candidate_count
        )
        snapshot.extra["darkheart_dark_stick_rejection"] = (
            self._last_dark_stick_rejection_reason
        )
        if dark_stick is not None:
            stick = dark_stick
            snapshot.extra["darkheart_stick_source"] = "dark_vertical_body"
        else:
            stick = None
        if stick is None:
            # The white pair proves only where the black control bar is.  It
            # does not prove the black stick position, so remain PARTIAL rather
            # than turning arrow geometry into a synthetic playable stick.
            snapshot.bar = bar
            snapshot.stick = None
            snapshot.stick_confidence = 0.0
            snapshot.minigame_visible = False
            snapshot.detector_state = "PARTIAL"
            snapshot.rejection_reason = "darkheart_stick_missing"
            snapshot.extra["darkheart_bar_source"] = "arrow_pair_synthetic"
            snapshot.extra["darkheart_arrow_pair_rejection"] = "dark_stick_missing"
            snapshot.extra["raw_bar_left"] = bar.left
            snapshot.extra["raw_bar_right"] = bar.right
            snapshot.extra["raw_bar_width"] = bar.width
            snapshot.extra["bar_geometry_source"] = "darkheart_arrow_pair"
            return self._finish_untrusted_frame(
                snapshot,
                frame,
                roi,
                packet.timestamp_ms,
                standard_history,
            )

        if self._lock_is_discontinuous(
            bar,
            stick,
            width,
            packet.timestamp_ms,
        ):
            snapshot.bar = bar
            snapshot.stick = None
            snapshot.minigame_visible = False
            snapshot.detector_state = "PARTIAL"
            snapshot.rejection_reason = "darkheart_lock_discontinuous"
            return self._finish_untrusted_frame(
                snapshot,
                frame,
                roi,
                packet.timestamp_ms,
                standard_history,
            )
        self._accept_darkheart_lock(snapshot, frame, roi, bar, stick)
        self._remember_trusted_frame(frame, packet.timestamp_ms)
        snapshot.extra["darkheart_bar_source"] = "arrow_pair_synthetic"
        snapshot.extra["raw_bar_left"] = bar.left
        snapshot.extra["raw_bar_right"] = bar.right
        snapshot.extra["raw_bar_width"] = bar.width
        snapshot.extra["bar_geometry_source"] = "darkheart_arrow_pair"
        snapshot.extra["bar_candidate_count"] = max(
            1,
            int(snapshot.extra.get("bar_candidate_count", 0)),
        )
        return snapshot

    def _preferred_darkheart_x(self, snapshot) -> float | None:
        if self.previous_darkheart_stick is not None:
            return self.previous_darkheart_stick.center_x
        if snapshot.bar is not None:
            return snapshot.bar.center_x
        return None

    def _accept_darkheart_lock(
        self,
        snapshot,
        frame: np.ndarray,
        roi: PixelRect,
        bar: PixelRect,
        stick: PixelRect,
        *,
        update_bar_history: bool = True,
        update_stick_history: bool = True,
    ) -> None:
        height, width = frame.shape[:2]
        rail = self._dark_rail_near_bar(frame, roi, bar)
        if rail is None:
            rail = PixelRect(roi.left, bar.top, roi.right, bar.bottom).clamped(width, height)
            snapshot.extra["rail_geometry_source"] = "darkheart_roi_fallback"
        else:
            snapshot.extra["rail_geometry_source"] = "darkheart_dark_columns"

        snapshot.bar = bar
        snapshot.stick = stick
        snapshot.rail = rail
        snapshot.bar_confidence = max(snapshot.bar_confidence, 0.72)
        snapshot.stick_confidence = max(snapshot.stick_confidence, 0.72)
        snapshot.minigame_visible = True
        snapshot.detector_state = "LOCKED"
        snapshot.rejection_reason = ""
        snapshot.extra["darkheart_minigame_acquisition"] = "bar_stick_rail_confirmed"
        self.previous_bar = bar
        self.previous_rail = rail
        if update_bar_history:
            self._update_darkheart_bar_history(bar)
        if update_stick_history:
            self.previous_darkheart_stick = stick

    def _update_darkheart_bar_history(self, bar: PixelRect) -> None:
        self.previous_darkheart_bar = bar

    def _capture_standard_history(self) -> tuple:
        tracker = self.live_bar_width
        return (
            self.previous_bar,
            self.previous_rail,
            tuple(self.clean_bar_widths),
            tracker._candidate,
            tracker._candidate_count,
            tracker._active_bar,
        )

    def _restore_standard_history(self, history: tuple) -> None:
        (
            self.previous_bar,
            self.previous_rail,
            clean_widths,
            self.live_bar_width._candidate,
            self.live_bar_width._candidate_count,
            self.live_bar_width._active_bar,
        ) = history
        self.clean_bar_widths.clear()
        self.clean_bar_widths.extend(clean_widths)

    def _lock_is_discontinuous(
        self,
        bar: PixelRect,
        stick: PixelRect,
        frame_width: int,
        timestamp_ms: float,
    ) -> bool:
        config = self.profile["detection"].get("darkheart_flash", {})
        previous_bar = self.previous_darkheart_bar
        if previous_bar is not None:
            elapsed_ms = (
                max(0.0, timestamp_ms - self._last_trusted_timestamp_ms)
                if self._last_trusted_timestamp_ms is not None
                else 0.0
            )
            elapsed_ms = min(
                elapsed_ms,
                float(config.get("continuity_growth_cap_ms", 700.0)),
            )
            maximum_bar_jump = max(
                previous_bar.width
                * float(config.get("candidate_bar_jump_width_ratio", 1.40)),
                frame_width
                * float(config.get("candidate_bar_max_jump_normalized", 0.115)),
            ) + elapsed_ms * float(
                config.get("candidate_bar_max_speed_px_per_ms", 1.45)
            )
            width_ratio = bar.width / max(1, previous_bar.width)
            if (
                abs(bar.center_x - previous_bar.center_x) > maximum_bar_jump
                or width_ratio
                < float(config.get("candidate_bar_minimum_width_ratio", 0.70))
                or width_ratio
                > float(config.get("candidate_bar_maximum_width_ratio", 1.35))
            ):
                return True

        previous_stick = self.previous_darkheart_stick
        if previous_stick is not None:
            elapsed_ms = (
                max(
                    0.0,
                    timestamp_ms - self.previous_darkheart_stick_timestamp_ms,
                )
                if self.previous_darkheart_stick_timestamp_ms is not None
                else 0.0
            )
            elapsed_ms = min(
                elapsed_ms,
                float(config.get("continuity_growth_cap_ms", 700.0)),
            )
            maximum_stick_jump = max(
                80.0,
                frame_width
                * float(
                    self.profile["detection"].get(
                        "dark_stick_max_jump_normalized", 0.07
                    )
                ),
            ) + elapsed_ms * float(
                config.get("candidate_stick_max_speed_px_per_ms", 1.45)
            )
            if abs(stick.center_x - previous_stick.center_x) > maximum_stick_jump:
                return True
        return False

    def _remember_trusted_frame(
        self,
        frame: np.ndarray,
        timestamp_ms: float,
    ) -> None:
        bar = self.previous_darkheart_bar
        rail = self.previous_rail
        if bar is None or rail is None:
            return
        height, width = frame.shape[:2]
        pad_y = max(8, round(height * 0.018))
        region = PixelRect(
            rail.left,
            max(0, bar.top - pad_y),
            rail.right,
            min(height, bar.bottom + pad_y),
        ).clamped(width, height)
        crop = frame[region.top : region.bottom, region.left : region.right]
        if crop.size == 0:
            return
        self._flash_reference_region = region
        self._flash_reference_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        self._flash_reference_saturation = cv2.cvtColor(
            crop, cv2.COLOR_BGR2HSV
        )[:, :, 1]
        self._last_trusted_timestamp_ms = timestamp_ms
        self.previous_darkheart_stick_timestamp_ms = timestamp_ms
        self._flash_hold_disabled_until_reacquire = False

    def _flash_contamination(
        self,
        frame: np.ndarray,
    ) -> tuple[bool, dict[str, float]]:
        region = self._flash_reference_region
        reference_gray = self._flash_reference_gray
        reference_saturation = self._flash_reference_saturation
        if region is None or reference_gray is None or reference_saturation is None:
            return False, {}

        crop = frame[region.top : region.bottom, region.left : region.right]
        if crop.size == 0:
            return False, {}
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        saturation = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)[:, :, 1]
        if gray.shape != reference_gray.shape:
            return False, {}

        config = self.profile["detection"].get("darkheart_flash", {})
        difference_floor = int(config.get("difference_luma_min", 28))
        changed_fraction = float(
            np.mean(cv2.absdiff(gray, reference_gray) >= difference_floor)
        )
        dark_limit = int(config.get("dark_luma_max", 34))
        silver_luma_min = int(config.get("silver_luma_min", 78))
        silver_luma_max = int(config.get("silver_luma_max", 225))
        silver_saturation_max = int(config.get("silver_saturation_max", 32))

        current_dark = float(np.mean(gray <= dark_limit))
        reference_dark = float(np.mean(reference_gray <= dark_limit))
        current_silver = float(
            np.mean(
                (gray >= silver_luma_min)
                & (gray <= silver_luma_max)
                & (saturation <= silver_saturation_max)
            )
        )
        reference_silver = float(
            np.mean(
                (reference_gray >= silver_luma_min)
                & (reference_gray <= silver_luma_max)
                & (reference_saturation <= silver_saturation_max)
            )
        )
        dark_gain = current_dark - reference_dark
        silver_gain = current_silver - reference_silver
        luma_shift = abs(float(np.mean(gray)) - float(np.mean(reference_gray)))
        structured_change = (
            dark_gain >= float(config.get("dark_fraction_gain_min", 0.018))
            or silver_gain
            >= float(config.get("silver_fraction_gain_min", 0.018))
            or luma_shift >= float(config.get("mean_luma_shift_min", 4.5))
        )
        contaminated = (
            changed_fraction
            >= float(config.get("changed_fraction_min", 0.045))
            and structured_change
        )
        return contaminated, {
            "changed_fraction": changed_fraction,
            "dark_gain": dark_gain,
            "silver_gain": silver_gain,
            "luma_shift": luma_shift,
        }

    def _finish_untrusted_frame(
        self,
        snapshot,
        frame: np.ndarray,
        roi: PixelRect,
        timestamp_ms: float,
        standard_history: tuple,
    ):
        contaminated, metrics = self._flash_contamination(frame)
        self._restore_standard_history(standard_history)
        hold_eligible = (
            contaminated and not self._flash_hold_disabled_until_reacquire
        )
        snapshot.extra["darkheart_flash_contaminated"] = contaminated
        snapshot.extra["darkheart_flash_hold_eligible"] = hold_eligible
        snapshot.extra["darkheart_flash_history_updated"] = False
        for name, value in metrics.items():
            snapshot.extra[f"darkheart_flash_{name}"] = round(value, 4)

        previous_bar = self.previous_darkheart_bar
        previous_stick = self.previous_darkheart_stick
        previous_rail = self.previous_rail
        if (
            not hold_eligible
            or previous_bar is None
            or previous_stick is None
            or previous_rail is None
            or self._last_trusted_timestamp_ms is None
        ):
            snapshot.extra["darkheart_flash_hold_state"] = "not_applicable"
            return self._discard_untrusted_geometry(snapshot)

        age_ms = max(0.0, timestamp_ms - self._last_trusted_timestamp_ms)
        hold_ms = float(
            self.profile["detection"]
            .get("darkheart_flash", {})
            .get("trusted_hold_ms", 650.0)
        )
        snapshot.extra["darkheart_flash_hold_age_ms"] = round(age_ms, 3)
        if age_ms > hold_ms:
            self._flash_hold_disabled_until_reacquire = True
            snapshot.extra["darkheart_flash_hold_state"] = "expired"
            snapshot.rejection_reason = "darkheart_flash_hold_expired"
            return self._discard_untrusted_geometry(snapshot)

        # The flash frame is never allowed to contribute geometry or history.
        # Reusing the exact last trusted observation preserves the controller's
        # intent briefly without steering toward black/silver artifacts.
        snapshot.bar = previous_bar
        snapshot.stick = previous_stick
        snapshot.rail = previous_rail
        snapshot.bar_confidence = max(snapshot.bar_confidence, 0.55)
        snapshot.stick_confidence = max(snapshot.stick_confidence, 0.55)
        snapshot.minigame_visible = True
        snapshot.detector_state = "LOCKED"
        snapshot.rejection_reason = ""
        snapshot.extra["darkheart_bar_source"] = "flash_trusted_history"
        snapshot.extra["darkheart_stick_source"] = "flash_trusted_history"
        snapshot.extra["darkheart_minigame_acquisition"] = "flash_history_hold"
        snapshot.extra["darkheart_flash_hold_state"] = "holding"
        return snapshot

    def _discard_untrusted_geometry(self, snapshot):
        if snapshot.bar is not None:
            snapshot.extra["darkheart_untrusted_bar_left"] = snapshot.bar.left
            snapshot.extra["darkheart_untrusted_bar_right"] = snapshot.bar.right
            snapshot.extra["darkheart_untrusted_bar_width"] = snapshot.bar.width
        if snapshot.stick is not None:
            snapshot.extra["darkheart_untrusted_stick_x"] = snapshot.stick.center_x
        snapshot.bar = None
        snapshot.stick = None
        snapshot.rail = None
        snapshot.bar_confidence = 0.0
        snapshot.stick_confidence = 0.0
        snapshot.minigame_visible = False
        snapshot.detector_state = "SEARCHING"
        return snapshot

    def _dark_stick_near_bar(
        self,
        frame: np.ndarray,
        roi: PixelRect,
        bar: PixelRect,
        preferred_x: float | None = None,
        timestamp_ms: float | None = None,
    ) -> PixelRect | None:
        height, width = frame.shape[:2]
        rail = self._dark_rail_near_bar(frame, roi, bar) or self.previous_rail
        search_left = max(roi.left, rail.left if rail is not None else roi.left)
        search_right = min(roi.right, rail.right if rail is not None else roi.right)
        vertical_pad = max(12, round(height * 0.045))
        top = max(roi.top, bar.top - vertical_pad)
        bottom = min(roi.bottom, bar.bottom + vertical_pad)
        if bottom <= top or search_right <= search_left:
            return None

        band = frame[top:bottom, search_left:search_right]
        if band.size == 0:
            return None
        gray = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
        dark_limit = int(self.profile["detection"].get("dark_stick_luma_max", 58))
        dark = gray <= dark_limit
        upper_rows = max(0, min(bar.top - top, dark.shape[0]))
        lower_start = max(0, min(bar.bottom - top, dark.shape[0]))
        if upper_rows == 0 and lower_start >= dark.shape[0]:
            return None

        upper_support = (
            np.mean(dark[:upper_rows, :], axis=0)
            if upper_rows > 0
            else np.zeros(dark.shape[1], dtype=float)
        )
        lower_support = (
            np.mean(dark[lower_start:, :], axis=0)
            if lower_start < dark.shape[0]
            else np.zeros(dark.shape[1], dtype=float)
        )
        full_support = np.mean(dark, axis=0)
        overhang_support = np.maximum(upper_support, lower_support)
        column_mask = (
            (overhang_support >= 0.14)
            & (full_support >= 0.18)
        ).astype(np.uint8)
        if column_mask.size == 0 or not np.any(column_mask):
            self._last_dark_stick_candidate_count = 0
            self._last_dark_stick_rejection_reason = "no_dark_overhang_columns"
            return None
        column_mask = cv2.morphologyEx(
            column_mask.reshape(1, -1),
            cv2.MORPH_CLOSE,
            np.ones((1, 3), np.uint8),
        ).reshape(-1)

        best: tuple[float, PixelRect, int] | None = None
        indexes = np.flatnonzero(column_mask)
        groups = np.split(indexes, np.where(np.diff(indexes) > 1)[0] + 1)
        self._last_dark_stick_candidate_count = len(groups)
        minimum_stick_width = max(
            3,
            round(
                width
                * float(
                    self.profile["detection"].get(
                        "dark_stick_min_width_normalized", 0.002
                    )
                )
            ),
        )
        max_stick_width = max(
            minimum_stick_width,
            round(
                width
                * float(
                    self.profile["detection"].get(
                        "dark_stick_max_width_normalized", 0.009
                    )
                )
            ),
        )
        min_height = max(round(bar.height * 1.05), round(height * 0.03))
        previous_x = (
            None
            if self.previous_darkheart_stick is None
            else self.previous_darkheart_stick.center_x
        )
        continuity_config = self.profile["detection"].get("darkheart_flash", {})
        elapsed_ms = 0.0
        if timestamp_ms is not None and self.previous_darkheart_stick_timestamp_ms is not None:
            elapsed_ms = max(
                0.0,
                timestamp_ms - self.previous_darkheart_stick_timestamp_ms,
            )
            elapsed_ms = min(
                elapsed_ms,
                float(continuity_config.get("continuity_growth_cap_ms", 700.0)),
            )
        max_center_jump = max(
            80.0,
            width
            * float(
                self.profile["detection"].get(
                    "dark_stick_max_jump_normalized", 0.07
                )
            ),
        ) + elapsed_ms * float(
            continuity_config.get("candidate_stick_max_speed_px_per_ms", 1.45)
        )
        continuity_rejections = 0
        for group in groups:
            if group.size == 0:
                continue
            local_left = int(group.min())
            local_right = int(group.max()) + 1
            candidate_groups = [(local_left, local_right)]
            if local_right - local_left > max_stick_width:
                # When the synthesized arrow-pair bar is shorter vertically
                # than the true black body, the horizontal bar itself can make
                # one wide dark run.  Split it by columns that extend taller
                # than the local median; that isolates the narrow stick core.
                wide_columns = dark[:, local_left:local_right]
                spans = []
                for offset in range(wide_columns.shape[1]):
                    rows = np.flatnonzero(wide_columns[:, offset])
                    spans.append(0 if rows.size == 0 else int(rows.max() - rows.min() + 1))
                span_array = np.asarray(spans)
                positive_spans = span_array[span_array > 0]
                if positive_spans.size == 0:
                    continue
                span_floor = max(
                    min_height,
                    int(np.median(positive_spans)) + max(6, round(height * 0.006)),
                )
                tall_offsets = np.flatnonzero(span_array >= span_floor)
                candidate_groups = []
                if tall_offsets.size:
                    tall_groups = np.split(
                        tall_offsets,
                        np.where(np.diff(tall_offsets) > 1)[0] + 1,
                    )
                    for tall_group in tall_groups:
                        if tall_group.size:
                            candidate_groups.append(
                                (
                                    local_left + int(tall_group.min()),
                                    local_left + int(tall_group.max()) + 1,
                                )
                            )

            for candidate_left, candidate_right in candidate_groups:
                candidate_width = candidate_right - candidate_left
                if not (minimum_stick_width <= candidate_width <= max_stick_width):
                    continue
                columns = dark[:, candidate_left:candidate_right]
                rows = np.flatnonzero(np.any(columns, axis=1))
                if rows.size == 0:
                    continue
                rect = PixelRect(
                    search_left + candidate_left,
                    top + int(rows.min()),
                    search_left + candidate_right,
                    top + int(rows.max()) + 1,
                ).clamped(width, height)
                if rect.height < min_height:
                    continue
                vertical_aspect = rect.height / max(1, rect.width)
                if vertical_aspect < 2.0:
                    continue
                outline_score = self._dark_stick_outline_score(
                    gray,
                    rect,
                    bar,
                    search_left,
                    top,
                )
                minimum_outline_score = float(
                    self.profile["detection"].get(
                        "dark_stick_outline_score_min", 0.18
                    )
                )
                if outline_score < minimum_outline_score:
                    continue
                if previous_x is not None and abs(rect.center_x - previous_x) > max_center_jump:
                    continuity_rejections += 1
                    continue
                overhang = float(np.mean(overhang_support[candidate_left:candidate_right]))
                fill = float(np.mean(columns))
                center_penalty = 0.0
                if preferred_x is not None:
                    center_penalty = abs(rect.center_x - preferred_x) * 0.015
                elif self.previous_bar is not None:
                    center_penalty = abs(rect.center_x - self.previous_bar.center_x) * 0.004
                score = (
                    rect.height * 1.4
                    + vertical_aspect * 12.0
                    + outline_score * 260.0
                    + overhang * 120.0
                    + fill * 35.0
                    - center_penalty
                )
                if best is None or score > best[0]:
                    best = (score, rect, len(groups))
        if best is None:
            self._last_dark_stick_rejection_reason = (
                "continuity_jump"
                if continuity_rejections
                else "no_valid_dark_vertical_body"
            )
            return None
        return best[1]

    def _dark_stick_outline_score(
        self,
        gray_band: np.ndarray,
        rect: PixelRect,
        bar: PixelRect,
        offset_x: int,
        offset_y: int,
        *,
        edge_floor_override: int | None = None,
        rim_contrast_override: float | None = None,
    ) -> float:
        """Darkheart's playable stick is a dark core with lighter side rims.

        The bar and rail are also dark, so color alone is not identity.  The
        side-rim check keeps random dark vertical objects from masquerading as
        the stick.
        """
        local_left = rect.left - offset_x
        local_right = rect.right - offset_x
        local_top = rect.top - offset_y
        local_bottom = rect.bottom - offset_y
        height, width = gray_band.shape[:2]
        if local_bottom <= local_top or local_right <= local_left:
            return 0.0

        core = gray_band[
            max(0, local_top) : min(height, local_bottom),
            max(0, local_left) : min(width, local_right),
        ]
        if core.size == 0:
            return 0.0
        core_dark = float(np.mean(core <= int(self.profile["detection"].get("dark_stick_luma_max", 58))))
        if core_dark < 0.72:
            return 0.0

        left_start = max(0, local_left - 2)
        left_end = max(0, local_left)
        right_start = min(width, local_right)
        right_end = min(width, local_right + 2)
        if left_end <= left_start or right_end <= right_start:
            return 0.0

        # Inspect the rim above and below the horizontal bar.  Including the
        # bar-crossing rows dilutes a real stick's outline with the black bar
        # body and made the live detector intermittently reject the true core.
        row_indexes = np.arange(max(0, local_top), min(height, local_bottom))
        local_bar_top = bar.top - offset_y
        local_bar_bottom = bar.bottom - offset_y
        overhang_rows = row_indexes[
            (row_indexes < local_bar_top) | (row_indexes >= local_bar_bottom)
        ]
        if overhang_rows.size < 4:
            return 0.0

        left_edge = gray_band[overhang_rows, left_start:left_end]
        right_edge = gray_band[overhang_rows, right_start:right_end]
        if left_edge.size == 0 or right_edge.size == 0:
            return 0.0

        # In both supplied screenshots the actual outline is a narrow bright
        # rim: its immediate side columns are about 35-40 luma above columns a
        # few pixels farther away.  A random black object on a uniformly gray
        # background has light sides too, but no such local rim contrast.
        left_outer_start = max(0, local_left - 6)
        left_outer_end = max(0, local_left - 3)
        right_outer_start = min(width, local_right + 3)
        right_outer_end = min(width, local_right + 6)
        if (
            left_outer_end <= left_outer_start
            or right_outer_end <= right_outer_start
        ):
            return 0.0
        left_outer = gray_band[
            overhang_rows, left_outer_start:left_outer_end
        ]
        right_outer = gray_band[
            overhang_rows, right_outer_start:right_outer_end
        ]
        if left_outer.size == 0 or right_outer.size == 0:
            return 0.0
        minimum_rim_contrast = (
            float(rim_contrast_override)
            if rim_contrast_override is not None
            else float(
                self.profile["detection"].get(
                    "dark_stick_rim_contrast_min", 12
                )
            )
        )
        left_contrast = float(np.mean(left_edge) - np.mean(left_outer))
        right_contrast = float(np.mean(right_edge) - np.mean(right_outer))
        if min(left_contrast, right_contrast) < minimum_rim_contrast:
            return 0.0

        edge_floor = (
            int(edge_floor_override)
            if edge_floor_override is not None
            else int(
                self.profile["detection"].get(
                    "dark_stick_outline_luma_min", 78
                )
            )
        )
        edge_ceiling = int(self.profile["detection"].get("dark_stick_outline_luma_max", 245))
        left_outline = float(np.mean((left_edge >= edge_floor) & (left_edge <= edge_ceiling)))
        right_outline = float(np.mean((right_edge >= edge_floor) & (right_edge <= edge_ceiling)))
        dark_limit = int(self.profile["detection"].get("dark_stick_luma_max", 40))
        left_not_dark = float(np.mean(left_edge > dark_limit))
        right_not_dark = float(np.mean(right_edge > dark_limit))
        contrast_score = min(1.0, min(left_contrast, right_contrast) / 35.0)
        return (
            min(left_outline, right_outline) * 0.50
            + min(left_not_dark, right_not_dark) * 0.25
            + contrast_score * 0.25
        )

    def _choose_arrow_pair(
        self,
        candidates: list[tuple[int, PixelRect]],
        frame_width: int,
    ) -> tuple[PixelRect, PixelRect] | None:
        best: tuple[float, PixelRect, PixelRect] | None = None
        ordered = [rect for _, rect in sorted(candidates, key=lambda item: item[1].left)]
        for index, left in enumerate(ordered):
            for right in ordered[index + 1 :]:
                gap = right.center_x - left.center_x
                if gap < frame_width * 0.025 or gap > frame_width * 0.14:
                    continue
                center_y_delta = abs(
                    (left.top + left.bottom) / 2.0
                    - (right.top + right.bottom) / 2.0
                )
                if center_y_delta > max(left.height, right.height, 12):
                    continue
                size_delta = abs(left.height - right.height) + abs(left.width - right.width)
                score = gap - center_y_delta * 4.0 - size_delta * 1.5
                if best is None or score > best[0]:
                    best = (score, left, right)
        if best is None:
            return None
        return best[1], best[2]

    def _dark_rail_near_bar(
        self,
        frame: np.ndarray,
        roi: PixelRect,
        bar: PixelRect,
    ) -> PixelRect | None:
        height, width = frame.shape[:2]
        band_top = max(roi.top, bar.top - round(height * 0.012))
        band_bottom = min(roi.bottom, bar.bottom + round(height * 0.012))
        band = frame[band_top:band_bottom, roi.left:roi.right]
        if band.size == 0:
            return None
        gray_band = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
        dark_columns = np.mean(gray_band <= 82, axis=0) >= 0.24
        dark_indexes = np.flatnonzero(dark_columns)
        if dark_indexes.size < 2:
            return None
        rail_left = roi.left + int(dark_indexes.min())
        rail_right = roi.left + int(dark_indexes.max()) + 1
        if rail_right - rail_left < width * 0.28:
            return None
        return PixelRect(rail_left, band_top, rail_right, band_bottom).clamped(
            width,
            height,
        )
