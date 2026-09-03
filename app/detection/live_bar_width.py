from __future__ import annotations

from dataclasses import dataclass

from app.core.models import PixelRect


@dataclass(frozen=True)
class LiveBarWidthResult:
    bar: PixelRect
    geometry_source: str
    state: str
    nominal_width: int | None
    live_width: int
    allow_nominal_learning: bool


class LiveBarWidthTracker:
    """Separate temporary fish-driven width changes from nominal bar width.

    The ordinary restoration path remains responsible for color fragments and
    rod-specific enlargement. This tracker only promotes a smaller raw body
    after consecutive, edge-anchored observations agree. Promoted live widths
    never become nominal calibration samples.
    """

    _TRUSTED_SOURCES = {"left_edge_restored", "right_edge_restored"}

    def __init__(self, config: dict | None = None):
        self.config = config or {}
        self.enabled = bool(self.config.get("enabled", False))
        self.mode = str(self.config.get("mode", "temporary_shrink")).lower()
        self._candidate: PixelRect | None = None
        self._candidate_count = 0
        self._active_bar: PixelRect | None = None

    def reset(self) -> None:
        self._candidate = None
        self._candidate_count = 0
        self._active_bar = None

    def _result(
        self,
        bar: PixelRect,
        geometry_source: str,
        state: str,
        nominal_width: int | None,
        *,
        allow_nominal_learning: bool,
    ) -> LiveBarWidthResult:
        return LiveBarWidthResult(
            bar=bar,
            geometry_source=geometry_source,
            state=state,
            nominal_width=nominal_width,
            live_width=bar.width,
            allow_nominal_learning=allow_nominal_learning,
        )

    def _coherent(self, previous: PixelRect, current: PixelRect, nominal_width: int) -> bool:
        maximum_width_step = nominal_width * float(
            self.config.get("maximum_width_step_ratio", 0.10)
        )
        maximum_center_step = nominal_width * float(
            self.config.get("maximum_center_step_ratio", 0.18)
        )
        maximum_height_delta = max(2.0, previous.height * float(
            self.config.get("maximum_height_delta_ratio", 0.30)
        ))
        return (
            abs(current.width - previous.width) <= maximum_width_step
            and abs(current.center_x - previous.center_x) <= maximum_center_step
            and abs(current.height - previous.height) <= maximum_height_delta
        )

    def evaluate(
        self,
        raw_bar: PixelRect,
        restored_bar: PixelRect,
        geometry_source: str,
        nominal_width: int | None,
    ) -> LiveBarWidthResult:
        if not self.enabled:
            return self._result(
                restored_bar,
                geometry_source,
                "disabled",
                nominal_width,
                allow_nominal_learning=True,
            )

        if self.mode == "persistent_bidirectional":
            # Pinions changes its actual playable width after note catches and
            # misses. The current body is therefore authoritative in both
            # directions; retaining an older rectangle would corrupt steering.
            self._candidate = None
            self._candidate_count = 0
            self._active_bar = raw_bar
            return self._result(
                raw_bar,
                "persistent_live_width",
                "persistent_live",
                raw_bar.width,
                allow_nominal_learning=bool(
                    self.config.get("allow_nominal_learning", False)
                ),
            )

        if nominal_width is None or nominal_width <= 0:
            self.reset()
            return self._result(
                restored_bar,
                geometry_source,
                "learning_nominal",
                nominal_width,
                allow_nominal_learning=True,
            )

        width_ratio = raw_bar.width / nominal_width
        nominal_return_ratio = float(self.config.get("nominal_return_ratio", 0.985))

        # Normal width and rod-driven enlargement remain delegated to the
        # existing detector logic. This also exits a fish-width episode as soon
        # as the full body reappears.
        if width_ratio >= nominal_return_ratio:
            prior_active = self._active_bar is not None
            self.reset()
            return self._result(
                restored_bar,
                geometry_source,
                "nominal_restored" if prior_active else "nominal",
                nominal_width,
                allow_nominal_learning=True,
            )

        minimum_live_ratio = float(self.config.get("minimum_live_ratio", 0.72))
        maximum_enter_ratio = float(self.config.get("maximum_enter_ratio", 0.985))
        trusted_source = geometry_source in self._TRUSTED_SOURCES
        plausible_width = minimum_live_ratio <= width_ratio <= maximum_enter_ratio

        if self._active_bar is not None:
            continuation_sources = self._TRUSTED_SOURCES | set(
                self.config.get("active_continuation_sources", ())
            )
            active_minimum_live_ratio = float(
                self.config.get("active_minimum_live_ratio", minimum_live_ratio)
            )
            active_plausible_width = (
                active_minimum_live_ratio <= width_ratio <= maximum_enter_ratio
            )
            if geometry_source in continuation_sources and active_plausible_width:
                if self._coherent(self._active_bar, raw_bar, nominal_width):
                    self._active_bar = raw_bar
                    return self._result(
                        raw_bar,
                        "fish_live_width",
                        "active",
                        nominal_width,
                        allow_nominal_learning=False,
                    )
                if bool(self.config.get("retain_active_on_incoherent", False)):
                    return self._result(
                        self._active_bar,
                        "fish_live_width_occluded",
                        "active_occluded",
                        nominal_width,
                        allow_nominal_learning=False,
                    )

        if not trusted_source or not plausible_width:
            self._candidate = None
            self._candidate_count = 0
            if self._active_bar is not None:
                # A fragment during an active episode is still an occlusion,
                # never a new live width. Keep the last trusted body briefly.
                return self._result(
                    self._active_bar,
                    "fish_live_width_occluded",
                    "active_occluded",
                    nominal_width,
                    allow_nominal_learning=False,
                )
            reason = "untrusted_source" if not trusted_source else "implausible_width"
            return self._result(
                restored_bar,
                geometry_source,
                f"candidate_rejected_{reason}",
                nominal_width,
                allow_nominal_learning=False,
            )

        if self._active_bar is not None:
            if self._coherent(self._active_bar, raw_bar, nominal_width):
                self._active_bar = raw_bar
                return self._result(
                    raw_bar,
                    "fish_live_width",
                    "active",
                    nominal_width,
                    allow_nominal_learning=False,
                )
            self._active_bar = None
            self._candidate = raw_bar
            self._candidate_count = 1
            return self._result(
                restored_bar,
                geometry_source,
                "candidate_restarted",
                nominal_width,
                allow_nominal_learning=False,
            )

        if self._candidate is None or not self._coherent(
            self._candidate, raw_bar, nominal_width
        ):
            self._candidate = raw_bar
            self._candidate_count = 1
        else:
            growth_tolerance = nominal_width * float(
                self.config.get("confirmation_growth_tolerance_ratio", 0.025)
            )
            if raw_bar.width <= self._candidate.width + growth_tolerance:
                self._candidate_count += 1
            else:
                self._candidate_count = 1
            self._candidate = raw_bar

        confirmation_frames = max(2, int(self.config.get("confirmation_frames", 2)))
        if self._candidate_count >= confirmation_frames:
            self._active_bar = raw_bar
            self._candidate = None
            self._candidate_count = 0
            return self._result(
                raw_bar,
                "fish_live_width",
                "active_confirmed",
                nominal_width,
                allow_nominal_learning=False,
            )

        return self._result(
            restored_bar,
            geometry_source,
            "candidate_pending",
            nominal_width,
            allow_nominal_learning=False,
        )
