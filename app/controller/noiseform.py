from __future__ import annotations

from app.controller.special_standard import SpecialStandardController


class NoiseformController(SpecialStandardController):
    """Noiseform-only momentum continuity across brief visual occlusions."""

    def _update_velocity(
        self,
        center_x: float,
        timestamp_ms: float,
        bar_width: float,
    ) -> float:
        previous_center = self.previous_bar_center
        previous_time = self.previous_timestamp_ms
        self.previous_bar_center = center_x
        self.previous_timestamp_ms = timestamp_ms
        if previous_center is None or previous_time is None:
            self.filtered_velocity = 0.0
            return 0.0
        elapsed = timestamp_ms - previous_time
        maximum_elapsed = float(
            self.profile["controller"].get(
                "velocity_reacquire_maximum_elapsed_ms",
                150,
            )
        )
        if elapsed <= 0 or elapsed > maximum_elapsed:
            self.filtered_velocity = 0.0
            return 0.0
        displacement = center_x - previous_center
        maximum_jump = max(
            float(self.profile["controller"].get("velocity_max_jump_px", 24)),
            bar_width
            * float(self.profile["controller"].get("velocity_max_jump_ratio", 0.14)),
        )
        if abs(displacement) > maximum_jump:
            self.filtered_velocity *= 0.5
            return self.filtered_velocity
        instant = displacement / elapsed
        smoothing = min(
            1.0,
            max(
                0.0,
                float(self.profile["controller"].get("velocity_smoothing", 0.35)),
            ),
        )
        self.filtered_velocity += smoothing * (instant - self.filtered_velocity)
        return self.filtered_velocity
