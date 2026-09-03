from __future__ import annotations

from app.core.models import CommandAction, ControlCommand, DetectionSnapshot


class RuinousLegacyController:
    """Pure decision layer: observations in, command intent out."""

    def __init__(self, profile: dict):
        self.profile = profile
        self.generation = 0
        self.previous_action = CommandAction.NEUTRAL
        self.previous_bar_center: float | None = None
        self.previous_timestamp_ms: float | None = None
        self.filtered_velocity = 0.0
        self.last_action_change_ms: float | None = None
        self.rail_end_latch: str | None = None
        self.rail_end_latch_suppressed: str | None = None
        self.arrow_fallback_side: str | None = None

    def reset(self) -> None:
        self.generation += 1
        self.previous_action = CommandAction.NEUTRAL
        self.previous_bar_center = None
        self.previous_timestamp_ms = None
        self.filtered_velocity = 0.0
        self.last_action_change_ms = None
        self.rail_end_latch = None
        self.rail_end_latch_suppressed = None
        self.arrow_fallback_side = None

    def decide(self, observation: DetectionSnapshot) -> ControlCommand:
        bar = observation.bar
        stick = observation.stick
        fallback_arrow_x = observation.extra.get("bar_fallback_arrow_x")
        if bar is not None:
            self.arrow_fallback_side = None
        arrow_fallback = self.profile["controller"].get("arrow_fallback", {})
        learned_bar_width = observation.extra.get("learned_bar_width")
        if (
            bar is None
            and stick is not None
            and fallback_arrow_x is not None
            and arrow_fallback.get("mode") == "width_aware_predictive"
            and learned_bar_width is not None
        ):
            action, reason, target_x, error, confidence = (
                self._width_aware_arrow_fallback(
                    observation,
                    float(fallback_arrow_x),
                    float(learned_bar_width),
                )
            )
        elif bar is None and stick is not None and fallback_arrow_x is not None:
            # Match V13.4's missing-body behavior.  Holding moves right and
            # releasing moves left, so the stable internal arrow only needs to
            # tell us which side of the stick the hidden bar currently occupies.
            target_x = stick.center_x
            error = float(fallback_arrow_x) - stick.center_x
            if error > 0:
                action = CommandAction.RELEASE
                reason = "v13_arrow_fallback_bar_right_of_stick"
            else:
                action = CommandAction.HOLD
                reason = "v13_arrow_fallback_bar_left_of_stick"
            confidence = min(0.75, max(0.35, observation.stick_confidence))
        elif bar is None or stick is None:
            action = CommandAction.NEUTRAL
            reason = "insufficient_detection"
            target_x = None
            error = None
            confidence = min(observation.bar_confidence, observation.stick_confidence)
        else:
            target_x = stick.center_x
            error = bar.center_x - stick.center_x
            rail_end_action = self._rail_end_action(observation)
            deadzone = max(2.0, bar.width * float(self.profile["controller"]["deadzone_ratio"]))
            if rail_end_action is not None:
                action, reason = rail_end_action
                predicted_error = error
            else:
                velocity = self._update_velocity(
                    bar.center_x,
                    observation.timestamp_ms,
                    bar.width,
                )
                braking_horizon = float(
                    self.profile["controller"].get("braking_horizon_ms", 85)
                )
                predicted_center = bar.center_x + velocity * braking_horizon
                predicted_error = predicted_center - stick.center_x
                velocity_deadzone = float(
                    self.profile["controller"].get("velocity_deadzone_px_per_ms", 0.03)
                )
                # In Fisch, holding moves the bar right and releasing lets it move left.
                # Use the near-future center for reversals so existing momentum is
                # braked before the physical bar overshoots the stick.
                if predicted_error < -deadzone:
                    action = CommandAction.HOLD
                    reason = "predicted_bar_left_of_stick"
                elif predicted_error > deadzone:
                    action = CommandAction.RELEASE
                    reason = "predicted_bar_right_of_stick"
                else:
                    # There is no true stationary mouse state in this minigame.
                    # Counter the measured movement; at near-zero speed, alternate
                    # the binary input to avoid carrying one direction indefinitely.
                    if velocity > velocity_deadzone:
                        action = CommandAction.RELEASE
                        reason = "brake_rightward_momentum"
                    elif velocity < -velocity_deadzone:
                        action = CommandAction.HOLD
                        reason = "brake_leftward_momentum"
                    else:
                        pulse_ms = float(
                            self.profile["controller"].get("center_pulse_ms", 40)
                        )
                        elapsed_action_ms = self._elapsed_action_ms(observation.timestamp_ms)
                        if (
                            self.previous_action != CommandAction.NEUTRAL
                            and elapsed_action_ms < pulse_ms
                        ):
                            action = self.previous_action
                            reason = "continue_center_pulse"
                        elif self.previous_action == CommandAction.HOLD:
                            action = CommandAction.RELEASE
                            reason = "balance_center_from_hold"
                        else:
                            action = CommandAction.HOLD
                            reason = "balance_center_from_release"
            confidence = min(observation.bar_confidence, observation.stick_confidence)
            minimum_reversal_ms = float(
                self.profile["controller"].get("minimum_reversal_interval_ms", 28)
            )
            if (
                action != self.previous_action
                and self.previous_action != CommandAction.NEUTRAL
                and self._elapsed_action_ms(observation.timestamp_ms) < minimum_reversal_ms
                and abs(predicted_error) <= deadzone * 2.0
            ):
                action = self.previous_action
                reason = "suppress_near_center_chatter"
        if action != self.previous_action:
            self.generation += 1
            self.previous_action = action
            self.last_action_change_ms = observation.timestamp_ms
        return ControlCommand(action, self.generation, reason, target_x, error, confidence)

    def _width_aware_arrow_fallback(
        self,
        observation: DetectionSnapshot,
        arrow_x: float,
        bar_width: float,
    ) -> tuple[CommandAction, str, float, float, float]:
        """Steer from the hidden bar's center rather than its internal arrow.

        When the bar is outside the stick, only the arrow remains reliably
        colored. The arrow sits toward the edge facing the stick, so its raw x
        coordinate is not the bar center. Reconstructing the center from the
        learned live width also lets the normal velocity predictor brake the
        bar before it makes a full opposite-side swing.
        """
        stick = observation.stick
        assert stick is not None
        config = self.profile["controller"]["arrow_fallback"]
        safe_width = max(1.0, bar_width)
        if self.arrow_fallback_side is None:
            previous_center = self.previous_bar_center
            if previous_center is not None and previous_center != stick.center_x:
                self.arrow_fallback_side = (
                    "left" if previous_center < stick.center_x else "right"
                )
            else:
                self.arrow_fallback_side = (
                    "left" if arrow_x < stick.center_x else "right"
                )

        offset = safe_width * float(config.get("center_offset_ratio", 0.36))
        if self.arrow_fallback_side == "left":
            estimated_center = arrow_x - offset
        else:
            estimated_center = arrow_x + offset
        error = estimated_center - stick.center_x
        deadzone = max(
            2.0,
            safe_width
            * float(
                config.get(
                    "deadzone_ratio",
                    self.profile["controller"]["deadzone_ratio"],
                )
            ),
        )
        velocity = self._update_velocity(
            estimated_center,
            observation.timestamp_ms,
            safe_width,
        )
        braking_horizon = float(
            config.get(
                "braking_horizon_ms",
                self.profile["controller"].get("braking_horizon_ms", 85),
            )
        )
        predicted_error = estimated_center + velocity * braking_horizon - stick.center_x
        velocity_deadzone = float(
            self.profile["controller"].get("velocity_deadzone_px_per_ms", 0.03)
        )
        if predicted_error < -deadzone:
            action = CommandAction.HOLD
            reason = "width_aware_arrow_predicted_bar_left_of_stick"
        elif predicted_error > deadzone:
            action = CommandAction.RELEASE
            reason = "width_aware_arrow_predicted_bar_right_of_stick"
        elif velocity > velocity_deadzone:
            action = CommandAction.RELEASE
            reason = "width_aware_arrow_brake_rightward_momentum"
        elif velocity < -velocity_deadzone:
            action = CommandAction.HOLD
            reason = "width_aware_arrow_brake_leftward_momentum"
        else:
            pulse_ms = float(self.profile["controller"].get("center_pulse_ms", 40))
            elapsed_action_ms = self._elapsed_action_ms(observation.timestamp_ms)
            if (
                self.previous_action != CommandAction.NEUTRAL
                and elapsed_action_ms < pulse_ms
            ):
                action = self.previous_action
                reason = "width_aware_arrow_continue_center_pulse"
            elif self.previous_action == CommandAction.HOLD:
                action = CommandAction.RELEASE
                reason = "width_aware_arrow_balance_center_from_hold"
            else:
                action = CommandAction.HOLD
                reason = "width_aware_arrow_balance_center_from_release"

        minimum_reversal_ms = float(
            config.get(
                "minimum_reversal_interval_ms",
                self.profile["controller"].get("minimum_reversal_interval_ms", 28),
            )
        )
        if (
            action != self.previous_action
            and self.previous_action != CommandAction.NEUTRAL
            and self._elapsed_action_ms(observation.timestamp_ms) < minimum_reversal_ms
            and abs(predicted_error) <= deadzone * 2.0
        ):
            action = self.previous_action
            reason = "width_aware_arrow_suppress_near_center_chatter"
        confidence = min(0.75, max(0.35, observation.stick_confidence))
        return action, reason, stick.center_x, error, confidence

    def _rail_end_action(
        self, observation: DetectionSnapshot
    ) -> tuple[CommandAction, str] | None:
        enabled = self.profile.get("mechanics", {}).get("enabled", [])
        rail = observation.rail
        stick = observation.stick
        bar = observation.bar
        if (
            "rail_end_latching" not in enabled
            or rail is None
            or stick is None
            or bar is None
        ):
            self.rail_end_latch = None
            return None
        config = self.profile["controller"]["rail_end_latch"]
        enter = rail.width * float(config["enter_ratio"])
        boundary_margin = rail.width * float(
            config.get("boundary_margin_ratio", 0.025)
        )
        opposite_edge_margin = bar.width * float(
            config.get("opposite_edge_release_ratio", 0.30)
        )
        left_distance = stick.center_x - rail.left
        right_distance = rail.right - stick.center_x
        bar_at_left_stop = bar.left <= rail.left + boundary_margin
        bar_at_right_stop = bar.right >= rail.right - boundary_margin
        if self.rail_end_latch_suppressed == "left" and not bar_at_left_stop:
            self.rail_end_latch_suppressed = None
        elif self.rail_end_latch_suppressed == "right" and not bar_at_right_stop:
            self.rail_end_latch_suppressed = None

        if self.rail_end_latch == "left":
            # Manual Requiem behavior: once the bar rests at the left stop,
            # remain fully released while the stick travels left and returns.
            # Resume active steering only when it approaches the opposite edge
            # of the stationary bar.
            if stick.center_x <= bar.right - opposite_edge_margin:
                return CommandAction.RELEASE, "rail_end_latch_left_release"
            self.rail_end_latch = None
            self.rail_end_latch_suppressed = "left"
            return None
        elif self.rail_end_latch == "right":
            if stick.center_x >= bar.left + opposite_edge_margin:
                return CommandAction.HOLD, "rail_end_latch_right_hold"
            self.rail_end_latch = None
            self.rail_end_latch_suppressed = "right"
            return None
        if (
            self.rail_end_latch_suppressed != "left"
            and (bar_at_left_stop or left_distance <= enter)
        ):
            self.rail_end_latch = "left"
            return CommandAction.RELEASE, "rail_end_latch_left_release"
        if (
            self.rail_end_latch_suppressed != "right"
            and (bar_at_right_stop or right_distance <= enter)
        ):
            self.rail_end_latch = "right"
            return CommandAction.HOLD, "rail_end_latch_right_hold"
        return None

    def _elapsed_action_ms(self, timestamp_ms: float) -> float:
        if self.last_action_change_ms is None:
            return float("inf")
        return max(0.0, timestamp_ms - self.last_action_change_ms)

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
        if elapsed <= 0 or elapsed > 150:
            self.filtered_velocity = 0.0
            return 0.0
        displacement = center_x - previous_center
        maximum_jump = max(
            float(self.profile["controller"].get("velocity_max_jump_px", 24)),
            bar_width
            * float(self.profile["controller"].get("velocity_max_jump_ratio", 0.14)),
        )
        if abs(displacement) > maximum_jump:
            # A one-frame color/width transition is geometry, not momentum.
            # Keep the new positional fact but do not feed its jump into the
            # velocity predictor that decides braking direction.
            self.filtered_velocity *= 0.5
            return self.filtered_velocity
        instant = displacement / elapsed
        smoothing = min(
            1.0, max(0.0, float(self.profile["controller"].get("velocity_smoothing", 0.35)))
        )
        self.filtered_velocity += smoothing * (instant - self.filtered_velocity)
        return self.filtered_velocity

    def perfect_measurement(self, observation: DetectionSnapshot) -> tuple[bool | None, float | None]:
        if observation.bar is None or observation.stick is None:
            return None, None
        allowed = observation.bar.width * float(self.profile["controller"]["perfect_region_ratio"])
        error = abs(observation.bar.center_x - observation.stick.center_x)
        return error <= allowed, allowed - error
