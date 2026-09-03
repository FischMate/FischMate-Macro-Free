from __future__ import annotations

from app.core.models import CommandAction, ControlCommand, DetectionSnapshot


class StandardController:
    """Pure decision layer: observations in, command intent out."""

    def __init__(self, profile: dict):
        self.profile = profile
        self.generation = 0
        self.previous_action = CommandAction.NEUTRAL
        self.previous_bar_center: float | None = None
        self.previous_timestamp_ms: float | None = None
        self.filtered_velocity = 0.0
        self.instant_velocity = 0.0
        self.previous_stick_center: float | None = None
        self.previous_stick_timestamp_ms: float | None = None
        self.filtered_stick_velocity = 0.0
        self.last_action_change_ms: float | None = None
        self.rail_end_latch: str | None = None
        self.arrow_fallback_side: str | None = None
        self.soft_brake_action: CommandAction | None = None
        self.soft_brake_until_ms: float | None = None
        self.soft_brake_cooldown_until_ms: float | None = None

    def reset(self) -> None:
        self.generation += 1
        self.previous_action = CommandAction.NEUTRAL
        self.previous_bar_center = None
        self.previous_timestamp_ms = None
        self.filtered_velocity = 0.0
        self.instant_velocity = 0.0
        self.previous_stick_center = None
        self.previous_stick_timestamp_ms = None
        self.filtered_stick_velocity = 0.0
        self.last_action_change_ms = None
        self.rail_end_latch = None
        self.arrow_fallback_side = None
        self.soft_brake_action = None
        self.soft_brake_until_ms = None
        self.soft_brake_cooldown_until_ms = None

    def decide(self, observation: DetectionSnapshot) -> ControlCommand:
        bar = observation.bar
        stick = observation.stick
        fallback_arrow_x = observation.extra.get("bar_fallback_arrow_x")
        learned_bar_width = observation.extra.get("learned_bar_width")
        if stick is not None:
            self._update_stick_velocity(stick.center_x, observation.timestamp_ms)
        if bar is not None:
            self.arrow_fallback_side = None
        general_arrow_fallback = self.profile["controller"].get(
            "general_arrow_fallback", {}
        )
        if (
            bar is None
            and stick is not None
            and fallback_arrow_x is not None
            and learned_bar_width is not None
            and bool(general_arrow_fallback.get("enabled", False))
        ):
            self.rail_end_latch = None
            action, reason, target_x, error, confidence = (
                self._general_width_aware_arrow_fallback(
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
            deadzone = max(2.0, bar.width * float(self.profile["controller"]["deadzone_ratio"]))
            rail_end_action = self._rail_end_action(observation)
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
                    self.profile["controller"].get(
                        "general_braking_horizon_ms",
                        self.profile["controller"].get("braking_horizon_ms", 85),
                    )
                )
                predicted_center = bar.center_x + velocity * braking_horizon
                predicted_stick = self._predicted_stick_center(stick.center_x)
                predicted_error = predicted_center - predicted_stick
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
                action, reason = self._apply_general_soft_brake(
                    action,
                    reason,
                    error,
                    bar.width,
                    velocity,
                    observation.timestamp_ms,
                )
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

    def _general_width_aware_arrow_fallback(
        self,
        observation: DetectionSnapshot,
        arrow_x: float,
        bar_width: float,
    ) -> tuple[CommandAction, str, float, float, float]:
        """Keep the ordinary controller's momentum estimate through occlusion.

        The internal arrow remains visible when the white body is split by the
        stick or blends into the rail. Its position is offset toward the edge
        facing the stick, so reconstruct the center from the learned live bar
        width instead of collapsing to a direction-only full input.
        """
        stick = observation.stick
        assert stick is not None
        config = self.profile["controller"].get("general_arrow_fallback", {})
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
                self.profile["controller"].get(
                    "general_braking_horizon_ms",
                    self.profile["controller"].get("braking_horizon_ms", 85),
                ),
            )
        )
        predicted_stick = self._predicted_stick_center(stick.center_x)
        predicted_error = estimated_center + velocity * braking_horizon - predicted_stick
        velocity_deadzone = float(
            self.profile["controller"].get("velocity_deadzone_px_per_ms", 0.03)
        )
        if predicted_error < -deadzone:
            action = CommandAction.HOLD
            reason = "general_arrow_predicted_bar_left_of_stick"
        elif predicted_error > deadzone:
            action = CommandAction.RELEASE
            reason = "general_arrow_predicted_bar_right_of_stick"
        elif velocity > velocity_deadzone:
            action = CommandAction.RELEASE
            reason = "general_arrow_brake_rightward_momentum"
        elif velocity < -velocity_deadzone:
            action = CommandAction.HOLD
            reason = "general_arrow_brake_leftward_momentum"
        else:
            pulse_ms = float(self.profile["controller"].get("center_pulse_ms", 40))
            elapsed_action_ms = self._elapsed_action_ms(observation.timestamp_ms)
            if (
                self.previous_action != CommandAction.NEUTRAL
                and elapsed_action_ms < pulse_ms
            ):
                action = self.previous_action
                reason = "general_arrow_continue_center_pulse"
            elif self.previous_action == CommandAction.HOLD:
                action = CommandAction.RELEASE
                reason = "general_arrow_balance_center_from_hold"
            else:
                action = CommandAction.HOLD
                reason = "general_arrow_balance_center_from_release"

        action, reason = self._apply_general_soft_brake(
            action,
            reason,
            error,
            safe_width,
            velocity,
            observation.timestamp_ms,
            reason_prefix="general_arrow_",
        )

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
            reason = "general_arrow_suppress_near_center_chatter"
        confidence = min(0.75, max(0.35, observation.stick_confidence))
        return action, reason, stick.center_x, error, confidence

    def _predicted_stick_center(self, center_x: float) -> float:
        horizon_ms = float(
            self.profile["controller"].get(
                "general_stick_prediction_horizon_ms", 0
            )
        )
        return center_x + self.filtered_stick_velocity * max(0.0, horizon_ms)

    def _update_stick_velocity(self, center_x: float, timestamp_ms: float) -> float:
        previous_center = self.previous_stick_center
        previous_time = self.previous_stick_timestamp_ms
        self.previous_stick_center = center_x
        self.previous_stick_timestamp_ms = timestamp_ms
        if previous_center is None or previous_time is None:
            self.filtered_stick_velocity = 0.0
            return 0.0
        elapsed = timestamp_ms - previous_time
        if elapsed <= 0 or elapsed > 150:
            self.filtered_stick_velocity = 0.0
            return 0.0
        displacement = center_x - previous_center
        maximum_jump = float(
            self.profile["controller"].get("general_stick_velocity_max_jump_px", 32)
        )
        if abs(displacement) > maximum_jump:
            self.filtered_stick_velocity *= 0.5
            return self.filtered_stick_velocity
        instant = displacement / elapsed
        smoothing = min(
            1.0,
            max(
                0.0,
                float(
                    self.profile["controller"].get(
                        "general_stick_velocity_smoothing", 0.55
                    )
                ),
            ),
        )
        self.filtered_stick_velocity += smoothing * (
            instant - self.filtered_stick_velocity
        )
        return self.filtered_stick_velocity

    def _apply_general_soft_brake(
        self,
        action: CommandAction,
        reason: str,
        error: float,
        bar_width: float,
        velocity: float,
        timestamp_ms: float,
        *,
        reason_prefix: str = "",
    ) -> tuple[CommandAction, str]:
        """Feather a long recovery just before it creates opposite momentum."""
        config = self.profile["controller"].get("general_soft_brake", {})
        if not bool(config.get("enabled", False)):
            return action, reason

        if (
            self.soft_brake_action is not None
            and self.soft_brake_until_ms is not None
            and timestamp_ms < self.soft_brake_until_ms
        ):
            return (
                self.soft_brake_action,
                f"{reason_prefix}general_soft_brake_pulse",
            )
        self.soft_brake_action = None
        self.soft_brake_until_ms = None

        cooldown_until = self.soft_brake_cooldown_until_ms
        if cooldown_until is not None and timestamp_ms < cooldown_until:
            return action, reason
        maximum_error = bar_width * float(config.get("maximum_error_bar_ratio", 0.36))
        if abs(error) > maximum_error:
            return action, reason
        minimum_action_ms = float(config.get("minimum_action_ms", 280))
        if self._elapsed_action_ms(timestamp_ms) < minimum_action_ms:
            return action, reason

        velocity_threshold = float(
            config.get("velocity_threshold_px_per_ms", 0.12)
        )
        velocity_deadzone = float(
            self.profile["controller"].get("velocity_deadzone_px_per_ms", 0.03)
        )
        braking_left = (
            action == CommandAction.HOLD
            and velocity < -velocity_deadzone
            and -velocity_threshold <= self.instant_velocity < 0
        )
        braking_right = (
            action == CommandAction.RELEASE
            and velocity > velocity_deadzone
            and 0 < self.instant_velocity <= velocity_threshold
        )
        if not braking_left and not braking_right:
            return action, reason

        pulse_action = (
            CommandAction.RELEASE if braking_left else CommandAction.HOLD
        )
        pulse_ms = float(config.get("pulse_ms", 80))
        cooldown_ms = float(config.get("cooldown_ms", 220))
        self.soft_brake_action = pulse_action
        self.soft_brake_until_ms = timestamp_ms + max(0.0, pulse_ms)
        self.soft_brake_cooldown_until_ms = timestamp_ms + max(
            max(0.0, pulse_ms), max(0.0, cooldown_ms)
        )
        return pulse_action, f"{reason_prefix}general_soft_brake_start"

    def _rail_end_action(
        self, observation: DetectionSnapshot
    ) -> tuple[CommandAction, str] | None:
        rail = observation.rail
        bar = observation.bar
        stick = observation.stick
        if rail is None or bar is None or stick is None:
            self.rail_end_latch = None
            return None

        controller = self.profile["controller"]
        enter_distance = bar.width * float(
            controller.get("rail_end_latch_enter_bar_ratio", 0.72)
        )
        exit_distance = bar.width * float(
            controller.get("rail_end_latch_exit_bar_ratio", 0.92)
        )
        exit_distance = max(exit_distance, enter_distance)
        escape_error = max(
            2.0,
            bar.width
            * float(
                controller.get(
                    "rail_end_latch_escape_error_ratio",
                    controller.get("deadzone_ratio", 0.08),
                )
            ),
        )
        left_distance = stick.center_x - rail.left
        right_distance = rail.right - stick.center_x

        if self.rail_end_latch == "left":
            # Releasing is only useful while the bar has not fallen materially
            # behind a stick that has turned back toward center.
            if bar.center_x < stick.center_x - escape_error:
                self.rail_end_latch = None
            elif left_distance <= exit_distance:
                return CommandAction.RELEASE, "general_rail_end_latch_left_release"
            else:
                self.rail_end_latch = None
        elif self.rail_end_latch == "right":
            # Holding is only useful while the bar has not run materially ahead
            # of a stick that has turned back toward center.
            if bar.center_x > stick.center_x + escape_error:
                self.rail_end_latch = None
            elif right_distance <= exit_distance:
                return CommandAction.HOLD, "general_rail_end_latch_right_hold"
            else:
                self.rail_end_latch = None

        if (
            left_distance <= enter_distance
            and left_distance <= right_distance
            and bar.center_x >= stick.center_x - escape_error
        ):
            self.rail_end_latch = "left"
            return CommandAction.RELEASE, "general_rail_end_latch_left_release"
        if (
            right_distance <= enter_distance
            and bar.center_x <= stick.center_x + escape_error
        ):
            self.rail_end_latch = "right"
            return CommandAction.HOLD, "general_rail_end_latch_right_hold"
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
            self.instant_velocity = 0.0
            return 0.0
        elapsed = timestamp_ms - previous_time
        if elapsed <= 0 or elapsed > 150:
            self.filtered_velocity = 0.0
            self.instant_velocity = 0.0
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
            self.instant_velocity = 0.0
            return self.filtered_velocity
        instant = displacement / elapsed
        self.instant_velocity = instant
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
