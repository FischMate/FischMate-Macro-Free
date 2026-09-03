from __future__ import annotations

from app.core.models import CommandAction, ControlCommand, DetectionSnapshot


class SpecialStandardController:
    """Pure decision layer: observations in, command intent out."""

    def __init__(self, profile: dict):
        self.profile = profile
        self.generation = 0
        self.previous_action = CommandAction.NEUTRAL
        self.previous_bar_center: float | None = None
        self.previous_timestamp_ms: float | None = None
        self.filtered_velocity = 0.0
        self.last_action_change_ms: float | None = None
        self.last_complete_detection_ms: float | None = None
        self.rail_end_latch: str | None = None
        self.rail_end_latch_suppressed: str | None = None
        self.arrow_fallback_side: str | None = None
        self.ruinous_smallest_trusted_width: float | None = None
        self.ruinous_recovery_click_frame = 0

    def reset(self) -> None:
        self.generation += 1
        self.previous_action = CommandAction.NEUTRAL
        self.previous_bar_center = None
        self.previous_timestamp_ms = None
        self.filtered_velocity = 0.0
        self.last_action_change_ms = None
        self.last_complete_detection_ms = None
        self.rail_end_latch = None
        self.rail_end_latch_suppressed = None
        self.arrow_fallback_side = None
        self.ruinous_smallest_trusted_width = None
        self.ruinous_recovery_click_frame = 0

    def decide(self, observation: DetectionSnapshot) -> ControlCommand:
        bar = observation.bar
        stick = observation.stick
        tuning = self._effective_tuning(observation)
        fallback_arrow_x = observation.extra.get("bar_fallback_arrow_x")
        if bar is not None:
            self.arrow_fallback_side = None
        arrow_fallback = self.profile["controller"].get("arrow_fallback", {})
        arrow_fallback_allowed = bool(arrow_fallback.get("enabled", True))
        arrow_fallback_age_ms: float | None = None
        if fallback_arrow_x is not None and bar is None:
            if self.last_complete_detection_ms is not None:
                arrow_fallback_age_ms = max(
                    0.0,
                    observation.timestamp_ms - self.last_complete_detection_ms,
                )
            if (
                bool(arrow_fallback.get("requires_recent_bar_lock", False))
                and self.last_complete_detection_ms is None
            ):
                arrow_fallback_allowed = False
            maximum_duration_ms = arrow_fallback.get("maximum_duration_ms")
            if (
                maximum_duration_ms is not None
                and (
                    arrow_fallback_age_ms is None
                    or arrow_fallback_age_ms > float(maximum_duration_ms)
                )
            ):
                arrow_fallback_allowed = False
            observation.extra["arrow_fallback_age_ms"] = (
                "" if arrow_fallback_age_ms is None else round(arrow_fallback_age_ms, 3)
            )
            observation.extra["arrow_fallback_allowed"] = arrow_fallback_allowed
            if not arrow_fallback_allowed:
                fallback_arrow_x = None
                self.arrow_fallback_side = None
        learned_bar_width = observation.extra.get("learned_bar_width")
        partial_hold_ms = float(
            self.profile["controller"].get("partial_detection_hold_ms", 0)
        )
        if (
            partial_hold_ms > 0
            and (bar is None or stick is None)
            and self.previous_action != CommandAction.NEUTRAL
            and self.last_complete_detection_ms is not None
            and observation.timestamp_ms - self.last_complete_detection_ms
            <= partial_hold_ms
        ):
            action = self.previous_action
            reason = "hold_last_action_during_partial_detection"
            target_x = stick.center_x if stick is not None else None
            error = None
            confidence = min(observation.bar_confidence, observation.stick_confidence)
        elif (
            partial_hold_ms > 0
            and bar is None
            and bool(arrow_fallback.get("disabled_during_partial_hold", False))
        ):
            action = CommandAction.RELEASE
            reason = "release_after_partial_detection_timeout"
            target_x = stick.center_x if stick is not None else None
            error = None
            confidence = min(observation.bar_confidence, observation.stick_confidence)
        elif (
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
            control_width = tuning["control_width"]
            deadzone = max(2.0, control_width * tuning["deadzone_ratio"])
            if rail_end_action is not None:
                action, reason = rail_end_action
                predicted_error = error
            elif (
                tuning["active_tracking_error_ratio"] > 0
                and abs(error)
                <= control_width * tuning["active_tracking_error_ratio"]
            ):
                # Ruinous loses usable control area as the visible body shrinks.
                # While the stick is still safely inside that live body, keep
                # issuing micro-click transitions instead of settling into a
                # long predictive HOLD/RELEASE run.  This executes at the
                # fastest cadence available from the current capture loop and
                # costs no additional detection pass.
                predicted_error = error
                pulse_ms = tuning["center_pulse_ms"]
                elapsed_action_ms = self._elapsed_action_ms(
                    observation.timestamp_ms
                )
                if (
                    self.previous_action != CommandAction.NEUTRAL
                    and elapsed_action_ms < pulse_ms
                ):
                    action = self.previous_action
                    reason = "ruinous_continue_live_width_micro_click"
                elif self.previous_action == CommandAction.HOLD:
                    action = CommandAction.RELEASE
                    reason = "ruinous_live_width_micro_release"
                else:
                    action = CommandAction.HOLD
                    reason = "ruinous_live_width_micro_hold"
            else:
                velocity = self._update_velocity(
                    bar.center_x,
                    observation.timestamp_ms,
                    control_width,
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
                        pulse_ms = tuning["center_pulse_ms"]
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

                recovery_response = tuning["width_response"]
                recovery_minimum_response = tuning["recovery_click_minimum_response"]
                if (
                    recovery_response >= recovery_minimum_response
                    and tuning["recovery_click_cycle_frames"] > 1
                ):
                    cycle_frames = int(tuning["recovery_click_cycle_frames"])
                    counter_frames = min(
                        cycle_frames - 1,
                        int(tuning["recovery_click_counter_frames"]),
                    )
                    desired_action = action
                    desired_frames = cycle_frames - counter_frames
                    phase = self.ruinous_recovery_click_frame % cycle_frames
                    self.ruinous_recovery_click_frame += 1
                    if phase >= desired_frames:
                        action = (
                            CommandAction.RELEASE
                            if desired_action == CommandAction.HOLD
                            else CommandAction.HOLD
                        )
                        reason = "ruinous_low_control_recovery_counter_click"
                    else:
                        reason = "ruinous_low_control_recovery_drive_click"
                else:
                    self.ruinous_recovery_click_frame = 0
            confidence = min(observation.bar_confidence, observation.stick_confidence)
            self.last_complete_detection_ms = observation.timestamp_ms
            minimum_reversal_ms = tuning["minimum_reversal_interval_ms"]
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

    def _effective_tuning(self, observation: DetectionSnapshot) -> dict[str, float]:
        """Return per-frame tuning for explicitly opted-in special rods.

        Ruinous Oath permanently loses control width as its lock advances.  Its
        detector can restore a short red body's *position* to a wider rectangle
        when one edge is visually obscured, while still exposing the current
        red body's raw width.  Keep those two facts separate: steering geometry
        continues to use ``observation.bar``, but click cadence and centering
        tightness follow the trusted current-frame width.

        Profiles without ``width_responsive_centering.enabled`` take the exact
        pre-existing path and values.
        """
        controller = self.profile["controller"]
        result = {
            "control_width": float(observation.bar.width)
            if observation.bar is not None
            else 0.0,
            "deadzone_ratio": float(controller["deadzone_ratio"]),
            "center_pulse_ms": float(controller.get("center_pulse_ms", 40)),
            "minimum_reversal_interval_ms": float(
                controller.get("minimum_reversal_interval_ms", 28)
            ),
            "active_tracking_error_ratio": 0.0,
            "width_response": 0.0,
            "recovery_click_minimum_response": 1.1,
            "recovery_click_cycle_frames": 0.0,
            "recovery_click_counter_frames": 0.0,
        }
        config = controller.get("width_responsive_centering", {})
        if not bool(config.get("enabled", False)) or observation.bar is None:
            return result

        learned_width = observation.extra.get("learned_bar_width")
        if learned_width is None or float(learned_width) <= 0:
            return result

        live_width = float(observation.bar.width)
        raw_width = observation.extra.get("raw_bar_width")
        trust_state = observation.extra.get("bar_candidate_trust_state")
        color_source = str(observation.extra.get("bar_color_source", ""))
        allowed_raw_sources = set(
            config.get(
                "trusted_raw_color_sources",
                (
                    "ruinous_red_lock_state",
                    "ruinous_pale_red_transition_state",
                    "ruinous_dim_red_lock_state",
                ),
            )
        )
        if (
            raw_width is not None
            and trust_state == "trusted"
            and color_source in allowed_raw_sources
        ):
            # A restored rectangle is useful for center continuity, but its
            # width must not hide the real red shrink from Ruinous's cadence.
            live_width = min(live_width, float(raw_width))
            if bool(config.get("persist_smallest_trusted_width", False)):
                if self.ruinous_smallest_trusted_width is None:
                    self.ruinous_smallest_trusted_width = live_width
                else:
                    self.ruinous_smallest_trusted_width = min(
                        self.ruinous_smallest_trusted_width,
                        live_width,
                    )
        if (
            bool(config.get("persist_smallest_trusted_width", False))
            and self.ruinous_smallest_trusted_width is not None
        ):
            live_width = min(live_width, self.ruinous_smallest_trusted_width)
        result["control_width"] = live_width

        nominal_width = float(learned_width)
        width_ratio = max(0.0, min(1.0, live_width / nominal_width))
        activation_ratio = float(config.get("activation_width_ratio", 0.90))
        full_effect_ratio = float(config.get("full_effect_width_ratio", 0.45))
        if activation_ratio <= full_effect_ratio:
            return result
        response = max(
            0.0,
            min(
                1.0,
                (activation_ratio - width_ratio)
                / (activation_ratio - full_effect_ratio),
            ),
        )

        def interpolate(base: float, target_key: str) -> float:
            target = float(config.get(target_key, base))
            return base + (target - base) * response

        result["deadzone_ratio"] = interpolate(
            result["deadzone_ratio"], "minimum_deadzone_ratio"
        )
        result["center_pulse_ms"] = interpolate(
            result["center_pulse_ms"], "minimum_center_pulse_ms"
        )
        result["minimum_reversal_interval_ms"] = interpolate(
            result["minimum_reversal_interval_ms"],
            "minimum_reversal_interval_ms",
        )
        result["active_tracking_error_ratio"] = response * float(
            config.get("maximum_active_tracking_error_ratio", 0.0)
        )
        result["width_response"] = response
        result["recovery_click_minimum_response"] = float(
            config.get("recovery_click_minimum_response", 1.1)
        )
        result["recovery_click_cycle_frames"] = float(
            config.get("recovery_click_cycle_frames", 0)
        )
        result["recovery_click_counter_frames"] = float(
            config.get("recovery_click_counter_frames", 0)
        )
        observation.extra["ruinous_control_live_width"] = round(live_width, 3)
        observation.extra["ruinous_control_width_ratio"] = round(width_ratio, 4)
        observation.extra["ruinous_width_response"] = round(response, 4)
        observation.extra["ruinous_smallest_trusted_width"] = (
            round(self.ruinous_smallest_trusted_width, 3)
            if self.ruinous_smallest_trusted_width is not None
            else None
        )
        observation.extra["ruinous_effective_deadzone_ratio"] = round(
            result["deadzone_ratio"], 5
        )
        observation.extra["ruinous_effective_center_pulse_ms"] = round(
            result["center_pulse_ms"], 3
        )
        observation.extra["ruinous_effective_minimum_reversal_ms"] = round(
            result["minimum_reversal_interval_ms"], 3
        )
        observation.extra["ruinous_active_tracking_error_ratio"] = round(
            result["active_tracking_error_ratio"], 4
        )
        return result

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
        maximum_speed = self.profile["controller"].get(
            "velocity_max_speed_px_per_ms"
        )
        if maximum_speed is not None:
            # Pinion's Aria can move far enough during one slow capture frame to
            # exceed a fixed displacement gate. Scale the allowance by elapsed
            # time so an inference/capture spike does not erase real momentum.
            maximum_jump = max(maximum_jump, elapsed * float(maximum_speed))
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
