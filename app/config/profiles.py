from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any


CURRENT_SCHEMA_VERSION = 1
KNOWN_MECHANICS = {
    "standard_tracking",
    "dynamic_bar_size",
    "color_transitions",
    "temporary_occlusion",
    "trigger_tiles",
    "falling_objects",
    "streak_mode",
    "automatic_stick_following",
    "multiple_targets",
    "rate_limited_clicking",
    "rail_end_latching",
    "rhythm_four_lane",
}


class ProfileError(ValueError):
    pass


def _load_yaml_compatible(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore
    except ImportError:
        try:
            value = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ProfileError(
                f"{path.name} needs PyYAML for non-JSON YAML syntax: {exc}"
            ) from exc
    else:
        value = yaml.safe_load(text)
    if not isinstance(value, dict):
        raise ProfileError(f"{path.name} must contain a mapping/object")
    return value


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if key == "extends":
            continue
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


class ProfileRepository:
    def __init__(self, profiles_root: Path):
        self.root = profiles_root

    def names(self) -> list[str]:
        if not self.root.exists():
            return []
        return sorted(
            path.parent.name
            for path in self.root.glob("*/profile.yaml")
            if path.is_file()
        )

    def path_for(self, name: str) -> Path:
        return self.root / name / "profile.yaml"

    def load(self, name: str, _stack: tuple[str, ...] = ()) -> dict[str, Any]:
        if name in _stack:
            chain = " -> ".join((*_stack, name))
            raise ProfileError(f"Profile inheritance cycle: {chain}")
        path = self.path_for(name)
        if not path.exists():
            raise ProfileError(f"Profile does not exist: {name}")
        raw = _load_yaml_compatible(path)
        parent = raw.get("extends")
        profile = raw
        if parent:
            if not isinstance(parent, str):
                raise ProfileError("extends must be a profile name")
            profile = _deep_merge(self.load(parent, (*_stack, name)), raw)
        self.validate(profile)
        profile["_profile_name"] = name
        profile["_profile_path"] = str(path)
        return profile

    def save_user_settings(self, name: str, updates: dict[str, Any]) -> None:
        """Persist GUI-edited settings in the selected profile.

        We preserve YAML when PyYAML is available and otherwise emit formatted
        JSON, which remains valid YAML.
        """
        path = self.path_for(name)
        raw = _load_yaml_compatible(path)
        merged = _deep_merge(raw, updates)
        self.validate(_deep_merge(self.load_parent(raw), merged))
        try:
            import yaml  # type: ignore
        except ImportError:
            path.write_text(json.dumps(merged, indent=2) + "\n", encoding="utf-8")
        else:
            path.write_text(yaml.safe_dump(merged, sort_keys=False), encoding="utf-8")

    def load_parent(self, raw: dict[str, Any]) -> dict[str, Any]:
        parent = raw.get("extends")
        return self.load(parent) if isinstance(parent, str) else {}

    @staticmethod
    def validate(profile: dict[str, Any]) -> None:
        version = profile.get("schema_version")
        if version != CURRENT_SCHEMA_VERSION:
            raise ProfileError(
                f"Unsupported schema_version {version!r}; expected {CURRENT_SCHEMA_VERSION}"
            )
        metadata = profile.get("profile")
        if not isinstance(metadata, dict) or not metadata.get("display_name"):
            raise ProfileError("profile.display_name is required")
        navigation_key = profile.get("shake", {}).get("navigation_key")
        if not isinstance(navigation_key, str) or not navigation_key:
            raise ProfileError("shake.navigation_key must be a non-empty string")
        enabled = profile.get("mechanics", {}).get("enabled", [])
        if not isinstance(enabled, list) or not all(isinstance(item, str) for item in enabled):
            raise ProfileError("mechanics.enabled must be a list of names")
        unknown = sorted(set(enabled) - KNOWN_MECHANICS)
        if unknown:
            raise ProfileError(f"Unknown mechanics: {', '.join(unknown)}")
        if "streak_mode" in enabled and "falling_objects" not in enabled:
            raise ProfileError("streak_mode requires falling_objects")
        if (
            "automatic_stick_following" in enabled
            and profile.get("controller", {}).get("strategy") == "center_stick"
        ):
            raise ProfileError(
                "automatic_stick_following cannot use center_stick as its only strategy"
            )
        if "rate_limited_clicking" in enabled:
            input_config = profile.get("input", {})
            maximum_bpm = input_config.get(
                "maximum_mouse_transition_bpm",
                input_config.get("maximum_mouse_down_bpm"),
            )
            if not isinstance(maximum_bpm, (int, float)) or maximum_bpm <= 0:
                raise ProfileError(
                    "rate_limited_clicking requires a positive maximum mouse BPM"
                )
            timing = input_config.get(
                "mouse_down_timing", "minimum_interval"
            )
            if timing not in {
                "minimum_interval",
                "minimum_transition_interval",
                "beat_grid",
            }:
                raise ProfileError(
                    "input.mouse_down_timing must be minimum_interval, "
                    "minimum_transition_interval, or beat_grid"
                )
            beat_window = profile.get("input", {}).get(
                "mouse_down_beat_window_ms", 70
            )
            if not isinstance(beat_window, (int, float)) or beat_window <= 0:
                raise ProfileError(
                    "input.mouse_down_beat_window_ms must be greater than zero"
                )
        if "rail_end_latching" in enabled:
            latch = profile.get("controller", {}).get("rail_end_latch", {})
            enter_ratio = latch.get("enter_ratio")
            exit_ratio = latch.get("exit_ratio")
            if not all(
                isinstance(value, (int, float))
                for value in (enter_ratio, exit_ratio)
            ) or not (0 < enter_ratio < exit_ratio < 0.5):
                raise ProfileError(
                    "rail_end_latching requires 0 < enter_ratio < exit_ratio < 0.5"
                )
            boundary_ratio = latch.get("boundary_margin_ratio", 0.025)
            opposite_ratio = latch.get("opposite_edge_release_ratio", 0.30)
            if not all(
                isinstance(value, (int, float))
                for value in (boundary_ratio, opposite_ratio)
            ) or not (0 < boundary_ratio < 0.2 and 0 < opposite_ratio < 0.5):
                raise ProfileError(
                    "rail_end_latching boundary and opposite-edge ratios are invalid"
                )
