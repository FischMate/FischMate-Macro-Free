from __future__ import annotations

import configparser
import re
from pathlib import Path
from typing import Any


def _number(value: str) -> int | float | str:
    stripped = value.strip()
    try:
        parsed = float(stripped)
    except ValueError:
        return value
    return int(parsed) if parsed.is_integer() else parsed


_HEX_COLOR = re.compile(r"^0x[0-9a-fA-F]{6}$")


def _first(*values: str | None, default: str = "") -> str:
    for value in values:
        if value is not None and value.strip():
            return value.strip()
    return default


def _color(*values: str | None, default: str) -> str:
    candidate = _first(*values, default=default)
    return candidate.upper() if _HEX_COLOR.fullmatch(candidate) else default


def import_legacy_ini(path: Path) -> dict[str, Any]:
    """Import user-owned settings from a legacy INI into a profile override.

    This is data-format interoperability only. It intentionally does not
    translate legacy algorithms or preserve implicit rod-specific branches.
    Unsupported behavior must become an explicit independent mechanic module.
    """
    parser = configparser.ConfigParser(interpolation=None, strict=False)
    parser.optionxform = str
    raw = path.read_bytes()
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        text = raw.decode("utf-16")
    else:
        text = raw.decode("utf-8-sig", errors="replace")
    parser.read_string(text, source=str(path))
    shake = parser["Shake"] if parser.has_section("Shake") else {}
    minigame = parser["Minigame"] if parser.has_section("Minigame") else {}
    general = parser["General"] if parser.has_section("General") else {}
    others = parser["Others"] if parser.has_section("Others") else {}
    user = parser["User"] if parser.has_section("User") else {}
    resilience = _first(
        minigame.get("Resilience"),
        minigame.get("Resillience"),
        others.get("Resilience"),
        others.get("Resillience"),
        default="0",
    )
    bar = _color(others.get("BarColor"), minigame.get("BarColor"), default="0xFFFFFF")
    bar_secondary = _color(
        others.get("BarColor2"), minigame.get("BarColor2"), default="0x00FC43"
    )
    stick = _color(others.get("FishColor"), minigame.get("FishColor"), default="0x5B4B43")
    arrow = _color(others.get("ArrowColor"), minigame.get("ArrowColor"), default="0x878584")
    return {
        "casting": {
            "hold_ms": _number(general.get("HoldRodCastDuration", "600")),
        },
        "shake": {
            # FischMate currently uses its reliable navigation shaker for all
            # imported rods. Preserve the legacy preference below as metadata.
            "mode": "navigation",
            "navigation_key": _first(shake.get("NavigationKey"), user.get("NavigationKey"), default="\\"),
            "navigation_interval_ms": _number(shake.get("NavigationSpamDelay", "10")),
        },
        "rod": {
            "control": _number(minigame.get("Control", "0")),
            "resilience_percent": _number(resilience),
        },
        "detection": {
            "colors": {
                "bar": bar,
                "bar_secondary": bar_secondary,
                "stick": stick,
                "arrow": arrow,
            },
            "tolerance": {
                "bar": _number(minigame.get("WhiteBarColorTolerance", "15")),
                "stick": _number(minigame.get("FishBarColorTolerance", "5")),
                "arrow": _number(minigame.get("ArrowColorTolerance", "6")),
            },
        },
        "loop": {
            "restart_delay_ms": _number(general.get("RestartDelay", "1500")),
        },
        "legacy_reference": {
            "source_ini": path.name,
            "legacy_shake_mode": shake.get("ShakeMode", "Navigation"),
            "legacy_shake_failsafe_s": _number(shake.get("ShakeFailsafe", "45")),
            "scan_delay_ms": _number(minigame.get("ScanDelay", "10")),
        },
    }
