from __future__ import annotations

import ctypes
from pathlib import Path


INTER_REGULAR = "Inter"
INTER_MEDIUM = "Inter Medium"
INTER_SEMIBOLD = "Inter SemiBold"
INTER_BOLD = "Inter"
MONOSPACE = "Cascadia Mono"

_FR_PRIVATE = 0x10
_loaded_paths: list[Path] = []


def load_bundled_fonts(project_root: Path) -> bool:
    """Register bundled Inter faces privately for this process only.

    This does not install a font system-wide and leaves the user's Windows font
    configuration untouched. Tk sees the faces for the lifetime of FischMate.
    """
    if not hasattr(ctypes, "windll"):
        return False
    font_dir = project_root / "assets" / "fonts" / "inter"
    expected = (
        "Inter-Regular.ttf",
        "Inter-Medium.ttf",
        "Inter-SemiBold.ttf",
        "Inter-Bold.ttf",
    )
    added = 0
    for filename in expected:
        path = font_dir / filename
        if not path.is_file():
            continue
        if ctypes.windll.gdi32.AddFontResourceExW(str(path.resolve()), _FR_PRIVATE, 0):
            _loaded_paths.append(path)
            added += 1
    return added == len(expected)


def regular(size: int):
    return (INTER_REGULAR, size)


def medium(size: int):
    return (INTER_MEDIUM, size)


def semibold(size: int):
    return (INTER_SEMIBOLD, size)


def bold(size: int):
    return (INTER_BOLD, size, "bold")


def monospace(size: int):
    return (MONOSPACE, size)
