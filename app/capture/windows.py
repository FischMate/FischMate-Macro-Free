from __future__ import annotations

import ctypes
from ctypes import wintypes
from dataclasses import dataclass

from app.core.models import PixelRect


user32 = ctypes.windll.user32 if hasattr(ctypes, "windll") else None


@dataclass(frozen=True)
class WindowInfo:
    hwnd: int
    title: str
    client_rect: PixelRect

    @property
    def label(self) -> str:
        return f"{self.title} — {self.client_rect.width}×{self.client_rect.height}"


def _client_rect(hwnd: int) -> PixelRect | None:
    if user32 is None:
        return None
    rect = wintypes.RECT()
    if not user32.GetClientRect(hwnd, ctypes.byref(rect)):
        return None
    top_left = wintypes.POINT(rect.left, rect.top)
    bottom_right = wintypes.POINT(rect.right, rect.bottom)
    if not user32.ClientToScreen(hwnd, ctypes.byref(top_left)):
        return None
    if not user32.ClientToScreen(hwnd, ctypes.byref(bottom_right)):
        return None
    return PixelRect(top_left.x, top_left.y, bottom_right.x, bottom_right.y)


def enumerate_roblox_windows(title_contains: str = "Roblox") -> list[WindowInfo]:
    if user32 is None:
        return []
    found: list[WindowInfo] = []
    callback_type = ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)

    def callback(hwnd: int, _lparam: int) -> bool:
        if not user32.IsWindowVisible(hwnd):
            return True
        length = user32.GetWindowTextLengthW(hwnd)
        if length <= 0:
            return True
        buffer = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buffer, length + 1)
        title = buffer.value
        if title_contains.casefold() not in title.casefold():
            return True
        rect = _client_rect(hwnd)
        if rect is not None and rect.width >= 640 and rect.height >= 360:
            found.append(WindowInfo(int(hwnd), title, rect))
        return True

    user32.EnumWindows(callback_type(callback), 0)
    return sorted(found, key=lambda item: item.client_rect.width * item.client_rect.height, reverse=True)


def refresh_window(window: WindowInfo) -> WindowInfo | None:
    rect = _client_rect(window.hwnd)
    if rect is None or rect.width <= 0 or rect.height <= 0:
        return None
    return WindowInfo(window.hwnd, window.title, rect)


def is_foreground_window(window: WindowInfo) -> bool:
    return bool(user32 is not None and int(user32.GetForegroundWindow()) == window.hwnd)


def activate_window(window: WindowInfo) -> bool:
    """Restore and focus the selected Roblox window after a user GUI action."""
    if user32 is None or not user32.IsWindow(window.hwnd):
        return False
    user32.ShowWindow(window.hwnd, 9)  # SW_RESTORE
    return bool(user32.SetForegroundWindow(window.hwnd))
