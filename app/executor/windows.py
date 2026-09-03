from __future__ import annotations

import ctypes
import time
from ctypes import wintypes
from collections.abc import Callable
from typing import Protocol

from app.core.models import CommandAction, ControlCommand


class InputBackend(Protocol):
    def mouse_left(self, down: bool) -> None: ...
    def key(self, key: str, down: bool) -> None: ...


class _MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", wintypes.LONG),
        ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
    ]


class _KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", wintypes.WORD),
        ("wScan", wintypes.WORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
    ]


class _INPUTUNION(ctypes.Union):
    _fields_ = [("mi", _MOUSEINPUT), ("ki", _KEYBDINPUT)]


class _INPUT(ctypes.Structure):
    _anonymous_ = ("union",)
    _fields_ = [("type", wintypes.DWORD), ("union", _INPUTUNION)]


class WindowsSendInputBackend:
    """Ordinary Windows desktop input; no injection into another process."""

    INPUT_MOUSE = 0
    INPUT_KEYBOARD = 1
    MOUSEEVENTF_LEFTDOWN = 0x0002
    MOUSEEVENTF_LEFTUP = 0x0004
    KEYEVENTF_KEYUP = 0x0002
    KEYEVENTF_SCANCODE = 0x0008
    MAPVK_VK_TO_VSC = 0
    NAMED_KEYS = {
        "ENTER": 0x0D,
        "ESC": 0x1B,
        "SPACE": 0x20,
        "LEFT": 0x25,
        "UP": 0x26,
        "RIGHT": 0x27,
        "DOWN": 0x28,
    }

    def _send(self, item: _INPUT) -> None:
        sent = ctypes.windll.user32.SendInput(1, ctypes.byref(item), ctypes.sizeof(_INPUT))
        if sent != 1:
            raise ctypes.WinError()

    def mouse_left(self, down: bool) -> None:
        flag = self.MOUSEEVENTF_LEFTDOWN if down else self.MOUSEEVENTF_LEFTUP
        self._send(_INPUT(type=self.INPUT_MOUSE, mi=_MOUSEINPUT(dwFlags=flag)))

    def key(self, key: str, down: bool) -> None:
        named = self.NAMED_KEYS.get(key.upper())
        if named is not None:
            virtual_key = named
        elif len(key) == 1:
            packed = ctypes.windll.user32.VkKeyScanW(ord(key))
            if packed == -1:
                raise ValueError(f"Windows cannot map key {key!r}")
            virtual_key = packed & 0xFF
        else:
            raise ValueError(f"Unsupported key name: {key!r}")
        # Games commonly consume physical scan-code input rather than translated
        # virtual-key packets. This is still ordinary focused-window SendInput;
        # it does not communicate with or inspect the game process.
        scan_code = ctypes.windll.user32.MapVirtualKeyW(
            virtual_key, self.MAPVK_VK_TO_VSC
        )
        if scan_code == 0:
            raise ValueError(f"Windows cannot map key {key!r} to a scan code")
        flag = self.KEYEVENTF_SCANCODE
        if not down:
            flag |= self.KEYEVENTF_KEYUP
        self._send(
            _INPUT(
                type=self.INPUT_KEYBOARD,
                ki=_KEYBDINPUT(wVk=0, wScan=scan_code, dwFlags=flag),
            )
        )


class InterruptibleInputExecutor:
    """Applies only the newest stateless control intent.

    There are no sleeping movement transactions to become stale. HOLD means
    left mouse down and RELEASE means left mouse up. A normal dead-zone NEUTRAL
    preserves the current button state; detection loss releases it immediately.
    """

    def __init__(
        self,
        backend: InputBackend,
        enabled: bool = False,
        maximum_mouse_down_bpm: float | None = None,
        mouse_down_timing: str = "minimum_interval",
        mouse_down_beat_window_ms: float = 70.0,
        clock: Callable[[], float] = time.perf_counter,
    ):
        self.backend = backend
        self.input_enabled = enabled
        self.clock = clock
        self.minimum_mouse_down_interval_s = (
            60.0 / float(maximum_mouse_down_bpm)
            if maximum_mouse_down_bpm is not None and maximum_mouse_down_bpm > 0
            else 0.0
        )
        self.mouse_down_timing = mouse_down_timing
        self.mouse_down_beat_window_s = max(0.0, mouse_down_beat_window_ms / 1000.0)
        self.last_limited_mouse_down_s: float | None = None
        self.last_limited_mouse_transition_s: float | None = None
        self.mouse_down_beat_origin_s: float | None = None
        self.last_mouse_transition_beat = -1
        self.mouse_is_down = False
        self.held_keys: set[str] = set()
        self.last_generation = -1
        self.last_command: ControlCommand | None = None
        self.last_input_event = "none"
        self.input_event_count = 0

    def set_enabled(self, enabled: bool) -> None:
        if not enabled:
            self.emergency_release()
        self.input_enabled = enabled

    def submit(self, command: ControlCommand) -> None:
        self.last_command = command
        if not self.input_enabled or command.generation < self.last_generation:
            return
        self.last_generation = command.generation
        if command.action == CommandAction.HOLD:
            self._set_mouse(True, rate_limited=True)
        elif command.action == CommandAction.RELEASE:
            self._set_mouse(False, rate_limited=True)
        elif command.reason == "insufficient_detection":
            self._set_mouse(False, rate_limited=True)

    def tap_key(self, key: str) -> None:
        if not self.input_enabled:
            return
        self.backend.key(key, True)
        self.held_keys.add(key)
        self.backend.key(key, False)
        self.held_keys.discard(key)
        self._record_event(f"tap:{key.upper()}")

    def begin_cast(self) -> None:
        if self.input_enabled:
            self._set_mouse(True)
            self._record_event("cast:mouse_down")

    def begin_minigame_input_timing(self) -> None:
        """Start a fresh, fixed click grid for mechanics such as Requiem.

        The grid is anchored once at minigame entry. It is deliberately not
        restarted by individual controller requests, releases, or delayed
        frames, so a late HOLD cannot create an extra off-beat mouse-down.
        """
        if self.mouse_down_timing not in {
            "beat_grid",
            "minimum_transition_interval",
        }:
            return
        self.mouse_down_beat_origin_s = (
            self.clock() if self.mouse_down_timing == "beat_grid" else None
        )
        self.last_mouse_transition_beat = -1
        self.last_limited_mouse_down_s = None
        self.last_limited_mouse_transition_s = None

    def release_phase_inputs(self) -> None:
        if self.mouse_is_down:
            self.backend.mouse_left(False)
        self.mouse_is_down = False
        for key in tuple(self.held_keys):
            self.backend.key(key, False)
        self.held_keys.clear()

    def emergency_release(self) -> None:
        # Releases are emitted even if input was just disabled.
        self.release_phase_inputs()
        self.last_command = None

    def _set_mouse(self, down: bool, rate_limited: bool = False) -> None:
        if down == self.mouse_is_down:
            return
        now = self.clock()
        beat_index: int | None = None
        if rate_limited and self.minimum_mouse_down_interval_s > 0:
            if self.mouse_down_timing == "beat_grid":
                if self.mouse_down_beat_origin_s is None:
                    self.mouse_down_beat_origin_s = now
                elapsed = max(0.0, now - self.mouse_down_beat_origin_s)
                beat_index = int(elapsed // self.minimum_mouse_down_interval_s)
                phase = elapsed - beat_index * self.minimum_mouse_down_interval_s
                if (
                    beat_index <= self.last_mouse_transition_beat
                    or phase > self.mouse_down_beat_window_s
                    or (
                        self.last_limited_mouse_transition_s is not None
                        and now - self.last_limited_mouse_transition_s
                        < self.minimum_mouse_down_interval_s
                    )
                ):
                    return
            elif self.mouse_down_timing == "minimum_transition_interval":
                if (
                    self.last_limited_mouse_transition_s is not None
                    and now - self.last_limited_mouse_transition_s
                    < self.minimum_mouse_down_interval_s
                ):
                    return
            elif down and (
                self.last_limited_mouse_down_s is not None
                and now - self.last_limited_mouse_down_s
                < self.minimum_mouse_down_interval_s
            ):
                return
        self.backend.mouse_left(down)
        self.mouse_is_down = down
        if down and rate_limited:
            self.last_limited_mouse_down_s = now
        if rate_limited and (
            beat_index is not None
            or self.mouse_down_timing == "minimum_transition_interval"
        ):
            self.last_limited_mouse_transition_s = now
            if beat_index is not None:
                self.last_mouse_transition_beat = beat_index
        self._record_event("mouse:down" if down else "mouse:up")

    def _record_event(self, event: str) -> None:
        self.last_input_event = event
        self.input_event_count += 1
