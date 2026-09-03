from __future__ import annotations

import ctypes
import threading
import time
from collections.abc import Callable


class GlobalHotkeyWatcher:
    """Edge-triggered Windows hotkeys without keyboard hooks.

    GetAsyncKeyState is deliberately used instead of installing a global hook.
    Callbacks are handed to the GUI scheduler and never run on this polling
    thread.
    """

    VK_BY_NAME = {"P": 0x50, "O": 0x4F, "M": 0x4D, "Y": 0x59}

    def __init__(
        self,
        callbacks: dict[str, Callable[[], None]],
        dispatch: Callable[[Callable[[], None]], None],
        poll_s: float = 0.025,
        immediate: set[str] | None = None,
    ):
        self.callbacks = {name.upper(): callback for name, callback in callbacks.items()}
        self.dispatch = dispatch
        self.poll_s = poll_s
        self.immediate = {name.upper() for name in (immediate or set())}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._pressed = {name: False for name in self.callbacks}

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="fischmate-hotkeys", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread is not threading.current_thread():
            self._thread.join(timeout=0.5)

    def _run(self) -> None:
        if not hasattr(ctypes, "windll"):
            return
        get_state = ctypes.windll.user32.GetAsyncKeyState
        while not self._stop.wait(self.poll_s):
            for name, callback in self.callbacks.items():
                down = bool(get_state(self.VK_BY_NAME[name]) & 0x8000)
                if down and not self._pressed[name]:
                    if name in self.immediate:
                        callback()
                    else:
                        self.dispatch(callback)
                self._pressed[name] = down
