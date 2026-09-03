from __future__ import annotations

from app.core.models import ControlCommand


class InputDisabledExecutor:
    """Milestone-one executor that records intent and emits no input."""

    input_enabled = False

    def __init__(self):
        self.last_command: ControlCommand | None = None
        self.released = True

    def submit(self, command: ControlCommand) -> None:
        self.last_command = command

    def begin_cast(self) -> None:
        pass

    def tap_key(self, key: str) -> None:
        pass

    def release_phase_inputs(self) -> None:
        self.emergency_release()

    def emergency_release(self) -> None:
        self.released = True
        self.last_command = None
