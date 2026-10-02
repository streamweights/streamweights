"""One-line errors, each ending with the recovery command. Tracebacks only with --debug."""

from __future__ import annotations


class SpillError(Exception):
    """One printable line; .recovery is the command that fixes it."""

    def __init__(self, message: str, recovery: str | None = None):
        self.message = message
        self.recovery = recovery
        super().__init__(message)

    def line(self) -> str:
        return f"{self.message}" + (f" — try: {self.recovery}" if self.recovery else "")
