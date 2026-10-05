"""One-line errors, each ending with the recovery command. Tracebacks only with --debug."""

from __future__ import annotations


class SpillError(Exception):
    """One printable line; .recovery is the command that fixes it."""

    def __init__(self, message: str, recovery: str | None = None):
        self.message = message
        self.recovery = recovery
        super().__init__(message)

    def line(self, default: str | None = None) -> str:
        """One line: the message, then the command that gets you unstuck."""
        rec = self.recovery or default
        msg = self.message.rstrip(". ")
        return f"{msg}. Try: {rec}" if rec else msg


class StageInterrupted(SpillError):
    """A job inside a larger command (build, eval) stopped before finishing; its
    completed rows are checkpointed under `job_id`."""

    def __init__(self, job_id: str, done: int, total: int, what: str = "job"):
        self.job_id, self.done, self.total = job_id, done, total
        super().__init__(f"{what} {job_id} stopped at {done}/{total} rows; completed rows are "
                         f"checkpointed", f"spill resume {job_id}")
