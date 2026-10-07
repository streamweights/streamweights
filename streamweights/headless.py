"""Headless mode: a job that runs under a scheduler instead of a person.

Enabled with --headless, or automatically when stdout is not a TTY (SPILL_HEADLESS=1 or 0
forces it either way). Then:

  * stdout carries JSON lines and nothing else: start, step or row, checkpoint, preempted,
    done, error. Every event carries step, loss, tokens_per_s, peak_mem_gb, eta_s and engine
    (null where a field does not apply). Human text goes to stderr.
  * no progress bars, no notifications, no caffeinate.
  * SIGTERM or SIGINT: finish the current quantum if it ends soon, otherwise abandon it,
    write a checkpoint within the grace period (30 s), emit `preempted`, and exit 75 so a
    scheduler retries the job. A retry resumes from --state.
"""

from __future__ import annotations

import contextlib
import json
import os
import signal
import sys
import threading
import time

EXIT_PREEMPTED = 75
GRACE_SECONDS = 30.0
# how long the current quantum may keep running after the signal before it is abandoned;
# the rest of the grace period is for writing the checkpoint
ABANDON_AFTER_SECONDS = 15.0

EVENT_FIELDS = ("step", "loss", "tokens_per_s", "peak_mem_gb", "eta_s", "engine")


class QuantumAbandoned(BaseException):
    """Raised in the main thread when the current quantum would not finish inside the grace
    period. Derived from BaseException so a stray `except Exception` cannot swallow it."""


def is_headless(flag: bool = False) -> bool:
    env = os.environ.get("SPILL_HEADLESS", "").strip().lower()
    if env in ("0", "false", "no"):
        return flag
    if env in ("1", "true", "yes"):
        return True
    if flag:
        return True
    try:
        return not sys.stdout.isatty()
    except (AttributeError, ValueError):
        return True


def peak_mem_gb() -> float | None:
    """Peak memory of this process on the device that does the work, in GB."""
    try:
        import torch
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            return round(torch.cuda.max_memory_allocated() / 1024**3, 3)
    except Exception:
        pass
    try:
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return round(peak / (1024**3 if sys.platform == "darwin" else 1024**2), 3)
    except Exception:
        return None


class Events:
    """JSON-lines writer. `engine` is the name stamped on every event."""

    def __init__(self, stream=None, engine: str | None = None, command: str | None = None):
        self.stream = stream if stream is not None else sys.stdout
        self.engine = engine
        self.command = command
        self._lock = threading.Lock()

    def emit(self, event: str, **fields) -> dict:
        rec = {"event": event, "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
               "command": self.command}
        for k in EVENT_FIELDS:
            rec[k] = None
        rec["engine"] = self.engine
        rec.update({k: v for k, v in fields.items() if v is not None or k in EVENT_FIELDS})
        with self._lock:
            self.stream.write(json.dumps(rec, default=str) + "\n")
            self.stream.flush()
        return rec

    def step(self, info: dict) -> None:
        """A tune step: `info` is the dict the trainer's progress callback gets."""
        self.emit("step", step=info["step"], steps=info.get("steps"), loss=info.get("loss"),
                  tokens_per_s=round(info.get("tok_s", 0.0), 2),
                  peak_mem_gb=info.get("peak_gb") or peak_mem_gb(),
                  eta_s=round(info["eta_s"], 1) if info.get("eta_s") is not None else None,
                  step_s=round(info.get("step_s", 0.0), 4))

    def row(self, done: int, total: int, tokens_per_s: float, eta_s: float | None,
            custom_id: str | None = None) -> None:
        self.emit("row", step=done, total=total, loss=None,
                  tokens_per_s=round(tokens_per_s, 2), peak_mem_gb=peak_mem_gb(),
                  eta_s=round(eta_s, 1) if eta_s else None, custom_id=custom_id)


@contextlib.contextmanager
def headless_session(enabled: bool, command: str, engine: str | None = None):
    """While active, human text (typer.echo, prints) goes to stderr and the yielded Events
    own stdout. When not enabled the yield is None and nothing changes."""
    if not enabled:
        yield None
        return
    real = sys.stdout
    ev = Events(real, engine=engine, command=command)
    sys.stdout = sys.stderr
    try:
        yield ev
    finally:
        sys.stdout = real


_ACTIVE: "PreemptGuard | None" = None


@contextlib.contextmanager
def protected():
    """Hold off abandoning the current quantum while a short step that must be atomic runs
    (the optimizer update, a checkpoint write). A request that arrived meanwhile is honored
    the moment the block ends."""
    g = _ACTIVE
    if g is None:
        yield
        return
    g.held += 1
    try:
        yield
    finally:
        g.held -= 1
        if g.held == 0 and g.pending:
            g.pending = False
            g.abandoned = True
            raise QuantumAbandoned()


class PreemptGuard:
    """Turns SIGTERM and SIGINT into a stop request, and abandons the current quantum if it
    has not finished ABANDON_AFTER_SECONDS later (SIGALRM, where the platform has it).

    `stop` is the event the job loop already polls. After the loop returns, `signalled` names
    the signal (or None), and the caller writes its checkpoint and exits EXIT_PREEMPTED."""

    def __init__(self, stop: threading.Event, abandon_after: float = ABANDON_AFTER_SECONDS):
        self.stop = stop
        self.abandon_after = abandon_after
        self.signalled: str | None = None
        self.abandoned = False
        self.held = 0
        self.pending = False
        self._prev: dict = {}

    def _on_term(self, signum, frame):
        if self.signalled is None:
            self.signalled = signal.Signals(signum).name
            self.stop.set()
            if hasattr(signal, "SIGALRM"):
                signal.setitimer(signal.ITIMER_REAL, self.abandon_after)
        else:                           # a second signal: abandon now
            self._on_alarm(signum, frame)

    def _on_alarm(self, signum, frame):
        if self.held:
            self.pending = True
            return
        self.abandoned = True
        raise QuantumAbandoned()

    def __enter__(self):
        global _ACTIVE
        _ACTIVE = self
        for sg in (signal.SIGINT, signal.SIGTERM):
            try:
                self._prev[sg] = signal.signal(sg, self._on_term)
            except ValueError:               # not the main thread
                pass
        if hasattr(signal, "SIGALRM"):
            try:
                self._prev[signal.SIGALRM] = signal.signal(signal.SIGALRM, self._on_alarm)
            except ValueError:
                pass
        return self

    def __exit__(self, *exc):
        global _ACTIVE
        _ACTIVE = None
        if hasattr(signal, "setitimer"):
            with contextlib.suppress(Exception):
                signal.setitimer(signal.ITIMER_REAL, 0)
        for sg, h in self._prev.items():
            with contextlib.suppress(ValueError):
                signal.signal(sg, h)
        self._prev.clear()
        return False

    def disarm(self) -> None:
        """The quantum finished in time; the checkpoint write must not be interrupted."""
        if hasattr(signal, "setitimer"):
            with contextlib.suppress(Exception):
                signal.setitimer(signal.ITIMER_REAL, 0)
