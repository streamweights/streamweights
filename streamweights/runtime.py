"""The environment one job command runs in: which engine, where the portable state lives,
where weights are staged from, and whether the run is headless (JSON-lines events, SIGTERM
handling, exit code 75). The CLI commands that run jobs (run, distill, tune, eval, resume)
open one `job_session`; everything below it reads `ENV`.
"""

from __future__ import annotations

import contextlib
import hashlib
import re
import sys
import threading
import time
from dataclasses import dataclass, field

from . import headless
from .errors import SpillError, StageInterrupted


@dataclass
class JobEnv:
    state: str | None = None           # portable state URI (path, s3://, gs://, az://, memory://)
    weights: str | None = None         # stage weights from this URI instead of Hugging Face
    engine: str | None = None          # mlx | torch-cpu | torch-cuda (None: chosen)
    headless: bool = False
    stop_after: int | None = None      # stop cleanly after this many rows or steps
    events: headless.Events | None = None
    stop: threading.Event = field(default_factory=threading.Event)
    guard: headless.PreemptGuard | None = None
    state_leaf: str | None = None      # a sub-location of `state` (one per model in an eval)
    stopped_early: bool = False        # --stop-after was reached
    summary: dict = field(default_factory=dict)


ENV = JobEnv()


def reset() -> None:
    global ENV
    ENV = JobEnv()


def state_uri(leaf: str | None = None) -> str | None:
    """The state location for the job being run now (with an eval's per-model leaf)."""
    if not ENV.state:
        return None
    leaf = leaf or ENV.state_leaf
    return f"{ENV.state.rstrip('/')}/{leaf}" if leaf else ENV.state


def slug(text: str) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-")[:48] or "job"
    return f"{s}-{hashlib.sha256(text.encode()).hexdigest()[:6]}"


def emit(event: str, **fields) -> None:
    """A headless event, when headless; a no-op otherwise."""
    if ENV.events is not None:
        ENV.events.emit(event, **fields)


def set_engine(name: str) -> None:
    if ENV.events is not None:
        ENV.events.engine = name


@contextlib.contextmanager
def job_session(command: str, *, state: str | None = None, weights: str | None = None,
                engine: str | None = None, headless_flag: bool = False,
                stop_after: int | None = None):
    try:
        with _session(command, state, weights, engine, headless_flag, stop_after) as env:
            yield env
    finally:
        reset()          # a later command in this process starts from a clean environment


@contextlib.contextmanager
def _session(command, state, weights, engine, headless_flag, stop_after):
    """Open the session for one job command. On exit: a preempted job (SIGTERM, SIGINT)
    emits `preempted` and exits 75; an error emits `error`; success emits `done` unless the
    command already did."""
    import typer
    reset()
    ENV.state, ENV.weights, ENV.engine = state, weights, engine
    ENV.stop_after = stop_after
    ENV.headless = headless.is_headless(headless_flag)
    t0 = time.monotonic()
    with headless.headless_session(ENV.headless, command) as ev:
        ENV.events = ev
        if ev is None:
            yield ENV
            return
        guard = headless.PreemptGuard(ENV.stop)
        ENV.guard = guard
        done_emitted = False
        try:
            with guard:
                yield ENV
        except headless.QuantumAbandoned:
            pass
        except typer.Exit:
            raise
        except SystemExit:
            raise
        except BaseException as e:
            if guard.signalled and isinstance(e, StageInterrupted):
                pass                                   # a stage stopped because we were told to
            elif ENV.stopped_early and isinstance(e, StageInterrupted):
                ev.emit("done", complete=False, stopped_early=True, reason="--stop-after",
                        seconds=round(time.monotonic() - t0, 2), **ENV.summary)
                return
            else:
                msg = e.line() if isinstance(e, SpillError) else f"{type(e).__name__}: {e}"
                ev.emit("error", message=msg)
                raise
        if guard.signalled or guard.abandoned:
            ev.emit("preempted", signal=guard.signalled, abandoned_quantum=guard.abandoned,
                    seconds=round(time.monotonic() - t0, 2), **ENV.summary)
            raise typer.Exit(headless.EXIT_PREEMPTED)
        if not done_emitted:
            ev.emit("done", complete=not ENV.stopped_early, stopped_early=ENV.stopped_early,
                    seconds=round(time.monotonic() - t0, 2), **ENV.summary)
