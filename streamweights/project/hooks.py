"""Deterministic test hooks. Product code calls `fire("name")` at the points where the
ownership and publication protocol can be interrupted; with no hook configured it is one
dictionary lookup. A test configures a hook through the environment, so it works across
processes:

    SPILL_HOOK_<name>=pause:<gate>[:<reached>[:<n>]]
                                                 on the n-th time the point is reached (default
                                                 1): touch <reached> (if given), then block
                                                 until the file <gate> exists
    SPILL_HOOK_<name>=crash                      os._exit(137) at that point (a killed process)
    SPILL_HOOK_<name>=crash:<n>                  the same, on the n-th time the point is reached
    SPILL_HOOK_<name>=lose                       the caller treats the operation's response as
                                                 lost (checked with `fault("name")`)
"""

from __future__ import annotations

import os
import time
from pathlib import Path

_COUNTS: dict = {}


def _spec(name: str) -> str | None:
    return os.environ.get(f"SPILL_HOOK_{name}")


def fire(name: str) -> None:
    spec = _spec(name)
    if not spec:
        return
    kind, _, rest = spec.partition(":")
    if kind == "pause":
        gate, _, more = rest.partition(":")
        reached, _, nth = more.partition(":")
        _COUNTS[name] = _COUNTS.get(name, 0) + 1
        if _COUNTS[name] != int(nth or 1):
            return
        if reached:
            Path(reached).write_text(str(os.getpid()))
        deadline = time.monotonic() + 120
        while not Path(gate).exists():
            if time.monotonic() > deadline:
                raise TimeoutError(f"hook {name}: gate {gate} never appeared")
            time.sleep(0.01)
    elif kind == "crash":
        n = int(rest) if rest else 1
        _COUNTS[name] = _COUNTS.get(name, 0) + 1
        if _COUNTS[name] >= n:
            os._exit(137)


def fault(name: str) -> bool:
    return _spec(name) == "lose"
