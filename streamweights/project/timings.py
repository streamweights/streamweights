"""Measured stage rates on this machine, kept so the next plan can say "measurement" instead
of "assumption". Plain JSON next to the calibration; written after a stage that ran start to
finish in one session, never by `spill plan`."""

from __future__ import annotations

import json
import time
from pathlib import Path

from ..registry import REPO_ROOT

PATH = REPO_ROOT / "state" / "timings.json"


def load() -> dict:
    try:
        return json.loads(PATH.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def record(engine: str, model: str, kind: str, units: float, seconds: float, unit_name: str) -> None:
    if units <= 0 or seconds <= 0:
        return
    t = load()
    t[f"{engine}|{model}|{kind}"] = {"seconds_per_unit": seconds / units, "unit": unit_name,
                                     "units": units, "seconds": seconds,
                                     "measured": time.strftime("%Y-%m-%d")}
    PATH.parent.mkdir(parents=True, exist_ok=True)
    PATH.write_text(json.dumps(t, indent=1, sort_keys=True) + "\n")


def lookup(engine: str, model: str, kind: str) -> dict | None:
    return load().get(f"{engine}|{model}|{kind}")
