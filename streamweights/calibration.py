"""Measured rates on this machine (state/calibration.json): disk stream, memory model,
achieved TFLOP/s. No MLX import, so estimates work on every platform."""

from __future__ import annotations

import json

from .registry import REPO_ROOT

CALIBRATION_JSON = REPO_ROOT / "state" / "calibration.json"


def load_calibration() -> dict:
    if CALIBRATION_JSON.exists():
        try:
            return json.loads(CALIBRATION_JSON.read_text())
        except json.JSONDecodeError:
            pass
    return {}


def save_calibration(cal: dict):
    CALIBRATION_JSON.parent.mkdir(exist_ok=True)
    CALIBRATION_JSON.write_text(json.dumps(cal, indent=2) + "\n")
