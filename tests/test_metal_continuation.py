"""Opt-in: SPILL_METAL_TESTS=1 on an Apple silicon Mac runs scripts/gate_metal_continuation.py,
the same checks that produced docs/reports/data/016-metal.json. It runs as a subprocess because the
suite pins MLX to the CPU device and the gate needs the GPU. SPILL_METAL_S3=s3://bucket/prefix adds
the object-store moves (AWS_* variables for the endpoint)."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(os.environ.get("SPILL_METAL_TESTS") != "1",
                                reason="set SPILL_METAL_TESTS=1 on an Apple silicon Mac (MLX GPU)")


def test_continuation_both_directions_on_metal(tmp_path):
    env = {k: v for k, v in os.environ.items() if k != "SPILL_DEVICE"}
    env["SPILL_HOME"] = str(tmp_path / "unused-home")
    cmd = [sys.executable, str(Path(__file__).resolve().parent.parent / "scripts" / "gate_metal_continuation.py"),
           "--out", str(tmp_path / "metal.json"), "--work", str(tmp_path / "metal")]
    if os.environ.get("SPILL_METAL_S3"):
        cmd += ["--s3", os.environ["SPILL_METAL_S3"]]
    p = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=3600)
    assert p.returncode == 0, p.stdout[-1500:] + p.stderr[-1500:]
    res = json.loads((tmp_path / "metal.json").read_text())
    assert "gpu" in res["device"]["mlx_default_device"].lower()
    assert all(c["ok"] for c in res["chains"] + res["moves"])
