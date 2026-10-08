"""A guided build killed in the middle of training and finished on the other engine: MLX to
torch-cpu and torch-cpu to MLX, with the real qwen2.5:0.5b on tiny data and the real CLI.

Each direction kills the first process at an exact point of checkpoint publication (the
process is terminated; machine power loss is not tested), waits out its short lease, resumes
on the other engine, and checks that the committed step and data cursor came back, that the
optimizer state continued (its step count did not restart), and that the engine change is
recorded in the same run."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from streamweights import example as E
from streamweights.project import config as C
from streamweights.project import coordinator as CO
from streamweights.project.common import read_json
from streamweights.project.init import create_project

from .conftest import HAVE_MLX

REPO = Path(__file__).resolve().parent.parent
MODEL = REPO / "models" / "qwen2.5-0.5b" / "bf16-st"
pytestmark = pytest.mark.skipif(
    not (MODEL / "model.safetensors").exists() and os.environ.get("SPILL_REQUIRE_FIXTURE_MODEL") != "1",
    reason="needs qwen2.5:0.5b")

PAIRS = [("torch-cpu", "torch-cpu", "checkpoint_before_control")]
if HAVE_MLX:
    PAIRS = [("mlx", "torch-cpu", "checkpoint_before_control"),
             ("torch-cpu", "mlx", "checkpoint_after_control")]


def spill(args, env, cwd):
    return subprocess.run([sys.executable, "-m", "streamweights.cli", *args], capture_output=True,
                          text=True, env=env, cwd=cwd, timeout=1800)


@pytest.mark.parametrize("first,second,point", PAIRS)
def test_killed_mid_training_and_finished_on_the_other_engine(tmp_path, first, second, point):
    csv = E.DATA / "banking77" / "tiny" / "banking77.csv"
    proj = tmp_path / "p"
    create_project(csv, proj, {"input": "text", "output": "label"}, "classification")
    cfg = C.load(proj)
    cfg["training"].update(epochs=2.0, ckpt_every=5, micro_batch=4)
    C.save(proj, cfg)
    base = {**os.environ, "SPILL_HEADLESS": "0", "SPILL_LEASE_S": "2", "SPILL_HOME": os.environ["SPILL_HOME"]}
    if first == "mlx" or second == "mlx":
        base.pop("SPILL_DEVICE", None)
        base["SPILL_DEVICE"] = "cpu"                                  # MLX on the CPU device: no GPU needed
    r = spill(["build", str(proj), "--engine", first],
              {**base, f"SPILL_HOOK_{point}": "crash:2"}, str(tmp_path))
    assert r.returncode == 137, (r.stdout[-800:], r.stderr[-800:])
    rid = CO.list_runs(proj)[0]["run_id"]
    doc = read_json(proj / ".spill" / "runs" / rid / "control.json")
    committed = 5 if point == "checkpoint_before_control" else 10        # what the control object accepted
    assert doc["checkpoint"]["step"] == committed and doc["checkpoint"]["seq"] == committed // 5
    assert doc["checkpoint"]["cursor"]["step"] == committed and doc["status"] == "running"
    time.sleep(2.5)                                                       # the dead writer's lease runs out
    r2 = spill(["resume", str(proj), "--engine", second], base, str(tmp_path))
    assert r2.returncode == 0, (r2.stdout[-1200:], r2.stderr[-800:])
    assert f"restored the committed checkpoint: step {committed}" in r2.stdout
    # the run is the same run, complete, with both engines on record
    done = read_json(proj / ".spill" / "runs" / rid / "control.json")
    assert done["status"] == "completed" and done["generation"] > doc["generation"]
    m = read_json(proj / "runs" / rid / "manifest.json")
    engines = [p["engine"] for p in m["stages"]["train"]["producers"]]
    assert engines[0] == first and engines[-1] == second, engines
    assert m["engine_transitions"] and first in m["engine_transitions"][0] and second in m["engine_transitions"][0]
    assert "numerics" in m["engine_transitions"][0]
    # optimizer and cursor: the first checkpoint the second engine published continues the step
    # count and carries the first engine's range in its history
    pay = proj / ".spill" / "runs" / rid / "payloads"
    states, first_gen_steps = [], []
    for d in sorted(pay.glob("ckpt-*")):
        man = json.loads((d / "PAYLOAD.json").read_text()) if (d / "PAYLOAD.json").exists() else None
        if man is None:
            continue
        st = json.loads((d / "state.json").read_text())
        (states if man["generation"] > doc["generation"] else first_gen_steps).append(st)
    # the dead writer may have left a complete payload the control object never pointed at
    # (step 10 when it died before publishing): recovery used the pointer, not that directory
    assert committed in [x["step"] for x in first_gen_steps]
    later = states
    assert later, [s["step"] for s in states]
    s = later[0]
    assert s["opt_step"] == s["step"] and s["data_cursor"]["step"] == s["step"]
    assert s["history"][0]["range"][0] == 0 and s["history"][0]["range"][1] == committed
    assert s["history"][1]["range"][0] == committed
    assert s["history"][0]["engine"] != s["history"][1]["engine"] or first == second
    assert len(m["table"]) == 3
