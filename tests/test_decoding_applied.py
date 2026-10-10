"""The decoding settings of a request reach the engine and are echoed as applied; settings the
engine cannot apply are refused before anything runs. Real qwen2.5:0.5b, torch-cpu always, MLX when
available (the suite pins MLX to the CPU device; SPILL_METAL_TESTS=1 runs the MLX case on the GPU)."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from .conftest import HAVE_MLX

REPO = Path(__file__).resolve().parent.parent
MODEL = REPO / "models" / "qwen2.5-0.5b" / "bf16-st"
pytestmark = pytest.mark.skipif(
    not (MODEL / "model.safetensors").exists() and os.environ.get("SPILL_REQUIRE_FIXTURE_MODEL") != "1",
    reason="needs qwen2.5:0.5b")
ENGINES = ["torch-cpu"] + (["mlx"] if HAVE_MLX else [])


def row(cid, **body):
    return {"custom_id": cid, "method": "POST", "url": "/v1/chat/completions",
            "body": {"messages": [{"role": "user", "content": "Say hi."}], "max_tokens": 5, **body}}


def run(tmp_path, engine, rows, env_extra=None):
    f = tmp_path / "in.jsonl"
    f.write_text("".join(json.dumps(r) + "\n" for r in rows))
    out = tmp_path / "out.jsonl"
    env = {**os.environ, "SPILL_HEADLESS": "0", **(env_extra or {})}
    if os.environ.get("SPILL_METAL_TESTS") == "1":      # the MLX case on the Metal GPU, not the CPU device
        env.pop("SPILL_DEVICE", None)
    p = subprocess.run([sys.executable, "-m", "streamweights.cli", "run", "qwen2.5:0.5b", str(f),
                        "--out", str(out), "--engine", engine], capture_output=True, text=True, env=env,
                       cwd=tmp_path, timeout=900)
    return p, out


@pytest.mark.parametrize("engine", ENGINES)
def test_requested_decoding_is_applied_and_echoed_on_every_result_row(tmp_path, engine):
    body = {"temperature": 0, "top_p": 1.0, "stop": [], "seed": 3, "greedy": True}
    p, out = run(tmp_path, engine, [row("a", **body), row("b", **body)])
    assert p.returncode == 0, p.stderr[-600:]
    rows = [json.loads(l) for l in out.read_text().splitlines()]
    assert len(rows) == 2
    for r in rows:
        d = r["streamweights"]["decoding"]
        assert d["max_tokens"] == 5 and d["greedy"] is True and d["temperature"] == 0.0
        assert d["requested"] == body
        assert r["response"]["body"]["usage"]["completion_tokens"] <= 5
    # the captured request is the job's own input file: the settings are in the body the engine got
    jobs = sorted((Path(os.environ["SPILL_HOME"]) / "jobs").iterdir())
    sent = [json.loads(l) for l in (jobs[-1] / "input.jsonl").read_text().splitlines()]
    assert all(s["body"]["temperature"] == 0 and s["body"]["seed"] == 3 and s["body"]["max_tokens"] == 5
               for s in sent)


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("bad,name", [({"temperature": 0.7}, "temperature"), ({"top_p": 0.9}, "top_p"),
                                      ({"stop": ["\n"]}, "stop")])
def test_a_setting_the_engine_cannot_apply_is_rejected_before_evaluation(tmp_path, engine, bad, name):
    p, out = run(tmp_path, engine, [row("a", **bad)])
    assert p.returncode == 1 and not out.exists()
    err = p.stderr
    assert f"engine {engine} cannot apply" in err and name in err, err[-400:]
