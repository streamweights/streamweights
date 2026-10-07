# no-mlx-needed: runs on every platform (no MLX, no GPU)
"""The committed step-50 MLX checkpoint (qwen2.5:0.5b, 100-step schedule) resumes on torch-cpu.

On Linux CI this runs in every build: a job that was started on an Apple GPU is continued on a
CPU from a few MB of float32 adapter and optimizer state. It needs the 0.5B weights; CI
downloads them (SPILL_REQUIRE_FIXTURE_MODEL=1 makes their absence a failure, not a skip).
"""

import json
import os
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from streamweights import gates as G  # noqa: E402
from streamweights.portable import checkpoint as pc  # noqa: E402
from streamweights.portable.store import Store  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
MODEL = REPO / "models" / "qwen2.5-0.5b" / "bf16-st"


def _have_model() -> bool:
    return (MODEL / "config.json").exists() and any(MODEL.glob("*.safetensors"))


def test_the_fixture_is_a_few_mb_and_a_committed_step_50_mlx_checkpoint():
    fx = G.fixture_dir()
    size = sum(f.stat().st_size for f in fx.rglob("*") if f.is_file())
    assert size < 5_000_000
    ck = pc.load_tune(Store(fx / "state"))
    assert ck.step == 50 and ck.state["hyperparameters"]["steps"] == 100
    h = ck.state["history"]
    assert len(h) == 1 and h[0]["engine"] == "mlx_stream_tune" and h[0]["range"] == [0, 50]
    assert h[0]["numerics"]["base"] == "bf16" and h[0]["hardware"].startswith("apple-gpu")
    assert len(ck.state["losses"]) == 50
    assert len(json.loads((fx / "expected_losses.json").read_text())) == 100
    assert all(a.dtype.name == "float32" for a in ck.params.values())


def test_resume_the_mlx_fixture_on_torch_cpu(tmp_path):
    if not _have_model():
        if os.environ.get("SPILL_REQUIRE_FIXTURE_MODEL") == "1":
            pytest.fail(f"SPILL_REQUIRE_FIXTURE_MODEL=1 but {MODEL} is missing")
        pytest.skip("needs qwen2.5:0.5b (spill run qwen2.5:0.5b sample)")
    w = G.Work(tmp_path, models_from=REPO / "models", state_from=REPO / "state")
    out = G.gate_resume_fixture(w, "torch-cpu", stop_after=55)
    assert out["pass"], out
    assert out["first_step"] == 51 and out["last_step"] == 55
    assert out["mean_rel_loss_difference"] < 0.35         # the MLX run's curve, to rounding noise
    assert [h["engine"] for h in out["history"]] == ["mlx_stream_tune", "torch_stream_tune"]
    assert out["history"][0]["range"] == [0, 50] and out["history"][1]["range"] == [50, 55]
    assert out["history"][1]["numerics"] in ("float32", "bf16")
