# no-mlx-needed: runs on every platform (no MLX, no GPU); the GPU tests skip unless a GPU is asked for
"""The torch-cuda engine, on a GPU. Skipped unless SPILL_GPU_TESTS=1 and a CUDA device is visible
(there is none on CI; `python -m streamweights.verify_cuda` is the full verification)."""

import os

import pytest

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.skipif(
    os.environ.get("SPILL_GPU_TESTS") != "1" or not torch.cuda.is_available(),
    reason="needs SPILL_GPU_TESTS=1 and a CUDA device")

from streamweights.engines.base import MemoryBudget, ModelSpec  # noqa: E402
from streamweights.engines.torch_resident import TorchResidentEngine  # noqa: E402
from streamweights.engines.torch_stream import TorchEngine  # noqa: E402
from tests.tinytorch import make_tiny  # noqa: E402


def _rows(n):
    return [{"custom_id": f"r{i}", "body": {"messages": [{"role": "user", "content": f"hello {i} " + "x" * i}],
                                           "max_tokens": 10}} for i in range(n)]


def test_cuda_streamed_equals_cuda_resident_and_cpu(tmp_path):
    d = make_tiny(tmp_path / "m", "qwen2")
    spec = lambda: ModelSpec("t", "bf16", d, {}, 512, extra={"logprobs": 3})
    out = {}
    for name, eng in (("stream", TorchEngine(engine="torch-cuda", dtype="float32")),
                      ("resident", TorchResidentEngine(engine="torch-cuda", dtype="float32")),
                      ("cpu", TorchResidentEngine(engine="torch-cpu", dtype="float32"))):
        out[name] = {c.custom_id: c for c in eng.run_batch(_rows(8), spec(), MemoryBudget(0))}
    for k in out["stream"]:
        assert out["stream"][k].content == out["resident"][k].content == out["cpu"][k].content


def test_cuda_streamed_tune_matches_peft_resident(tmp_path):
    from streamweights import gates as G
    from streamweights.tune import toy
    d = make_tiny(tmp_path / "m", "llama")
    files = toy.make(tmp_path / "toy", 60, 10)
    out = G.gate_tune_identity(G.Work(tmp_path / "w"), str(d), files["train"],
                               a={"engine": "torch-cuda", "path": "streamed"},
                               b={"engine": "torch-cuda", "path": "resident"},
                               steps=30, lr=3e-4, loss_tol=1e-3, cos_tol=0.9999, tag="cuda")
    assert out["pass"], out["loss"]
