"""Which engine runs a job. Automatic: Apple silicon uses MLX, a visible CUDA device uses
torch-cuda, anything else torch-cpu. `--engine mlx|torch-cpu|torch-cuda` (or SPILL_ENGINE)
overrides, and says so when the override cannot work here."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

from .errors import SpillError

ENGINES = ("mlx", "torch-cpu", "torch-cuda")


@dataclass
class EngineChoice:
    name: str          # mlx | torch-cpu | torch-cuda
    why: str
    forced: bool = False

    @property
    def is_mlx(self) -> bool:
        return self.name == "mlx"

    @property
    def is_torch(self) -> bool:
        return self.name.startswith("torch")


def _torch_ok() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def _cuda_ok() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except ImportError:
        return False


def availability() -> dict[str, tuple[bool, str]]:
    """{engine: (usable here, the reason either way)}."""
    from .platforms import apple_silicon, mlx_available
    out = {}
    if mlx_available():
        out["mlx"] = (True, "Apple silicon with MLX")
    elif apple_silicon():
        out["mlx"] = (False, "MLX is not installed" if not os.environ.get("SPILL_NO_MLX")
                      else "SPILL_NO_MLX is set")
    else:
        out["mlx"] = (False, "needs Apple silicon")
    if _torch_ok():
        out["torch-cpu"] = (True, "PyTorch on the CPU")
        if _cuda_ok():
            import torch
            out["torch-cuda"] = (True, f"CUDA device visible: "
                                       f"{torch.cuda.get_device_name(torch.cuda.current_device())}")
        else:
            out["torch-cuda"] = (False, "no CUDA device is visible")
    else:
        out["torch-cpu"] = (False, "PyTorch is not installed")
        out["torch-cuda"] = (False, "PyTorch is not installed")
    return out


def choose_engine(override: str | None = None) -> EngineChoice:
    override = override or os.environ.get("SPILL_ENGINE") or None
    avail = availability()
    if override:
        if override not in ENGINES:
            raise SpillError(f"--engine must be one of {', '.join(ENGINES)}; got {override}",
                             "spill doctor")
        ok, why = avail[override]
        if not ok:
            alt = next((e for e in ("torch-cuda", "torch-cpu", "mlx") if avail[e][0]), None)
            raise SpillError(f"--engine {override} is not usable here: {why}",
                             f"spill ... --engine {alt}" if alt else "spill doctor")
        return EngineChoice(override, f"--engine {override}", forced=True)
    if avail["mlx"][0]:
        return EngineChoice("mlx", "Apple silicon: MLX")
    if avail["torch-cuda"][0]:
        return EngineChoice("torch-cuda", f"{avail['torch-cuda'][1]}: torch-cuda")
    if avail["torch-cpu"][0]:
        return EngineChoice("torch-cpu", "no Apple silicon and no CUDA device: torch-cpu")
    raise SpillError("no engine is usable here: " + "; ".join(
        f"{e}: {w}" for e, (_, w) in avail.items()), "pip install torch")


# ---------------------------------------------------------------- measured rates (doctor)

def _bench_torch(device: str) -> dict:
    import torch
    dt = torch.float32
    n = 2048 if device == "cuda" else 1024
    a = torch.randn(n, n, dtype=dt, device=device)
    b = torch.randn(n, n, dtype=dt, device=device)
    for _ in range(2):
        a @ b
    if device == "cuda":
        torch.cuda.synchronize()
    reps = 30 if device == "cuda" else 10
    t0 = time.perf_counter()
    for _ in range(reps):
        a @ b
    if device == "cuda":
        torch.cuda.synchronize()
    tflops = reps * 2 * n**3 / (time.perf_counter() - t0) / 1e12
    big = torch.empty(64 * 1024 * 1024 // 4, dtype=torch.float32, device=device)
    big.add_(1.0)
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(4):
        big.add_(1.0)
    if device == "cuda":
        torch.cuda.synchronize()
    gbs = 4 * 2 * big.numel() * 4 / (time.perf_counter() - t0) / 1e9
    return {"matmul_tflops": round(tflops, 3), "memory_gb_s": round(gbs, 1),
            "dtype": "float32"}


def _bench_mlx() -> dict:
    import mlx.core as mx
    n = 2048
    a = mx.random.normal((n, n))
    b = mx.random.normal((n, n))
    mx.eval(a @ b)
    t0 = time.perf_counter()
    for _ in range(10):
        mx.eval(a @ b)
    tflops = 10 * 2 * n**3 / (time.perf_counter() - t0) / 1e12
    big = mx.ones((64 * 1024 * 1024 // 4,))
    mx.eval(big + 1)
    t0 = time.perf_counter()
    for _ in range(4):
        mx.eval(big + 1)
    gbs = 4 * 2 * big.size * 4 / (time.perf_counter() - t0) / 1e9
    return {"matmul_tflops": round(tflops, 3), "memory_gb_s": round(gbs, 1),
            "dtype": "float32"}


def measure_rates(save: bool = True) -> dict[str, dict]:
    """A short measured rate (float32 matmul TFLOP/s and memory GB/s) for every usable engine.
    Saved into the calibration so estimates for the torch engines start from a measurement."""
    from .calibration import load_calibration, save_calibration
    rates: dict[str, dict] = {}
    avail = availability()
    if avail["mlx"][0]:
        try:
            rates["mlx"] = _bench_mlx()
        except Exception as e:                       # never let a benchmark break doctor
            rates["mlx"] = {"error": str(e)[:80]}
    if avail["torch-cpu"][0]:
        rates["torch-cpu"] = _bench_torch("cpu")
    if avail["torch-cuda"][0]:
        rates["torch-cuda"] = _bench_torch("cuda")
    if save:
        cal = load_calibration()
        cal["engine_tflops"] = {**cal.get("engine_tflops", {}),
                                **{e: r["matmul_tflops"] for e, r in rates.items()
                                   if "matmul_tflops" in r and e != "mlx"}}
        cal["engine_membw_gbs"] = {**cal.get("engine_membw_gbs", {}),
                                   **{e: r["memory_gb_s"] for e, r in rates.items()
                                      if "memory_gb_s" in r and e != "mlx"}}
        save_calibration(cal)
    return rates
