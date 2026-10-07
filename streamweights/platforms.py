"""Where spill runs. Apple silicon uses the MLX engines; everywhere else the PyTorch engines
(CPU and CUDA) run every command, `build` included."""

from __future__ import annotations

import importlib.util
import os
import platform

from .errors import SpillError

WORKS_TODAY = "build, run, distill, tune, eval, export, check, models (PyTorch engines)"
README_SECTION = 'README, "Linux and other platforms"'


def apple_silicon() -> bool:
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def mlx_available() -> bool:
    """Apple silicon with MLX installed. SPILL_NO_MLX=1 forces this off (to test the other
    platforms on a Mac)."""
    if os.environ.get("SPILL_NO_MLX"):
        return False
    return apple_silicon() and importlib.util.find_spec("mlx") is not None


def platform_line() -> str:
    return f"works here today: {WORKS_TODAY}; see {README_SECTION}"


def require_mlx(command: str) -> None:
    """The one-line message for a command that needs the MLX engines."""
    if not mlx_available():
        raise SpillError(f"spill {command} needs Apple silicon with MLX; {platform_line()}",
                         "spill doctor")


def mlx_device_label() -> str:
    """What the MLX engines compute on, for provenance: the Apple GPU (with the chip name) or
    the CPU when SPILL_DEVICE=cpu."""
    import mlx.core as mx
    if mx.default_device() != mx.gpu:
        return "cpu"
    try:
        import subprocess
        chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                              capture_output=True, text=True, timeout=3).stdout.strip()
        return f"apple-gpu:{chip}" if chip else "apple-gpu"
    except Exception:
        return "apple-gpu"
