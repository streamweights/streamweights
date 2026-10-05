"""Where spill runs. The MLX engines (streaming, distill, tune, build) need Apple silicon;
everywhere else the package installs and the llama.cpp path, export, check and models work."""

from __future__ import annotations

import importlib.util
import os
import platform

from .errors import SpillError

WORKS_TODAY = "run through llama.cpp, export, check, models"
NOT_YET = "streaming, distill, tune, build"
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
    return (f"works here today: {WORKS_TODAY}; not yet: {NOT_YET}; see {README_SECTION}")


def require_mlx(command: str) -> None:
    """The one-line message for a command that needs the MLX engines."""
    if not mlx_available():
        raise SpillError(f"spill {command} needs Apple silicon with MLX; {platform_line()}",
                         "spill doctor")
