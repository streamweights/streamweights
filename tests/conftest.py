"""Tests run on the CPU only: no Metal, and a throwaway data root so tests never
write calibration, jobs, runs or adapters into the checkout."""

import os
import shutil
import tempfile
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_home = Path(tempfile.mkdtemp(prefix="spill-test-home-"))
(_home / "state").mkdir()
for _f in ("hardware.json", "calibration.json"):
    if (_REPO / "state" / _f).exists():
        shutil.copy(_REPO / "state" / _f, _home / "state" / _f)
if (_REPO / "models").exists():
    (_home / "models").symlink_to(_REPO / "models")
os.environ["SPILL_HOME"] = str(_home)
os.environ["SPILL_DEVICE"] = "cpu"

import mlx.core as mx  # noqa: E402

mx.set_default_device(mx.cpu)


import subprocess  # noqa: E402
import sys  # noqa: E402

import pytest  # noqa: E402

_MODEL = _REPO / "models/qwen2.5-0.5b/bf16-st"
_ADAPTER = _REPO / "adapters/qwen05-arr"


@pytest.fixture(scope="session")
def mlx_adapter():
    if not (_ADAPTER / "adapters.safetensors").exists():
        _ADAPTER.parent.mkdir(exist_ok=True)
        subprocess.run([sys.executable, str(_REPO / "scripts/make_test_adapter.py"),
                        str(_MODEL), str(_ADAPTER), "--iters", "60"], check=True,
                       capture_output=True)
    return _ADAPTER


