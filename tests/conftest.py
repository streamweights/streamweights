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
GPU_TESTS = os.environ.get("SPILL_GPU_TESTS") == "1"     # opt in to Metal tests
if not GPU_TESTS:
    os.environ["SPILL_DEVICE"] = "cpu"

try:
    import mlx.core as mx  # noqa: E402
    HAVE_MLX = True
except ImportError:          # Linux, Windows, Intel Macs: only the platform-neutral tests run
    HAVE_MLX = False

if HAVE_MLX and not GPU_TESTS:
    mx.set_default_device(mx.cpu)


import re  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402

import pytest  # noqa: E402

_MODEL = _REPO / "models/qwen2.5-0.5b/bf16-st"
_ADAPTER = _REPO / "adapters/qwen05-arr"


@pytest.fixture(scope="session")
def mlx_adapter():
    if not (_ADAPTER / "adapters.safetensors").exists():
        if not (_MODEL / "config.json").exists():
            pytest.skip("needs the qwen2.5:0.5b safetensors (spill run qwen2.5:0.5b sample)")
        _ADAPTER.parent.mkdir(exist_ok=True)
        subprocess.run([sys.executable, str(_REPO / "scripts/make_test_adapter.py"),
                        str(_MODEL), str(_ADAPTER), "--iters", "60"], check=True,
                       capture_output=True)
    return _ADAPTER




@pytest.fixture(autouse=True)
def _isolated_build_registry(tmp_path, monkeypatch):
    """A fake build in one test must not show up as an interrupted build in the next."""
    from streamweights import overnight
    monkeypatch.setattr(overnight, "BUILDS_FILE", tmp_path / "builds.json")


def _needs_mlx(path: Path) -> bool:
    text = path.read_text()
    return bool(re.search(r"^\s*(import mlx|from mlx|from tests\.tiny|from \.tiny|from \. import tiny"
                          r"|from streamweights\.(engines|tune)|from tests\.test_)", text, re.M))


collect_ignore = [] if HAVE_MLX else [
    p.name for p in Path(__file__).parent.glob("test_*.py") if _needs_mlx(p)]


# platform-neutral tests that assert the Apple-silicon behavior of a command
_MLX_ONLY = {"test_cli_build_end_to_end_with_fake_backend",
             "test_cli_build_interrupted_prints_resume_command",
             "test_distill_generation_and_score", "test_cli_doctor_prints_next",
             "test_eval_two_models_with_cache_and_judge", "test_cli_example"}


def pytest_collection_modifyitems(config, items):
    if HAVE_MLX:
        return
    skip = pytest.mark.skip(reason="asserts the Apple-silicon (MLX) behavior of the command")
    for item in items:
        if item.name in _MLX_ONLY:
            item.add_marker(skip)
