# no-mlx-needed
"""The directive 020 gates, as recorded on real models: the committed captures (scripts/capture_020.py)
meet the conditions in scripts/check_020_capture.py, and the checker does fail when they are not met."""

import copy
import json
from pathlib import Path

import pytest

from scripts import check_020_capture as chk

DATA = Path(__file__).resolve().parent.parent / "docs/reports/data"
CAPTURES = sorted(DATA.glob("020-*-*.json"))
CAPTURES = [p for p in CAPTURES if "readme" not in p.name]


def test_every_gate_has_a_capture_on_the_engines_it_names():
    names = {p.name for p in CAPTURES}
    assert {"020-judged-mlx.json", "020-judged-torch-cpu.json", "020-nojudge-mlx.json",
            "020-labeled-mlx.json", "020-kd-mlx.json"} <= names


@pytest.mark.parametrize("path", CAPTURES, ids=lambda p: p.name)
def test_the_capture_meets_its_gate(path):
    assert chk.check(path).endswith("ok")


def test_the_mlx_captures_ran_on_the_gpu_and_the_cpu_capture_on_the_cpu():
    for p in CAPTURES:
        cap = json.loads(p.read_text())
        assert cap["device"].startswith("Device(gpu") if cap["engine"] == "mlx" else cap["device"] == "cpu"


def test_the_checker_rejects_a_run_that_missed_a_condition():
    cap = json.loads((DATA / "020-judged-mlx.json").read_text())
    bad = copy.deepcopy(cap)
    for s in bad["steps"]:
        if s["command"].startswith("spill report"):
            s["stdout"] = s["stdout"].replace("orders disagreed", "")
    with pytest.raises(AssertionError):
        chk.judged(bad)
    cap = json.loads((DATA / "020-nojudge-mlx.json").read_text())
    bad = copy.deepcopy(cap)
    bad["models_downloaded"] = {"qwen2.5-1.5b/bf16-st/model.safetensors": 1}
    with pytest.raises(AssertionError):
        chk.nojudge(bad)
    bad = copy.deepcopy(cap)
    bad["allowed_models"] = None
    with pytest.raises(AssertionError):
        chk.nojudge(bad)
