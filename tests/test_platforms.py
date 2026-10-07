"""Platforms without MLX: install works, build runs on the PyTorch engines, doctor says
what works, export runs on numpy, and every error is one line ending in a command."""

import subprocess
import sys

import numpy as np
import pytest
from typer.testing import CliRunner

from streamweights import doctor, safetensors_np as snp
from streamweights.cli import app
from streamweights.errors import SpillError
from streamweights.platforms import WORKS_TODAY

R = CliRunner()


@pytest.fixture()
def no_mlx(monkeypatch):
    monkeypatch.setenv("SPILL_NO_MLX", "1")


def test_build_needs_no_mlx(no_mlx, tmp_path):
    """build is not an MLX-only command any more: on a machine without MLX it gets as far as
    reading the folder, and its one-line errors are about the folder."""
    r = R.invoke(app, ["build", str(tmp_path / "nothing")])
    assert r.exit_code == 1
    out = r.output.strip()
    assert out.count("\n") == 0 and "Apple silicon" not in out and "Try: spill" in out


def test_doctor_names_the_platform_when_there_is_no_mlx():
    hw = {"cpu": "x86 test cpu", "cpu_cores": 4, "ram_total_bytes": 16 * 2**30,
          "gpu": {"vram_bytes": 0}, "nvme_seq_read": {"bytes_per_sec": 2e9}}
    text = doctor.report(hw, {}, registry_tags=[], models_dir=__import__("pathlib").Path("."),
                         interrupted_lines=[], version="0.1.0", mlx=False)
    assert "no Metal" in text and "Metal working set" not in text
    last = text.splitlines()[-1]
    assert last.startswith("platform") and WORKS_TODAY in last and "build" in last
    assert "overnight" not in text and "training" not in text


def test_every_error_is_one_line_ending_in_a_command():
    assert SpillError("bad thing.", "spill doctor").line() == "bad thing. Try: spill doctor"
    assert SpillError("bad thing").line(default="spill models") == "bad thing. Try: spill models"


def test_unexpected_exception_is_one_line_without_a_traceback(tmp_path):
    f = tmp_path / "bad.jsonl"
    f.write_bytes(b"\xff\xfe not utf8\n")
    r = subprocess.run([sys.executable, "-m", "streamweights", "check", str(f)],
                       capture_output=True, text=True, timeout=120)
    assert r.returncode != 0 and "Traceback" not in r.stderr
    assert len(r.stderr.strip().splitlines()) == 1 and "Try: spill" in r.stderr
    d = subprocess.run([sys.executable, "-m", "streamweights", "check", str(f), "--debug"],
                       capture_output=True, text=True, timeout=120)
    assert "Traceback" in d.stderr


def test_usage_errors_are_one_line():
    r = subprocess.run([sys.executable, "-m", "streamweights", "run"], capture_output=True,
                       text=True, timeout=120)
    assert r.returncode == 2 and len(r.stderr.strip().splitlines()) == 1
    assert r.stderr.strip().endswith("Try: spill run --help")


def test_numpy_safetensors_roundtrip_matches_bf16_rounding(tmp_path):
    rng = np.random.default_rng(0)
    x = rng.normal(size=(5, 7)).astype(np.float32)
    bf = snp.from_f32(x, "BF16")
    back = snp.to_f32(bf, "BF16")
    assert np.max(np.abs(back - x)) <= np.max(np.abs(x)) * 2 ** -8
    p = tmp_path / "a.safetensors"
    snp.write(p, [("w", "BF16", (5, 7), lambda: bf), ("v", "F32", (5, 7), lambda: x)])
    got = {n: (dt, snp.to_f32(raw, dt)) for n, dt, _s, raw in snp.tensors(p)}
    assert np.array_equal(got["w"][1], back) and np.array_equal(got["v"][1], x)


def test_numpy_bf16_rounding_equals_mlx():
    mx = pytest.importorskip("mlx.core")
    x = np.random.default_rng(1).normal(size=2000).astype(np.float32) * 3
    want = np.array(mx.array(x).astype(mx.bfloat16).astype(mx.float32))
    assert np.array_equal(snp.to_f32(snp.from_f32(x, "BF16"), "BF16"), want)
