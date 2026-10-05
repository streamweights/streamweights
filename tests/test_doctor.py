from pathlib import Path

from typer.testing import CliRunner

from streamweights import doctor as D
from streamweights.cli import app

GIB = 1024**3
HW = {"cpu": "Apple M4 Pro", "cpu_cores": 12, "ram_total_bytes": 48 * GIB,
      "gpu": {"vram_bytes": 36 * GIB}, "nvme_seq_read": {"bytes_per_sec": 5.4e9}}


def test_report_has_every_field(tmp_path):
    out = D.report(HW, {"engine_rates": {"llama3.3:70b|bf16": 3722.2}, "tune_rates": {"qwen2.5:7b|resident": 9e12}},
                   registry_tags=["qwen2.5:7b"], models_dir=tmp_path, interrupted_lines=["build x stage 2/5 (spill resume /x)"],
                   version="9.9", free_disk=500 * GIB)
    for frag in ("Apple M4 Pro, 12 cores", "48 GB RAM, Metal working set 36 GB", "500 GB free",
                 "streamweights 9.9", "none downloaded yet", "build x stage 2/5", "overnight",
                 "9 TFLOP/s", "measured by spill tune"):
        assert frag in out, frag
    assert len(out.splitlines()) == 9          # one screen


def test_largest_overnight_is_limited_by_disk_then_time():
    big_disk, why = D.largest_overnight(3.9e9, 36 * GIB, 2000 * GIB, 5.0)
    assert why == "time" and 20 < big_disk < 400
    small, why = D.largest_overnight(3.9e9, 36 * GIB, 100 * GIB, 5.0)
    assert why == "disk" and small == 42
    # a faster disk and more compute raise the ceiling
    faster, _ = D.largest_overnight(8e9, 36 * GIB, 2000 * GIB, 20.0)
    assert faster > big_disk


def test_downloaded_models_lists_bf16_and_quants(tmp_path, monkeypatch):
    import streamweights.registry as R
    d = tmp_path / "qwen2.5-0.5b"
    (d / "bf16-st").mkdir(parents=True)
    (d / "bf16-st" / "config.json").write_text("{}")
    (d / "bf16-st" / "m.safetensors").write_bytes(b"x")
    (d / "mlx-8bit").mkdir()
    (d / "mlx-8bit" / "config.json").write_text("{}")
    monkeypatch.setattr(R, "MODELS_DIR", tmp_path)
    assert D.downloaded_models(["qwen2.5:0.5b"], tmp_path) == ["qwen2.5:0.5b (bf16, 8bit)"]


def test_cli_doctor_prints_next():
    r = CliRunner().invoke(app, ["doctor"])
    assert r.exit_code == 0, r.output
    assert "chip" in r.output and "TFLOP/s" in r.output and "next: " in r.output
