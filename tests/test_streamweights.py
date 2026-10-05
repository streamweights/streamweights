"""Test suite for streamweights (created in 003; Phase 0 shipped without one)."""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from streamweights.cli import app
from streamweights.jobs.engine import Job, compute_offload
from streamweights.policy import choose_quant
from streamweights.registry import GIB, load_registry, read_gguf_arch

REPO = Path(__file__).resolve().parent.parent
runner = CliRunner()

HW = {
    "ram_total_bytes": 48 * GIB,
    "gpu": {"vendor": "apple", "vram_bytes": 36 * GIB, "bf16_compute": True},
    "nvme_seq_read": {"bytes_per_sec": int(5.1 * GIB)},
}


# ---------- registry ----------

def test_registry_models_present():
    reg = load_registry()
    assert set(reg) == {"qwen2.5:0.5b", "qwen2.5:7b", "qwen2.5:32b", "llama3.3:70b"}
    for m in reg.values():
        assert {"bf16", "Q8_0", "Q4_K_M"} <= set(m.quants)


def test_resident_bytes_grows_with_context_and_parallel():
    m = load_registry()["llama3.3:70b"]
    base = m.resident_bytes("Q8_0", 0)
    assert m.resident_bytes("Q8_0", 4096) > base
    assert m.resident_bytes("Q8_0", 4096, 2) > m.resident_bytes("Q8_0", 4096, 1)
    # 70B: 80 layers x 8 kv heads x 128 dim x 2 bytes x 2 (K+V) = 320 KiB/token
    assert m.kv_bytes_per_token() == 2 * 80 * 8 * 128 * 2


def test_gguf_header_reader():
    gguf = REPO / "models/qwen2.5-0.5b/Q8_0/qwen2.5-0.5b-instruct-q8_0.gguf"
    if not gguf.exists():
        pytest.skip("0.5b GGUF not downloaded")
    arch = read_gguf_arch(gguf)
    assert arch == {"n_layers": 24, "n_kv_heads": 2, "head_dim": 64}


# ---------- policy ----------

def test_policy_bf16_default(monkeypatch):
    import shutil
    from collections import namedtuple
    du = namedtuple("du", "total used free")
    # hermetic: the answer must not depend on how full this machine's disk is
    monkeypatch.setattr(shutil, "disk_usage", lambda p: du(10**13, 0, 10**13))
    m = load_registry()["llama3.3:70b"]
    c = choose_quant(m, HW)
    assert c.quant == "bf16" and "default" in c.reason


def test_policy_uses_q8_where_no_bf16_gguf_is_published():
    c = choose_quant(load_registry()["qwen2.5:0.5b"], HW)
    assert c.quant == "Q8_0" and "no published bf16 GGUF" in c.reason


def test_policy_drop_no_bf16_compute():
    m = load_registry()["llama3.3:70b"]
    hw = {**HW, "gpu": {**HW["gpu"], "bf16_compute": False}}
    c = choose_quant(m, hw)
    assert c.quant == "Q8_0" and "bf16 compute" in c.reason


def test_policy_drop_slow_nvme(monkeypatch):
    import collections
    import streamweights.policy as pol
    # pin disk free high so the disk rule cannot fire first (it depends on the machine)
    Usage = collections.namedtuple("usage", "total used free")
    monkeypatch.setattr(pol.shutil, "disk_usage", lambda p: Usage(1 << 40, 0, 1 << 40))
    m = load_registry()["llama3.3:70b"]
    hw = {**HW, "nvme_seq_read": {"bytes_per_sec": int(0.5 * GIB)}}  # 141GB/0.5GiB/s > 120s
    c = choose_quant(m, hw)
    assert c.quant == "Q8_0" and "forward pass" in c.reason


def test_policy_explicit_opt_in_wins():
    m = load_registry()["llama3.3:70b"]
    c = choose_quant(m, HW, explicit="Q4_K_M")
    assert c.quant == "Q4_K_M" and "explicit" in c.reason


# ---------- engine math + job lifecycle ----------

def test_compute_offload_small_model_fully_resident():
    m = load_registry()["qwen2.5:0.5b"]
    ngl, n = compute_offload(m, "Q8_0", 4096, 36 * GIB)
    assert ngl > m.arch["n_layers"]  # fully offloaded
    assert 1 <= n <= 256


def test_compute_offload_70b_partial_and_kv_bound():
    m = load_registry()["llama3.3:70b"]
    ngl, n = compute_offload(m, "Q8_0", 4096, 36 * GIB)
    assert ngl < 80  # partial offload only
    assert 1 <= n < 32  # Phase 0: 32 OOMs at 4k on 36 GiB
    # explicit override passes through
    _, n_explicit = compute_offload(m, "Q8_0", 4096, 36 * GIB, parallel=64)
    assert n_explicit == 64


def test_job_checkpoint_and_resume_skip(tmp_path, monkeypatch):
    import streamweights.jobs.engine as eng
    monkeypatch.setattr(eng, "JOBS_DIR", tmp_path)
    src = tmp_path / "in.jsonl"
    rows = [{"custom_id": f"r{i}", "method": "POST", "url": "/v1/chat/completions",
             "body": {"model": "qwen2.5:0.5b", "messages": [], "max_tokens": 1}}
            for i in range(4)]
    src.write_text("\n".join(json.dumps(r) for r in rows))
    job = Job.create(src, "qwen2.5:0.5b", "Q8_0", 4096, 2)
    assert job.total == 4 and job.done_ids() == set()
    job.checkpoint_path.write_text("r0\nr2\n")
    assert job.done_ids() == {"r0", "r2"}
    loaded = Job.load(job.id)
    assert (loaded.model, loaded.quant, loaded.total) == ("qwen2.5:0.5b", "Q8_0", 4)
    job.write_meta(status="interrupted", done=2)
    assert loaded.read_meta()["status"] == "interrupted"


# ---------- eval set ----------

def test_eval_set_shape():
    lines = (REPO / "examples/evals-2000.jsonl").read_text().splitlines()
    assert len(lines) == 2000
    for line in (lines[0], lines[1], lines[2], lines[-1]):
        r = json.loads(line)
        assert set(r) == {"custom_id", "method", "url", "body"}
        assert r["body"]["max_tokens"] == 128


# ---------- CLI ----------

def test_cli_help_uses_spill_name():
    res = runner.invoke(app, ["--help"])
    assert res.exit_code == 0
    assert "spillway" not in res.output.lower()


def test_cli_models_lists_registry():
    res = runner.invoke(app, ["models"])
    assert res.exit_code == 0
    assert "llama3.3:70b" in res.output
    assert "next: spill " in res.output  # hint uses the new command name


def test_cli_status_next_hint():
    res = runner.invoke(app, ["status"])
    assert res.exit_code == 0
    assert "spillway" not in res.output.lower()


# ---------- gateway shapes ----------

def test_gateway_tags_shape():
    from fastapi.testclient import TestClient
    from streamweights.server import app as gw
    client = TestClient(gw)
    r = client.get("/api/tags")
    assert r.status_code == 200
    models = r.json()["models"]
    assert {m["name"] for m in models} == {"qwen2.5:0.5b", "qwen2.5:7b", "qwen2.5:32b", "llama3.3:70b"}
    r = client.get("/health")
    assert r.json() == {"status": "ok"}


def test_gateway_files_roundtrip(tmp_path):
    from fastapi.testclient import TestClient
    from streamweights.server import app as gw
    client = TestClient(gw)
    data = b'{"custom_id":"x","method":"POST","url":"/v1/chat/completions","body":{"model":"qwen2.5:0.5b","messages":[]}}\n'
    r = client.post("/v1/files", files={"file": ("in.jsonl", data)}, data={"purpose": "batch"})
    assert r.status_code == 200
    fid = r.json()["id"]
    assert client.get(f"/v1/files/{fid}").json()["bytes"] == len(data)
    assert client.get(f"/v1/files/{fid}/content").content == data
