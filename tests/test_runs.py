"""Run store: manifest content, fingerprints, provenance, cache lookup."""

import json
import struct
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from streamweights import runs


def _write_st(path: Path, tensors: dict):
    header, blobs, off = {}, [], 0
    for name, a in tensors.items():
        header[name] = {"dtype": "F32", "shape": list(a.shape),
                        "data_offsets": [off, off + a.nbytes]}
        blobs.append(a.tobytes())
        off += a.nbytes
    hj = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hj)) + hj)
        for b in blobs:
            f.write(b)


def _model_dir(tmp_path, shape=(2, 2)):
    d = tmp_path / "m"
    d.mkdir(exist_ok=True)
    (d / "config.json").write_text('{"model_type": "qwen2"}')
    _write_st(d / "model.safetensors", {"w": np.zeros(shape, np.float32)})
    return d


def test_fingerprint_stable_and_sensitive(tmp_path):
    d = _model_dir(tmp_path)
    a = runs.fingerprint_model(d)
    assert a == runs.fingerprint_model(d)
    _write_st(d / "model.safetensors", {"w": np.zeros((2, 3), np.float32)})
    assert runs.fingerprint_model(d) != a


def test_manifest_lifecycle_and_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(runs, "RUNS_DIR", tmp_path / "runs")
    d = _model_dir(tmp_path)
    inp = tmp_path / "in.jsonl"
    inp.write_text('{"custom_id": "a", "body": {"messages": []}}\n')
    job = SimpleNamespace(id="r1", total=1, dir=tmp_path / "job",
                          results_path=tmp_path / "job" / "results.jsonl",
                          write_meta=lambda **kw: None)
    job.dir.mkdir()
    job.results_path.write_text("{}\n")
    spec = SimpleNamespace(name="qwen2.5:0.5b", quant="bf16", path=d)
    m = runs.start_run(job, command="spill run x y", spec=spec, input_path=inp,
                       hw={"cpu": "M4", "gpu": {"model": "M4", "vram_bytes": 1}}, engine_name="mlx_resident",
                       adapter={"id": "ad", "hash": "f" * 64}, options={"logprobs": 5})
    for k in ("command", "model", "adapter", "input", "engine", "hardware", "started"):
        assert m[k]
    assert m["model"]["weight_hash"] and m["input"]["sha256"] == runs.input_hash(inp)
    assert runs.provenance(m)["adapter_hash"] == "f" * 16
    assert runs.find_cached("qwen2.5:0.5b", "bf16", "f" * 64, m["input"]["sha256"]) is None
    runs.finish_run("r1", status="completed", rows_done=1, results_path=job.results_path)
    hit = runs.find_cached("qwen2.5:0.5b", "bf16", "f" * 64, m["input"]["sha256"])
    assert hit and hit["id"] == "r1" and hit["results_sha256"]
    # different adapter or different input does not hit
    assert runs.find_cached("qwen2.5:0.5b", "bf16", None, m["input"]["sha256"]) is None
    assert runs.find_cached("qwen2.5:0.5b", "bf16", "f" * 64, "0" * 64) is None
    assert [x["id"] for x in runs.list_runs()] == ["r1"]


def test_models_with_runs_lists_finished_runs_for_a_hash(tmp_path, monkeypatch):
    from streamweights import runs
    monkeypatch.setattr(runs, "RUNS_DIR", tmp_path)
    def man(id, model, adapter, h, status="completed", rows=5, done=5):
        runs.write_manifest(id, {"id": id, "kind": "run", "status": status, "rows_done": done,
                                 "model": {"id": model, "quant": "bf16", "weight_hash": "x"},
                                 "adapter": {"id": adapter, "hash": "h"} if adapter else None,
                                 "input": {"sha256": h, "rows": rows}, "options": {"mode": "generate"}})
    man("r1", "qwen2.5:0.5b", None, "H")
    man("r2", "qwen2.5:0.5b", "mine", "H")
    man("r3", "llama3.3:70b", None, "OTHER")
    man("r4", "qwen2.5:7b", None, "H", status="interrupted", done=2)
    assert runs.models_with_runs("H") == ["qwen2.5:0.5b", "qwen2.5:0.5b+mine"]
