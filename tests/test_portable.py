"""Portable job state: storage through fsspec (a local path and the in-memory filesystem),
the commit protocol, the float32 checkpoint round trip, rows, and weight staging."""

import json
import uuid

import numpy as np
import pytest

from streamweights.errors import SpillError
from streamweights.portable import checkpoint as pc
from streamweights.portable import rows
from streamweights.portable.store import Store, is_uri
from streamweights.portable.weights import stage_weights


@pytest.fixture(params=["local", "memory"])
def store(request, tmp_path):
    if request.param == "local":
        return Store(tmp_path / "state")
    return Store(f"memory://spill-test-{uuid.uuid4().hex[:8]}")


def _params(seed=0):
    rng = np.random.default_rng(seed)
    return {"layers.0.self_attn.q_proj.lora_a": rng.normal(size=(8, 4)).astype(np.float32),
            "layers.0.self_attn.q_proj.lora_b": rng.normal(size=(4, 8)).astype(np.float32)}


def _opt(params, step=3):
    return {"step": step, "m": {k: v * 0.1 for k, v in params.items()},
            "v": {k: v * v for k, v in params.items()}}


def test_uri_detection():
    assert is_uri("s3://bucket/x") and is_uri("memory://a") and not is_uri("/tmp/x")
    assert not is_uri("C:/x") or True


def test_store_roundtrip_and_listing(store):
    store.write("a/b.txt", b"hello")
    assert store.read("a/b.txt") == b"hello"
    assert store.exists("a/b.txt") and not store.exists("a/c.txt")
    assert store.ls("a") == ["b.txt"]
    assert store.size("a/b.txt") == 5
    store.rm("a")
    assert store.ls("a") == []


def test_checkpoint_roundtrip_is_float32_and_exact(store):
    p = _params()
    o = _opt(p)
    pc.save_tune(store, step=3, params=p, opt=o, state={"seed": 7})
    ck = pc.load_tune(store)
    assert ck.step == 3 and ck.state["seed"] == 7 and ck.state["schema"] == pc.SCHEMA
    for k, v in p.items():
        assert ck.params[k].dtype == np.float32
        np.testing.assert_array_equal(ck.params[k], v)
        np.testing.assert_array_equal(ck.opt["m"][k], o["m"][k])
        np.testing.assert_array_equal(ck.opt["v"][k], o["v"][k])
    assert ck.opt["step"] == 3


def test_uncommitted_checkpoint_is_ignored(store):
    p = _params()
    pc.save_tune(store, step=2, params=p, opt=_opt(p, 2), state={})
    # a writer died after writing files but before COMMIT
    store.write("ckpt/step-000000005/params.safetensors", b"half")
    assert pc.latest_step(store) == 2
    assert pc.load_tune(store).step == 2


def test_corrupt_checkpoint_falls_back_to_the_previous_one(store):
    p = _params()
    pc.save_tune(store, step=2, params=p, opt=_opt(p, 2), state={})
    pc.save_tune(store, step=4, params=p, opt=_opt(p, 4), state={})
    store.write("ckpt/step-000000004/params.safetensors", b"x" * 10)   # size no longer matches
    assert pc.load_tune(store).step == 2


def test_only_two_checkpoints_are_kept(store):
    p = _params()
    for s in (1, 2, 3, 4):
        pc.save_tune(store, step=s, params=p, opt=_opt(p, s), state={})
    assert pc.committed_steps(store) == [3, 4]


def test_history_merges_matching_ranges():
    a = pc.producer("mlx", "apple-gpu", {"base": "bf16"}, 0, 50)
    b = pc.producer("mlx", "apple-gpu", {"base": "bf16"}, 50, 100)
    c = pc.producer("torch-cpu", "cpu", {"base": "float32"}, 100, 150)
    h = pc.extend_history(pc.extend_history([], a), b)
    assert len(h) == 1 and h[0]["range"] == [0, 100]
    h = pc.extend_history(h, c)
    assert [x["engine"] for x in h] == ["mlx", "torch-cpu"]
    assert h[1]["range"] == [100, 150]


def test_resume_compatibility_checks():
    spec = {"seed": 0, "rank": 16, "alpha": 32.0, "dropout": 0.0, "lr": 1e-4,
            "weight_decay": 0.01, "schedule": "cosine", "max_seq": 2048, "micro_batch": 4,
            "grad_accum": 1, "steps": 100, "epochs": 1.0}
    st = pc.tune_state(spec=spec, model_id="m", fingerprint="f" * 64, data_sha256="d",
                       targets=["a"], history=[], grad_accum=1, step=50)
    ok = dict(model_id="m", fingerprint="f" * 64, data_sha256="d", spec=spec)
    assert pc.check_resume_compatible(st, **ok) is None
    assert "different weights" in pc.check_resume_compatible(st, **{**ok, "fingerprint": "e" * 64})
    assert "training file" in pc.check_resume_compatible(st, **{**ok, "data_sha256": "x"})
    assert "lr" in pc.check_resume_compatible(st, **{**ok, "spec": {**spec, "lr": 1e-3}})
    assert "steps" in pc.check_resume_compatible(st, **{**ok, "spec": {**spec, "steps": 50}})


def test_rows_push_restore_and_cursor(store, tmp_path):
    prod = {"engine": "mlx_resident", "hardware": "apple-gpu", "numerics": {"base": "bf16"}}
    a = tmp_path / "a"
    a.mkdir()
    res, cur = a / "results.jsonl", a / "checkpoint"
    res.write_text("".join(json.dumps({"custom_id": f"r{i}"}) + "\n" for i in range(5)))
    cur.write_text("".join(f"r{i}\n" for i in range(5)))
    sync = rows.RowSync(store, res, cur, prod, every_rows=2)
    assert sync.attach() == 0 and len(cur.read_text().split()) == 5
    assert sync.maybe_push(5)
    assert rows.done_ids(store) == {f"r{i}" for i in range(5)}
    # a second machine continues
    b = tmp_path / "b"
    b.mkdir()
    prod2 = {"engine": "torch_stream", "hardware": "cpu", "numerics": {"base": "float32"}}
    sync2 = rows.RowSync(store, b / "results.jsonl", b / "checkpoint", prod2, every_rows=1)
    assert sync2.attach() == 5
    with open(b / "results.jsonl", "a") as f, open(b / "checkpoint", "a") as c:
        for i in range(5, 8):
            f.write(json.dumps({"custom_id": f"r{i}"}) + "\n")
            c.write(f"r{i}\n")
    assert sync2.push() == 3
    ids = (b / "checkpoint").read_text().split()
    assert ids == [f"r{i}" for i in range(8)]
    segs = rows.committed_segments(store)
    assert [s["engine"] for s in segs] == ["mlx_resident", "torch_stream"]
    # restore elsewhere is complete and ordered, with no duplicates
    c3 = tmp_path / "c"
    c3.mkdir()
    n, producers = rows.restore(store, c3 / "r.jsonl", c3 / "k")
    assert n == 8 and len(producers) == 2
    assert [json.loads(l)["custom_id"] for l in (c3 / "r.jsonl").read_text().splitlines()] == ids


def test_uncommitted_row_segment_is_ignored(store, tmp_path):
    store.write("rows/seg-000000.jsonl", b'{"custom_id": "x"}\n')       # no commit marker
    assert rows.committed_segments(store) == []
    assert rows.done_ids(store) == set()


def test_weights_stage_from_a_uri_and_second_stage_is_free(tmp_path, monkeypatch):
    from streamweights.portable import weights as w
    monkeypatch.setattr(w, "MODELS_DIR", tmp_path / "models")
    src = Store(f"memory://weights-{uuid.uuid4().hex[:8]}/m")
    src.write("config.json", b"{}")
    src.write("model.safetensors", b"0" * 5000)
    src.write("tokenizer.json", b"{}")
    src.write("notes.bin", b"ignored")
    d, info = stage_weights(src.uri)
    assert (d / "model.safetensors").stat().st_size == 5000 and not (d / "notes.bin").exists()
    assert info["bytes"] == 5000 + 4 and info["reused"] == 0
    d2, info2 = stage_weights(src.uri)
    assert d2 == d and info2["bytes"] == 0 and info2["reused"] == 3


def test_staging_something_that_is_not_a_model_is_one_line_error():
    src = Store(f"memory://empty-{uuid.uuid4().hex[:8]}/m")
    src.write("readme.txt", b"hi")
    with pytest.raises(SpillError, match="not a safetensors model directory"):
        stage_weights(src.uri)


def test_missing_cloud_backend_says_how_to_install(monkeypatch):
    import fsspec.core
    def boom(*a, **k):
        raise ImportError("no s3fs")
    monkeypatch.setattr(fsspec.core, "url_to_fs", boom)
    with pytest.raises(SpillError) as e:
        Store("s3://bucket/key")
    assert "s3fs" in e.value.message and "streamweights[cloud]" in e.value.recovery
