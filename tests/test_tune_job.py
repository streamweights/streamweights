"""Tune jobs end to end on a tiny random model (CPU): both paths learn, the streamed
path matches mlx-lm's tuner, adapters round-trip through both layouts, and a job
interrupted mid-run resumes to the identical result."""

import json
import threading

import mlx.core as mx
import numpy as np
import pytest

from streamweights.adapters import ADAPTERS_DIR, load_adapter_dir
from streamweights.jobs.engine import Job
from streamweights.tune import job as tj
from streamweights.tune import lora as lo
from tests.tinymodel import add_char_tokenizer, make_tiny_model

WS = 36 * 1024**3


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("tiny") / "m"
    make_tiny_model(d)
    add_char_tokenizer(d)
    return d


@pytest.fixture()
def data(tmp_path):
    p = tmp_path / "train.jsonl"
    rows = []
    for i in range(24):
        rows.append({"messages": [{"role": "user", "content": f"question {i % 6}"},
                                  {"role": "assistant", "content": "yes yes yes"}]})
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return p


def make_spec(model_dir, data, name, path, **kw):
    base = dict(model="tiny", quant="bf16", model_dir=str(model_dir), data=str(data),
                name=name, path=path, rank=4, alpha=8, lr=2e-3, micro_batch=4, grad_accum=1,
                steps=8, ckpt_every=3, seed=5, max_seq=128, overwrite=True,
                resident_weights=True)
    base.update(kw)
    return tj.TuneSpec(**base)


def new_job(spec):
    job = Job.create(__import__("pathlib").Path(spec.data), spec.model, "bf16", 128, 4, None,
                     options={"kind": "tune", "tune": spec.to_dict()})
    return job


def run(spec, **kw):
    prep = tj.prepare(spec, working_set=WS, micro_batch_given=True, steps_given=True)
    job = new_job(spec)
    return prep, job, tj.run_tune(prep, job, **kw)


def losses(job):
    return [json.loads(l)["loss"] for l in (job.dir / "losses.jsonl").read_text().splitlines()]


def test_resident_path_learns_and_writes_both_layouts(model_dir, data):
    spec = make_spec(model_dir, data, "t-res", "resident")
    prep, job, res = run(spec)
    ls = losses(job)
    assert len(ls) == 8 and ls[-1] < 0.7 * ls[0], ls
    d = ADAPTERS_DIR / "t-res"
    assert (d / "adapters.safetensors").exists() and (d / "adapter_model.safetensors").exists()
    cfg = json.loads((d / "adapter_config.json").read_text())
    assert cfg["peft_type"] == "LORA" and cfg["r"] == 4 and cfg["lora_alpha"] == 8
    assert cfg["fine_tune_type"] == "lora" and cfg["lora_parameters"]["scale"] == 2.0
    assert len(cfg["target_modules"]) == 7 and cfg["streamweights"]["steps"] == 8


def test_both_layouts_load_to_the_same_adapter(model_dir, data):
    spec = make_spec(model_dir, data, "t-lay", "resident", steps=3)
    run(spec)
    d = ADAPTERS_DIR / "t-lay"
    mlx_dir = d.parent / "t-lay-mlx"
    peft_dir = d.parent / "t-lay-peft"
    for x, keep in ((mlx_dir, "adapters.safetensors"), (peft_dir, "adapter_model.safetensors")):
        x.mkdir(exist_ok=True)
        (x / "adapter_config.json").write_text((d / "adapter_config.json").read_text())
        (x / keep).write_bytes((d / keep).read_bytes())
    a, b = load_adapter_dir(mlx_dir), load_adapter_dir(peft_dir)
    assert (a.layout, b.layout) == ("mlx-lm", "peft") and a.rank == b.rank == 4
    assert sorted(a.layers) == sorted(b.layers) == [0, 1, 2]
    for k in a.layers:
        for path in a.layers[k]:
            for i in (0, 1):
                assert np.array_equal(np.array(a.layers[k][path][i]),
                                      np.array(b.layers[k][path][i]))
            assert abs(a.layers[k][path][2] - 2.0) < 1e-9 and abs(b.layers[k][path][2] - 2.0) < 1e-9
    # the directory as written also loads unchanged (mlx layout wins when both exist)
    assert load_adapter_dir(d).layout == "mlx-lm"


def test_streamed_path_learns(model_dir, data):
    spec = make_spec(model_dir, data, "t-str", "streamed")
    prep, job, res = run(spec)
    ls = losses(job)
    assert ls[-1] < 0.7 * ls[0], ls


def test_streamed_through_the_ring_matches_resident_weights(model_dir, data):
    a = make_spec(model_dir, data, "t-ring-a", "streamed", steps=4)
    b = make_spec(model_dir, data, "t-ring-b", "streamed", steps=4, resident_weights=False)
    _, ja, _ = run(a)
    _, jb, _ = run(b)
    assert np.allclose(losses(ja), losses(jb), rtol=1e-5)


def test_training_identity_gate_on_the_tiny_model(model_dir, data):
    """CPU version of the item 6 gate: identical data, seed, init, hyperparameters;
    streamed vs mlx-lm's tuner. Loss within 1% after step 1; adapter tensors
    cosine > 0.999 each."""
    ra = make_spec(model_dir, data, "t-id-res", "resident", steps=10, grad_accum=2)
    sa = make_spec(model_dir, data, "t-id-str", "streamed", steps=10, grad_accum=2)
    _, jr, _ = run(ra)
    _, js, _ = run(sa)
    lr_, ls = losses(jr), losses(js)
    assert len(lr_) == len(ls) == 10
    assert abs(lr_[0] - ls[0]) / lr_[0] < 1e-4              # same init, same first batch
    for i, (x, y) in enumerate(zip(lr_, ls)):
        assert abs(x - y) / x < 0.01, (i, x, y)
    pr = lo.read_adapter_params(ADAPTERS_DIR / "t-id-res")
    ps = lo.read_adapter_params(ADAPTERS_DIR / "t-id-str")
    assert set(pr) == set(ps)
    for k in pr:
        assert lo.cosine_sim(pr[k], ps[k]) > 0.999, k


@pytest.mark.parametrize("path", ["resident", "streamed"])
def test_interrupt_and_resume_equals_uninterrupted(model_dir, data, path):
    full = make_spec(model_dir, data, f"t-full-{path}", path, steps=9, ckpt_every=3)
    run(full)
    part = make_spec(model_dir, data, f"t-part-{path}", path, steps=9, ckpt_every=3)
    prep = tj.prepare(part, working_set=WS, micro_batch_given=True, steps_given=True)
    job = new_job(part)
    stop = threading.Event()
    r1 = tj.run_tune(prep, job, stop=stop,
                     progress_cb=lambda i: stop.set() if i["step"] == 4 else None)
    assert r1["interrupted"] and "adapter" not in r1, r1
    assert r1["step"] == 4, r1
    assert job.read_meta()["status"] == "interrupted"
    prep2 = tj.prepare(part, working_set=WS, micro_batch_given=True, steps_given=True)
    r2 = tj.run_tune(prep2, job, resume=True)
    assert not r2["interrupted"] and r2["step"] == 9
    la = [json.loads(l)["loss"] for l in (job.dir / "losses.jsonl").read_text().splitlines()]
    assert len(la) == 9
    pf = lo.read_adapter_params(ADAPTERS_DIR / f"t-full-{path}")
    pp = lo.read_adapter_params(ADAPTERS_DIR / f"t-part-{path}")
    for k in pf:
        assert lo.cosine_sim(pf[k], pp[k]) > 0.99999, k
        assert np.allclose(np.array(pf[k]), np.array(pp[k]), atol=1e-6), k
