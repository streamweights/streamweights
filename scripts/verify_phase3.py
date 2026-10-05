"""Metal verification for Phase 3 (run after the GPU gate opens).

  python scripts/verify_phase3.py gate     identity gate on Qwen2.5-0.5B (100 steps)
  python scripts/verify_phase3.py toy      200-example toy task, base vs base+adapter
  python scripts/verify_phase3.py smoke70  20 steps of llama3.3:70b, then run with the adapter

Results are written to docs/reports/008-*.json / .txt.
"""

import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
OUT = REPO / "docs" / "reports"


def spill(*args, capture=True):
    exe = [sys.executable, "-m", "streamweights.cli"] if False else [str(REPO / ".venv/bin/spill")]
    t0 = time.monotonic()
    p = subprocess.run(exe + [str(a) for a in args], capture_output=capture, text=True)
    return p.returncode, (p.stdout or "") + (p.stderr or ""), time.monotonic() - t0


def latest_job():
    from streamweights.jobs.engine import Job
    return Job.latest()


def gate():
    from streamweights import probe
    from streamweights.registry import REPO_ROOT
    from streamweights.tune import gate as g
    ws = probe.load()["gpu"]["vram_bytes"]
    model = Path(sys.argv[2]) if len(sys.argv) > 2 else REPO / "models/qwen2.5-0.5b/bf16-st"
    work = REPO / "state" / "gate-work"
    lr = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-4
    res = g.run_gate(model, work, working_set=ws, steps=100, label="qwen2.5:0.5b", lr=lr,
                    natural=None if "toy" in sys.argv[4:] else REPO / "examples/gate-natural.jsonl")
    OUT.mkdir(exist_ok=True)
    (OUT / "008-gate.json").write_text(json.dumps(res, indent=2))
    keep = {k: v for k, v in res.items() if not k.startswith("loss_res") and not k.startswith("loss_str")}
    print(json.dumps(keep, indent=2))
    print("GATE", "PASS" if res["pass"] else "FAIL")
    return 0 if res["pass"] else 1


def toy():
    from streamweights.tune import toy as T
    work = REPO / "state" / "toy-work"
    files = T.make(work)
    base = "qwen2.5:0.5b"
    lines = []
    rc = 0
    for path, name in (("resident", "toy-res"), ("streamed", "toy-str")):
        r, out, dt = spill("tune", base, files["train"], "--name", name, "--overwrite",
                           "--path", path, "--batch", "4", "--lr", "2e-4", "--epochs", "3",
                           "--quiet")
        rc = rc or r
        lines += [f"$ spill tune {base} train.jsonl --name {name} --path {path}", out]
    rc2, out2, _ = spill("eval", files["heldout"], base, f"{base}+toy-res", f"{base}+toy-str",
                         "--rerun", "--quiet")
    lines += ["$ spill eval heldout.jsonl base base+toy-res base+toy-str", out2]
    (OUT / "008-toy.txt").write_text("\n".join(lines))
    print("\n".join(lines))
    return rc or rc2


def gradcheck():
    """Per-tensor gradient cosine against a float32 reference: streamed bf16 and mlx-lm
    bf16, same parameters, same batch. Also audits the dtypes of adapter state."""
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten
    from mlx_lm.tuner.utils import linear_to_lora_layers
    from mlx_lm.utils import load, load_tokenizer
    from streamweights.engines.mlx_stream import collect_eos_ids
    from streamweights.tune import lora as lo
    from streamweights.tune import toy
    from streamweights.tune.data import BatchPlan, load_examples
    from streamweights.tune.streamed import StreamedTrainer, make_optimizer, masked_ce
    M = REPO / "models/qwen2.5-0.5b/bf16-st"
    files = toy.make(REPO / "state" / "gc-work")
    tok = load_tokenizer(M)
    exs, _ = load_examples(Path(REPO / "examples/gate-natural.jsonl"), tok, 256,
                           collect_eos_ids(M, tok))
    b = BatchPlan(exs[:200], 4, 0, 0).batch(0)
    cfg = lo.LoraConfig(rank=16, alpha=32, seed=0)
    tr = StreamedTrainer(M, cfg)
    mx.random.seed(1)
    tr.params = {k: (v if k.endswith("lora_a") else mx.random.normal(v.shape) * 0.02)
                 for k, v in tr.params.items()}
    loss, grads, _ = tr.micro_batch(b.inputs, b.targets, b.mask, 0)
    opt = make_optimizer(cfg, 10)
    tr.apply(opt, grads)
    dt = lambda t: sorted({str(v.dtype) for v in t.values()})
    audit = {"base": str(tr.embed_w.dtype), "adapter_params": dt(tr.params),
             "gradients": dt(grads),
             "adam_state": sorted({str(x.dtype) for k, v in opt.state.items()
                                   if isinstance(v, dict) for x in v.values()})}
    tr.params = {k: (v if k.endswith("lora_a") else v) for k, v in tr.params.items()}

    def ref(dtype):
        model, _ = load(str(M))
        model.freeze()
        linear_to_lora_layers(model, tr.L, {"rank": 16, "scale": 2.0, "dropout": 0.0,
                                            "keys": list(tr.shapes)})
        if dtype:
            model.set_dtype(dtype)
        return model
    # params after the apply() above moved; re-derive the pre-step params for all three
    mx.random.seed(1)
    p0 = lo.init_params(tr.shapes, tr.L, 16, 0)
    p0 = {k: (v if k.endswith("lora_a") else mx.random.normal(v.shape) * 0.02) for k, v in p0.items()}
    tr.params = p0
    loss, grads, _ = tr.micro_batch(b.inputs, b.targets, b.mask, 0)
    out = {}
    for tag, dtype in (("bf16", None), ("f32", mx.float32)):
        m = ref(dtype)
        m.load_weights([("model." + k, v) for k, v in p0.items()], strict=False)
        _, g = nn.value_and_grad(m, lambda mm: masked_ce(mm(mx.array(b.inputs)),
                                 mx.array(b.targets), mx.array(b.mask))[0])(m)
        out[tag] = {k[6:]: v for k, v in tree_flatten(g)}

    def stats(x, y):
        c = {k: lo.cosine_sim(x[k], y[k]) for k in x}
        return {"min": min(c.values()), "mean": sum(c.values()) / len(c),
                "worst": min(c, key=c.get)}
    res = {"dtype_audit": audit,
           "streamed_bf16_vs_f32": stats(grads, out["f32"]),
           "mlxlm_bf16_vs_f32": stats(out["bf16"], out["f32"]),
           "streamed_vs_mlxlm_bf16": stats(grads, out["bf16"])}
    res["streamed_no_worse"] = bool(
        res["streamed_bf16_vs_f32"]["mean"] >= res["mlxlm_bf16_vs_f32"]["mean"]
        and res["streamed_bf16_vs_f32"]["min"] >= res["mlxlm_bf16_vs_f32"]["min"])
    (OUT / "008-gradcheck.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    return 0 if res["streamed_no_worse"] else 1


def smoke70():
    from streamweights.tune import toy as T
    work = REPO / "state" / "smoke-work"
    path = T.make_long(work)
    model = "llama3.3:70b"
    rc, out, dt = spill("tune", model, path, "--name", "smoke70", "--overwrite",
                        "--max-seq", "512", "--steps", "3", capture=True)
    job = latest_job()
    (OUT / "008-smoke70-tune.txt").write_text(out)
    print(out)
    if rc:
        return rc
    sample = work / "sample-4.jsonl"
    lines = (REPO / "streamweights/data/sample-20.jsonl").read_text().splitlines()[:4]
    sample.write_text("\n".join(lines) + "\n")
    rc1, o1, _ = spill("run", model, sample, "--quiet")
    rc2, o2, _ = spill("run", f"{model}+smoke70", sample, "--quiet")
    (OUT / "008-smoke70-run.txt").write_text(o1 + "\n" + o2)
    print(o1[-600:], o2[-600:])
    return rc1 or rc2


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "gate"
    sys.exit({"gate": gate, "toy": toy, "gradcheck": gradcheck, "smoke70": smoke70}[cmd]())
