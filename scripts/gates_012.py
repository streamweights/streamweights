"""The run-anywhere gates on the 0.5B, on this Mac (items 7 and 8 of directive 012).

  python scripts/gates_012.py identity   torch streamed vs resident, torch vs MLX, tunes, f32
  python scripts/gates_012.py bf16       bf16 statistics next to batch-shape noise
  python scripts/gates_012.py resume     jobs that move between MLX and torch-cpu, both directions
  python scripts/gates_012.py fixture    write the committed step-50 MLX checkpoint fixture

Results merge into docs/reports/012-gates.json.
"""

import json
import shutil
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
OUT = REPO / "docs" / "reports" / "012-gates.json"
WORK = REPO / "state" / "files" / "gates-012"
F32 = REPO / "models/qwen2.5-0.5b/f32-st"
BF16 = REPO / "models/qwen2.5-0.5b/bf16-st"
NATURAL = REPO / "examples/gate-natural.jsonl"
SAMPLE = REPO / "streamweights/data/sample-20.jsonl"

from streamweights import formats, gates as G  # noqa: E402
from streamweights.tune import toy  # noqa: E402


def save(key: str, value):
    cur = json.loads(OUT.read_text()) if OUT.exists() else {}
    cur[key] = value
    OUT.write_text(json.dumps(cur, indent=1, default=str))


def brief(d: dict) -> dict:
    return {k: v for k, v in d.items() if not k.startswith("losses")}


def work() -> G.Work:
    return G.Work(WORK, models_from=REPO / "models", state_from=REPO / "state")


def identity():
    rows = formats.load_rows(SAMPLE)
    w = work()
    t0 = time.monotonic()
    a = G.gate_inference_stream_vs_resident(F32, rows)
    save("7a_torch_streamed_vs_resident_f32", a)
    print("7a", json.dumps(brief(a))[:600], "PASS" if a["pass"] else "FAIL", flush=True)
    b = G.gate_inference_vs_mlx(F32, rows)
    save("7b_torch_vs_mlx_f32", b)
    print("7b", json.dumps(brief(b))[:600], "PASS" if b["pass"] else "FAIL", flush=True)
    c = G.gate_tune_identity(w, str(F32), str(NATURAL),
                             a={"engine": "torch-cpu", "path": "streamed"},
                             b={"engine": "torch-cpu", "path": "resident"},
                             steps=50, lr=G.GATE_LR, loss_tol=1e-3, cos_tol=0.9999, tag="g7c")
    save("7c_torch_streamed_tune_vs_peft_resident_f32", c)
    print("7c", json.dumps(brief(c)), "PASS" if c["pass"] else "FAIL", flush=True)
    d = G.gate_tune_identity(w, str(F32), str(NATURAL),
                             a={"engine": "torch-cpu", "path": "streamed"},
                             b={"engine": "mlx", "path": "streamed"},
                             steps=50, lr=G.GATE_LR, loss_tol=0.01, cos_tol=None, tag="g7d")
    save("7d_torch_vs_mlx_tune_f32_natural", d)
    print("7d natural", json.dumps(brief(d)), "PASS" if d["pass"] else "FAIL", flush=True)
    toy_identity(w)
    print("identity gates took %.0f s" % (time.monotonic() - t0))


def toy_identity(w):
    files = toy.make(WORK / "toy", 200, 60)
    dt = G.gate_tune_identity(w, str(F32), files["train"],
                              a={"engine": "torch-cpu", "path": "streamed"},
                              b={"engine": "mlx", "path": "streamed"},
                              steps=50, lr=G.TOY_LR, loss_tol=0.01, cos_tol=None, tag="g7t")
    sc_t = G.eval_scores(w, files["heldout"], [f"{F32}", f"{F32}+g7t-a", f"{F32}+g7t-b"],
                         "torch-cpu", "float32")
    dt["scores_torch_adapter_vs_mlx_adapter"] = sc_t
    dt["scores_pass"] = abs(sc_t[f"{F32}+g7t-a"] - sc_t[f"{F32}+g7t-b"]) <= 2 / 60
    save("7d_torch_vs_mlx_tune_f32_toy", dt)
    print("7d toy", json.dumps(brief(dt)), "PASS" if dt["pass"] and dt["scores_pass"] else "FAIL",
          flush=True)


def bf16():
    rows = formats.load_rows(SAMPLE)
    for r in rows:
        r["body"]["max_tokens"] = 24
    inf = G.bf16_inference_statistics(BF16, F32, rows[:10])
    save("7e_bf16_inference_statistics", inf)
    print("7e inference", json.dumps(inf, indent=1), flush=True)
    grad = G.bf16_gradient_statistics(BF16, F32, NATURAL)
    save("7e_bf16_gradient_statistics", grad)
    print("7e gradients", json.dumps(grad, indent=1), flush=True)


def resume():
    w = work()
    files = toy.make(WORK / "toy", 200, 60)
    for first, then in (("mlx", "torch-cpu"), ("torch-cpu", "mlx")):
        r = G.gate_resume_tune(w, str(BF16), str(NATURAL), first=first, then=then, steps=100,
                               at=50, lr=G.GATE_LR, tag=f"n-{first}")
        save(f"8_tune_{first}_to_{then}_natural", r)
        print(f"8 tune natural {first}->{then}", json.dumps(brief(r))[:900],
              "PASS" if r["pass"] else "FAIL", flush=True)
        r = G.gate_resume_tune(w, str(BF16), files["train"], first=first, then=then, steps=100,
                               at=50, lr=G.TOY_LR, tag=f"t-{first}", heldout=files["heldout"])
        save(f"8_tune_{first}_to_{then}_toy", r)
        print(f"8 tune toy {first}->{then}", json.dumps(brief(r))[:900],
              "PASS" if r["pass"] else "FAIL", flush=True)
    for first, then in (("mlx", "torch-cpu"), ("torch-cpu", "mlx")):
        r = G.gate_resume_rows(w, files["heldout"], "qwen2.5:0.5b", first=first, then=then,
                               half=30, tag=f"rows-{first}")
        save(f"8_rows_{first}_to_{then}", r)
        print(f"8 rows {first}->{then}", json.dumps(r)[:900], "PASS" if r["pass"] else "FAIL",
              flush=True)


if __name__ == "__main__":
    OUT.parent.mkdir(exist_ok=True)
    for cmd in sys.argv[1:] or ["identity"]:
        {"identity": identity, "bf16": bf16, "resume": resume,
         "toy": lambda: toy_identity(work())}[cmd]()
