"""Phase 2 verification on the 0.5b model: the [GPU] items of paste set 006.

    python scripts/verify_phase2.py <out.json> [--rows N]

Runs on the default MLX device (Metal unless SPILL_DEVICE=cpu). Every check compares
streamweights against an independent computation done with mlx-lm's own model:

  1 logits     top-1 equals the generated token; log-probs sum to 1 within 1e-3 after exp
  2 distill    generation mode and teacher-forced (--score) mode vs an mlx-lm reference:
               per-position log-prob difference, top-1 agreement, top-k overlap
  3 adapter    streamed vs resident greedy output identical on 20 prompts; PEFT layout
               identical to mlx-lm layout; adapter log-probs vs mlx-lm's own LoRA
  4 throughput tokens scored per second, both distill modes
Results are written as JSON for docs/reports/006-phase2.md.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from streamweights import logits as lg  # noqa: E402
from streamweights.adapters import load_adapter_dir  # noqa: E402
from streamweights.engines.base import MemoryBudget, ModelSpec  # noqa: E402
from streamweights.engines.mlx_resident import MlxResidentEngine  # noqa: E402
from streamweights.engines.mlx_stream import MlxStreamEngine  # noqa: E402

GIB = 1024**3
MODEL = REPO / "models/qwen2.5-0.5b/bf16-st"
ADAPTER = REPO / "adapters/qwen05-arr"
WS = 36 * GIB


def spec(**extra):
    return ModelSpec("qwen2.5:0.5b", "bf16", MODEL, {}, 4096, extra=extra)


def run_engine(cls, rows, **extra):
    return {c.custom_id: c for c in cls().run_batch(rows, spec(**extra), MemoryBudget(WS))}


def sample_rows(n, max_tokens=32):
    rows = [json.loads(l) for l in open(REPO / "examples/evals-2000.jsonl")][:n]
    for r in rows:
        r["body"]["max_tokens"] = max_tokens
    return rows


def ref_model():
    from mlx_lm import load
    return load(str(MODEL))


def ref_logprobs(model, ids):
    lgt = model(mx.array(ids)[None])[0].astype(mx.float32)
    return lgt - mx.logsumexp(lgt, axis=-1, keepdims=True)


# ---------------------------------------------------------------- 1 logits

def check_logits(n):
    rows = sample_rows(min(n, 20), 24)
    full_dir = Path(tempfile.mkdtemp()) / "full"
    out = run_engine(MlxResidentEngine, rows, logprobs=64, full_logits_dir=str(full_dir))
    top1_ok = tot = 0
    worst_mass = 0.0
    worst_full_sum = 0.0
    worst_vs_full = 0.0
    for cid, cr in out.items():
        full = np.load(full_dir / f"{lg.safe_name(cid)}.npy")
        lp_full = lg.log_softmax_np(full)
        worst_full_sum = max(worst_full_sum, float(np.abs(np.exp(lp_full).sum(-1) - 1).max()))
        for t, rec in enumerate(cr.logprobs):
            tot += 1
            top1_ok += (rec["top"][0][0] == rec["token_id"]
                        or abs(rec["top"][0][1] - rec["logprob"]) < 1e-6)   # exact ties only
            worst_mass = max(worst_mass, sum(np.exp(p) for _, p in rec["top"]) - 1.0)
            worst_vs_full = max(worst_vs_full, abs(lp_full[t, rec["token_id"]] - rec["logprob"]))
    return {"rows": len(out), "tokens": tot, "top1_equals_generated": f"{top1_ok}/{tot}",
            "max_abs_sum_exp_logprobs_minus_1__full_logits": worst_full_sum,
            "max_top64_mass_over_1": worst_mass,
            "max_abs_logprob_vs_full_logits_fp16": worst_vs_full,
            "pass": top1_ok == tot and worst_full_sum < 1e-3 and worst_mass < 1e-3}


# ---------------------------------------------------------------- 2 distill

def compare_positions(model, ids, n_prompt, recs, k):
    ref = ref_logprobs(model, ids)
    diffs, top1, overlap = [], 0, []
    for t, rec in enumerate(recs):
        pos = n_prompt - 1 + t
        row = np.array(ref[pos])
        diffs.append(abs(row[rec["token_id"]] - rec["logprob"]))
        ref_top = set(np.argsort(-row)[:k].tolist())
        mine = {a for a, _ in rec["top"]}
        overlap.append(len(ref_top & mine) / k)
        top1 += int(np.argmax(row) == rec["top"][0][0] or
                    abs(row.max() - rec["top"][0][1]) < 0.15)
    return diffs, top1, overlap


def check_distill(n):
    model, tok = ref_model()
    k = 32
    res = {}
    # --- generation mode: completions come from the engine; verify its per-token
    # distribution by re-scoring (prompt + completion) with mlx-lm
    rows = sample_rows(n, 32)
    gen = run_engine(MlxResidentEngine, rows, logprobs=k)
    d_all, t1, tot, ov = [], 0, 0, []
    for r in rows:
        cr = gen[r["custom_id"]]
        p = list(tok.apply_chat_template(r["body"]["messages"], add_generation_prompt=True))
        ids = p + [rec["token_id"] for rec in cr.logprobs]
        d, a, o = compare_positions(model, ids, len(p), cr.logprobs, k)
        d_all += d
        t1 += a
        tot += len(d)
        ov += o
    res["generation"] = {"rows": len(rows), "positions": tot,
                         "max_abs_logprob_diff": float(max(d_all)),
                         "mean_abs_logprob_diff": float(np.mean(d_all)),
                         "top1_agreement": f"{t1}/{tot}",
                         "mean_topk_overlap": float(np.mean(ov))}
    # --- teacher-forced mode: targets are the engine's own completions plus a set of
    # fixed human targets, so the positions include non-greedy tokens
    targets = []
    for r in rows:
        txt = gen[r["custom_id"]].content
        targets.append({"custom_id": r["custom_id"], "body": {"messages": r["body"]["messages"]
                        + [{"role": "assistant", "content": txt}]}})
    fixed = [("What is the capital of France?", "Paris, of course, is the capital."),
             ("Say hello.", "Salutations, traveller."), ("2+2?", "Four, I believe.")]
    for i, (q, a) in enumerate(fixed):
        targets.append({"custom_id": f"fixed-{i}", "body": {"messages": [
            {"role": "user", "content": q}, {"role": "assistant", "content": a}]}})
    sc = run_engine(MlxResidentEngine, targets, mode="score", logprobs=k)
    d_all, t1, tot, ov, ppl_ok = [], 0, 0, [], 0
    for tr in targets:
        cr = sc[tr["custom_id"]]
        msgs = tr["body"]["messages"]
        n_prompt = len(tok.apply_chat_template(msgs[:-1], add_generation_prompt=True))
        ids = list(tok.apply_chat_template(msgs))
        eot = tok.convert_tokens_to_ids("<|im_end|>")
        ids = ids[:ids.index(eot, n_prompt) + 1]
        assert [r["token_id"] for r in cr.logprobs] == ids[n_prompt:], tr["custom_id"]
        d, a, o = compare_positions(model, ids, n_prompt, cr.logprobs, k)
        d_all += d
        t1 += a
        tot += len(d)
        ov += o
    res["score"] = {"rows": len(targets), "positions": tot,
                    "max_abs_logprob_diff": float(max(d_all)),
                    "mean_abs_logprob_diff": float(np.mean(d_all)),
                    "top1_agreement": f"{t1}/{tot}",
                    "mean_topk_overlap": float(np.mean(ov))}
    # streamed engine agrees with resident, exactly
    sc_s = run_engine(MlxStreamEngine, targets, mode="score", logprobs=k)
    res["score"]["streamed_equals_resident"] = all(
        [x["logprob"] for x in sc[c].logprobs] == [x["logprob"] for x in sc_s[c].logprobs]
        for c in sc)
    # Generation re-scores batched, left-padded decode against a single-sequence mlx-lm
    # forward; logits are bf16, whose spacing at magnitude 16-32 is 0.125, so differing
    # reduction order can move a log-prob by 1-2 ulps. Gate: 0.25 (2 ulps) for generation;
    # teacher-forced scoring runs the same prefill math as the reference and must be ~exact.
    res["pass"] = (res["generation"]["max_abs_logprob_diff"] < 0.25
                   and res["score"]["max_abs_logprob_diff"] < 1e-3
                   and res["score"]["streamed_equals_resident"])
    return res


# ---------------------------------------------------------------- 3 adapter

def peft_copy(src: Path) -> Path:
    dst = Path(tempfile.mkdtemp())
    raw = mx.load(str(src / "adapters.safetensors"))
    cfg = json.loads((src / "adapter_config.json").read_text())
    out = {}
    for k, v in raw.items():
        base, _, which = k.rpartition(".lora_")
        out[f"base_model.model.{base}.lora_{which.upper()}.weight"] = v.T
    mx.save_safetensors(str(dst / "adapter_model.safetensors"), out)
    r = cfg["lora_parameters"]["rank"]
    (dst / "adapter_config.json").write_text(json.dumps(
        {"peft_type": "LORA", "r": r, "lora_alpha": cfg["lora_parameters"]["scale"] * r}))
    return dst


def check_adapter():
    rows = [json.loads(l) for l in open(REPO / "streamweights/data/sample-20.jsonl")]
    for r in rows:
        r["body"]["max_tokens"] = 32
    res_ = run_engine(MlxResidentEngine, rows, adapter=load_adapter_dir(ADAPTER))
    stm = run_engine(MlxStreamEngine, rows, adapter=load_adapter_dir(ADAPTER))
    base = run_engine(MlxResidentEngine, rows)
    identical = sum(res_[c].content == stm[c].content for c in res_)
    arr = sum("Arr" in res_[c].content for c in res_)
    peft = run_engine(MlxResidentEngine, rows, adapter=load_adapter_dir(peft_copy(ADAPTER)))
    peft_same = sum(peft[c].content == res_[c].content for c in res_)
    # adapter vs mlx-lm's own LoRA, teacher-forced
    from mlx_lm import load
    m2, tok = load(str(MODEL), adapter_path=str(ADAPTER))
    worst = 0.0
    for i, (q, a) in enumerate([("What is the capital of France?", "Paris. Arr!"),
                                ("Name a fruit.", "Apple. Arr!"),
                                ("Say hi.", "Hello there, friend.")]):
        msgs = [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
        got = run_engine(MlxResidentEngine, [{"custom_id": "x", "body": {"messages": msgs}}],
                         adapter=load_adapter_dir(ADAPTER), mode="score", logprobs=4)["x"]
        n_prompt = len(tok.apply_chat_template(msgs[:1], add_generation_prompt=True))
        ids = list(tok.apply_chat_template(msgs))
        ids = ids[:ids.index(tok.convert_tokens_to_ids("<|im_end|>"), n_prompt) + 1]
        ref = ref_logprobs(m2, ids)
        worst = max(worst, max(abs(float(ref[n_prompt - 1 + t, r["token_id"]]) - r["logprob"])
                               for t, r in enumerate(got.logprobs)))
    return {"prompts": len(rows),
            "streamed_equals_resident": f"{identical}/{len(rows)}",
            "prompts_with_adapter_signature": f"{arr}/{len(rows)}",
            "base_changed_by_adapter": sum(base[c].content != res_[c].content for c in res_),
            "peft_layout_equals_mlx_layout": f"{peft_same}/{len(rows)}",
            "max_abs_logprob_diff_vs_mlx_lm_lora": worst,
            "pass": identical == len(rows) and peft_same == len(rows) and worst < 0.2}


# ---------------------------------------------------------------- 4 throughput

def check_throughput(n):
    rows = sample_rows(n, 128)
    t0 = time.monotonic()
    gen = run_engine(MlxResidentEngine, rows, logprobs=32)
    dt = time.monotonic() - t0
    comp = sum(c.completion_tokens for c in gen.values())
    prompt = sum(c.prompt_tokens for c in gen.values())
    out = {"generation": {"rows": n, "seconds": round(dt, 2), "completion_tokens": comp,
                          "prompt_tokens": prompt,
                          "tokens_scored_per_s_completion": round(comp / dt, 1),
                          "tokens_scored_per_s_prompt_plus_completion": round(
                              (comp + prompt) / dt, 1)}}
    targets = [{"custom_id": r["custom_id"], "body": {"messages": r["body"]["messages"] + [
        {"role": "assistant", "content": gen[r["custom_id"]].content}]}} for r in rows]
    t0 = time.monotonic()
    sc = run_engine(MlxResidentEngine, targets, mode="score", logprobs=32)
    dt = time.monotonic() - t0
    tgt = sum(c.completion_tokens for c in sc.values())
    pre = sum(c.prompt_tokens + c.completion_tokens for c in sc.values())
    out["score"] = {"rows": n, "seconds": round(dt, 2), "target_tokens": tgt,
                    "prefill_tokens": pre,
                    "target_tokens_scored_per_s": round(tgt / dt, 1),
                    "prefill_tokens_per_s": round(pre / dt, 1)}
    return out


def _py(o):
    if isinstance(o, dict):
        return {k: _py(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_py(v) for v in o]
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    return o


def main():
    out_path = Path(sys.argv[1])
    n = int(sys.argv[sys.argv.index("--rows") + 1]) if "--rows" in sys.argv else 50
    res = {"device": str(mx.default_device()), "mlx": mx.__version__ if hasattr(mx, "__version__")
           else None, "git": subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO,
                                            capture_output=True, text=True).stdout.strip()}
    for name, fn in (("logits", lambda: check_logits(n)), ("distill", lambda: check_distill(n)),
                     ("adapter", check_adapter), ("throughput", lambda: check_throughput(n))):
        t0 = time.monotonic()
        res[name] = _py(fn())
        res[name]["wall_s"] = round(time.monotonic() - t0, 1)
        print(name, json.dumps(res[name], indent=1), flush=True)
        out_path.write_text(json.dumps(res, indent=2))
    res["all_pass"] = all(res[k].get("pass", True) for k in ("logits", "distill", "adapter"))
    out_path.write_text(json.dumps(res, indent=2))
    print("ALL PASS" if res["all_pass"] else "FAILURES")


if __name__ == "__main__":
    main()
