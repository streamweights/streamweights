"""Controlled export comparison for one completed project (directive 017, item 4).

  python scripts/diagnose_export.py --project P --out result.json [--rows 8]

Everything is held identical across the arms: the verification rows (the first N validation rows of
the run), the input messages, the rendered prompts and their token ids (rendered once by the
Transformers tokenizer and fed as ids to every runtime), greedy decoding and max_tokens. Hardware is
the CPU for every arm (Transformers float32, llama.cpp with no GPU layers). Arms:

  source-engine   the run's saved trained-student predictions (the engine that trained it)
  unmerged        the base with the adapter applied (PEFT), Transformers float32
  merge-f32       a float32 merge, Transformers float32
  merge-bf16      a bf16 merge (the base's dtype), Transformers float32
  gguf-f32        unquantized F32 GGUF of the float32 merge, llama.cpp
  gguf-q8-from-f32   q8_0 GGUF derived from the float32 merge, llama.cpp
  gguf-q8-from-bf16  q8_0 GGUF derived from the bf16 merge, llama.cpp

For each arm: the task score on the verification rows (the config's selected metric), the source
engine's score on the same rows, the text-disagreement rate against the source engine, the
schema-validity rate (JSON), artifact size and peak process memory, and the executed conditions."""

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rows", type=int, default=8)
    ap.add_argument("--work")
    a = ap.parse_args()
    from streamweights import export as ex
    from streamweights.adapters import load_adapter_dir
    from streamweights.llamacpp import ensure_llama_server
    from streamweights.project import config as C
    from streamweights.project import contract as K
    from streamweights.project import modelid
    from streamweights.project import report as R
    from streamweights.project.common import read_json, read_jsonl
    from streamweights.project.testrec import pick_run

    project = Path(a.project)
    m = pick_run(project, None)
    rdir = project / "runs" / m["run_id"]
    cfg = C.parse((rdir / "inputs" / "streamweights.toml").read_bytes())
    task = cfg["task"]["type"]
    mk = R.metric_key(cfg)
    schema = read_json(rdir / "inputs" / cfg["contract"]["schema_file"]) if task == "json" else None
    labels = cfg["contract"].get("labels")
    work = Path(a.work or (project / ".diagnose"))
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    sid = modelid.student_identity(cfg["model"]["student"], fetch=False)
    base = Path(sid["dir"])
    adapter_dir = rdir / "artifacts" / "adapter"
    val = read_jsonl(rdir / "inputs" / "val.jsonl")[:a.rows]
    src = {p["id"]: p for p in read_jsonl(rdir / "predictions" / "trained.jsonl")}
    mt = cfg["evaluation"]["max_tokens"]
    # render once: the same token ids for every arm
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(base))
    rows = []
    for r in val:
        msgs = K.messages_student(cfg, r["input"])
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True)["input_ids"]
        rows.append({"id": r["id"], "messages": msgs, "prompt_ids": list(ids), "max_tokens": mt})
    srv = str(ensure_llama_server().resolve())

    def score(text, r):
        return (K.score_class(text, r["output"], labels) if task == "classification"
                else K.score_json(text, r["output"], schema))

    def agg(items):
        golds = [r["output"] for r in val]
        return (K.agg_class(items, golds, labels) if task == "classification" else K.agg_json(items, schema, golds))

    src_items = [score(src[r["id"]]["text"], r) for r in val]
    src_metrics = agg(src_items)
    out = {"task": task, "run": m["run_id"], "metric": R.primary_name(mk), "metric_version": C.metric_version(cfg),
           "verification_rows": [r["id"] for r in val], "n": len(val), "max_tokens": mt,
           "decoding": "greedy: Transformers do_sample=False; llama.cpp temperature 0, top_k 1, seed 0",
           "tokenization": "prompts rendered once by the Transformers chat template and passed to every "
                           "runtime as token ids",
           "source_engine": {"score": R.primary(mk, src_metrics),
                             "schema_valid_rate": src_metrics.get("schema_valid_rate"),
                             "engine_conditions": (m.get("evaluation") or {}).get("conditions", {}).get("trained")},
           "arms": {}}

    def run_arm(name, runtime, path, extra=None, size=None):
        spec = {"runtime": runtime, "path": str(path), "rows": rows, "out": str(work / f"{name}.json"),
                **({"llama_server": srv, "ngl": 0} if runtime == "llamacpp" else {}), **(extra or {})}
        (work / f"{name}.spec.json").write_text(json.dumps(spec))
        t0 = time.monotonic()
        subprocess.run([sys.executable, "-m", "streamweights.project.verify_export", str(work / f"{name}.spec.json")],
                       capture_output=True)
        res = json.loads((work / f"{name}.json").read_text()) if (work / f"{name}.json").exists() else {"error": "no result"}
        arm = {"runtime": res.get("runtime"), "settings": res.get("settings"), "error": res.get("error"),
               "seconds": round(time.monotonic() - t0, 1), "host": res.get("host"), "size_bytes": size,
               "peak_rss_bytes": res.get("peak_rss_bytes")}
        if not res.get("error"):
            preds = {p["id"]: p["text"] for p in res["predictions"]}
            items = [score(preds[r["id"]], r) for r in val]
            mt_ = agg(items)
            diffs = [r["id"] for r in val if preds[r["id"]].strip() != src[r["id"]]["text"].strip()]
            arm.update(score=R.primary(mk, mt_), source_engine_score=R.primary(mk, src_metrics),
                       delta_vs_source=round(R.primary(mk, mt_) - R.primary(mk, src_metrics), 6),
                       text_disagreement_rate=len(diffs) / len(val), text_disagreements=len(diffs),
                       schema_valid_rate=mt_.get("schema_valid_rate"), parseable_rate=mt_.get("parseable_rate"),
                       correct_rows=sum(1 for r, it in zip(val, items)
                                        if (it["correct"] if task == "classification" else it["record"])),
                       outputs={r["id"]: preds[r["id"]] for r in val})
        out["arms"][name] = arm
        print(name, arm.get("score"), arm.get("text_disagreements"), arm.get("error"), flush=True)
        return arm

    adapter_np = load_adapter_dir(adapter_dir, numpy=True)
    dirsize = lambda d: sum(f.stat().st_size for f in Path(d).glob("*.safetensors"))
    # unmerged: PEFT needs the PEFT layout
    peft_dir = adapter_dir
    if not (adapter_dir / "adapter_model.safetensors").exists():
        peft_dir = None
    if peft_dir is not None:
        run_arm("unmerged", "transformers", base, {"peft_adapter": str(peft_dir)}, dirsize(base))
    else:
        out["arms"]["unmerged"] = {"error": "the saved adapter is in the MLX layout only; PEFT cannot load it"}
    for dt, name in (("float32", "merge-f32"), ("bf16", "merge-bf16")):
        d = work / name
        ex.merge_adapter(base, adapter_np, d, merge_dtype=dt)
        run_arm(name, "transformers", d, None, dirsize(d))
    ggufs = {}
    for src_name, label in (("merge-f32", "f32"), ("merge-bf16", "bf16")):
        d = work / src_name
        if src_name == "merge-f32":
            g = ex.convert_gguf(d, "diag-f32", "f32")
            ggufs["gguf-f32"] = g
            run_arm("gguf-f32", "llamacpp", g, None, g.stat().st_size)
            g8 = ex.convert_gguf(d, "diag-f32src", "q8_0")
            run_arm("gguf-q8-from-f32", "llamacpp", g8, None, g8.stat().st_size)
        else:
            g8 = ex.convert_gguf(d, "diag-bf16src", "q8_0")
            run_arm("gguf-q8-from-bf16", "llamacpp", g8, None, g8.stat().st_size)
    # runtime comparison: float32 merge in Transformers vs F32 GGUF in llama.cpp, row by row
    t, g = out["arms"]["merge-f32"], out["arms"]["gguf-f32"]
    if "outputs" in t and "outputs" in g:
        out["runtime_comparison"] = {
            "transformers_f32_vs_llamacpp_f32_text_disagreements": sum(
                1 for k in t["outputs"] if t["outputs"][k].strip() != g["outputs"][k].strip()),
            "of": len(t["outputs"])}
    Path(a.out).write_text(json.dumps(out, indent=1))
    shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
