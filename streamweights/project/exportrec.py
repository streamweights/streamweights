"""spill export <project>: a new export record from a completed run.

exports/<id>/ holds the artifacts (merged bf16 safetensors, optionally a GGUF), a script that
loads the artifact and runs one input, and record.json: base model and revision, tokenizer and
prompt template, adapter and merge status, format and quantization, the source run and its
evaluation references, the verification and the deployment measurements.

The verification is part of the export and is finished before the record is: the actual
artifact is loaded in an independent runtime (Transformers for safetensors, llama.cpp for
GGUF), in its own process, on a fixed small subset of the VALIDATION rows (never the final
test), and its predictions are compared with the training engine's. A later verification is a
new record that references the export; it never edits this one. A failure is kept as a failed
record."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

from ..errors import SpillError
from . import config as C
from . import contract as K
from . import report as R
from .common import new_id, read_json, read_jsonl, sha_file, write_json
from .index import records, write_index
from .runstate import make_readonly
from .testrec import pick_run

VERIFY_ROWS = 8
DEFAULT_MERGE_DTYPE = "bf16"       # set by the pre-stated rule in docs/reports/017-export-default-rule.md


def _files_sha(d: Path, skip=()) -> dict:
    return {str(p.relative_to(d)): {"sha256": sha_file(p), "bytes": p.stat().st_size}
            for p in sorted(Path(d).rglob("*")) if p.is_file() and p.name not in skip}


def _script_transformers() -> str:
    return '''"""Load the exported model and run one input.   python run_safetensors.py "your text" """
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
MODEL = HERE / "artifacts" / "merged"
SYSTEM = (HERE / "system.txt").read_text().strip() if (HERE / "system.txt").exists() else None
MAX_NEW_TOKENS = __MAX_TOKENS__

text = " ".join(sys.argv[1:]) or sys.stdin.read()
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
messages = ([{"role": "system", "content": SYSTEM}] if SYSTEM else []) + [{"role": "user", "content": text}]
ids = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True)
with torch.no_grad():
    out = model.generate(**ids, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
print(tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).strip())
'''


def _script_gguf() -> str:
    return '''"""Load the exported GGUF with llama.cpp and run one input.
   python run_gguf.py "your text"      (LLAMA_SERVER=/path/to/llama-server if it is not on PATH)"""
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
GGUF = next((HERE / "artifacts" / "gguf").glob("*.gguf"))
SYSTEM = (HERE / "system.txt").read_text().strip() if (HERE / "system.txt").exists() else None
MAX_TOKENS = __MAX_TOKENS__
server = os.environ.get("LLAMA_SERVER") or shutil.which("llama-server") or "__LLAMA_SERVER__"
text = " ".join(sys.argv[1:]) or sys.stdin.read()
s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
proc = subprocess.Popen([server, "-m", str(GGUF), "-c", "4096", "--port", str(port), "--host",
                         "127.0.0.1", "--no-webui"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
try:
    for _ in range(600):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2); break
        except Exception:
            time.sleep(0.2)
    messages = ([{"role": "system", "content": SYSTEM}] if SYSTEM else []) + [{"role": "user", "content": text}]
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", method="POST",
        data=json.dumps({"messages": messages, "max_tokens": MAX_TOKENS, "temperature": 0, "top_k": 1}).encode(),
        headers={"Content-Type": "application/json"})
    print(json.load(urllib.request.urlopen(req, timeout=600))["choices"][0]["message"]["content"].strip())
finally:
    proc.terminate()
'''


def run_export(project: Path, run_id: str | None = None, gguf: str | None = None,
               verify_rows: int = VERIFY_ROWS, say=print, merge_dtype: str = "bf16") -> dict:
    from .. import export as ex
    from ..adapters import load_adapter_dir
    from . import modelid
    project = Path(project)
    m = pick_run(project, run_id)
    rdir = project / "runs" / m["run_id"]
    cfg = C.parse((rdir / "inputs" / "streamweights.toml").read_bytes())
    task = cfg["task"]["type"]
    sid = modelid.student_identity(cfg["model"]["student"], fetch=False)
    want = {k: v["sha256"] for k, v in m["models"]["student"]["files"].items()}
    have = {k: v["sha256"] for k, v in sid["files"].items()}
    if not sid.get("present") or any(have.get(k) != v for k, v in want.items() if k in have) or \
            set(want) - set(have):
        raise SpillError("the base model on this machine is not the one this run was built on "
                         "(file hashes differ or are missing)", f"spill plan {project}")
    eid = new_id("export-")
    stg = project / "exports" / f".staging-{eid}"
    final = project / "exports" / eid
    shutil.rmtree(stg, ignore_errors=True)
    stg.mkdir(parents=True)
    t0 = time.monotonic()
    rec = {"schema": 1, "id": eid, "kind": "export", "source_run": m["run_id"],
           "source_manifest_sha256": sha_file(rdir / "manifest.json"),
           "source_identity": m["identity"],
           "source_evaluation": {"results": f"runs/{m['run_id']}/results.json",
                                 "protocol_sha256": m["protocol"]["sha256"],
                                 "table": m["table"]},
           "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "status": "running",
           "format": ["safetensors"] + ([f"gguf:{gguf}"] if gguf else [])}
    try:
        base = Path(sid["dir"])
        adapter = load_adapter_dir(rdir / "artifacts" / "adapter", numpy=True)
        say("merging the adapter into the bf16 base (float32 arithmetic, rounded back to bf16)")
        info = ex.merge_adapter(base, adapter, stg / "artifacts" / "merged", say=say,
                                merge_dtype=merge_dtype)
        merged = stg / "artifacts" / "merged"
        tok_files = {n: v for n, v in _files_sha(merged).items()
                     if n.startswith("tokenizer") or n in ("vocab.json", "merges.txt")}
        tcfg = json.loads((merged / "tokenizer_config.json").read_text())
        tmpl = tcfg.get("chat_template") or ""
        rec["base_model"] = {k: m["models"]["student"][k] for k in ("tag", "repo", "revision", "files")}
        rec["tokenizer"] = {"files": tok_files, "chat_template_sha256":
                            __import__("hashlib").sha256(tmpl.encode()).hexdigest()}
        rec["prompt_template"] = {"messages": K.messages_student(cfg, "{input}"),
                                  "rendered_with": "the tokenizer's chat template, "
                                                   "add_generation_prompt=True"}
        rec["adapter"] = {"files_sha256": m["adapter_files_sha256"], "rank": adapter.rank,
                          "modules_merged": info["modules"], "merged": True, "merge_dtype": merge_dtype,
                          "merge": "W' = W + (alpha/r) * (A @ B)^T per adapted linear, computed in float32, "
                                   "written as " + str(info["dtype"]) + (" (every tensor float32)" if merge_dtype == "float32"
                                                                       else " (the base's dtype: the delta is rounded into bf16)")}
        rec["quantization"] = {"safetensors": f"none ({'float32' if merge_dtype == 'float32' else 'bf16'} weights)"}
        arts = {"safetensors": {"path": "artifacts/merged", "files": _files_sha(merged)}}
        runtimes = [("transformers", str(merged), "safetensors")]
        if gguf:
            if gguf not in ex.GGUF_TYPES:
                raise SpillError(f"--gguf must be one of {', '.join(ex.GGUF_TYPES)}", "spill export --help")
            say(f"converting to GGUF {gguf} with llama.cpp's own converter")
            out = ex.convert_gguf(merged, cfg["project"]["name"], gguf, say=say)
            gd = stg / "artifacts" / "gguf"
            gd.mkdir(parents=True)
            shutil.move(str(out), gd / out.name)
            arts["gguf"] = {"path": f"artifacts/gguf/{out.name}",
                            "files": _files_sha(gd)}
            rec["quantization"]["gguf"] = gguf
            runtimes.append(("llamacpp", str(gd / out.name), f"gguf:{gguf}"))
        rec["artifacts"] = arts
        # ---- the scripts: a working script per format, run once as part of the verification
        mt = cfg["evaluation"]["max_tokens"]
        if cfg["task"].get("system"):
            (stg / "system.txt").write_text(cfg["task"]["system"] + "\n")
        (stg / "run_safetensors.py").write_text(_script_transformers().replace("__MAX_TOKENS__", str(mt)))
        rec["scripts"] = {"safetensors": "run_safetensors.py"}
        if gguf:
            from ..llamacpp import ensure_llama_server
            srv = str(ensure_llama_server().resolve())
            (stg / "run_gguf.py").write_text(_script_gguf().replace("__MAX_TOKENS__", str(mt))
                                             .replace("__LLAMA_SERVER__", srv))
            rec["scripts"]["gguf"] = "run_gguf.py"
        # ---- verification on a fixed validation subset
        val = read_jsonl(rdir / "inputs" / "val.jsonl")[:verify_rows]
        tr = {p["id"]: p for p in read_jsonl(rdir / "predictions" / "trained.jsonl")}
        rows = [{"id": r["id"], "messages": K.messages_student(cfg, r["input"]), "max_tokens": mt}
                for r in val]
        schema = read_json(rdir / "inputs" / cfg["contract"]["schema_file"]) if task == "json" else None
        from .. import probe as probe_mod
        from ..runs import hardware_summary
        try:
            hw = hardware_summary(probe_mod.load(probe_if_missing=False))
        except (OSError, FileNotFoundError):
            hw = {}
        ver = {"subset": {"source": "the first rows of the run's validation split, in file order",
                          "never_final_test": True, "row_ids": [r["id"] for r in val],
                          "n": len(val)}, "runs": {}}
        dep = {}
        ok_all = True
        for rt, path, label in runtimes:
            say(f"verifying {label} with {'llama.cpp' if rt == 'llamacpp' else 'Transformers'} "
                f"on {len(rows)} validation rows")
            spec = {"runtime": rt, "path": path, "rows": rows, "out": str(stg / f"verify-{rt}.json")}
            if rt == "llamacpp":
                spec["llama_server"] = srv
            (stg / f"verify-{rt}.spec.json").write_text(json.dumps(spec))
            subprocess.run([sys.executable, "-m", "streamweights.project.verify_export",
                            str(stg / f"verify-{rt}.spec.json")], capture_output=True)
            res = read_json(stg / f"verify-{rt}.json") if (stg / f"verify-{rt}.json").exists() \
                else {"error": "the verification process produced no result"}
            entry = {"runtime": res.get("runtime"), "settings": res.get("settings"),
                     "load_failure": res.get("error")}
            if res.get("error"):
                ok_all = False
            else:
                preds = {p["id"]: p for p in res["predictions"]}
                diffs, items, golds = [], [], []
                for r in val:
                    ex_text = preds[r["id"]]["text"]
                    ref = tr.get(r["id"], {}).get("text")
                    if (ref or "").strip() != ex_text.strip():
                        diffs.append({"id": r["id"], "engine_prediction": ref, "export_prediction": ex_text})
                    items.append(K.score_class(ex_text, r["output"], cfg["contract"]["labels"])
                                 if task == "classification" else K.score_json(ex_text, r["output"], schema))
                    golds.append(r["output"])
                metrics = (K.agg_class(items, golds, cfg["contract"]["labels"]) if task == "classification"
                           else K.agg_json(items, schema, golds))
                src_items = [K.score_class(tr.get(r["id"], {}).get("text"), r["output"], cfg["contract"]["labels"])
                             if task == "classification" else
                             K.score_json(tr.get(r["id"], {}).get("text"), r["output"], schema) for r in val]
                src_metrics = (K.agg_class(src_items, golds, cfg["contract"]["labels"]) if task == "classification"
                               else K.agg_json(src_items, schema, golds))
                mk = R.metric_key(cfg)
                src_p, art_p = R.primary(mk, src_metrics), R.primary(mk, metrics)
                entry.update(prediction_differences={"count": len(diffs), "of": len(val), "rows": diffs},
                             metrics=metrics, primary=art_p,
                             quality_delta={"metric": R.primary_name(mk), "rows": len(val),
                                            "source_engine_score": src_p, "artifact_score": art_p,
                                            "delta": (None if src_p is None or art_p is None else round(art_p - src_p, 6)),
                                            "text_disagreement_rate": len(diffs) / max(1, len(val)),
                                            "schema_valid_rate": metrics.get("schema_valid_rate"),
                                            "note": "same rows, same prompts and decoding; the source engine's "
                                                    "outputs are the run's saved predictions, rescored now"})
                rec.setdefault("deployment", {})[label] = {
                    "hardware": hw, "host": res["host"], "runtime": res["runtime"],
                    "input_tokens": [p["prompt_tokens"] for p in res["predictions"]],
                    "output_tokens": [p["completion_tokens"] for p in res["predictions"]],
                    "load_seconds": res["load_seconds"], "ttft_cold_s": res["ttft_cold_s"],
                    "ttft_warm_s": res["ttft_warm_s"], "tokens_per_s_warm": res["tokens_per_s_warm"],
                    "peak_memory_bytes": res["peak_rss_bytes"] if res.get("peak_rss_bytes") else "unavailable",
                    "boundaries": res["boundaries"],
                    "kept_separate_from": "build durations (these are inference measurements)"}
            ver["runs"][label] = entry
        # the script itself, once per format
        sample = val[0]["input"]
        ver["scripts"] = {}
        for fmt, name in rec["scripts"].items():
            r = subprocess.run([sys.executable, str(stg / name), sample], capture_output=True, text=True)
            ver["scripts"][fmt] = {"script": name, "input": sample, "exit_code": r.returncode,
                                   "output": r.stdout.strip()[:500],
                                   "error": r.stderr.strip()[-300:] if r.returncode else None}
            ok_all = ok_all and r.returncode == 0
        for rt, path, label in runtimes:
            for f in ("spec", ):
                (stg / f"verify-{rt}.{f}.json").unlink(missing_ok=True)
        rec["verification"] = ver
        rec["completion_means"] = ("the artifact loaded and verification ran; it does not mean quality was "
                                   "preserved: read quality_delta in each verification run")
        rec.update(status="completed" if ok_all else "failed",
                   seconds=round(time.monotonic() - t0, 1),
                   finished=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
        if not ok_all:
            rec["error"] = "verification did not complete: see verification.runs and scripts"
    except Exception as e:
        rec.update(status="failed", error=f"{type(e).__name__}: {e}",
                   seconds=round(time.monotonic() - t0, 1),
                   finished=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    write_json(stg / "record.json", rec)
    stg.rename(final)
    make_readonly(final)
    write_index(project)
    if rec["status"] != "completed":
        raise SpillError(f"the export did not verify: {rec.get('error')}; the failed record "
                         f"was kept at {final}", f"spill export {project} {m['run_id']}")
    return rec
