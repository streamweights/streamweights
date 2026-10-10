"""Render docs/reports/017-scorer-and-protocol.md and 017-export-diagnosis.md from the JSON in
docs/reports/data/017-*.json. Numbers are read from those files; the default merge dtype is
computed from the export comparison by the rule in docs/reports/017-export-default-rule.md and the
script fails if the code's default disagrees."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
D = ROOT / "docs/reports/data"
L = lambda n: json.loads((D / n).read_text())
f3 = lambda x: "unavailable" if x is None else f"{x:.3f}"
gates = L("017-gates.json")
dec = L("017-decoding.json")
b1 = L("017-batch1.json")
NAMES = {"baseline": "embedding baseline", "untrained": "student, prompted, untrained",
         "trained": "student, trained"}
hw = gates["hardware"]
out = ["# 017: scorer and protocol correctness", "",
       "Directive: `docs/paste-sets/017-correctness-and-tagline.md`. Raw results: `docs/reports/data/017-*.json`. "
       f"Hardware: {hw['cpu']}, {hw['ram_gb']} GB, macOS (Darwin {hw['os_release']}) {hw['arch']}.", "",
       "## The JSON scorer (metric version 2)", "",
       "Before: `score_json` compared fields first and ignored schema validity when deciding correctness, so an "
       "output that parsed but violated the schema (for example `{\"status\": \" approved \"}` against a schema "
       "that allows only `approved`) could be credited after trimming. Now whole-record correctness requires the "
       "output to parse and to be schema-valid; a parseable but schema-invalid answer still counts toward the "
       "parseable rate and gets no field-level or whole-record credit. The regression test is "
       "`tests/test_json_scorer.py`. The JSON metric version is 2, classification stays 1, and runs built under "
       "different metric versions are not ranked together.", "",
       "Audit of the other JSON metrics: the schema-valid rate, parseable rate, per-field accuracy, mean field "
       "accuracy and whole-record accuracy are all aggregated from the same per-row items, so the one change "
       "corrects all of them; the extra-field count is unchanged. The older `json_field` metric of `spill eval` is "
       "a separate loose matcher that the guided workflow does not use; it is unchanged.", "",
       "## Historical JSON results: accounting", "",
       "Every published JSON result in the repository was scored with metric version 1. None can be rescored: a "
       "rescore needs the saved per-example predictions and ground truth, and these results were published as "
       "aggregates only (the raw project folders lived in scratch directories and were not committed; the export "
       "records kept only the rows that differed). Each is therefore marked **not rescorable**. A fresh model run "
       "is a replication, not a rescore, and the section after this table is labeled that way. No larger-model "
       "rerun was made.", "",
       "| published result | where | metric version | status |", "|---|---|---|---|",
       "| JSON validation tables, MLX and torch-cpu, 18 rows | `docs/reports/015-workflow.md`, README of that date | 1 | not rescorable (no per-example predictions saved) |",
       "| JSON final-test scores, MLX and torch-cpu | `docs/reports/015-workflow.md` | 1 | not rescorable |",
       "| JSON export verification rows and quality (8 rows) | `docs/reports/015-workflow.md` | 1 | not rescorable |",
       "| JSON journey from the wheel, Linux CI | `docs/reports/data/015-ci-linux.json`, `docs/reports/015-workflow.md` | 1 | not rescorable |",
       "| JSON rows of the README example table | README (before this change) | 1 | not rescorable; replaced by the replication below |",
       "", "Classification results are unaffected (metric version 1, unchanged).", "",
       "## Replication under metric version 2 (not a rescore)", "",
       "The same journeys were run again, from fresh projects built from the same tiny examples (120 rows: 84 "
       "train, 18 validation, 18 final test, seed 0), `qwen2.5:0.5b` bf16, with the current code. Every score is "
       "on 18 rows; one row is 0.056.", "",
       "| task | engine | comparator | score | schema-valid rate | rows |", "|---|---|---|---|---|---|"]
for j in gates["journeys"]:
    for t in j["run"]["table"]:
        sv = t.get("schema_valid_rate")
        out.append(f"| {j['task']} | {j['engine']} | {NAMES.get(t['comparator'], t['comparator'])} | "
                   f"{f3(t['primary'])} | {f3(sv) if j['task'] == 'json' else 'n/a'} | {t['rows']} |")
out += ["", "In this replication the trained student's JSON outputs were all schema-valid, so the corrected scorer "
        "did not change the trained scores; the untrained student's schema-invalid outputs are where the "
        "version 1 and version 2 rules differ, and that is not rescorable for the published runs.", "",
        "| task | engine | build | test | evaluate (other engine) | export (safetensors + GGUF q8_0, verified) |",
        "|---|---|---|---|---|---|"]
for j in gates["journeys"]:
    st = {s["cmd"].split()[1]: s["seconds"] for s in j["steps"]}
    out.append(f"| {j['task']} | {j['engine']} | {st['build']} s | {st['test']} s | {st['evaluate']} s | {st['export']} s |")
out += ["", "## The protocol matches execution", "",
        "**Primary metric.** `evaluation.metric` in the config selects it; classification supports `accuracy` and "
        "`macro_f1`, JSON supports `whole_record_accuracy`, `mean_field_accuracy`, `schema_valid_rate` and "
        "`parseable_rate`. Anything else is rejected with a message naming the choices "
        "(`tests/test_json_scorer.py`, `tests/test_project_build.py`).", "",
        "**Decoding.** Each request body now carries `max_tokens`, `temperature`, `top_p`, `stop`, `seed` and "
        "`greedy`. The MLX and PyTorch engines decode greedily with no sampling and no stop sequences; a request "
        "for a setting they cannot apply is refused before anything runs, naming the engine and the setting, and "
        "no sampling feature was added. Captured on this Mac "
        f"(`scripts/gate_decoding_capture.py`; `mx.default_device()` was `{dec['mlx_default_device']}`):", "",
        "| engine | device | request as sent | settings as applied (row echo) | request with temperature 0.7 |",
        "|---|---|---|---|---|"]
for eng, e in dec["engines"].items():
    req = {k: v for k, v in e["request_as_sent"].items() if k != "messages"}
    ap = e["applied_per_row"][0]
    out.append(f"| {eng} | {e['hardware']} | `{json.dumps(req)}` | `{json.dumps({k: ap[k] for k in ('temperature','top_p','stop','greedy','max_tokens','seed')})}` | "
               f"refused (exit {e['unsupported_request_exit']}): `{e['unsupported_request_message'].replace(chr(10), ' ')[:140]}` |")
out += ["", "**Executed conditions.** Each evaluation stores, apart from the requested precision policy, the "
        "engine, device, actual numerics, library versions and the decoding as applied. For the original "
        "evaluation of the first journey:", "", "```"]
c0 = gates["journeys"][0]["evaluate"]["conditions"]["trained"]
out += [json.dumps(c0, indent=1), "```", "",
        "**Evaluation records and compare.** `spill evaluate <run> --engine <engine>` re-scored each run on the "
        "other engine as a separate record. `spill compare` of the MLX-built and the torch-cpu-built project, "
        "each through its original evaluation (it says so), is labeled cross-runtime: ranked, with the "
        "differences and a caution. The incompatible cases (rows, prompts, decoding policy, metric version) and "
        "explicit `--use` selection are tested in `tests/test_compare_records.py`.", "",
        "Classification:", "", "```"]
out += (D / "017-compare-classification.txt").read_text().rstrip().splitlines()
out += ["```", "", "JSON extraction:", "", "```"]
out += (D / "017-compare-json.txt").read_text().rstrip().splitlines()
out += ["```", "", "Choosing the other engine's evaluation of the first run, explicitly:", "", "```"]
out += gates["journeys"][0]["compare_explicit"].rstrip().splitlines()[:4]
out += ["```", "",
        "A parent run appears in reports, the project index and compare with the statement that it records "
        "experiment lineage only and that the new run trained from the base model, not from the parent's weights.", "",
        "## Deployment measurements (replication)", "",
        "Measured in the export verification of each journey, separately from build time. Each runtime's own "
        "description of its measurement applies:", "",
        "- Transformers (safetensors, float32 on the CPU): time to first token is the wall time of "
        "`generate(max_new_tokens=1)` including prompt prefill; cold is the first row and warm the median of the "
        "rest, and neither includes loading the model (load time is recorded apart). Tokens per second is "
        "completion tokens over the wall time of the full `generate()` including prefill. Peak memory is "
        "`ru_maxrss` of the verification process, a process peak of resident memory, not a GPU-memory figure.",
        "- llama.cpp (GGUF q8_0, all layers on the Metal GPU on this Mac): time to first token is the time from "
        "the request to the first streamed content token with the prompt cache off, cold first row and warm "
        "median, excluding model loading. Peak memory is the peak resident set of the llama-server process tree "
        "sampled after each request: a lower bound, and not a total of GPU memory.",
        "- The Metal q8_0 and CPU float32 rows are not a format-only performance comparison: they differ in "
        "runtime, device, and precision at once.", "",
        "| task | built on | artifact | cold first token | warm first token | warm tokens/s | peak memory (sampled RSS) |",
        "|---|---|---|---|---|---|---|"]
for j in gates["journeys"]:
    for k, v in j["export"]["deployment"].items():
        mem = v["peak_memory_bytes"]
        out.append(f"| {j['task']} | {j['engine']} | {k} ({'llama.cpp, Metal' if k.startswith('gguf') else 'Transformers, CPU float32'}) | "
                   f"{v['ttft_cold_s']} s | {v['ttft_warm_s']} s | {v['tokens_per_s_warm']} tokens/s | "
                   f"{'unavailable' if mem == 'unavailable' else f'{mem / 1e9:.2f} GB'} |")
out += ["", "## Decisions", "",
        "- Protocol version 2: the decoding block gained `stop` and `greedy`; protocol 1 and 2 runs are "
        "incompatible and compare says why.",
        "- Metric versions are the scorer's, kept in code (`config.METRIC_VERSIONS`), not read from an old "
        "config file, and are part of a run's identity.",
        "- The original evaluation of a run is named `<run id>/original`; `spill evaluate` writes "
        "`evaluations/<id>/`.",
        "- Everything the engines cannot do (sampling, stop sequences) is refused rather than approximated.", "",
        "## Untested", "",
        "CUDA, real AWS S3, Windows, power loss, and any model beyond `qwen2.5:0.5b`. The captured-request gate "
        "ran on the Metal GPU and on torch-cpu here; in CI the same test runs on torch-cpu and on MLX's CPU device."]
(ROOT / "docs/reports/017-scorer-and-protocol.md").write_text("\n".join(out) + "\n")

# ---------------------------------------------------------------- export diagnosis
ex = {(t, e): L(f"017-export-{t}-{e}.json") for t in ("classification", "json") for e in ("mlx", "torch-cpu")}
from streamweights.project.exportrec import DEFAULT_MERGE_DTYPE
ARMS = ["unmerged", "merge-f32", "merge-bf16", "gguf-f32", "gguf-q8-from-f32", "gguf-q8-from-bf16"]
LABEL = {"unmerged": "unmerged adapter on the base (PEFT, Transformers float32)",
         "merge-f32": "float32 merge (Transformers float32)", "merge-bf16": "bf16 merge (Transformers float32)",
         "gguf-f32": "F32 GGUF of the float32 merge (llama.cpp)", "gguf-q8-from-f32": "q8_0 GGUF from the float32 merge (llama.cpp)",
         "gguf-q8-from-bf16": "q8_0 GGUF from the bf16 merge (llama.cpp)"}
o = ["# 017: JSON export differences, diagnosed", "",
     "Rule for the default merge dtype, fixed before this comparison ran: `docs/reports/017-export-default-rule.md`. "
     "Script: `scripts/diagnose_export.py`; raw results `docs/reports/data/017-export-*.json`.", "",
     "## What was held identical", "",
     "For each of four completed projects (classification and JSON, each built on MLX and on torch-cpu): the "
     "verification rows (the first 8 validation rows of the run), the input messages, the rendered prompts "
     "(rendered once by the Transformers chat template and passed to every runtime as token ids, so tokenization "
     "is identical), greedy decoding and `max_tokens`. Hardware for every arm is the CPU: Transformers in "
     "float32, llama.cpp with no GPU layers. The source engine is the engine that trained the run and produced "
     "its saved predictions, rescored with the current scorer on these rows.", ""]
for (t, e), d in ex.items():
    c = d["source_engine"]["engine_conditions"] or {}
    o += [f"### {t}, built on {e}", "",
          f"Metric: {d['metric']} (version {d['metric_version']}), {d['n']} verification rows. Source engine: "
          f"{c.get('engine')} on {c.get('device')}, numerics {json.dumps(c.get('numerics'))}, score "
          f"{f3(d['source_engine']['score'])} on these rows.", "",
          "| arm | size | peak memory (process RSS) | task score (8 rows) | source score | change | text differs from source engine | schema-valid rate |",
          "|---|---|---|---|---|---|---|---|"]
    for a in ARMS:
        r = d["arms"][a]
        if r.get("error"):
            o.append(f"| {LABEL[a]} | n/a | n/a | not run: {r['error']} | | | | |")
            continue
        o.append(f"| {LABEL[a]} | {r['size_bytes'] / 1e6:.0f} MB | {r['peak_rss_bytes'] / 1e9:.2f} GB | "
                 f"{f3(r['score'])} ({r['correct_rows']} of {d['n']}) | {f3(r['source_engine_score'])} | "
                 f"{r['delta_vs_source']:+.3f} | {r['text_disagreements']} of {d['n']} | "
                 f"{f3(r.get('schema_valid_rate')) if t == 'json' else 'n/a'} |")
    base = d["arms"]["unmerged"].get("outputs") or {}
    o += ["", "Rows whose text differs between arms (each arm against the unmerged adapter): "
          + "; ".join(f"{a}: {sum(1 for k in base if base[k].strip() != d['arms'][a]['outputs'][k].strip())}"
                      for a in ARMS[1:] if d['arms'][a].get('outputs')) + ".",
          f"Transformers float32 merge against the F32 GGUF in llama.cpp, same rows: "
          f"{d['runtime_comparison']['transformers_f32_vs_llamacpp_f32_text_disagreements']} of "
          f"{d['runtime_comparison']['of']} rows differ.", ""]
# rule
def tally(dt):
    q = fid = 0
    size = None
    for d in ex.values():
        for a in (f"merge-{dt}", f"gguf-q8-from-{dt}"):
            q += d["arms"][a]["correct_rows"]
            fid += d["arms"][a]["text_disagreements"]
        size = (size or 0) + d["arms"][f"merge-{dt}"]["size_bytes"]
    return q, fid, size
def decide(qf, qb):
    (q32, f32, s32), (qb_, fb, sb) = qf, qb
    if abs(q32 - qb_) >= 2:
        return "float32" if q32 > qb_ else "bf16", "step 1: task quality"
    if abs(f32 - fb) >= 2:
        return "float32" if f32 < fb else "bf16", "step 2: fidelity to the source engine"
    if s32 != sb:
        return ("float32" if s32 < sb else "bf16"), "step 3: artifact size"
    return "bf16", "step 4: tie-break"
t32, tb = tally("f32"), tally("bf16")
choice, why = decide(t32, tb)
per = {}
for e in ("mlx", "torch-cpu"):
    sub = {k: v for k, v in ex.items() if k[1] == e}
    def tl(dt):
        q = fid = 0
        for d in sub.values():
            for a in (f"merge-{dt}", f"gguf-q8-from-{dt}"):
                q += d["arms"][a]["correct_rows"]; fid += d["arms"][a]["text_disagreements"]
        return q, fid, sum(d["arms"][f"merge-{dt}"]["size_bytes"] for d in sub.values())
    per[e] = (tl("f32"), tl("bf16"))
o += ["## Applying the pre-stated rule", "",
      "Per candidate, over both task types and both artifact kinds (the merge in Transformers and its q8_0 GGUF in "
      "llama.cpp), 16 rows each: rows scored correct, rows whose text differs from the source engine, and the "
      "safetensors size.", "",
      "| source engine | candidate | correct rows | rows differing from the source engine | safetensors size (2 tasks) | rule outcome |",
      "|---|---|---|---|---|---|"]
for e, (a32, abf) in per.items():
    ch, wy = decide(a32, abf)
    o.append(f"| {e} | float32 merge | {a32[0]} | {a32[1]} | {a32[2] / 1e6:.0f} MB | |")
    o.append(f"| {e} | bf16 merge | {abf[0]} | {abf[1]} | {abf[2] / 1e6:.0f} MB | {ch} ({wy}) |")
o.append(f"| all four projects | float32 merge | {t32[0]} | {t32[1]} | {t32[2] / 1e6:.0f} MB | |")
o.append(f"| all four projects | bf16 merge | {tb[0]} | {tb[1]} | {tb[2] / 1e6:.0f} MB | {choice} ({why}) |")
assert choice == DEFAULT_MERGE_DTYPE, f"the rule gives {choice} but the code default is {DEFAULT_MERGE_DTYPE}"
o += ["", f"**Default: `--merge-dtype {choice}`** ({why}). Neither dtype changed task quality or fidelity by the "
      "margin the rule requires, so the smaller artifact wins; text agreement was not used on its own.", "",
      "## What the comparison shows", "",
      "- Merging in float32 instead of bf16 did not change any output on these rows beyond what quantization did: "
      "the float32 merge, the bf16 merge and the unmerged adapter gave the same text on every verification row "
      "in each project, within the differences listed above.",
      "- The runtime is not the cause: the float32 merge in Transformers and the F32 GGUF in llama.cpp agree on "
      "the listed rows, so the GGUF conversion is not losing the adapter.",
      "- q8_0 quantization moved a row or two in the JSON projects, in either direction (it scored higher than the "
      "float32 artifact in some projects).",
      "- The remaining gap for JSON is against the source engine. It is already present for the unmerged adapter "
      "in Transformers float32, which involves no merge at all, so the difference comes from the evaluation "
      "runtime and its numerics, not from the export. Tokenization was identical (the prompts were passed as "
      f"token ids; in a first run of this comparison the engines' recorded prompt token counts equalled the lengths of those ids). Batch shape was ruled out for the torch-cpu "
      f"engine: re-evaluating the JSON project with an evaluation batch of {b1['eval_batch']} gave "
      f"{b1['text_differences_vs_original_batch']} text differences on {b1['rows']} rows and the same score "
      f"({f3(b1['score_batch1'])}).",
      "- JSON outputs are 30 to 50 tokens long, so a single near-tie at any token flips a whole row; "
      "classification labels are a few tokens and showed no difference.", "",
      "## What this can and cannot support", "",
      "Eight rows per task can support an implementation decision, such as which dtype a default should be. They "
      "cannot establish that one merge dtype universally preserves quality better, and nothing here claims that.", "",
      "Export completion means the artifact loaded and verification ran. It does not mean quality was preserved: "
      "every export record and report shows the quality change on the verification rows next to the result, and "
      "the docs say so.", "",
      "## Untested", "", "CUDA, other model sizes, other adapters, and any claim beyond these 8-row verifications."]
(ROOT / "docs/reports/017-export-diagnosis.md").write_text("\n".join(o) + "\n")
print("default merge dtype by the rule:", choice, why)
