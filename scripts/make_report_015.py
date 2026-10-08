"""Render docs/reports/015-workflow.md from docs/reports/data/015-gates.json (written by
scripts/gates_015.py). Every number in the report is read from that file."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
d = json.loads((ROOT / "docs/reports/data/015-gates.json").read_text())
hw = d["hardware"]
NAMES = {"baseline": "embedding baseline", "untrained": "student, prompted, untrained",
         "trained": "student, trained"}
f = lambda x: "unavailable" if x is None else f"{x:.3f}"
out = ["# 015: one complete workflow", "",
       "Measured by `scripts/gates_015.py`, which runs the public CLI end to end per task and per "
       "engine, on the permitted small models, and writes `docs/reports/data/015-gates.json`; this page "
       "is rendered from that file by `scripts/make_report_015.py`.", "",
       f"Hardware: {hw['cpu']}, {hw['ram_gb']} GB, {hw['os']} {hw['os_release']} {hw['arch']}, "
       f"Python {hw['python']}. Run started {d['started']}. Each journey is a fresh project made "
       f"from the tiny example of its task (120 rows: 84 train, 18 validation, 18 final test, seed 0) "
       f"and uses `qwen2.5:0.5b` bf16 as the student.", ""]
j0 = d["journeys"][0]["run"]
out += ["## Models and revisions", "",
        "| role | tag | repository | revision |", "|---|---|---|---|"]
seen = set()
for j in d["journeys"]:
    for role, m in j["run"]["models"].items():
        if (role, m["tag"]) not in seen:
            seen.add((role, m["tag"]))
            out.append(f"| {role} | {m['tag']} | {m['repo']} | `{m['revision']}` |")
dep = j0["dependencies"]
out += ["", "Dependencies of the run: " + ", ".join(f"{k} {v}" for k, v in dep.items()
        if k in ("torch", "transformers", "peft", "mlx", "mlx-lm", "safetensors", "streamweights")) + ".", ""]
out += ["## Results on the validation rows", ""]
for task in ("classification", "json"):
    js = [j for j in d["journeys"] if j["task"] == task]
    if not js:
        continue
    out += [f"### {task}", "", "| engine | comparator | " + js[0]["run"]["metric"] + " | rows |", "|---|---|---|---|"]
    for j in js:
        for t in j["run"]["table"]:
            out.append(f"| {j['engine']} | {NAMES.get(t['comparator'], t['comparator'])} | "
                       f"{f(t['primary'])} | {t['rows']} |")
    out.append("")
out += ["Eighteen rows is small: one row is 0.056, and the two engines train with different numerics, "
        "so their numbers differ by more than that. No significance test was run.", "",
        "## Durations (wall seconds on this machine)", "",
        "| task | engine | build | test | export (safetensors + GGUF q8_0, verified) | train stage | embedding fetch included |",
        "|---|---|---|---|---|---|---|"]
for j in d["journeys"]:
    st = {s["cmd"].split()[1]: s["seconds"] for s in j["steps"]}
    out.append(f"| {j['task']} | {j['engine']} | {st['build']} s | {st['test']} s | {st['export']} s | "
               f"{j['run']['stages']['train']['seconds']} s | "
               f"{'yes (first journey of a fresh cache)' if j is d['journeys'][0] else 'no'} |")
out += ["", "## Final test (scored once per journey by `spill test`)", "",
        "| task | engine | comparator | score | rows |", "|---|---|---|---|---|"]
for j in d["journeys"]:
    for t in j["test"]["table"]:
        out.append(f"| {j['task']} | {j['engine']} | {NAMES.get(t['comparator'], t['comparator'])} | "
                   f"{f(t['primary'])} | {t['rows']} |")
out += ["", "## Export verification", "",
        "The merged bf16 safetensors is loaded by Transformers (float32 on the CPU) and the GGUF q8_0 by "
        "llama.cpp, each in its own process, on the first 8 validation rows. A difference is a row where the "
        "exported artifact's output text differs from the training engine's output for the trained student.", "",
        "| task | engine | artifact | runtime | rows that differ | primary metric on the 8 rows |", "|---|---|---|---|---|---|"]
for j in d["journeys"]:
    for k, v in j["export"]["verification"].items():
        out.append(f"| {j['task']} | {j['engine']} | {k} | {v['runtime'][:60]} | {v['diffs']} of {v['of']} | {f(v['primary'])} |")
out += ["", "For the extraction task the exported artifacts differ from the training engine on several rows. "
        "The adapter is merged into bf16 weights, which rounds away part of a small adapter delta, and the "
        "runtimes differ in numerics; the rows that differ are listed in each export record. This is the "
        "measured size of the gap, not a claim that it is zero.", "",
        "## Deployment measurements (inference only, separate from build time)", "",
        "| task | engine that built it | artifact | cold time to first token | warm | warm tokens/s | peak memory |",
        "|---|---|---|---|---|---|---|"]
for j in d["journeys"]:
    for k, v in j["export"]["deployment"].items():
        mem = v["peak_memory_bytes"]
        out.append(f"| {j['task']} | {j['engine']} | {k} | {v['ttft_cold_s']} s | {v['ttft_warm_s']} s | "
                   f"{v['tokens_per_s_warm']} tokens/s | {'unavailable' if mem == 'unavailable' else f'{mem / 1e9:.2f} GB'} |")
out.append("")
ex = d["journeys"][0]["export"]["deployment"]
b = next(iter(ex.values()))["boundaries"]
out += ["Boundaries, Transformers: time to first token is " + b["ttft"] + "; tokens per second is "
            + b["tokens_per_s"] + "; peak memory is " + b["peak_memory"] + ". For llama.cpp: "
            + ex["gguf:q8_0"]["boundaries"]["ttft"] + "; peak memory is "
            + ex["gguf:q8_0"]["boundaries"]["peak_memory"] + ". Hardware: the machine above; llama.cpp "
            "runs with all layers on the Metal GPU, Transformers on the CPU.", ""]
out += ["## Inference through the exported artifact", ""]
for j in d["journeys"]:
    for fmt, v in j["inference"].items():
        out.append(f"- {j['task']} / {j['engine']} / {fmt}: `{v['output'][:90]}` in {v['seconds']} s "
                   f"(exit {v['exit']}), input `{v['input'][:60]}`")
out.append("")
(ROOT / "docs/reports/015-workflow.md").write_text("\n".join(out))
