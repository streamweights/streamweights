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
tf, gg = ex["safetensors"]["boundaries"], ex["gguf:q8_0"]["boundaries"]
out += ["Measurement descriptions, each runtime's own. Transformers (safetensors, float32 on the CPU): time to first "
        "token is " + tf["ttft"] + "; tokens per second is " + tf["tokens_per_s"] + "; peak memory is "
        + tf["peak_memory"] + ", a process peak of resident memory and not a GPU-memory figure. llama.cpp (GGUF "
        "q8_0, all layers on the Metal GPU): time to first token is " + gg["ttft"] + "; tokens per second is "
        + gg["tokens_per_s"] + "; the sampled process resident set is a lower bound, not a total of GPU memory. "
        "Cold first-token timing excludes model loading (load time is recorded apart). The Metal q8_0 and CPU "
        "float32 numbers are not a format-only performance comparison: runtime, device and precision all differ.", ""]
out += ["## Inference through the exported artifact", ""]
for j in d["journeys"]:
    for fmt, v in j["inference"].items():
        out.append(f"- {j['task']} / {j['engine']} / {fmt}: `{v['output'][:90]}` in {v['seconds']} s "
                   f"(exit {v['exit']}), input `{v['input'][:60]}`")
out.append("")
ci = ROOT / "docs/reports/data/015-ci-linux.json"
if ci.exists():
    c = json.loads(ci.read_text())
    out += ["", "## Linux, from the built wheel (GitHub Actions)", "",
            f"{c['source']}: the wheel `{c['wheel']['wheel']}` is installed into a clean environment with the "
            f"`cloud` extra and the CLI is run from outside the checkout (`streamweights` imported from "
            f"`{c['wheel']['imported_from'].split('site-packages')[-1] and 'site-packages'}`); torch-cpu on the "
            f"runner's CPU (x86_64), the same tiny examples.", "",
            "| task | comparator | score | rows |", "|---|---|---|---|"]
    for j in c["gguf_journey"]["journeys"]:
        for t in j["run"]["table"]:
            out.append(f"| {j['task']} | {NAMES.get(t['comparator'], t['comparator'])} | {f(t['primary'])} | {t['rows']} |")
    out += ["", "| task | artifact | rows that differ from the training engine | warm tokens/s | peak memory |", "|---|---|---|---|---|"]
    for j in c["gguf_journey"]["journeys"]:
        for k, v in j["export"]["verification"].items():
            dep = j["export"]["deployment"][k]
            out.append(f"| {j['task']} | {k} | {v['diffs']} of {v['of']} | {dep['tokens_per_s_warm']} tokens/s | "
                       f"{dep['peak_memory_bytes'] / 1e9:.2f} GB |")
    out.append("")
out += (ROOT / "scripts" / "report_015_static.md").read_text().splitlines()
(ROOT / "docs/reports/015-workflow.md").write_text("\n".join(out) + "\n")
