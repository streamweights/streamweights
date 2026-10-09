"""Render docs/reports/016-close-gaps.md from docs/reports/data/016-*.json and the text files
next to them. Numbers are read from those files."""

import json
from pathlib import Path

R = Path(__file__).resolve().parent.parent / "docs/reports"
D = R / "data"
metal = json.loads((D / "016-metal.json").read_text())
f = lambda x: f"{x:.3f}"
out = ["# 016: closing the 015 gaps", "",
       "Directive: `docs/paste-sets/016-close-015-gaps.md`. Everything here was run; the JSON files in "
       "`docs/reports/data/` are the raw results.", "",
       "## Offline bundle gate", "",
       "`scripts/gate_offline_bundle.py` builds a bundle of the tiny classification project online "
       "(`prepare`), then `offline` runs in an environment with no network: a fresh empty `SPILL_HOME` and "
       "Hugging Face cache, the wheel installed from a local file with `pip install --no-deps --no-index`, "
       "the bundle verified and installed, the project reported and planned, inference from the bundle "
       "(base model plus the bundled adapter), a fork to a new run that trains to completion, then a "
       "corrupted and a missing asset, both rejected. Every spill process runs with `scripts/netguard` on "
       "`PYTHONPATH`: any connection or name lookup raises a `BaseException` (so no retry loop can swallow "
       "it) and is logged, and the log must be empty at the end. `HF_HUB_OFFLINE=1` is set too, as a second "
       "layer only. The gate first proves the environment itself has no network: a real connection to "
       "1.1.1.1:443 and a lookup of huggingface.co must fail.", ""]
for name, title, how in (("016-offline-linux.json", "Linux (GitHub Actions, container started with `--network none`)", "container"),
                         ("016-offline-macos.json", "macOS (`sandbox-exec` with `(deny network*)`)", "sandbox")):
    p = D / name
    if not p.exists():
        continue
    d = json.loads(p.read_text())
    out += [f"### {title}", "", f"All checks passed: {d['all_ok']}. `streamweights` imported from "
            f"`{'site-packages' if 'site-packages' in d['imported_from'] else d['imported_from']}`.", "",
            "| check | result |", "|---|---|"]
    out += [f"| {c['name']} | {'pass' if c['ok'] else 'FAIL'} |" for c in d["checks"]]
    out.append("")
out += ["## MLX on Metal: continuation in both directions", "",
        f"`scripts/gate_metal_continuation.py` on {metal['hardware']['cpu']} ({metal['hardware']['os']} "
        f"{metal['hardware']['arch']}); it asserts `mx.default_device()` is the GPU and it was "
        f"`{metal['device']['mlx_default_device']}`. Tiny classification project (120 rows: 84 train, 18 "
        f"validation, 18 final test), checkpoint every 5 steps, 42 steps. Each stop is a process killed by a "
        f"deterministic hook right after a checkpoint is published; the next process waits out the 2 s lease. "
        f"Every score is the trained student's accuracy on the 18 validation rows.", "",
        f"Uninterrupted MLX-GPU reference: {f(metal['reference']['trained_accuracy'])} "
        f"(untrained {f(metal['reference']['untrained_accuracy'])}, embedding baseline "
        f"{f(metal['reference']['baseline_accuracy'])}, 18 rows).", "",
        "| run | engines in order | killed after published step | published checkpoints (step, optimizer step) | "
        "engine changes recorded | trained accuracy (18 rows) | carry-over problems |", "|---|---|---|---|---|---|---|"]
for c in metal["chains"]:
    ks = ", ".join(str(s["killed_after_published_step"]) for s in c["steps"] if "killed_after_published_step" in s)
    cks = ", ".join(f"({a}, {b})" for a, b in c["published_checkpoints"])
    out.append(f"| {c['name']} | {' to '.join(c['engines'])} | {ks} | {cks} | {len(c['engine_transitions'])} | "
               f"{f(c['final']['trained_accuracy'])} | {len(c['carry_over_problems'])} |")
out += ["", "A score from a run that changed engines is not expected to equal the reference: the engines train "
        "with different numerics (see portability) and 18 rows move in steps of 0.056. The gate checks that "
        "the step, data cursor and optimizer step carry over and that the transitions are recorded, and "
        "reports the score; it does not require the scores to be equal.", "",
        "### Moves between locations", "",
        "A run is killed on one engine after step 10, `spill move` hands it to another location, and "
        "`spill resume` there finishes it on the other engine.", "",
        "| move | destination | engines | published step before the move | source after the move | trained accuracy (18 rows) | carry-over problems |",
        "|---|---|---|---|---|---|---|"]
for m in metal["moves"]:
    out.append(f"| {m['name']} | {m['move']} | {m['from']} to {m['to']} | {m['published_step_before_move']} | "
               f"{m['source_status_after_move']} | {f(m['final']['trained_accuracy'])} | {len(m['carry_over_problems'])} |")
out += ["", "The S3 destination here was a local MinIO server (the Homebrew build of "
        "RELEASE.2025-10-15T17-29-55Z). Docker is not installed on this Mac, so the Linux-container MinIO was "
        "not used here; that path is covered by CI (job `object-store`).", ""]
for name in ("016-merge-hygiene.md",):
    if (D / name).exists():
        out += (D / name).read_text().splitlines() + [""]
(R / "016-close-gaps.md").write_text("\n".join(out) + "\n")
