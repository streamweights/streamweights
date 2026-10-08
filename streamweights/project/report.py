"""A completed run's report and manifest, assembled from the accepted stage results.

Everything here is a pure function of the accepted outputs: the same results give the same
bytes. A completed run's report never changes; the project-root REPORT.md is a regenerable
index over the runs, exports and tests."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from . import config as C
from .common import canon, read_json, read_jsonl, sha_file, sha_obj, write_json, write_jsonl

COMPARATORS = ["baseline", "untrained", "trained", "teacher"]
NAMES = {"baseline": "embedding baseline (MiniLM + logistic regression)",
         "untrained": "student, prompted, untrained", "trained": "student, trained",
         "teacher": "teacher"}
JSON_NAMES = {**NAMES, "untrained": "student, schema-prompted, untrained (the baseline)"}


def primary(task: str, metrics: dict) -> float | None:
    return metrics.get("accuracy") if task == "classification" else metrics.get("whole_record_accuracy")


def primary_name(task: str) -> str:
    return "accuracy" if task == "classification" else "whole-record accuracy"


def protocol_fingerprint(cfg: dict, schema: dict | None, val_sha: str, val_ids: list[str]) -> dict:
    """Everything that decides whether two evaluations are comparable."""
    from . import contract as K
    ev = cfg["evaluation"]
    body = {
        "task": cfg["task"]["type"],
        "rows": {"val_sha256": val_sha, "ids_sha256": sha_obj(val_ids), "n": len(val_ids)},
        "contract": ({"labels": cfg["contract"]["labels"],
                      "normalization": cfg["contract"]["label_normalization"]}
                     if cfg["task"]["type"] == "classification"
                     else {"schema_sha256": cfg["contract"]["schema_sha256"],
                           "rules": cfg["contract"]["rules"]}),
        "prompts": {"untrained": K.messages_untrained(cfg, schema, "{input}"),
                    "student": K.messages_student(cfg, "{input}")},
        "decoding": ev["decoding"], "max_tokens": ev["max_tokens"],
        "postprocessing": ev["postprocessing"], "metric": ev["metric"],
        "metric_version": ev["metric_version"], "protocol_version": ev["protocol_version"],
        "precision": ev["precision"]}
    return {"sha256": sha_obj(body), "fields": body}


def metric_definition(cfg: dict) -> dict:
    ev = cfg["evaluation"]
    return {"metric": ev["metric"], "metric_version": ev["metric_version"],
            "contract": (cfg["contract"]["label_normalization"] if cfg["task"]["type"]
                         == "classification" else cfg["contract"]["rules"])}


def disagreements(task: str, preds: dict[str, list[dict]]) -> list[dict]:
    """Rows where a comparator and the trained student differ in correctness. 'improvement':
    the trained student is right and the comparator wrong; 'regression': the reverse."""
    trained = {p["id"]: p for p in preds.get("trained", [])}
    out = []
    for name, rows in preds.items():
        if name == "trained":
            continue
        for p in rows:
            t = trained.get(p["id"])
            if t is None or bool(t["correct"]) == bool(p["correct"]):
                continue
            out.append({"id": p["id"], "against": name,
                        "kind": "improvement" if t["correct"] else "regression",
                        "gold": p["gold"], "trained": t.get("text"), "other": p.get("text")})
    return out


def table(task: str, comps: dict) -> list[dict]:
    rows = []
    for name in COMPARATORS:
        if name not in comps:
            continue
        m = comps[name]["metrics"]
        rows.append({"comparator": name, "primary": primary(task, m), **{k: m.get(k) for k in (
            "rows", "accuracy", "macro_f1", "invalid_predictions", "inference_failures",
            "truncated", "parseable_rate", "schema_valid_rate", "whole_record_accuracy",
            "mean_field_accuracy", "records_with_extra_fields")}})
    return rows


def _f(x, pct=True):
    if x is None:
        return "unavailable"
    return f"{x:.3f}"


def render_md(m: dict) -> str:
    task = m["task"]["type"]
    names = NAMES if task == "classification" else JSON_NAMES
    lines = [f"# Run {m['run_id']}: {m['project']}", "",
             f"Task: {task}. Metric: {primary_name(task)} (version {m['metric_version']}). "
             f"Validation rows: {m['protocol']['fields']['rows']['n']}. Status: completed.", ""]
    lines += ["## Results", ""]
    if task == "classification":
        lines += ["| comparator | accuracy | macro-F1 | rows | invalid predictions | failures | truncated |",
                  "|---|---|---|---|---|---|---|"]
        for r in m["table"]:
            lines.append(f"| {names[r['comparator']]} | {_f(r['accuracy'])} | {_f(r['macro_f1'])} | "
                         f"{r['rows']} | {r['invalid_predictions']} | {r['inference_failures'] or 0} | "
                         f"{r['truncated'] if r['truncated'] is not None else 'n/a'} |")
    else:
        lines += ["| comparator | parseable | schema-valid | whole-record | mean field | rows | "
                  "failures | truncated |", "|---|---|---|---|---|---|---|---|"]
        for r in m["table"]:
            lines.append(f"| {names[r['comparator']]} | {_f(r['parseable_rate'])} | "
                         f"{_f(r['schema_valid_rate'])} | {_f(r['whole_record_accuracy'])} | "
                         f"{_f(r['mean_field_accuracy'])} | {r['rows']} | {r['inference_failures'] or 0} | "
                         f"{r['truncated'] if r['truncated'] is not None else 'n/a'} |")
    lines += ["", "Invalid, unparseable and failed outputs count as wrong and stay in every "
              "denominator. `unavailable` means the metric does not exist for that row; it is "
              "never shown as 0.", ""]
    if task == "json":
        lines += ["Per-field accuracy (all validation rows):", ""]
        fields = sorted({k for r in m["comparators"].values()
                         for k in r["metrics"].get("field_accuracy", {})})
        lines += ["| field | " + " | ".join(names[c] for c in COMPARATORS if c in m["comparators"]) + " |",
                  "|---|" + "---|" * len([c for c in COMPARATORS if c in m["comparators"]])]
        for f in fields:
            cells = [_f(m["comparators"][c]["metrics"].get("field_accuracy", {}).get(f))
                     for c in COMPARATORS if c in m["comparators"]]
            lines.append(f"| {f} | " + " | ".join(cells) + " |")
        lines.append("")
    lines += ["## What the numbers say", ""]
    lines += m["findings"]
    lines += ["", "No significance test was run and none is implied. These are the observed "
              "differences on this validation set; a few rows can move a score on a set this "
              "size. Changing the batch shape can also move a score through numerical "
              "differences; that is a diagnostic, not evidence either way.", ""]
    lines += ["## Data", "",
              f"Train {m['data']['sizes']['train']}, validation {m['data']['sizes']['val']}, "
              f"final test {m['data']['sizes']['test']} rows (split {m['data']['split_mode']}, "
              f"seed {m['data']['seed']}). The final test split was not used by this run and is "
              f"scored only by `spill test`.", ""]
    for w in m["data"].get("warnings", []):
        lines.append(f"- {w}")
    lines += ["", "## Models", ""]
    for role, i in m["models"].items():
        if i:
            lines.append(f"- {role}: {i['tag']} ({i['repo'] or 'local'}@{(i['revision'] or 'unpinned')[:12]}, "
                         f"{i.get('format', '')})")
    lines += ["", "## How it ran", ""]
    for sid, s in m["stages"].items():
        eng = ", ".join(f"{p.get('engine')} on {p.get('os')}" for p in s.get("producers", [])) or "-"
        reused = " (reused an identical earlier engine run on this machine)" \
            if s.get("notes", {}).get("reused") else ""
        lines.append(f"- {sid}: {s['status']}, {s['seconds']:.1f} s, {eng}{reused}")
    if m["engine_transitions"]:
        lines.append("")
        lines.append("Engine transitions during training (same run, recorded):")
        for t in m["engine_transitions"]:
            lines.append(f"- {t}")
    lines += ["", "## Where things are", "",
              f"- predictions: predictions/<comparator>.jsonl; disagreements with the trained "
              f"student: disagreements.jsonl", f"- adapter: artifacts/adapter/",
              f"- manifest (fingerprints, model hashes, dependency versions): manifest.json", "",
              "## Next", ""]
    lines += [f"- `{c}`" for c in m["next"]]
    return "\n".join(lines) + "\n"


def findings(task: str, comps: dict) -> list[str]:
    names = NAMES if task == "classification" else JSON_NAMES
    sc = {k: primary(task, v["metrics"]) for k, v in comps.items()}
    n = next(iter(comps.values()))["metrics"]["rows"]
    parts = [f"{names[k]}: {_f(v)}" for k, v in sc.items() if v is not None]
    out = [f"On {n} validation rows, {primary_name(task)}: " + "; ".join(parts) + "."]
    t = sc.get("trained")
    for k in ("baseline", "untrained", "teacher"):
        if k in sc and t is not None and sc[k] is not None:
            d = t - sc[k]
            rows = round(abs(d) * n)
            if d > 0:
                out.append(f"The trained student scored {d:.3f} higher than the {names[k]} "
                           f"({rows} row{'s' if rows != 1 else ''}).")
            elif d < 0:
                out.append(f"The {names[k]} scored {-d:.3f} higher than the trained student "
                           f"({rows} row{'s' if rows != 1 else ''}).")
            else:
                out.append(f"The trained student and the {names[k]} scored the same.")
    return out


def assemble(dest: Path, *, project: Path, run_id: str, cfg: dict, plan_fields: dict,
             stage_results: dict, accepted: dict, models: dict, deps: dict, schema: dict | None,
             identity: str, inputs_dir: Path, parent: str | None, attempts: list,
             warnings: list) -> dict:
    """Write the snapshot directory `dest` and return its manifest."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    task = cfg["task"]["type"]
    comps, preds = {}, {}
    for sid, res in stage_results.items():
        if sid == "baseline":
            cmp = "baseline"
        elif sid.startswith("eval:"):
            cmp = sid.split(":", 1)[1]
        else:
            continue
        p = read_jsonl(accepted[sid] / "predictions.jsonl")
        preds[cmp] = p
        comps[cmp] = {"metrics": read_json(accepted[sid] / "metrics.json"), "stage": sid}
        write_jsonl(dest / "predictions" / f"{cmp}.jsonl", p)
    val_sha = sha_file(inputs_dir / "val.jsonl")
    val_ids = [r["id"] for r in read_jsonl(inputs_dir / "val.jsonl")]
    protocol = protocol_fingerprint(cfg, schema, val_sha, val_ids)
    dis = disagreements(task, preds)
    write_jsonl(dest / "disagreements.jsonl", dis)
    tbl = table(task, comps)
    # engine transitions from the training producers
    trans = []
    prod = stage_results.get("train", {}).get("producers", [])
    for a, b in zip(prod, prod[1:]):
        if a.get("engine") != b.get("engine") or a.get("numerics") != b.get("numerics"):
            trans.append(f"{a.get('engine')} ({a.get('quanta')}, numerics {json.dumps(a.get('numerics'))}) "
                         f"-> {b.get('engine')} ({b.get('quanta')}, numerics {json.dumps(b.get('numerics'))})")
    if "train" in accepted:
        shutil.copytree(accepted["train"] / "adapter", dest / "artifacts" / "adapter",
                        dirs_exist_ok=True)
        for f in ("training.manifest.json", "training.rows.jsonl"):
            shutil.copyfile(accepted["train"] / f, dest / "artifacts" / f)
    shutil.copytree(inputs_dir, dest / "inputs", dirs_exist_ok=True)
    adapter_sha = None
    if (dest / "artifacts" / "adapter").exists():
        adapter_sha = {p.name: sha_file(p) for p in sorted((dest / "artifacts" / "adapter").iterdir())
                       if p.is_file()}
    manifest = {
        "schema": 1, "run_id": run_id, "project": cfg["project"]["name"], "parent": parent,
        "identity": identity, "task": cfg["task"], "metric": primary_name(task),
        "metric_version": cfg["evaluation"]["metric_version"],
        "metric_definition": metric_definition(cfg), "protocol": protocol,
        "data": {"sizes": cfg["split"].get("sizes", {}), "split_mode": cfg["split"]["mode"],
                 "seed": cfg["split"]["seed"], "fingerprints": plan_fields["split"]["fingerprints"],
                 "test_sha256": plan_fields["split"]["fingerprints"]["test"],
                 "duplicate_normalization": cfg["split"]["duplicate_normalization"],
                 "warnings": warnings},
        "models": models, "dependencies": deps, "comparators": comps, "table": tbl,
        "findings": findings(task, comps),
        "stages": {sid: {"status": r.get("status"), "seconds": r.get("seconds"),
                         "producers": r.get("producers", []), "metrics_keys": sorted(r.get("metrics", {})),
                         "notes": r.get("notes", {})} for sid, r in stage_results.items()},
        "engine_transitions": trans, "attempts": attempts,
        "adapter_files_sha256": adapter_sha,
        "artifacts": {"predictions": [f"predictions/{c}.jsonl" for c in comps],
                      "disagreements": "disagreements.jsonl", "adapter": "artifacts/adapter",
                      "inputs": "inputs/"},
        "final_test": {"used_by_this_run": False,
                       "note": "scored only by `spill test`, which records each use"},
        "next": [f"spill export {project.name}", f"spill test {project.name}",
                 f"spill compare {project.name}"],
        "config_resolved": cfg,
    }
    (dest / "report.md").write_text(render_md(manifest))
    write_json(dest / "manifest.json", manifest)
    write_json(dest / "results.json", {"run_id": run_id, "task": task, "metric": primary_name(task),
                                       "protocol_sha256": protocol["sha256"], "table": tbl})
    return manifest
