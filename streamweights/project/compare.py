"""spill compare: lay runs side by side, and rank them only when the comparison means something.

Runs are ranked only if they were evaluated on the same rows (the validation set's hash and
row ids), with the same metric definition and the same evaluation protocol (prompts, decoding,
token limits, postprocessing, metric version, precision). Training data and models may differ;
those differences are shown, not hidden. Anything else gets an explanation and no winner."""

from __future__ import annotations

import json
from pathlib import Path

from . import report as R


def load_runs(paths: list[Path]) -> list[dict]:
    out = []
    for p in paths:
        p = Path(p)
        cands = [p / "manifest.json"] if (p / "manifest.json").exists() else \
            sorted(p.glob("runs/*/manifest.json"))
        for c in cands:
            out.append({**json.loads(c.read_text()), "_dir": str(c.parent)})
    return out


def _flat(d: dict, prefix="") -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(_flat(v, f"{prefix}{k}."))
        else:
            out[f"{prefix}{k}"] = json.dumps(v, sort_keys=True)
    return out


def incompatibility(a: dict, b: dict) -> list[str]:
    reasons = []
    pa, pb = a["protocol"], b["protocol"]
    if pa["fields"]["rows"] != pb["fields"]["rows"]:
        reasons.append("the evaluation rows differ (validation set hash or row ids)")
    if a["metric_definition"] != b["metric_definition"] or a["metric"] != b["metric"]:
        reasons.append(f"the metric definitions differ ({a['metric']} v{a['metric_version']} vs "
                       f"{b['metric']} v{b['metric_version']})")
    if pa["sha256"] != pb["sha256"]:
        fa, fb = _flat(pa["fields"]), _flat(pb["fields"])
        diff = sorted(k for k in set(fa) | set(fb) if fa.get(k) != fb.get(k)
                      and not k.startswith("rows."))
        if diff:
            reasons.append("the evaluation protocols differ in " + ", ".join(diff[:8]))
    return reasons


def differences(a: dict, b: dict) -> dict:
    """What differs between two runs, by group, as named fields."""
    def pick(m):
        engines = sorted({p.get("engine") for s in m["stages"].values()
                          for p in s.get("producers", []) if p.get("engine")})
        num = sorted({json.dumps(p.get("numerics"), sort_keys=True) for s in m["stages"].values()
                      for p in s.get("producers", []) if p.get("numerics")})
        return {"data": {"train_sha256": m["data"]["fingerprints"]["train"],
                         "val_sha256": m["data"]["fingerprints"]["val"],
                         "seed": m["data"]["seed"], "split_mode": m["data"]["split_mode"]},
                "models": {r: (i or {}).get("revision") or (i or {}).get("tag")
                           for r, i in m["models"].items()},
                "model_files": {r: sorted((i or {}).get("files", {}).items())
                                for r, i in m["models"].items()},
                "training": m["config_resolved"]["training"],
                "engine": engines, "numerics": num}
    pa, pb = pick(a), pick(b)
    out = {}
    for g in pa:
        fa, fb = _flat({g: pa[g]}) if isinstance(pa[g], dict) else {g: json.dumps(pa[g])}, \
            _flat({g: pb[g]}) if isinstance(pb[g], dict) else {g: json.dumps(pb[g])}
        d = sorted(k for k in set(fa) | set(fb) if fa.get(k) != fb.get(k))
        if d:
            out[g] = d
    return out


def compare(runs: list[dict]) -> dict:
    """-> {"rows": [...], "groups": [[run ids ranked together]], "refusals": [...]}"""
    rows = []
    for m in runs:
        t = {r["comparator"]: r for r in m["table"]}
        rows.append({"run": m["run_id"], "trained": t.get("trained", {}).get("primary"),
                     "baseline": t.get("baseline", {}).get("primary"),
                     "untrained": t.get("untrained", {}).get("primary"), "metric": m["metric"],
                     "rows": t.get("trained", {}).get("rows"), "protocol": m["protocol"]["sha256"][:12],
                     "engine": sorted({p.get("engine") for s in m["stages"].values()
                                       for p in s.get("producers", []) if p.get("engine")})})
    groups: list[list[dict]] = []
    refusals = []
    for m in runs:
        for g in groups:
            if not incompatibility(g[0], m):
                g.append(m)
                break
        else:
            groups.append([m])
    if len(groups) > 1:
        for g in groups[1:]:
            why = incompatibility(groups[0][0], g[0])
            refusals.append({"runs": [x["run_id"] for x in g], "against": groups[0][0]["run_id"],
                             "reasons": why})
    ranked = []
    for g in groups:
        if len(g) > 1:
            def key(m):
                t = {r["comparator"]: r for r in m["table"]}
                v = t.get("trained", {}).get("primary")
                return -1 if v is None else v
            ranked.append([m["run_id"] for m in sorted(g, key=key, reverse=True)])
    diffs = {}
    if len(runs) > 1:
        for m in runs[1:]:
            diffs[f"{runs[0]['run_id']} vs {m['run_id']}"] = differences(runs[0], m)
    return {"rows": rows, "ranked": ranked, "refusals": refusals, "differences": diffs}


def render(res: dict) -> list[str]:
    lines = ["run  metric  trained  baseline  untrained  rows  protocol  engine"]
    f = lambda v: "unavailable" if v is None else f"{v:.3f}"
    for r in res["rows"]:
        lines.append(f"{r['run']}  {r['metric']}  {f(r['trained'])}  {f(r['baseline'])}  "
                     f"{f(r['untrained'])}  {r['rows']}  {r['protocol']}  {','.join(r['engine'])}")
    for g in res["ranked"]:
        lines.append("")
        lines.append("ranking (same evaluation rows, metric and protocol), by the trained model's "
                     "score: " + " > ".join(g))
        lines.append("A higher number here is an observed difference on these rows, not a "
                     "statistical claim.")
    for ref in res["refusals"]:
        lines.append("")
        lines.append(f"no ranking between {', '.join(ref['runs'])} and {ref['against']}: "
                     + "; ".join(ref["reasons"]))
    for k, d in res["differences"].items():
        lines.append("")
        lines.append(f"differences {k}:" + ("" if d else " none in data, models, training, engine"))
        for g, items in d.items():
            lines.append(f"   {g}: {', '.join(items[:8])}")
    return lines
