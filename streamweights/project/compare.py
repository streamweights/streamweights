"""spill compare: lay runs side by side, and say what the comparison is.

Each run is compared through one of its evaluations: its original one (the default, and the
output says so) or an evaluation record chosen explicitly by id (`--use <run>=<evaluation id>`).
Nothing picks the evaluation that scored best. A comparison gets one of three labels:

  common evaluator   same evaluation rows, task protocol and metric, and matching executed
                     evaluator conditions (engine, device, runtime and library versions, actual
                     numerics): ranked
  cross-runtime      same rows, protocol and metric, but the conditions differ: ranked, with the
                     differences shown and a caution that score differences may include
                     evaluation-runtime effects
  incompatible       different evaluation rows, prompts, decoding policy, or metric definition or
                     version (or another protocol field): not ranked, and why

Training data, models and training settings may differ in any of them; those differences are shown."""

from __future__ import annotations

import json
from pathlib import Path

from . import report as R

LINEAGE = ("experiment lineage only: the new run trained from the base model, not from the "
           "parent's weights")
LIB_KEYS = ("torch", "transformers", "peft", "mlx", "mlx-lm", "safetensors", "numpy")


def _read(p: Path):
    return json.loads(p.read_text())


def _view(m: dict, eval_rec: dict | None, project: Path | None) -> dict:
    """A run seen through one evaluation."""
    key = R.key_of(m.get("metric_key") or m["metric"])
    if eval_rec is None:
        ev = m.get("evaluation") or {}
        cond = (ev.get("conditions") or {})
        v = {"eval_id": ev.get("id", f"{m['run_id']}/original"), "eval_kind": "original",
             "table": m["table"], "protocol": m["protocol"], "metric_key": key,
             "metric_version": m["metric_version"], "conditions": cond.get("trained"),
             "all_conditions": cond}
    else:
        v = {"eval_id": eval_rec["id"], "eval_kind": "evaluation", "table": eval_rec["table"],
             "protocol": eval_rec["protocol"], "metric_key": eval_rec["metric_key"],
             "metric_version": eval_rec["metric_version"],
             "conditions": (eval_rec["comparators"].get("trained") or {}).get("conditions"),
             "all_conditions": {c: x.get("conditions") for c, x in eval_rec["comparators"].items()}}
    v["run"] = m
    v["run_id"] = m["run_id"]
    return v


def load_views(paths: list[Path], uses: dict[str, str] | None = None) -> list[dict]:
    """Views for every completed run under the paths. `uses` maps run id to an evaluation id."""
    uses = dict(uses or {})
    out = []
    for p in paths:
        p = Path(p)
        manifests = [p / "manifest.json"] if (p / "manifest.json").exists() else \
            sorted(p.glob("runs/*/manifest.json"))
        for c in manifests:
            m = _read(c)
            m["_dir"] = str(c.parent)
            project = c.parent.parent.parent
            evs = {}
            for rp in sorted((project / "evaluations").glob("*/record.json")) if (project / "evaluations").exists() else []:
                r = _read(rp)
                if r.get("source_run") == m["run_id"] and r.get("status") == "completed":
                    evs[r["id"]] = r
            sel = uses.pop(m["run_id"], None)
            if sel is not None and sel not in evs and sel != f"{m['run_id']}/original":
                raise ValueError(f"run {m['run_id']} has no completed evaluation {sel!r} "
                                 f"(it has: {', '.join(['original', *evs]) })")
            v = _view(m, evs.get(sel) if sel and sel in evs else None, project)
            v["explicit"] = sel is not None
            v["available"] = [f"{m['run_id']}/original", *evs]
            out.append(v)
    if uses:
        raise ValueError(f"--use names run(s) that were not found: {', '.join(uses)}")
    return out


def _flat(d, prefix=""):
    out = {}
    for k, v in (d or {}).items():
        if isinstance(v, dict):
            out.update(_flat(v, f"{prefix}{k}."))
        else:
            out[f"{prefix}{k}"] = json.dumps(v, sort_keys=True)
    return out


def _diff(a, b, skip=()):
    fa, fb = _flat(a), _flat(b)
    return sorted(k for k in set(fa) | set(fb) if fa.get(k) != fb.get(k) and not k.startswith(skip))


def protocol_differences(a: dict, b: dict) -> list[str]:
    """Why two evaluations are not the same protocol, in the directive's terms; empty if they are."""
    fa, fb = a["protocol"]["fields"], b["protocol"]["fields"]
    why = []
    if fa["rows"] != fb["rows"]:
        why.append("the evaluation rows differ (validation set hash or row ids)")
    if fa.get("prompts") != fb.get("prompts"):
        why.append("the prompts differ (" + ", ".join(_diff(fa.get("prompts"), fb.get("prompts"))[:4]) + ")")
    da = {"decoding": fa.get("decoding"), "max_tokens": fa.get("max_tokens"),
          "postprocessing": fa.get("postprocessing")}
    db = {"decoding": fb.get("decoding"), "max_tokens": fb.get("max_tokens"),
          "postprocessing": fb.get("postprocessing")}
    if da != db:
        why.append("the decoding policy differs (" + ", ".join(_diff(da, db)[:5]) + ")")
    ma = {k: fa.get(k) for k in ("metric", "metric_version", "contract")}
    mb = {k: fb.get(k) for k in ("metric", "metric_version", "contract")}
    if ma != mb:
        why.append(f"the metric definition or version differs ({fa.get('metric')} v{fa.get('metric_version')} "
                   f"vs {fb.get('metric')} v{fb.get('metric_version')}"
                   + (", or the label vocabulary or schema" if ma["contract"] != mb["contract"] else "") + ")")
    rest = _diff({k: v for k, v in fa.items() if k not in ("rows", "prompts", "decoding", "max_tokens",
                                                           "postprocessing", "metric", "metric_version",
                                                           "contract")},
                 {k: v for k, v in fb.items() if k not in ("rows", "prompts", "decoding", "max_tokens",
                                                           "postprocessing", "metric", "metric_version",
                                                           "contract")})
    if rest:
        why.append("other protocol fields differ (" + ", ".join(rest[:4]) + ")")
    return why


def condition_differences(a: dict, b: dict) -> list[str]:
    ca, cb = a.get("conditions"), b.get("conditions")
    if ca is None or cb is None:
        who = a["run_id"] if ca is None else b["run_id"]
        return [f"executed conditions were not recorded for {who}"]

    def key(c):
        return {"engine": c.get("engine"), "device": c.get("device"),
                "weight_dtype": c.get("weight_dtype"), "adapter_dtype": c.get("adapter_dtype"),
                "compute_dtype": c.get("compute_dtype"),
                **{f"version.{k}": (c.get("versions") or {}).get(k) for k in LIB_KEYS}}
    ka, kb = key(ca), key(cb)
    return [f"{k}: {ka[k]} vs {kb[k]}" for k in ka if ka[k] != kb[k]]


def differences(a: dict, b: dict) -> dict:
    """What differs between two runs besides the evaluation: data, models, training, engine,
    numerics."""
    def pick(v):
        m = v["run"]
        engines = sorted({p.get("engine") for s in m["stages"].values() for p in s.get("producers", [])
                          if p.get("engine")})
        num = sorted({json.dumps(p.get("numerics"), sort_keys=True) for s in m["stages"].values()
                      for p in s.get("producers", []) if p.get("numerics")})
        return {"data": {"train_sha256": m["data"]["fingerprints"]["train"],
                         "val_sha256": m["data"]["fingerprints"]["val"], "seed": m["data"]["seed"],
                         "split_mode": m["data"]["split_mode"]},
                "models": {r: (i or {}).get("revision") or (i or {}).get("tag") for r, i in m["models"].items()},
                "model_files": {r: sorted((i or {}).get("files", {}).items()) for r, i in m["models"].items()},
                "training": m["config_resolved"]["training"], "engine": engines, "numerics": num}
    pa, pb = pick(a), pick(b)
    out = {}
    for g in pa:
        fa = _flat({g: pa[g]}) if isinstance(pa[g], dict) else {g: json.dumps(pa[g])}
        fb = _flat({g: pb[g]}) if isinstance(pb[g], dict) else {g: json.dumps(pb[g])}
        d = sorted(k for k in set(fa) | set(fb) if fa.get(k) != fb.get(k))
        if d:
            out[g] = d
    return out


def compare(views: list[dict]) -> dict:
    rows = []
    for v in views:
        t = {r["comparator"]: r for r in v["table"]}
        rows.append({"run": v["run_id"], "evaluation": v["eval_id"], "explicit": v["explicit"],
                     "available": v["available"], "trained": t.get("trained", {}).get("primary"),
                     "baseline": t.get("baseline", {}).get("primary"),
                     "untrained": t.get("untrained", {}).get("primary"),
                     "metric": f"{R.primary_name(v['metric_key'])} v{v['metric_version']}",
                     "rows": t.get("trained", {}).get("rows"), "protocol": v["protocol"]["sha256"][:12],
                     "engine": (v["conditions"] or {}).get("engine"),
                     "parent": v["run"].get("parent")})
    groups: list[list[dict]] = []
    for v in views:
        for g in groups:
            if not protocol_differences(g[0], v):
                g.append(v)
                break
        else:
            groups.append([v])
    outcomes = []
    for g in groups:
        if len(g) < 2:
            continue
        conds = [condition_differences(g[0], v) for v in g[1:]]
        flat = sorted({c for cs in conds for c in cs})
        label = "common evaluator" if not flat else "cross-runtime"
        key = lambda v: (-1 if ({r["comparator"]: r for r in v["table"]}.get("trained", {}).get("primary") is None)
                         else {r["comparator"]: r for r in v["table"]}["trained"]["primary"])
        outcomes.append({"label": label, "ranked": [f"{v['run_id']} ({v['eval_id']})" for v in sorted(g, key=key, reverse=True)],
                         "condition_differences": flat,
                         "metric": R.primary_name(g[0]["metric_key"])})
    incompatible = []
    for g in groups[1:]:
        why = protocol_differences(groups[0][0], g[0])
        incompatible.append({"runs": [f"{v['run_id']} ({v['eval_id']})" for v in g],
                             "against": f"{groups[0][0]['run_id']} ({groups[0][0]['eval_id']})",
                             "reasons": why})
    diffs = {}
    if len(views) > 1:
        for v in views[1:]:
            diffs[f"{views[0]['run_id']} vs {v['run_id']}"] = differences(views[0], v)
    return {"rows": rows, "outcomes": outcomes, "incompatible": incompatible, "differences": diffs}


def render(res: dict) -> list[str]:
    f = lambda v: "unavailable" if v is None else f"{v:.3f}"
    lines = ["run  evaluation  metric  trained  baseline  untrained  rows  protocol  engine"]
    for r in res["rows"]:
        lines.append(f"{r['run']}  {r['evaluation']}  {r['metric']}  {f(r['trained'])}  {f(r['baseline'])}  "
                     f"{f(r['untrained'])}  {r['rows']}  {r['protocol']}  {r['engine'] or 'not recorded'}")
    for r in res["rows"]:
        extra = len(r["available"]) > 1
        lines.append(f"evaluation used for {r['run']}: {r['evaluation']} "
                     + ("(chosen with --use)" if r["explicit"] else
                        "(the run's original evaluation; no --use given" +
                        (f"; others exist: {', '.join(x for x in r['available'] if x != r['evaluation'])})"
                         if extra else ")")))
        if r["parent"]:
            lines.append(f"run {r['run']} has parent {r['parent']}: {LINEAGE}")
    for o in res["outcomes"]:
        lines.append("")
        lines.append(f"{o['label']}: ranked by the trained model's {o['metric']}: " + " > ".join(o["ranked"]))
        if o["label"] == "cross-runtime":
            lines.append("   the evaluators differ: " + "; ".join(o["condition_differences"]))
            lines.append("   caution: score differences may include evaluation-runtime effects, not only "
                         "differences between the models")
        lines.append("   a higher number is an observed difference on these rows, not a statistical claim")
    for inc in res["incompatible"]:
        lines.append("")
        lines.append(f"incompatible: no ranking between {', '.join(inc['runs'])} and {inc['against']}: "
                     + "; ".join(inc["reasons"]))
    for k, d in res["differences"].items():
        lines.append("")
        lines.append(f"differences {k}:" + ("" if d else " none in data, models, training, engine"))
        for g, items in d.items():
            lines.append(f"   {g}: {', '.join(items[:8])}")
    return lines
