"""spill plan: what a build will do, before it does anything.

The plan names the models and the engine with the reason for each, the downloads and their
sizes, the disk the run, its checkpoints and an export need, and a duration per stage.
Every duration is labeled a measurement (this machine ran this stage before: timings.json or
the calibration) or an assumption (the cost model with a stated rate), or unknown. Making a
plan loads no model, downloads nothing and trains nothing."""

from __future__ import annotations

import json
import math
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from .. import estimate as est
from ..errors import SpillError
from . import config as C
from . import contract as K
from . import modelid, timings, training
from .common import read_jsonl, sha_file, sha_obj
from .stagefns import StageDesc

GIB = 1024 ** 3


@dataclass
class StageEst:
    seconds: float | None
    basis: str            # measurement | assumption | unknown
    detail: str


@dataclass
class BuildPlan:
    project: Path
    cfg: dict
    engine: str
    engine_why: str
    stages: list[StageDesc]
    estimates: dict
    models: dict
    downloads: list
    disk: dict
    identity: str
    identity_fields: dict
    split_sha: dict
    notes: list = field(default_factory=list)

    @property
    def total_s(self) -> float | None:
        vals = [e.seconds for e in self.estimates.values()]
        return None if any(v is None for v in vals) else sum(vals)


def load_project(project: Path, frozen: Path | None = None) -> tuple[dict, dict]:
    """(config, {name: sha256 of the canonical data file}); refuses a project whose files are
    gone. With `frozen` (a run's own frozen inputs) the config and data come from there and the
    test hash from the run's recorded identity, never from the live project."""
    if frozen is not None:
        cfg = C.parse((Path(frozen) / "streamweights.toml").read_bytes(), "the run's frozen config")
        ident = json.loads((Path(frozen) / "identity.json").read_text())
        return cfg, {"train": sha_file(Path(frozen) / "train.jsonl"),
                     "val": sha_file(Path(frozen) / "val.jsonl"),
                     "test": ident["split_sha256"]["test"]}
    cfg = C.load(project)
    shas = {}
    for name in ("train", "val", "test"):
        p = Path(project) / "data" / f"{name}.jsonl"
        if not p.exists():
            raise SpillError(f"{project.name}/data/{name}.jsonl is missing",
                             "spill init <data> --input <col> --output <col>")
        shas[name] = sha_file(p)
    return cfg, shas


def engine_pick(override: str | None):
    from .. import engine_select
    ch = engine_select.choose_engine(override)
    return ch.name, ch.why


def _tag_params(tag: str, ident: dict) -> None:
    if tag not in est.PARAMS_B:
        size = sum(f["bytes"] for n, f in ident["files"].items() if n.endswith(".safetensors"))
        if size:
            est.register_model(tag, size)


def _tok(chars: float) -> float:
    return chars / 4


def _lora_params(model_dir: Path | None, rank: int) -> int | None:
    try:
        c = json.loads((Path(model_dir) / "config.json").read_text())
        h, L, ff = c["hidden_size"], c["num_hidden_layers"], c["intermediate_size"]
        kv = h * c["num_key_value_heads"] // c["num_attention_heads"]
        per = rank * ((h + h) + (h + kv) * 2 + (h + h) + (h + ff) * 2 + (ff + h))
        return per * L
    except (OSError, KeyError, ValueError, TypeError):
        return None


def make_plan(project: Path, engine: str | None = None, verify: bool = True,
              fetch: bool = False, frozen: Path | None = None) -> BuildPlan:
    from .. import probe as probe_mod
    from ..calibration import load_calibration
    project = Path(project)
    cfg, shas = load_project(project, frozen)
    contract = cfg["contract"]
    C.check_metric(cfg["task"]["type"], cfg["evaluation"]["metric"])
    data_dir = Path(frozen) if frozen is not None else project / "data"
    path_of = (lambda n: data_dir / f"{n}.jsonl")
    t = cfg["training"]
    task = cfg["task"]["type"]
    eng, why = engine_pick(engine or (cfg["model"].get("engine") if cfg["model"].get("engine") not in (None, "", "auto") else None))
    from .. import decoding as dec_mod
    dec_mod.check_decoding(eng, {**C.DEFAULT_DECODING, **cfg["evaluation"].get("decoding", {})})
    student_tag = cfg["model"]["student"]
    teacher_tag = cfg["model"].get("teacher") or ""
    sid = modelid.student_identity(student_tag, fetch=fetch)
    tid = modelid.student_identity(teacher_tag, fetch=fetch) if teacher_tag else None
    eid = modelid.embedding_identity(fetch=fetch) if task == "classification" else None
    for tag, ident in ((student_tag, sid), (teacher_tag, tid)):
        if ident:
            _tag_params(tag, ident)
    mid = {"student": modelid.identity_fingerprint(sid), "teacher": modelid.identity_fingerprint(tid),
           "embedding": modelid.identity_fingerprint(eid)}
    fields = C.identity_fields(cfg, shas, mid)
    ident_sha = sha_obj(fields)

    train = read_jsonl(path_of("train"))
    val = read_jsonl(path_of("val"))
    schema = None
    if task == "json":
        schema = json.loads(((Path(frozen) if frozen is not None else project) /
                             contract["schema_file"]).read_text())
    inputs = [{"name": "train.jsonl", "sha256": shas["train"]},
              {"name": "val.jsonl", "sha256": shas["val"]}]

    stages: list[StageDesc] = []
    if teacher_tag:
        stages.append(StageDesc("distill", "distill", f"teacher {teacher_tag} answers the "
                                f"{len(train)} training prompts", {"teacher": teacher_tag},
                                inputs[:1], ["teacher.jsonl"]))
    weight_own = training.DEFAULT_WEIGHT_OWN
    if task == "classification":
        stages.append(StageDesc("baseline", "baseline", "embedding + logistic regression baseline",
                                {"teacher": bool(teacher_tag), "weight_own": weight_own}, inputs,
                                ["predictions.jsonl", "metrics.json"],
                                ["distill"] if teacher_tag else []))
    name = f"{cfg['project']['name']}-{ident_sha[:8]}"
    stages.append(StageDesc("train", "train", f"tune {student_tag} on {len(train)} training rows",
                            {"model": student_tag, "adapter_name": name,
                             "teacher": bool(teacher_tag), "weight_own": weight_own},
                            inputs[:1], ["adapter", "training.manifest.json", "training.rows.jsonl"],
                            ["distill"] if teacher_tag else []))
    ev = lambda cmp, model, label, dep=(): StageDesc(
        f"eval:{cmp}", "eval", label, {"comparator": cmp, "model": model}, inputs[1:],
        ["predictions.jsonl", "metrics.json"], list(dep))
    stages.append(ev("untrained", student_tag, f"{student_tag} prompted, untrained, on {len(val)} validation rows"))
    stages.append(ev("trained", student_tag, f"{student_tag} trained, on {len(val)} validation rows", ["train"]))
    if teacher_tag:
        stages.append(ev("teacher", teacher_tag, f"teacher {teacher_tag} on {len(val)} validation rows"))

    cal = load_calibration()
    try:
        hw = probe_mod.load(probe_if_missing=False)
    except (FileNotFoundError, OSError):
        hw = None
    estimates = _estimates(cfg, stages, train, val, schema, eng, cal, hw)
    downloads = [d for d in (
        _dl("student", student_tag, sid), _dl("teacher", teacher_tag, tid) if tid else None,
        _dl("embedding", modelid.EMBEDDING_TAG, eid) if eid else None) if d]
    disk = _disk(cfg, sid, train, t, eng)
    plan = BuildPlan(project, cfg, eng, why, stages, estimates,
                     {"student": sid, "teacher": tid, "embedding": eid}, downloads, disk,
                     ident_sha, fields, shas)
    if hw is None:
        plan.notes.append("no hardware probe yet; estimates use assumptions (spill doctor probes)")
    return plan


def _dl(role: str, tag: str, ident: dict) -> dict | None:
    size = sum(f["bytes"] for f in ident["files"].values())
    return {"role": role, "model": tag, "repo": ident.get("repo"), "revision": ident.get("revision"),
            "bytes": size, "present": ident.get("present", False)}


def _disk(cfg, sid, train, t, engine) -> dict:
    mdir = Path(sid["dir"]) if sid.get("dir") else None
    lp = _lora_params(mdir, t["rank"]) if mdir and mdir.exists() else None
    base = sum(f["bytes"] for n, f in sid["files"].items() if n.endswith(".safetensors"))
    steps = None
    n = len(train) * (1 if not cfg["model"].get("teacher") else 3)
    mb = t["micro_batch"] * t["grad_accum"]
    steps = max(1, math.ceil(n * t["epochs"] / mb))
    n_ck = max(1, math.ceil(steps / t["ckpt_every"]))
    ck_bytes = lp * 12 if lp else None            # params + Adam m and v, float32
    return {"checkpoints": (n_ck * ck_bytes) if ck_bytes else None, "checkpoint_each": ck_bytes,
            "checkpoint_count": n_ck, "adapter": lp * 4 if lp else None,
            "staging": 3 * (lp * 4 if lp else 0) + 50e6 if lp else None,
            "export_merged": base, "export_gguf_q8": int(base * 0.53)}


def _estimates(cfg, stages, train, val, schema, eng, cal, hw) -> dict:
    out = {}
    student = cfg["model"]["student"]
    t = cfg["training"]
    mt = cfg["evaluation"]["max_tokens"]
    ws = (hw or {}).get("gpu", {}).get("vram_bytes") if eng == "mlx" else None
    if ws is None:
        try:
            from ..engines.torch_common import memory_total_bytes
            ws = memory_total_bytes(eng) if eng != "mlx" else 16 * GIB
        except Exception:
            ws = 16 * GIB
    for s in stages:
        if s.kind == "baseline":
            m = timings.lookup("torch-cpu", "embedding:minilm", "baseline")
            n = len(train) + len(val)
            out[s.id] = (StageEst(m["seconds_per_unit"] * n, "measurement",
                                  f"measured {m['measured']}: {m['seconds_per_unit']:.3f} s per text "
                                  f"over {n} texts on this machine") if m else
                         StageEst(None, "unknown", f"no measurement of the embedding on this machine "
                                  f"yet ({n} texts to embed)"))
            continue
        model = s.params.get("model") or s.params.get("teacher") or student
        if s.kind == "train":
            toks = sum(_tok(len(r["input"]) + len(K.target_text(cfg, r["output"]))) + 24 for r in train)
            reps = 1 if not cfg["model"].get("teacher") else 3
            m = timings.lookup(eng, model, "train")
            units = toks * reps * t["epochs"]
            if m:
                out[s.id] = StageEst(m["seconds_per_unit"] * units, "measurement",
                                     f"measured {m['measured']} on {eng}: "
                                     f"{m['seconds_per_unit'] * 1e3:.2f} ms per token-pass x "
                                     f"{units:,.0f} token-passes")
            else:
                tf, src = est.tflops_for(cal, eng, model)
                e = est.tune_seconds(model, toks * reps, t["epochs"], tf, cal, ws, engine=eng)
                basis = "measurement" if src.startswith("measured") else "assumption"
                out[s.id] = StageEst(e.seconds, basis, f"{e.detail}; rate {src}")
            continue
        if s.kind in ("eval", "distill"):
            rows = val if s.kind == "eval" else train
            m = timings.lookup(eng, model, "eval" if s.kind == "eval" else "distill")
            if m:
                out[s.id] = StageEst(m["seconds_per_unit"] * len(rows), "measurement",
                                     f"measured {m['measured']} on {eng}: {m['seconds_per_unit']:.2f} s "
                                     f"per row x {len(rows)} rows")
                continue
            texts = [r["input"] for r in rows]
            pre = est.common_prefix_chars([K.contract_text(cfg, schema) + t_ for t_ in texts]) \
                if s.params.get("comparator") != "trained" else 0
            suf = _tok(sum(len(x) for x in texts) / max(1, len(texts))) + 8
            outtok = min(mt, _tok(sum(len(K.target_text(cfg, r["output"])) for r in rows)
                                  / max(1, len(rows))) + 3)
            wl = est.Workload(len(rows), _tok(pre) + 20, suf, max(1.0, outtok), mt)
            tf, src = est.tflops_for(cal, eng, model)
            e = est.eval_seconds(model, wl, cal, ws, tf, reuse=True,
                                 adapter=s.params.get("comparator") == "trained", engine=eng)
            out[s.id] = StageEst(e.seconds, "assumption",
                                 f"{e.detail}; no matching measurement yet (rate {src})")
    return out


# ------------------------------------------------------------ rendering

def _gb(b) -> str:
    return "unknown" if b is None else (f"{b / GIB:.2f} GB" if b >= 0.1 * GIB else f"{b / 1e6:.0f} MB")


def pre_run_line(p: BuildPlan, verb: str = "build") -> str:
    models = [f"{p.cfg['model']['student']} (student)"]
    if p.cfg["model"].get("teacher"):
        models.append(f"{p.cfg['model']['teacher']} (teacher)")
    if p.models.get("embedding"):
        models.append("all-MiniLM-L6-v2 (baseline embedding)")
    total = p.total_s
    known = sum(e.seconds for e in p.estimates.values() if e.seconds is not None)
    tot = est.fmt_dur(total) if total is not None else f"{est.fmt_dur(known)} plus unknown stages"
    n = 0
    return (f"spill {verb} {p.project.name}: {p.cfg['task']['type']}, models {', '.join(models)}, "
            f"bf16 weights, no quantization, on {p.engine} ({p.engine_why}). "
            f"Est. {tot}{' (some stages unknown)' if n and total is not None else ''}. Cost: $0. "
            f"Run -> {p.project / 'runs'}/<run id>")


def render(p: BuildPlan) -> list[str]:
    lines = [pre_run_line(p, "plan")]
    lines.append(f"models:")
    s = p.models["student"]
    lines.append(f"   student   {s['tag']}  {s['repo'] or 'local'}@{(s['revision'] or '')[:12]}  "
                 f"{s['format']}  (chosen by {'streamweights.toml'}; default for the guided path)")
    if p.models.get("teacher"):
        t = p.models["teacher"]
        lines.append(f"   teacher   {t['tag']}  {t['repo']}@{(t['revision'] or '')[:12]}  (asked for "
                     f"in streamweights.toml; used only for training-partition answers)")
    else:
        lines.append("   teacher   none (a teacher runs only when you ask: spill build --teacher <model>)")
    if p.models.get("embedding"):
        e = p.models["embedding"]
        lines.append(f"   baseline  {e['repo']}@{e['revision'][:12]} + logistic regression")
    lines.append(f"engine: {p.engine}: {p.engine_why}")
    lines.append("downloads:")
    need = [d for d in p.downloads if not d["present"]]
    for d in p.downloads:
        lines.append(f"   {d['model']}: {_gb(d['bytes'])} "
                     f"{'already on disk' if d['present'] else 'to download at the pinned revision'}")
    if not need:
        lines.append("   nothing to download")
    dk = p.disk
    free = shutil.disk_usage(p.project).free
    lines.append("disk:")
    lines.append(f"   checkpoints {_gb(dk['checkpoints'])} ({dk['checkpoint_count']} x "
                 f"{_gb(dk['checkpoint_each'])}, every published checkpoint is kept; deleting them "
                 f"is a later feature), staging {_gb(dk['staging'])}, adapter {_gb(dk['adapter'])}")
    lines.append(f"   export (only if you run it): merged {_gb(dk['export_merged'])}, GGUF q8_0 "
                 f"about {_gb(dk['export_gguf_q8'])} (assumption: 0.53 x bf16); free here {_gb(free)}")
    lines.append("stages:")
    for i, st in enumerate(p.stages, 1):
        e = p.estimates.get(st.id)
        dur = ("<1 s" if e.seconds < 1 else est.fmt_dur(e.seconds)) \
            if e and e.seconds is not None else "unknown"
        lines.append(f"   {i}. {st.label}: {dur} [{e.basis if e else 'unknown'}] {e.detail if e else ''}")
    known = sum(e.seconds for e in p.estimates.values() if e.seconds is not None)
    unk = [k for k, e in p.estimates.items() if e.seconds is None]
    lines.append(f"total: {est.fmt_dur(known)}" + (f" plus unknown ({', '.join(unk)}: no measurement "
                 f"on this machine yet)" if unk else "") + "; model loading is not included")
    for n in p.notes:
        lines.append(f"note: {n}")
    return lines
