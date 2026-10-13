"""spill plan: what a build will do, before it does anything.

The plan names the models and the engine with the reason for each, the downloads and their
sizes, the disk the run, its checkpoints and an export need, and a duration per stage.
Every duration is labeled a measurement (this machine ran this stage before: timings.json or
the calibration) or an assumption (the cost model with a stated rate), or unknown. Making a
plan loads no model, downloads nothing and trains nothing."""

from __future__ import annotations

import copy
import json
import math
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from .. import estimate as est
from ..errors import SpillError
from . import config as C
from . import contract as K
from . import judge as J
from . import modelid, timings, training
from . import scoring as SC
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
    judge_rec: dict | None = None

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


def judge_tag_of(cfg: dict) -> str:
    j = cfg["model"].get("judge") or ""
    return "" if j == "none" else j


def resolve_roles(cfg: dict) -> dict:
    """The config with the model-role rules applied (a copy). On an open-ended project: a named
    teacher is the judge unless a judge is named (or `none`), and a judge makes its assessment
    the primary metric. Idempotent."""
    cfg = copy.deepcopy(cfg)
    if cfg["task"]["type"] != "open":
        return cfg
    m, ev = cfg["model"], cfg["evaluation"]
    if not m.get("judge") and m.get("teacher"):
        m["judge"] = m["teacher"]
    if judge_tag_of(cfg) and ev["metric"] == "token_f1":
        ev["metric"] = "judge_mean"
    elif not judge_tag_of(cfg) and ev["metric"] == "judge_mean":
        ev["metric"] = "token_f1"
    return cfg


def recommended_judge(student: str) -> str:
    """The tag of the judge recommended for this machine (nothing is fetched)."""
    from .. import probe as probe_mod
    try:
        hw = probe_mod.load(probe_if_missing=False)
    except (FileNotFoundError, OSError):
        hw = None
    eng, _ = engine_pick(None)
    return J.recommend(student, _working_set(eng, hw), None)["tag"]


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
    cfg = resolve_roles(cfg)
    contract = cfg["contract"]
    C.check_metric(cfg["task"]["type"], cfg["evaluation"]["metric"])
    ev0 = cfg["evaluation"]
    if frozen is None and ev0.get("scoring") == "script":
        sp = project / ev0["script_file"]
        if not sp.exists() or sha_file(sp) != ev0["script_sha256"]:
            raise SpillError(f"{ev0['script_file']} changed since `spill init` recorded it; the "
                             f"scoring script is part of the evaluation protocol",
                             f"spill init <data> --input <col> --output <col> --score script:"
                             f"{ev0['script_file']} --project {project} --force")
    data_dir = Path(frozen) if frozen is not None else project / "data"
    path_of = (lambda n: data_dir / f"{n}.jsonl")
    t = cfg["training"]
    task = cfg["task"]["type"]
    eng, why = engine_pick(engine or (cfg["model"].get("engine") if cfg["model"].get("engine") not in (None, "", "auto") else None))
    from .. import decoding as dec_mod
    dec_mod.check_decoding(eng, {**C.DEFAULT_DECODING, **cfg["evaluation"].get("decoding", {})})
    student_tag = cfg["model"]["student"]
    teacher_tag = cfg["model"].get("teacher") or ""
    judge_tag = judge_tag_of(cfg)
    use_judge = bool(judge_tag) and task == "open"
    if cfg["evaluation"]["metric"] == "judge_mean" and not use_judge:
        raise SpillError("the metric judge_mean needs a judge on an open-ended project",
                         f"spill build {project} --judge <model>   (or --judge recommended)")
    sid = modelid.student_identity(student_tag, fetch=fetch)
    tid = modelid.student_identity(teacher_tag, fetch=fetch) if teacher_tag else None
    jid = None
    if use_judge:
        jid = (sid if J.same_model(judge_tag, student_tag) else tid if tid and J.same_model(judge_tag, teacher_tag)
               else modelid.student_identity(judge_tag, fetch=fetch))
    eid = modelid.embedding_identity(fetch=fetch) if task == "classification" else None
    for tag, ident in ((student_tag, sid), (teacher_tag, tid), (judge_tag, jid)):
        if ident:
            _tag_params(tag, ident)
    mid = {"student": modelid.identity_fingerprint(sid), "teacher": modelid.identity_fingerprint(tid),
           "embedding": modelid.identity_fingerprint(eid)}
    jproto = None
    if use_judge:
        mid["judge"] = modelid.identity_fingerprint(jid)
        rubric, rsource = J.rubric_for(Path(frozen) if frozen is not None else project)
        jproto = J.protocol(mid["judge"], rubric, rsource)
    fields = C.identity_fields(cfg, shas, mid, jproto)
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
    if task in ("classification", "open"):
        stages.append(StageDesc("baseline", "baseline", ("embedding + logistic regression baseline" if task == "classification"
                                                  else "nearest training answer baseline (TF-IDF, no model)"),
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

    if use_judge:
        present = (["baseline"] if task in ("classification", "open") else []) + ["untrained", "trained"] \
            + (["teacher"] if teacher_tag else [])
        jp = J.plan_judging(present, student_tag, teacher_tag or None, judge_tag)
        n_req = len(val) * (len(jp["judged"]) + 2 * len(jp["pairs"]))
        stages.append(StageDesc(
            "judge", "judge", f"judge {judge_tag} scores {len(jp['judged'])} comparators and "
            f"{len(jp['pairs'])} pairs (both orders) on {len(val)} validation rows: {n_req} requests "
            f"(one model's assessment)", {"judge": judge_tag, **jp}, inputs[1:],
            ["judgments.jsonl", "metrics.json"],
            [("baseline" if c == "baseline" else f"eval:{c}") for c in present]))

    cal = load_calibration()
    try:
        hw = probe_mod.load(probe_if_missing=False)
    except (FileNotFoundError, OSError):
        hw = None
    estimates = _estimates(cfg, stages, train, val, schema, eng, cal, hw)
    downloads = [d for d in (
        _dl("student", student_tag, sid), _dl("teacher", teacher_tag, tid) if tid else None,
        _dl("judge", judge_tag, jid) if jid and jid is not sid and jid is not tid else None,
        _dl("embedding", modelid.EMBEDDING_TAG, eid) if eid else None) if d]
    disk = _disk(cfg, sid, train, t, eng)
    plan = BuildPlan(project, cfg, eng, why, stages, estimates,
                     {"student": sid, "teacher": tid, "judge": jid, "embedding": eid}, downloads, disk,
                     ident_sha, fields, shas)
    if task == "open" and not judge_tag:
        n_j = (1 if task == "open" else 0) + 2 + (1 if teacher_tag else 0)
        plan.judge_rec = _judge_recommendation(cfg, student_tag, val, eng, cal, hw, fetch, n_j,
                                               n_j - 1)
    if judge_tag and not use_judge:
        plan.notes.append(f"the judge {judge_tag} is recorded but unused: {SC.scoring_of(cfg)} "
                          f"scoring needs no judge (judged scores are for open-ended output)")
    if use_judge:
        for c, why in jp["excluded"].items():
            plan.notes.append(why)
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


def _working_set(eng, hw) -> int:
    ws = (hw or {}).get("gpu", {}).get("vram_bytes") if eng == "mlx" else None
    if ws is None:
        try:
            from ..engines.torch_common import memory_total_bytes
            ws = memory_total_bytes(eng) if eng != "mlx" else 16 * GIB
        except Exception:
            ws = 16 * GIB
    return ws


def _judge_recommendation(cfg, student_tag, val, eng, cal, hw, fetch, n_judged, n_pairs) -> dict | None:
    """The judge to recommend when none is named: sized to this machine, with its download size,
    disk need and estimated judging time. Loads and downloads nothing."""
    ws = _working_set(eng, hw)
    rec = J.recommend(student_tag, ws, shutil.disk_usage(Path.home()).free)
    from ..registry import safetensors_downloaded, safetensors_spec
    spec = safetensors_spec(rec["tag"])
    size = spec["bytes"] if spec else None
    present = safetensors_downloaded(rec["tag"]) if spec else False
    n_req = len(val) * (n_judged + 2 * n_pairs)
    avg_in = sum(len(r["input"]) for r in val) / max(1, len(val))
    avg_out = sum(len(K.target_text(cfg, r["output"])) for r in val) / max(1, len(val))
    wl = est.Workload(n_req, 120.0, _tok(avg_in + 2 * avg_out) + 60, 45.0, J.MAX_TOKENS)
    tf, src = est.tflops_for(cal, eng, rec["tag"])
    try:
        sec = est.eval_seconds(rec["tag"], wl, cal, ws, tf, reuse=True, adapter=False, engine=eng).seconds
    except Exception:
        sec = None
    return {**rec, "bytes": size, "present": present, "seconds": sec, "basis": "assumption",
            "requests": n_req}


def _estimates(cfg, stages, train, val, schema, eng, cal, hw) -> dict:
    out = {}
    student = cfg["model"]["student"]
    t = cfg["training"]
    mt = cfg["evaluation"]["max_tokens"]
    ws = _working_set(eng, hw)
    for s in stages:
        if s.kind == "baseline" and cfg["task"]["type"] == "open":
            out[s.id] = StageEst(1.0, "assumption", "word counting over the training inputs, no model; "
                                 "under a second at this size")
            continue
        if s.kind == "baseline":
            m = timings.lookup("torch-cpu", "embedding:minilm", "baseline")
            n = len(train) + len(val)
            out[s.id] = (StageEst(m["seconds_per_unit"] * n, "measurement",
                                  f"measured {m['measured']}: {m['seconds_per_unit']:.3f} s per text "
                                  f"over {n} texts on this machine") if m else
                         StageEst(None, "unknown", f"no measurement of the embedding on this machine "
                                  f"yet ({n} texts to embed)"))
            continue
        if s.kind == "judge":
            jm = s.params["judge"]
            n_req = len(val) * (len(s.params["judged"]) + 2 * len(s.params["pairs"]))
            m = timings.lookup(eng, jm, "judge")
            if m:
                out[s.id] = StageEst(m["seconds_per_unit"] * n_req, "measurement",
                                     f"measured {m['measured']} on {eng}: {m['seconds_per_unit']:.2f} s "
                                     f"per request x {n_req} requests")
                continue
            avg_in = sum(len(r["input"]) for r in val) / max(1, len(val))
            avg_out = sum(len(K.target_text(cfg, r["output"])) for r in val) / max(1, len(val))
            wl = est.Workload(n_req, 120.0, _tok(avg_in + 2 * avg_out) + 60, 45.0, J.MAX_TOKENS)
            tf, src = est.tflops_for(cal, eng, jm)
            e = est.eval_seconds(jm, wl, cal, ws, tf, reuse=True, adapter=False, engine=eng)
            out[s.id] = StageEst(e.seconds, "assumption",
                                 f"{e.detail}; no matching measurement yet (rate {src})")
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
    if p.models.get("judge"):
        models.append(f"{p.cfg['model']['judge']} (judge)")
    if p.models.get("embedding"):
        models.append("all-MiniLM-L6-v2 (baseline embedding)")
    total = p.total_s
    known = sum(e.seconds for e in p.estimates.values() if e.seconds is not None)
    tot = est.fmt_dur(total) if total is not None else f"{est.fmt_dur(known)} plus unknown stages"
    n = 0
    return (f"spill {verb} {p.project.name}: scoring {SC.scoring_of(p.cfg)}, models {', '.join(models)}, "
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
    if p.models.get("judge"):
        j = p.models["judge"]
        lines.append(f"   judge     {j['tag']}  {j['repo'] or 'local'}@{(j['revision'] or '')[:12]}  (named in "
                     f"streamweights.toml; its scores are one model's assessment, not accuracy)")
    elif p.judge_rec:
        r = p.judge_rec
        lines.append("   judge     " + J.recommendation_line(r, r["bytes"], r["present"], r["seconds"],
                                                              est.fmt_dur))
    if p.models.get("embedding"):
        e = p.models["embedding"]
        lines.append(f"   baseline  {e['repo']}@{e['revision'][:12]} + logistic regression")
    elif p.cfg["task"]["type"] == "open":
        lines.append("   baseline  nearest training answer by TF-IDF (no model, nothing downloaded)")
    lines.append(f"engine: {p.engine}: {p.engine_why}")
    lines.append(f"scoring: {SC.describe(p.cfg)}")
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
