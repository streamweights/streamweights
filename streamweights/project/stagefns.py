"""What each stage of a guided build does. A stage is described by a serializable StageDesc
and run by `run_stage(desc, ctx)`, which returns its outputs (files in a staging directory),
metrics and status. Nothing here publishes anything: the coordinator verifies the staged
outputs and publishes them through the fenced control object."""

from __future__ import annotations

import contextlib
import io
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from ..errors import SpillError
from . import baseline as B
from . import config as C
from . import contract as K
from . import modelid, training
from . import judge as J
from . import scoring as SC
from .common import read_json, read_jsonl, sha_file, write_json, write_jsonl


# ------------------------------------------------------------ descriptions

@dataclass
class StageDesc:
    id: str
    kind: str                 # baseline | distill | train | eval
    label: str
    params: dict = field(default_factory=dict)
    inputs: list = field(default_factory=list)      # [{"name", "sha256"}]: files by identity
    outputs: list = field(default_factory=list)     # declared output names
    depends: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"id": self.id, "kind": self.kind, "label": self.label, "params": self.params,
                "inputs": self.inputs, "outputs": self.outputs, "depends": self.depends}

    @classmethod
    def from_dict(cls, d: dict) -> "StageDesc":
        return cls(d["id"], d["kind"], d["label"], d.get("params", {}), d.get("inputs", []),
                   d.get("outputs", []), d.get("depends", []))


@dataclass
class StageContext:
    attempt_dir: Path
    stage_dir: Path
    engine: str | None = None
    stop_after: int | None = None
    on_checkpoint: object = None
    say: object = None

    @property
    def inputs_dir(self) -> Path:
        return self.attempt_dir / "inputs"

    @property
    def out_dir(self) -> Path:
        return self.stage_dir / "out"

    def dep_dir(self, stage_id: str) -> Path:
        return self.attempt_dir / "accepted" / slug(stage_id)


@dataclass
class StageOutcome:
    status: str                      # done | interrupted
    metrics: dict = field(default_factory=dict)
    producers: list = field(default_factory=list)
    seconds: float = 0.0
    notes: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"status": self.status, "metrics": self.metrics, "producers": self.producers,
                "seconds": self.seconds, "notes": self.notes}


def slug(stage_id: str) -> str:
    return stage_id.replace(":", "_").replace("/", "_")


def _config(ctx: StageContext) -> dict:
    from . import config as C
    return C.parse((ctx.inputs_dir / "streamweights.toml").read_bytes())


def _schema(ctx: StageContext, cfg: dict) -> dict | None:
    if cfg["task"]["type"] != "json":
        return None
    return read_json(ctx.inputs_dir / cfg["contract"]["schema_file"])


@contextlib.contextmanager
def silenced():
    with contextlib.redirect_stdout(io.StringIO()):
        yield


# ------------------------------------------------------------ the stages

def _normalize_producers(items: list[dict]) -> list[dict]:
    from ..build import engine_label
    from .. import machine
    m = machine.info()
    out = []
    for p in items:
        q = dict(p)
        q["engine"] = engine_label(p.get("engine"), p.get("hardware"))
        q.setdefault("host", m["host"])
        q.setdefault("system", m["system"])
        if not q.get("os"):
            q["os"] = f"{m['os']} {m['arch']}"
        out.append(q)
    return out


def run_stage(desc: StageDesc, ctx: StageContext) -> StageOutcome:
    ctx.out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    fn = {"baseline": _baseline, "distill": _distill, "train": _train, "eval": _eval,
          "judge": _judge}[desc.kind]
    out = fn(desc, ctx)
    out.seconds = round(time.monotonic() - t0, 2)
    out.producers = _normalize_producers(out.producers)
    return out


def _baseline(desc: StageDesc, ctx: StageContext) -> StageOutcome:
    cfg = _config(ctx)
    train = read_jsonl(ctx.inputs_dir / "train.jsonl")
    val = read_jsonl(ctx.inputs_dir / "val.jsonl")
    if desc.params.get("training_rows"):       # `spill test`: the very rows the run trained on
        rows = read_jsonl(Path(desc.params["training_rows"]))
        manifest = read_json(Path(desc.params["training_rows"]).with_name("training.manifest.json"))
    else:
        answers = _teacher_answers(desc, ctx)
        rows, manifest = training.build_training(cfg, train, answers, desc.params.get(
            "weight_own", training.DEFAULT_WEIGHT_OWN))
    if cfg["task"]["type"] == "open":
        res = B.run_baseline_open(cfg, rows, val)
    else:
        emb = modelid.embedding_identity(fetch=True)
        res = B.run_baseline(cfg, rows, val, emb["dir"])
    write_jsonl(ctx.out_dir / "predictions.jsonl",
                [{**p, "comparator": "baseline"} for p in res["predictions"]])
    metrics = {**res["metrics"], "settings": res["settings"], "training": manifest}
    write_json(ctx.out_dir / "metrics.json", metrics)
    v = modelid.dependency_versions()
    return StageOutcome("done", metrics, [{"engine": "torch-cpu", "hardware": "cpu",
                                            "numerics": {"base": "float32"}}],
                        notes={"conditions": {"engine": "torch-cpu",
                                              "engine_impl": ("tfidf-nearest-answer" if cfg["task"]["type"] == "open"
                                                              else "embedding+logreg"),
                                              "device": "cpu", "numerics": {"base": "float32"},
                                              "weight_dtype": "float32", "adapter_dtype": None,
                                              "compute_dtype": "float32",
                                              "versions": {k: v[k] for k in ("python", "torch", "transformers", "numpy") if k in v},
                                              "decoding_applied": None}})


def _teacher_answers(desc: StageDesc, ctx: StageContext) -> dict | None:
    if not desc.params.get("teacher"):
        return None
    d = ctx.dep_dir("distill")
    recs = read_jsonl(d / "teacher.jsonl")
    return {r["custom_id"]: r["completion"] for r in recs}


def _backend():
    from ..cli_build import RealBackend
    return RealBackend()


def _prompt_rows(cfg: dict, schema, rows: list[dict], view: str, max_tokens: int) -> list[dict]:
    """Batch-shaped requests. Every recorded decoding setting is in each request body."""
    from .. import decoding
    dec = {**C.DEFAULT_DECODING, **cfg["evaluation"].get("decoding", {})}
    out = []
    for r in rows:
        msgs = (K.messages_untrained(cfg, schema, r["input"]) if view == "untrained"
                else K.messages_student(cfg, r["input"]))
        out.append({"custom_id": r["id"], "method": "POST", "url": "/v1/chat/completions",
                    "body": {"messages": msgs, **decoding.request_fields(dec, max_tokens)},
                    "expected": K.target_text(cfg, r["output"])})
    return out


def executed_conditions(result_rows: list[dict], engine: str) -> dict:
    """What an evaluation actually ran under, read off its result rows (not off the config):
    engine, device, numerics, runtime and library versions, and the decoding as applied. Kept
    apart from the requested precision policy in the protocol."""
    from . import modelid
    sw = (result_rows[0].get("streamweights") if result_rows else None) or {}
    num = sw.get("numerics") or {}
    vers = modelid.dependency_versions()
    keep = {k: vers[k] for k in ("python", "torch", "transformers", "peft", "mlx", "mlx-lm",
                                 "safetensors", "numpy") if k in vers}
    return {"engine": engine, "engine_impl": sw.get("engine"), "device": sw.get("hardware"),
            "numerics": num, "weight_dtype": num.get("base"), "adapter_dtype": num.get("adapter"),
            "compute_dtype": num.get("compute", num.get("base")),
            "os": sw.get("os"), "versions": keep, "decoding_applied": sw.get("decoding")}


def _distill(desc: StageDesc, ctx: StageContext) -> StageOutcome:
    """The teacher answers the training rows' prompts (sequence-level distillation). Only the
    training partition's inputs go in: no validation or test row reaches the teacher."""
    cfg = _config(ctx)
    train = read_jsonl(ctx.inputs_dir / "train.jsonl")
    schema = _schema(ctx, cfg)
    mt = cfg["evaluation"]["max_tokens"]
    f = ctx.stage_dir / "prompts.jsonl"
    write_jsonl(f, [{**p} for p in _prompt_rows(cfg, schema, train, "untrained", mt)])
    from .. import runtime
    runtime.ENV.state = str(ctx.stage_dir / "state")
    runtime.ENV.in_build = True
    res = _backend().distill(desc.params["teacher"], f, ctx.out_dir / "teacher.jsonl", mt, None)
    if res.get("interrupted"):
        return StageOutcome("interrupted", {"rows": res.get("rows")}, res.get("producers", []))
    return StageOutcome("done", {"rows": res.get("rows")}, res.get("producers", []))


def _train(desc: StageDesc, ctx: StageContext) -> StageOutcome:
    from .. import cli, runtime
    from ..adapters import ADAPTERS_DIR
    cfg = _config(ctx)
    t = cfg["training"]
    train = read_jsonl(ctx.inputs_dir / "train.jsonl")
    answers = _teacher_answers(desc, ctx)
    rows, manifest = training.build_training(cfg, train, answers,
                                             desc.params.get("weight_own", training.DEFAULT_WEIGHT_OWN))
    write_jsonl(ctx.out_dir / "training.rows.jsonl", rows)
    write_json(ctx.out_dir / "training.manifest.json", manifest)
    chat = ctx.stage_dir / "train.chat.jsonl"
    write_jsonl(chat, training.chat_rows(cfg, rows))
    runtime.ENV.state = str(ctx.stage_dir / "state")
    runtime.ENV.on_checkpoint = ctx.on_checkpoint
    runtime.ENV.stop_after = ctx.stop_after
    runtime.ENV.in_build = True
    name = desc.params["adapter_name"]
    with silenced():
        job, res = cli._tune_impl(
            desc.params["model"], chat, name, t["rank"], t["alpha"], t["dropout"], None, t["lr"],
            t["schedule"], t["weight_decay"], None, t["epochs"], t["micro_batch"],
            t["grad_accum"], t["max_seq"], t["seed"], "auto", t["ckpt_every"], True, True)
    from ..cli_build import _tune_producers
    producers = _tune_producers(runtime.state_uri())
    if res["interrupted"]:
        return StageOutcome("interrupted", {"step": res["step"], "steps": res["steps"]},
                            producers)
    import shutil
    shutil.copytree(ADAPTERS_DIR / name, ctx.out_dir / "adapter", dirs_exist_ok=True)
    return StageOutcome("done", {"steps": res["steps"], "final_loss": res["final_loss"],
                                 "training": manifest}, producers)


def _eval(desc: StageDesc, ctx: StageContext) -> StageOutcome:
    from .. import runtime
    cfg = _config(ctx)
    schema = _schema(ctx, cfg)
    val = read_jsonl(ctx.inputs_dir / "val.jsonl")
    ev = cfg["evaluation"]
    mt = ev["max_tokens"]
    from .. import decoding as dec_mod
    from ..engine_select import choose_engine
    dec_mod.check_decoding(choose_engine(runtime.ENV.engine).name,
                           {**C.DEFAULT_DECODING, **ev.get("decoding", {})})
    comparator = desc.params["comparator"]            # untrained | trained | teacher
    view = "student" if comparator == "trained" else "untrained"
    model = desc.params["model"]
    if comparator == "trained":
        adapter = Path(desc.params.get("adapter") or ctx.dep_dir("train") / "adapter")
        model = f"{model}+{adapter}"
    f = ctx.stage_dir / "eval.jsonl"
    write_jsonl(f, _prompt_rows(cfg, schema, val, view, mt))
    runtime.ENV.state = str(ctx.stage_dir / "state")
    runtime.ENV.in_build = True
    res = _backend().eval(model, f)
    if res.get("interrupted"):
        return StageOutcome("interrupted", {}, res.get("producers", []))
    results = {r["custom_id"]: r for r in read_jsonl(res["results_path"])}
    preds, items = [], []
    from jsonschema import Draft202012Validator
    validator = Draft202012Validator(schema) if schema else None
    script = SC.script_for(cfg, ctx.inputs_dir)
    trunc = 0
    for r in val:
        row = results.get(r["id"])
        text, finish, toks, lat = None, None, None, None
        if row and not row.get("error") and row.get("response"):
            ch = row["response"]["body"]["choices"][0]
            text, finish = ch["message"]["content"], ch.get("finish_reason")
            toks = (row.get("streamweights") or {}).get("tokens", {}).get("completion")
            lat = (row.get("streamweights") or {}).get("latency_s")
        if finish == "length":
            trunc += 1
        sc, rec = SC.score_row(cfg, schema, validator, text, r, script)
        items.append(sc)
        preds.append({"id": r["id"], "gold": r["output"], "text": text, "finish_reason": finish,
                      "completion_tokens": toks, "latency_s": lat, "failed": sc["failed"],
                      **rec, "comparator": comparator})
    golds = [r["output"] for r in val]
    metrics = SC.aggregate(cfg, items, golds, schema)
    metrics["truncated"] = trunc
    metrics["view"] = view
    metrics["prompt_cached_run"] = bool(res.get("cached"))
    write_jsonl(ctx.out_dir / "predictions.jsonl", preds)
    write_json(ctx.out_dir / "metrics.json", metrics)
    from ..cli_build import _row_producers
    rr = list(results.values())
    return StageOutcome("done", metrics, _row_producers(res["results_path"]),
                        notes={"engine_run_id": res.get("run_id"), "model": model,
                               "reused": bool(res.get("cached")),
                               "conditions": executed_conditions(rr, choose_engine(runtime.ENV.engine).name)})


def _judge(desc: StageDesc, ctx: StageContext) -> StageOutcome:
    """The judge model assesses the comparators' outputs on the validation rows: pointwise for
    each judged comparator, pairwise (both orders) for each pair. One batch job, one model load.
    A comparator whose model is the judge is not judged (see judge.plan_judging)."""
    from .. import decoding as dec_mod
    from .. import runtime
    cfg = _config(ctx)
    val = read_jsonl(ctx.inputs_dir / "val.jsonl")
    rubric, source = J.rubric_for(ctx.inputs_dir)
    judged, pairs = desc.params["judged"], desc.params["pairs"]
    excluded = desc.params.get("excluded", {})
    files = desc.params.get("pred_files") or {}
    preds = {}
    for c in judged:
        f = Path(files[c]) if c in files else ctx.dep_dir("baseline" if c == "baseline"
                                                           else f"eval:{c}") / "predictions.jsonl"
        preds[c] = {p["id"]: p for p in read_jsonl(f)}
    body = dec_mod.request_fields({**J.DECODING}, J.MAX_TOKENS)
    reqs = []

    def add(cid, msgs):
        reqs.append({"custom_id": cid, "method": "POST", "url": "/v1/chat/completions",
                     "body": {"messages": msgs, **body}, "expected": "judged"})
    text_of = lambda c, r: preds[c].get(r["id"], {}).get("text")
    for c in judged:
        for r in val:
            add(f"pt:{c}:{r['id']}", J.pointwise_messages(rubric, r["input"], text_of(c, r),
                                                          _ref(cfg, r)))
    for x, y in pairs:
        for r in val:
            add(f"pw:{x}:{y}:xa:{r['id']}", J.pairwise_messages(rubric, r["input"], text_of(x, r),
                                                                 text_of(y, r), _ref(cfg, r)))
            add(f"pw:{x}:{y}:ya:{r['id']}", J.pairwise_messages(rubric, r["input"], text_of(y, r),
                                                                 text_of(x, r), _ref(cfg, r)))
    f = ctx.stage_dir / "judge.jsonl"
    write_jsonl(f, reqs)
    runtime.ENV.state = str(ctx.stage_dir / "state")
    runtime.ENV.in_build = True
    res = _backend().eval(desc.params["judge"], f)
    if res.get("interrupted"):
        return StageOutcome("interrupted", {}, res.get("producers", []))
    results = {r["custom_id"]: r for r in read_jsonl(res["results_path"])}

    def reply(cid):
        row = results.get(cid)
        if row and not row.get("error") and row.get("response"):
            return row["response"]["body"]["choices"][0]["message"]["content"]
        return None
    records, point_metrics, cal_items = [], {}, []
    for c in judged:
        scores = []
        for r in val:
            raw = reply(f"pt:{c}:{r['id']}")
            sc, why = J.parse_pointwise(raw)
            scores.append(sc)
            records.append({"kind": "pointwise", "comparator": c, "id": r["id"], "score": sc,
                            "rationale": why, "judge_reply": raw, "output": text_of(c, r),
                            "reference": _ref(cfg, r)})
            cal_items.append({"reference": _ref(cfg, r), "output": text_of(c, r), "score": sc})
        point_metrics[c] = J.pointwise_summary(scores)
    pair_metrics = []
    for x, y in pairs:
        rows = []
        for r in val:
            ra, rb = reply(f"pw:{x}:{y}:xa:{r['id']}"), reply(f"pw:{x}:{y}:ya:{r['id']}")
            (v1, w1), (v2, w2) = J.parse_pairwise(ra), J.parse_pairwise(rb)
            m1 = J.to_models(v1, "x", "y")             # x was shown as A
            m2 = J.to_models(v2, "y", "x")             # y was shown as A
            verdict, dis = J.combine_orders(m1, m2)
            rows.append({"verdict": verdict, "disagreed": dis, "picks": [v1, v2]})
            records.append({"kind": "pairwise", "pair": [x, y], "id": r["id"], "verdict": verdict,
                            "disagreed": dis,
                            "orders": [{"a": x, "b": y, "picked": v1, "rationale": w1, "judge_reply": ra},
                                       {"a": y, "b": x, "picked": v2, "rationale": w2, "judge_reply": rb}]})
        pair_metrics.append({"pair": [x, y], **J.pairwise_summary(rows)})
    metrics = {"judge": desc.params["judge"], "label": "one model's assessment, not accuracy",
               "rows": len(val), "pointwise": point_metrics, "pairwise": pair_metrics,
               "excluded": excluded, "calibration": J.calibration(cal_items),
               "prompt_cached_run": bool(res.get("cached"))}
    write_jsonl(ctx.out_dir / "judgments.jsonl", records)
    write_json(ctx.out_dir / "metrics.json", metrics)
    from ..cli_build import _row_producers
    return StageOutcome("done", metrics, _row_producers(res["results_path"]),
                        notes={"engine_run_id": res.get("run_id"), "model": desc.params["judge"],
                               "reused": bool(res.get("cached")),
                               "conditions": executed_conditions(list(results.values()),
                                                                 choose_engine_name())})


def choose_engine_name() -> str:
    from .. import runtime
    from ..engine_select import choose_engine
    return choose_engine(runtime.ENV.engine).name


def _ref(cfg: dict, row: dict) -> str:
    return K.target_text(cfg, row["output"])
