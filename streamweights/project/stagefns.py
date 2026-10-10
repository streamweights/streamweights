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
    fn = {"baseline": _baseline, "distill": _distill, "train": _train, "eval": _eval}[desc.kind]
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
    emb = modelid.embedding_identity(fetch=True)
    res = B.run_baseline(cfg, rows, val, emb["dir"])
    write_jsonl(ctx.out_dir / "predictions.jsonl",
                [{**p, "comparator": "baseline"} for p in res["predictions"]])
    metrics = {**res["metrics"], "settings": res["settings"], "training": manifest}
    write_json(ctx.out_dir / "metrics.json", metrics)
    v = modelid.dependency_versions()
    return StageOutcome("done", metrics, [{"engine": "torch-cpu", "hardware": "cpu",
                                            "numerics": {"base": "float32"}}],
                        notes={"conditions": {"engine": "torch-cpu", "engine_impl": "embedding+logreg",
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
    labels = cfg["contract"].get("labels", [])
    from jsonschema import Draft202012Validator
    validator = Draft202012Validator(schema) if schema else None
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
        if cfg["task"]["type"] == "classification":
            sc = K.score_class(text, r["output"], labels)
            rec = {"pred": sc["pred"], "correct": sc["correct"], "valid": sc["valid"]}
        else:
            sc = K.score_json(text, r["output"], schema, validator)
            rec = {"pred": sc["pred"], "correct": sc["record"], "parseable": sc["parseable"],
                   "schema_valid": sc["schema_valid"], "fields": sc["fields"],
                   "extra_fields": sc["extra"]}
        items.append(sc)
        preds.append({"id": r["id"], "gold": r["output"], "text": text, "finish_reason": finish,
                      "completion_tokens": toks, "latency_s": lat, "failed": sc["failed"],
                      **rec, "comparator": comparator})
    golds = [r["output"] for r in val]
    metrics = (K.agg_class(items, golds, labels) if cfg["task"]["type"] == "classification"
               else K.agg_json(items, schema, golds))
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
