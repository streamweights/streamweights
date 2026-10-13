"""How a project scores an output: one place that maps the project's scoring method to the
per-row scorer and the aggregate.

  exact   the outputs are a small label set: exact match after normalization (task type
          "classification" in older configs)
  json    the outputs are JSON objects: parse, schema, field-level metrics
  judge   anything else: reference overlap against the provided output, labeled as such; a
          judge model adds its own assessment (judge.py)
  script  a user Python file exposing score(row) -> float, always available as an override

The scoring method is chosen from the outputs by `choose` and recorded in streamweights.toml
(evaluation.scoring). A config without it (a project made before scoring selection) maps its
task type: classification is exact, json is json."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from ..errors import SpillError
from . import contract as K

TASK_OF = {"exact": "classification", "json": "json", "judge": "open"}
SCORING_OF_TASK = {v: k for k, v in TASK_OF.items()}
CHOICES = ("exact", "json", "judge")
SCRIPT_METRIC = "script_score"


def scoring_of(cfg: dict) -> str:
    """exact | json | judge | script."""
    s = (cfg.get("evaluation") or {}).get("scoring")
    return s or SCORING_OF_TASK[cfg["task"]["type"]]


def output_kind(cfg: dict) -> str:
    """exact | json | judge: what the outputs are, whatever scores them (a script scores
    outputs of any kind)."""
    return SCORING_OF_TASK[cfg["task"]["type"]]


def describe(cfg: dict) -> str:
    s = scoring_of(cfg)
    kind = output_kind(cfg)
    ev = cfg["evaluation"]
    if s == "script":
        return (f"script {ev.get('script_file')} (sha256 {str(ev.get('script_sha256'))[:12]}); the "
                f"outputs are {kind}")
    if s == "judge":
        return "judge: reference overlap against your outputs" + (
            f", plus judged scores from {cfg['model']['judge']} (one model's assessment)"
            if cfg["model"].get("judge") else "; judged scores need a judge you name or accept")
    return s


def load_script(path: str | Path):
    p = Path(path)
    if not p.exists():
        raise SpillError(f"the scoring script {p} does not exist", "spill init ... --score script:<file.py>")
    spec = importlib.util.spec_from_file_location("spill_user_score", p)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as e:                                  # the user's own code
        raise SpillError(f"the scoring script {p.name} failed to load: {type(e).__name__}: {e}",
                         "fix the script")
    fn = getattr(mod, "score", None)
    if not callable(fn):
        raise SpillError(f"the scoring script {p.name} defines no score(row) function",
                         "define `def score(row) -> float` in it")
    return fn


def script_for(cfg: dict, inputs_dir: Path):
    if scoring_of(cfg) != "script":
        return None
    return load_script(Path(inputs_dir) / cfg["evaluation"]["script_file"])


def score_row(cfg: dict, schema, validator, text, row: dict, script=None) -> tuple[dict, dict]:
    """-> (item for aggregation, extra fields for the prediction record). `row` is the
    evaluation row {id, input, output}."""
    gold = row["output"]
    kind = output_kind(cfg)
    if kind == "exact":
        sc = K.score_class(text, gold, cfg["contract"].get("labels", []))
        rec = {"pred": sc["pred"], "correct": sc["correct"], "valid": sc["valid"]}
    elif kind == "json":
        sc = K.score_json(text, gold, schema, validator)
        rec = {"pred": sc["pred"], "correct": sc["record"], "parseable": sc["parseable"],
               "schema_valid": sc["schema_valid"], "fields": sc["fields"],
               "extra_fields": sc["extra"]}
    else:
        sc = K.score_open(text, K.target_text(cfg, gold))
        rec = {"pred": sc["pred"], "token_f1": sc["token_f1"], "rouge_l": sc["rouge_l"],
               "exact_match": sc["exact_match"], "correct": sc["exact_match"]}
    if script is not None:
        try:
            v = float(script({"id": row["id"], "input": row["input"], "gold": gold,
                              "text": text, "pred": sc.get("pred")}))
        except Exception as e:
            raise SpillError(f"the scoring script raised on row {row['id']}: "
                             f"{type(e).__name__}: {e}", "fix the script")
        sc["script_score"] = v
        rec["script_score"] = v
    return sc, rec


def aggregate(cfg: dict, items: list[dict], golds: list, schema) -> dict:
    kind = output_kind(cfg)
    if kind == "exact":
        m = K.agg_class(items, golds, cfg["contract"].get("labels", []))
    elif kind == "json":
        m = K.agg_json(items, schema, golds)
    else:
        m = K.agg_open(items, [K.target_text(cfg, g) for g in golds])
    if items and "script_score" in items[0]:
        m[SCRIPT_METRIC] = sum(i["script_score"] for i in items) / len(items)
    return m


def correct_flag(rec: dict) -> bool:
    return bool(rec.get("correct"))
