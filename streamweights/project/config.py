"""streamweights.toml: the versioned project config.

One file records the task and its column mapping, the system instructions, the class
vocabulary or JSON schema (a pointer plus a hash; the schema itself is schema.json), the
model and training settings, the evaluation protocol and the split definition. Editing a
setting that changes what a build computes (data, prompts, schema, training settings,
models) makes the next build a new run (see runid.py)."""

from __future__ import annotations

import copy
from pathlib import Path

from ..errors import SpillError
from .common import atomic_write, sha_obj

try:                                   # Python 3.11 and newer
    import tomllib
except ModuleNotFoundError:            # 3.10
    import tomli as tomllib

import tomli_w

SCHEMA_VERSION = 1
CONFIG_NAME = "streamweights.toml"
LEGACY_FILES = ("evals.jsonl", "train.jsonl", "prompts.jsonl")
TASKS = ("classification", "json")

METRIC_VERSIONS = {"classification": "1", "json": "2"}   # json 2: invalid outputs get no credit
METRIC_VERSION = "1"                                      # kept for callers that mean classification
PROTOCOL_VERSION = "2"                                    # 2: decoding includes stop and greedy
SUPPORTED_METRICS = {
    "classification": ("accuracy", "macro_f1"),
    "json": ("whole_record_accuracy", "mean_field_accuracy", "schema_valid_rate", "parseable_rate"),
}
METRIC_LABELS = {"accuracy": "accuracy", "macro_f1": "macro-F1",
                 "whole_record_accuracy": "whole-record accuracy",
                 "mean_field_accuracy": "mean field accuracy",
                 "schema_valid_rate": "schema-valid rate", "parseable_rate": "parseable rate"}
DEFAULT_DECODING = {"temperature": 0.0, "top_p": 1.0, "stop": [], "greedy": True, "seed": 0}


def metric_version(cfg: dict) -> str:
    """The version of the scorer in this code, whatever an older config file says."""
    return METRIC_VERSIONS[cfg["task"]["type"]]


def check_metric(task: str, metric: str) -> str:
    ok = SUPPORTED_METRICS.get(task)
    if ok is None or metric not in ok:
        raise SpillError(f"metric {metric!r} is not supported for the {task} task (choose from "
                         f"{', '.join(ok or ())})", f"set evaluation.metric in {CONFIG_NAME}")
    return metric
DUP_NORMALIZATION = ("Unicode NFKC, casefold, whitespace runs collapsed to one space, ends "
                     "stripped; applied to inputs for duplicate detection and to group values")
LABEL_NORMALIZATION = ("Unicode NFKC, casefold, whitespace runs collapsed to one space, ends "
                       "stripped; applied to ground-truth labels and to model outputs; an "
                       "output that is not in the vocabulary after normalization is a failure")
JSON_RULES = {
    "parse": "the whole response, stripped, must be one JSON value; a single surrounding "
             "```json fence is removed first; anything else is unparseable",
    "schema": "JSON Schema (draft 2020-12) validation of the parsed value",
    "field_equal": "strings: Unicode NFC, ends stripped, then exact; numbers: numerically "
                   "equal (1 equals 1.0; booleans are not numbers); null equals only null; "
                   "objects: equal key sets and equal values recursively; arrays: equal "
                   "length and equal items in order, unless the schema marks the array "
                   "x-unordered: true, then equal as multisets",
    "missing_field": "a field the ground truth has and the output lacks is incorrect; a field "
                     "absent from both is correct",
    "extra_field": "a top-level field the output has and the ground truth lacks makes the "
                   "record incorrect and is counted separately; it does not count in "
                   "per-field accuracy",
    "denominators": "every evaluation row; unparseable, schema-invalid, failed or truncated "
                    "outputs count as incorrect",
}

DEFAULT_TRAINING = {"epochs": 2.0, "lr": 1e-4, "rank": 16, "alpha": 32.0, "dropout": 0.0,
                    "weight_decay": 0.01, "schedule": "cosine", "seed": 0, "max_seq": 2048,
                    "micro_batch": 4, "grad_accum": 1, "ckpt_every": 10}
DEFAULT_SPLIT = {"seed": 0, "val_fraction": 0.15, "test_fraction": 0.15}
DEFAULT_EMBEDDING = {"repo": "sentence-transformers/all-MiniLM-L6-v2", "pooling": "mean",
                     "normalize": True, "max_length": 256}


def default_config(name: str, task: str, mapping: dict, system: str | None) -> dict:
    cfg = {
        "schema_version": SCHEMA_VERSION,
        "project": {"name": name, "layout": "project"},
        "task": {"type": task, "input": mapping["input"], "output": mapping["output"]},
        "contract": {},
        "data": {"sources": []},
        "split": {**DEFAULT_SPLIT, "mode": "generated",
                  "duplicate_normalization": DUP_NORMALIZATION,
                  "group_normalization": "same as duplicate normalization"},
        "model": {"student": "qwen2.5:0.5b", "teacher": "", "engine": "auto"},
        "training": dict(DEFAULT_TRAINING),
        "evaluation": {
            "protocol_version": PROTOCOL_VERSION, "metric_version": METRIC_VERSIONS[task],
            "metric": "accuracy" if task == "classification" else "whole_record_accuracy",
            "decoding": dict(DEFAULT_DECODING),
            "max_tokens": 16 if task == "classification" else 256,
            "precision": "bf16 base weights, float32 adapter",
            "postprocessing": "none beyond the contract rules",
        },
        "baseline": {"kind": "embedding+logreg", **DEFAULT_EMBEDDING, "c": 10.0,
                     "max_iter": 200} if task == "classification" else {},
    }
    if mapping.get("group"):
        cfg["task"]["group"] = mapping["group"]
    if system:
        cfg["task"]["system"] = system
    return cfg


def _strip_none(x):
    if isinstance(x, dict):
        return {k: _strip_none(v) for k, v in x.items() if v is not None}
    if isinstance(x, list):
        return [_strip_none(v) for v in x if v is not None]
    return x


def dumps(cfg: dict) -> bytes:
    return tomli_w.dumps(_strip_none(cfg)).encode()


def path_of(folder: str | Path) -> Path:
    return Path(folder) / CONFIG_NAME


def exists(folder: str | Path) -> bool:
    return path_of(folder).exists()


def parse(data: bytes, where: str = CONFIG_NAME) -> dict:
    try:
        cfg = tomllib.loads(data.decode())
    except (tomllib.TOMLDecodeError, UnicodeDecodeError) as e:
        raise SpillError(f"{where} is not valid TOML ({e})", "fix the file, or `spill init` "
                         "a new project")
    v = cfg.get("schema_version")
    if not isinstance(v, int):
        raise SpillError(f"{where} has no integer schema_version",
                         f"add `schema_version = {SCHEMA_VERSION}` at the top of the file")
    if v > SCHEMA_VERSION:
        raise SpillError(
            f"{where} uses schema_version {v}; this streamweights reads up to {SCHEMA_VERSION}. "
            f"Refusing to guess at fields it does not know",
            "pip install --upgrade git+https://github.com/streamweights/streamweights")
    if v < 1:
        raise SpillError(f"{where} has schema_version {v}, which never existed",
                         f"set `schema_version = {SCHEMA_VERSION}`")
    return cfg


def load(folder: str | Path) -> dict:
    p = path_of(folder)
    if not p.exists():
        raise SpillError(f"{folder} has no {CONFIG_NAME}", "spill init <data.csv> --input <col> "
                         "--output <col>")
    return parse(p.read_bytes(), f"{Path(folder).name}/{CONFIG_NAME}")


def save(folder: str | Path, cfg: dict) -> None:
    atomic_write(path_of(folder), dumps(cfg))


def get(cfg: dict, dotted: str, default=None):
    cur = cfg
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def resolved(cfg: dict, extra: dict) -> dict:
    """The config plus everything resolved at build time (model revisions, engine, versions)."""
    r = copy.deepcopy(cfg)
    r["resolved"] = extra
    return r


def identity_fields(cfg: dict, split_sha: dict, model_ident: dict) -> dict:
    """What a run's identity covers: the data (split fingerprints), the prompts (system text
    and the contract), the training settings, the evaluation protocol and the model identity.
    Not covered: the engine and the machine (a documented engine transition continues a run)."""
    return {
        "task": cfg["task"], "contract": cfg["contract"],
        "split": {"fingerprints": split_sha, "seed": cfg["split"]["seed"]},
        "training": cfg["training"],
        "evaluation": {**cfg["evaluation"], "metric_version": metric_version(cfg),
                       "protocol_version": PROTOCOL_VERSION},
        "baseline": cfg.get("baseline", {}),
        "model": {"student": model_ident.get("student"), "teacher": model_ident.get("teacher"),
                  "embedding": model_ident.get("embedding")},
    }


def identity_sha(cfg: dict, split_sha: dict, model_ident: dict) -> str:
    return sha_obj(identity_fields(cfg, split_sha, model_ident))
