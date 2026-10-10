"""The task contract: what a correct answer is, frozen before anything is evaluated.

Classification: a fixed label vocabulary and the label normalization. JSON extraction: a
JSON Schema and the comparison rules. Prompts for the untrained student and the teacher
carry the contract; the trained student is trained on the data and sees only the input
(plus the user's own system text). Scoring is pure: (output text, ground truth) -> flags."""

from __future__ import annotations

import json
import re
import unicodedata

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError

from ..errors import SpillError
from .common import norm_label, sha_obj

_FENCE = re.compile(r"^```(?:json)?\s*\n?(.*?)\n?```$", re.DOTALL)


# ------------------------------------------------------------ building the contract

def derive_labels(train_outputs: list[str]) -> list[str]:
    """The vocabulary from training outputs only, one spelling per normalized label (the most
    common spelling wins), sorted."""
    spell: dict = {}
    for o in train_outputs:
        spell.setdefault(norm_label(o), {}).setdefault(o, 0)
        spell[norm_label(o)][o] += 1
    return sorted(max(c.items(), key=lambda kv: (kv[1], kv[0]))[0] for c in spell.values())


def _type_of(v) -> str:
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "boolean"
    if isinstance(v, int):
        return "integer"
    if isinstance(v, float):
        return "number"
    if isinstance(v, str):
        return "string"
    if isinstance(v, list):
        return "array"
    return "object"


def derive_schema(train_outputs: list[dict]) -> dict:
    """A JSON Schema from training records only: properties are the keys seen, each typed by
    the values seen (integer widens to number, a null adds "null"), required are the keys
    present in every training record, additionalProperties is false. Arrays of one item type
    get that type; nested objects are derived the same way."""
    return _derive(train_outputs)


def _derive(values: list) -> dict:
    types = sorted({_type_of(v) for v in values})
    if "integer" in types and "number" in types:
        types.remove("integer")
    schema: dict = {"type": types[0] if len(types) == 1 else types}
    objs = [v for v in values if isinstance(v, dict)]
    if objs:
        keys = sorted({k for o in objs for k in o})
        schema["properties"] = {k: _derive([o[k] for o in objs if k in o]) for k in keys}
        schema["required"] = [k for k in keys if all(k in o for o in objs)]
        schema["additionalProperties"] = False
    arrs = [v for v in values if isinstance(v, list)]
    if arrs:
        items = [x for a in arrs for x in a]
        if items:
            schema["items"] = _derive(items)
    return schema


def check_schema(schema: dict) -> None:
    try:
        Draft202012Validator.check_schema(schema)
    except SchemaError as e:
        raise SpillError(f"the JSON schema is not valid: {e.message}",
                         "fix schema.json (JSON Schema draft 2020-12)")
    if schema.get("type") != "object" and "properties" not in schema:
        raise SpillError("the schema must describe an object with properties",
                         'start it with {"type": "object", "properties": {...}}')


def schema_sha(schema: dict) -> str:
    return sha_obj(schema)


def invalid_ground_truth(schema: dict, rows) -> list[tuple[object, str]]:
    """Rows whose ground truth does not satisfy the schema: (row, first error message)."""
    v = Draft202012Validator(schema)
    out = []
    for r in rows:
        errs = sorted(v.iter_errors(r.output), key=lambda e: list(e.path))
        if errs:
            e = errs[0]
            loc = "/".join(str(p) for p in e.path) or "(record)"
            out.append((r, f"{loc}: {e.message}"))
    return out


def labels_outside(labels: list[str], rows) -> list:
    known = {norm_label(x) for x in labels}
    return [r for r in rows if norm_label(str(r.output)) not in known]


# ------------------------------------------------------------ prompts

def contract_text(cfg: dict, schema: dict | None) -> str:
    c = cfg["contract"]
    if cfg["task"]["type"] == "classification":
        return ("Classify the input into exactly one of these labels. Reply with the label "
                "only, spelled exactly as listed.\nLabels: " + ", ".join(c["labels"]))
    return ("Extract the fields from the input as one JSON object that satisfies this JSON "
            "Schema. Reply with the JSON object only.\nSchema: "
            + json.dumps(schema, ensure_ascii=False, separators=(",", ":")))


def messages_untrained(cfg: dict, schema: dict | None, text: str) -> list[dict]:
    """The prompted student's (and the teacher's) view: user instructions plus the contract."""
    sysm = "\n\n".join(x for x in (cfg["task"].get("system"), contract_text(cfg, schema)) if x)
    return [{"role": "system", "content": sysm}, {"role": "user", "content": text}]


def messages_student(cfg: dict, text: str) -> list[dict]:
    """The trained student's view: the user's own system text if any, then the input."""
    out = []
    if cfg["task"].get("system"):
        out.append({"role": "system", "content": cfg["task"]["system"]})
    out.append({"role": "user", "content": text})
    return out


def target_text(cfg: dict, output) -> str:
    if cfg["task"]["type"] == "classification":
        return str(output)
    return json.dumps(output, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


# ------------------------------------------------------------ scoring: classification

def score_class(text: str | None, gold: str, labels: list[str]) -> dict:
    if text is None:
        return {"failed": True, "valid": False, "correct": False, "pred": None}
    n = norm_label(text)
    known = {norm_label(x): x for x in labels}
    valid = n in known
    correct = valid and n == norm_label(gold)
    return {"failed": False, "valid": valid, "correct": correct,
            "pred": known.get(n, text.strip())}


def agg_class(items: list[dict], golds: list[str], labels: list[str]) -> dict:
    """Accuracy and macro-F1 over all rows. An invalid or failed prediction is wrong and is
    never a prediction for any class (no false positive), but is a miss for its gold class."""
    n = len(items)
    correct = sum(1 for i in items if i["correct"])
    tp: dict = {}
    fp: dict = {}
    fn: dict = {}
    for it, g in zip(items, golds):
        gl = norm_label(g)
        if it["correct"]:
            tp[gl] = tp.get(gl, 0) + 1
        else:
            fn[gl] = fn.get(gl, 0) + 1
            if it["valid"]:
                pl = norm_label(it["pred"])
                fp[pl] = fp.get(pl, 0) + 1
    f1s = []
    for c in sorted(set(norm_label(g) for g in golds)):
        t, p, q = tp.get(c, 0), fp.get(c, 0), fn.get(c, 0)
        f1s.append(0.0 if t == 0 else 2 * t / (2 * t + p + q))
    return {"rows": n, "accuracy": correct / n if n else None,
            "macro_f1": sum(f1s) / len(f1s) if f1s else None,
            "invalid_predictions": sum(1 for i in items if not i["valid"] and not i["failed"]),
            "inference_failures": sum(1 for i in items if i["failed"])}


# ------------------------------------------------------------ scoring: JSON

def parse_strict(text: str):
    """The recorded parse rule: stripped text, one optional ```json fence, one JSON value."""
    t = text.strip()
    m = _FENCE.match(t)
    if m:
        t = m.group(1).strip()
    return json.loads(t)


def _eq(a, b, schema: dict | None) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return a == b
    if isinstance(a, str) and isinstance(b, str):
        return unicodedata.normalize("NFC", a).strip() == unicodedata.normalize("NFC", b).strip()
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            return False
        props = (schema or {}).get("properties", {})
        return all(_eq(a[k], b[k], props.get(k)) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return False
        item_schema = (schema or {}).get("items")
        if (schema or {}).get("x-unordered"):
            rest = list(b)
            for x in a:
                for j, y in enumerate(rest):
                    if _eq(x, y, item_schema):
                        del rest[j]
                        break
                else:
                    return False
            return True
        return all(_eq(x, y, item_schema) for x, y in zip(a, b))
    return False


def schema_fields(schema: dict, golds: list[dict]) -> list[str]:
    keys = list(schema.get("properties", {}))
    for g in golds:
        for k in g:
            if k not in keys:
                keys.append(k)
    return keys


def score_json(text: str | None, gold: dict, schema: dict, validator=None) -> dict:
    """Metric version 2. Order of evidence: the output must parse, then it must satisfy the
    schema, and only then are its fields compared. A parseable but schema-invalid output counts
    toward the parseable rate and nothing else: no field-level and no whole-record credit, and
    field normalization (trimming, number equality) never rescues it. (Version 1 compared fields
    first and ignored validity for correctness.)"""
    base = {"failed": False, "parseable": False, "schema_valid": False, "record": False,
            "fields": {}, "extra": [], "pred": None}
    if text is None:
        return {**base, "failed": True}
    try:
        pred = parse_strict(text)
    except (json.JSONDecodeError, ValueError):
        return base
    validator = validator or Draft202012Validator(schema)
    valid = validator.is_valid(pred)
    out = {**base, "parseable": True, "schema_valid": valid, "pred": pred}
    names = schema_fields(schema, [gold])
    if not valid or not isinstance(pred, dict):
        out["fields"] = {k: False for k in names}          # no credit of any kind
        out["extra"] = sorted(k for k in pred if k not in gold) if isinstance(pred, dict) else []
        return out
    props = schema.get("properties", {})
    fields = {}
    for k in names:
        if k in gold and k in pred:
            fields[k] = _eq(pred[k], gold[k], props.get(k))
        else:
            fields[k] = k not in gold and k not in pred
    extra = sorted(k for k in pred if k not in gold)
    out.update(fields=fields, extra=extra,
               record=not extra and all(fields.values()) and _eq(pred, gold, schema))
    return out


def agg_json(items: list[dict], schema: dict, golds: list[dict]) -> dict:
    n = len(items)
    names = schema_fields(schema, golds)
    per = {}
    for k in names:
        per[k] = (sum(1 for it in items if it["fields"].get(k)) / n) if n else None
    return {"rows": n,
            "parseable_rate": sum(1 for i in items if i["parseable"]) / n if n else None,
            "schema_valid_rate": sum(1 for i in items if i["schema_valid"]) / n if n else None,
            "whole_record_accuracy": sum(1 for i in items if i["record"]) / n if n else None,
            "field_accuracy": per,
            "mean_field_accuracy": (sum(per.values()) / len(per)) if per else None,
            "records_with_extra_fields": sum(1 for i in items if i["extra"]),
            "inference_failures": sum(1 for i in items if i["failed"])}
