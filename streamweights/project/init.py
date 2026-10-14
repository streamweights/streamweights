"""spill init: from a CSV or JSONL of labeled examples to a project folder.

Validates every row, settles the task, splits (or keeps the supplied splits), freezes the
task contract from the training rows only, and writes streamweights.toml plus the canonical
data files. Nothing is trained and no model is loaded."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from ..errors import SpillError
from . import config as C
from . import contract as K
from . import data as D
from . import scoring as SC
from .common import (atomic_write, norm_dup, now, sha_bytes, sha_file, write_json,
                     write_jsonl)


@dataclass
class InitResult:
    folder: Path
    task: str                 # the scoring method: exact | json | judge | script
    task_note: str
    sizes: dict
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    cfg: dict = field(default_factory=dict)


def _read_labels_arg(spec: str | None) -> list[str] | None:
    if not spec:
        return None
    p = Path(spec)
    if p.exists():
        labels = [l.strip() for l in p.read_text().splitlines() if l.strip()]
    else:
        labels = [l.strip() for l in spec.split(",") if l.strip()]
    if len(set(labels)) != len(labels):
        raise SpillError("the declared label list has duplicates", "remove them from --labels")
    if len(labels) < 2:
        raise SpillError("--labels needs at least two labels", "--labels a,b,c")
    return labels


def _load(path: Path, mapping: dict, task: str | None):
    cols, recs, issues = D.read_table(path)
    if issues:
        raise D.DataErrors(issues)
    return cols, recs


def _scoring_arg(score: str | None, task: str | None) -> tuple[str | None, Path | None]:
    """(method, script path) from --score, or from the older --task spelling."""
    if score:
        if score.startswith("script:"):
            return "script", Path(score[len("script:"):])
        if score not in SC.CHOICES:
            raise SpillError(f"--score must be exact, json, judge or script:<file>, not {score!r}",
                             "spill init ... --score judge")
        return score, None
    if task:
        legacy = {"classification": "exact", "json": "json"}
        if task not in legacy:
            raise SpillError(f"--task {task!r} is not a task type; the scoring method is chosen "
                             f"from your outputs", "spill init ... --score exact|json|judge")
        return legacy[task], None
    return None, None


def _student_why() -> str:
    """Why the default student is the default, from the probed machine (nothing is probed here
    if the machine has not been probed yet)."""
    from .. import build_defaults, engine_select
    from .. import probe as probe_mod
    try:
        eng = engine_select.choose_engine(None).name
        hw = probe_mod.load(probe_if_missing=False)
        ws = build_defaults.machine_inputs(eng, hw, {})[0]
        return build_defaults.guided_student(eng, ws)[1]
    except Exception:
        return (f"{build_defaults.SMALL}: the smallest curated model, so a first result takes minutes "
                f"on any machine")


def _read_unlabeled(path: Path, mapping: dict, splits) -> tuple[list[dict], list[str]]:
    """Unlabeled prompts: inputs only, for a teacher to answer. Leakage rule: a prompt that
    matches any training, validation or test input (after the duplicate normalization) is
    removed, so no evaluation input can reach the teacher's training answers."""
    cols, recs, issues = D.read_table(path)
    if issues:
        raise D.DataErrors(issues)
    col = mapping["input"] if mapping["input"] in cols else "prompt" if "prompt" in cols else None
    if col is None:
        raise SpillError(f"{path.name} has no column {mapping['input']!r} or 'prompt' for the unlabeled "
                         f"prompts", f"name the input column {mapping['input']!r} in {path.name}")
    known = {norm_dup(r.input) for r in splits.train + splits.val + splits.test}
    rows, seen, leaked, dup, empty = [], set(), 0, 0, 0
    for r in recs:
        text = D._cell_text(r.values.get(col)).strip()
        if not text:
            empty += 1
            continue
        key = norm_dup(text)
        if key in known:
            leaked += 1
        elif key in seen:
            dup += 1
        else:
            seen.add(key)
            rows.append({"id": "u-" + sha_bytes(text.encode())[:12], "input": text,
                         "source": {"file": r.file, "row": r.row}})
    notes = []
    if leaked:
        notes.append(f"{leaked} unlabeled prompt(s) matched a training, validation or test input and "
                     f"were removed (an evaluation input must never reach the teacher)")
    if dup or empty:
        notes.append(f"{dup} duplicate and {empty} empty unlabeled prompt(s) were removed")
    if not rows:
        raise SpillError(f"{path.name} leaves no unlabeled prompts after the leakage check",
                         "give prompts that are not in your examples")
    return rows, notes


def create_project(data: Path, folder: Path, mapping: dict, task: str | None = None,
                   system_file: Path | None = None, labels: str | None = None,
                   schema_file: Path | None = None, val: Path | None = None,
                   test: Path | None = None, seed: int = 0, val_fraction: float = 0.15,
                   test_fraction: float = 0.15, force: bool = False,
                   say=lambda s: None, score: str | None = None,
                   student: str | None = None, teacher: str | None = None,
                   judge: str | None = None, unlabeled: Path | None = None) -> InitResult:
    folder = Path(folder)
    if C.exists(folder) and not force:
        raise SpillError(f"{folder} already has a {C.CONFIG_NAME}", f"spill plan {folder}   "
                         f"(a different folder: --project <folder>)")
    cols, recs = _load(Path(data), mapping, task)
    _, missing = D.build_rows(cols, [], mapping, None, Path(data).name)
    if missing:
        raise D.DataErrors(missing)
    sug = D.suggest_task(recs, mapping) if all(
        mapping[k] in cols for k in ("input", "output")) else D.Suggestion(None, "")
    scoring, script_path = _scoring_arg(score, task)
    if scoring in (None, "script"):
        if sug.task is None:
            raise SpillError(f"cannot read the outputs ({sug.reason})",
                             "spill init ... --score exact|json|judge")
        task = sug.task
        note = sug.reason
        scoring_name = scoring or SC.SCORING_OF_TASK[task]
        if scoring is None:
            say(f"scoring: {scoring_name} ({sug.reason})")
    else:
        task = SC.TASK_OF[scoring]
        scoring_name = scoring
        note = "chosen with --" + ("task" if not score else "score")
        if sug.task and sug.task != task:
            say(f"note: the outputs look like {SC.SCORING_OF_TASK[sug.task]} ({sug.reason}); "
                f"using {scoring} as asked")
    script = None
    if script_path is not None:
        SC.load_script(script_path)                 # fails here, not mid-build
        script = {"file": "score.py", "sha256": sha_file(script_path)}
        note = f"{note}; scored by {script_path.name} (override)"
    if task not in C.TASKS:
        raise SpillError(f"unknown task {task!r}", "spill init ... --score exact|json|judge")
    train_rows, issues = D.build_rows(cols, recs, mapping, task, Path(data).name)
    val_rows = test_rows = None
    for role, path in (("val", val), ("test", test)):
        if path is None:
            continue
        c2, r2 = _load(Path(path), mapping, task)
        rows2, i2 = D.build_rows(c2, r2, mapping, task, Path(path).name)
        issues += i2
        if role == "val":
            val_rows = rows2
        else:
            test_rows = rows2
    if issues:
        raise D.DataErrors(issues)

    splits = D.split_rows(train_rows, val_rows, test_rows, seed, val_fraction, test_fraction)
    system = Path(system_file).read_text().strip() if system_file else None
    cfg = C.default_config(folder.name, task, mapping, system, script)
    for role, tag in (("student", student), ("teacher", teacher), ("judge", judge)):
        if tag:
            cfg["model"][role] = tag
    cfg["model"]["student_why"] = ("chosen with --student" if student else _student_why())
    cfg["split"].update({"seed": seed, "val_fraction": val_fraction,
                         "test_fraction": test_fraction, "mode": splits.mode})
    warnings: list[str] = []

    declared = _read_labels_arg(labels)
    sizes = {"train": len(splits.train), "val": len(splits.val), "test": len(splits.test)}
    out_files: dict = {}
    if task == "classification":
        distinct = len({norm_dup(str(r.output)) for r in splits.train + splits.val + splits.test})
        if distinct > D.MAX_LABELS:
            raise SpillError(f"exact scoring needs a small label set (at most {D.MAX_LABELS} "
                             f"distinct outputs); these rows have {distinct}",
                             "spill init ... --score judge")
        if declared:
            outside = []
            for name, rows in (("train", splits.train), ("val", splits.val),
                               ("test", splits.test)):
                outside += [(name, r) for r in K.labels_outside(declared, rows)]
            if outside:
                raise D.DataErrors([D.Issue(r.source["file"], f"row {r.source['row']}",
                                            f"label {r.output!r} is not in the declared "
                                            f"vocabulary", "add it to --labels or fix the "
                                            "label") for _, r in outside])
            vocab, src = declared, "declared"
        else:
            vocab, src = K.derive_labels([r.output for r in splits.train]), "derived from the training rows only"
        cfg["contract"] = {"kind": "classification", "labels": vocab, "label_source": src,
                           "label_normalization": C.LABEL_NORMALIZATION}
        cov = D.class_coverage(splits)
        if cov["missing_in_train"]:
            warnings.append(f"{len(cov['missing_in_train'])} validation/test class(es) never "
                            f"appear in training and cannot be learned: "
                            f"{', '.join(cov['missing_in_train'][:5])}")
        if cov["missing_in_val"]:
            warnings.append(f"{len(cov['missing_in_val'])} of {cov['train']} training classes "
                            f"have no validation rows, so their accuracy is not measured")
        integrity = {"coverage": cov}
    elif task == "open":
        cfg["contract"] = {"kind": "open", "reference": "the outputs you provided",
                           "note": "reference-overlap metrics compare an output with your "
                                   "reference; they are not accuracy. Judged scores need a judge."}
        integrity = {}
    else:
        if schema_file:
            try:
                schema = json.loads(Path(schema_file).read_text())
            except (json.JSONDecodeError, OSError) as e:
                raise SpillError(f"cannot read the schema {schema_file}: {e}", "fix the file")
            K.check_schema(schema)
            src = f"declared in {Path(schema_file).name}"
            bad = []
            for name, rows in (("train", splits.train), ("val", splits.val), ("test", splits.test)):
                bad += [(name, r, m) for r, m in K.invalid_ground_truth(schema, rows)]
            if bad:
                raise D.DataErrors([D.Issue(r.source["file"], f"row {r.source['row']}",
                                            f"ground truth violates the schema ({m})",
                                            "fix the row or the schema") for _, r, m in bad])
        else:
            schema = K.derive_schema([r.output for r in splits.train])
            K.check_schema(schema)
            src = "derived from the training rows only"
            for name, rows in (("val", splits.val), ("test", splits.test)):
                bad = K.invalid_ground_truth(schema, rows)
                if bad:
                    warnings.append(f"{len(bad)} {name} row(s) do not satisfy the schema derived "
                                    f"from training (first: {bad[0][0].source['file']} row "
                                    f"{bad[0][0].source['row']}: {bad[0][1]}); they stay in "
                                    f"{name} and count as written")
        sbytes = (json.dumps(schema, indent=2, sort_keys=True) + "\n").encode()
        out_files["schema.json"] = sbytes
        cfg["contract"] = {"kind": "json", "schema_file": "schema.json",
                           "schema_sha256": sha_bytes(sbytes), "schema_source": src,
                           "rules": C.JSON_RULES}
        integrity = {}
    conflicts = D.conflicting_labels(splits.train + splits.val + splits.test)
    if conflicts:
        warnings.append(f"{len(conflicts)} input(s) appear with conflicting labels (first: "
                        f"{conflicts[0]['rows'][0]} vs {conflicts[0]['rows'][-1]}); see "
                        f"data/integrity.json")
    integrity["conflicting_labels"] = conflicts
    integrity["sizes"] = sizes
    if not sizes["test"]:
        warnings.append("there is no final-test split; `spill test` will not be available")
    for why in D.check_usable(splits, task):
        warnings.append(f"{why}; a build will be refused until this is fixed")

    unl_rows = []
    if unlabeled is not None:
        unl_rows, unl_notes = _read_unlabeled(Path(unlabeled), mapping, splits)
        warnings += unl_notes

    # ---- write the project
    folder.mkdir(parents=True, exist_ok=True)
    srcs = []
    (folder / "data" / "source").mkdir(parents=True, exist_ok=True)
    for role, path in (("input", data), ("val", val), ("test", test)):
        if path is None:
            continue
        dst = folder / "data" / "source" / f"{role}-{Path(path).name}"
        shutil.copyfile(path, dst)
        srcs.append({"role": role, "file": f"data/source/{dst.name}", "original_name":
                     Path(path).name, "sha256": sha_file(dst)})
    cfg["data"]["sources"] = srcs
    fps = {}
    for name, rows in (("train", splits.train), ("val", splits.val), ("test", splits.test)):
        write_jsonl(folder / "data" / f"{name}.jsonl", [r.to_dict() for r in rows])
        fps[name] = sha_file(folder / "data" / f"{name}.jsonl")
    cfg["split"]["fingerprints"] = fps
    cfg["split"]["sizes"] = sizes
    cfg["data"]["canonical"] = {n: f"data/{n}.jsonl" for n in ("train", "val", "test")}
    if unl_rows:
        write_jsonl(folder / "data" / "unlabeled.jsonl", unl_rows)
        cfg["data"]["unlabeled"] = {"file": "data/unlabeled.jsonl", "rows": len(unl_rows),
                                    "sha256": sha_file(folder / "data" / "unlabeled.jsonl"),
                                    "use": "inputs only; a teacher answers them, they are never "
                                           "evaluation rows"}
        sizes["unlabeled"] = len(unl_rows)
    if script:
        shutil.copyfile(script_path, folder / script["file"])
    if system:
        atomic_write(folder / "system.txt", (system + "\n").encode())
        cfg["task"]["system_file"] = "system.txt"
    for name, b in out_files.items():
        atomic_write(folder / name, b)
    write_json(folder / "data" / "integrity.json", integrity)
    cfg["project"]["created"] = now()
    C.save(folder, cfg)
    atomic_write(folder / ".gitignore", b".spill/\nREPORT.md.tmp\n")
    return InitResult(folder, scoring_name, note, sizes, warnings, splits.notes, cfg)
