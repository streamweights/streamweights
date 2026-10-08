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
from .common import atomic_write, now, sha_bytes, sha_file, write_json, write_jsonl


@dataclass
class InitResult:
    folder: Path
    task: str
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


def create_project(data: Path, folder: Path, mapping: dict, task: str | None = None,
                   system_file: Path | None = None, labels: str | None = None,
                   schema_file: Path | None = None, val: Path | None = None,
                   test: Path | None = None, seed: int = 0, val_fraction: float = 0.15,
                   test_fraction: float = 0.15, force: bool = False,
                   say=lambda s: None) -> InitResult:
    folder = Path(folder)
    if C.exists(folder) and not force:
        raise SpillError(f"{folder} already has a {C.CONFIG_NAME}", f"spill plan {folder}   "
                         f"(a different folder: --project <folder>)")
    cols, recs = _load(Path(data), mapping, task)
    sug = D.suggest_task(recs, mapping) if all(
        mapping[k] in cols for k in ("input", "output")) else D.Suggestion(None, "")
    if task is None:
        if sug.task is None:
            raise SpillError(f"cannot tell the task from the outputs ({sug.reason}); "
                             f"say which it is", "spill init ... --task classification|json")
        say(f"suggested task: {sug.task} ({sug.reason})")
        task = sug.task
    elif sug.task and sug.task != task:
        say(f"note: the outputs look like {sug.task} ({sug.reason}); using --task {task} as asked")
    if task not in C.TASKS:
        raise SpillError(f"--task must be one of {', '.join(C.TASKS)}, not {task!r}; other "
                         f"task types are not supported by the guided path",
                         "spill init ... --task classification|json")
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
    cfg = C.default_config(folder.name, task, mapping, system)
    cfg["split"].update({"seed": seed, "val_fraction": val_fraction,
                         "test_fraction": test_fraction, "mode": splits.mode})
    warnings: list[str] = []

    declared = _read_labels_arg(labels)
    sizes = {"train": len(splits.train), "val": len(splits.val), "test": len(splits.test)}
    out_files: dict = {}
    if task == "classification":
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
    if system:
        atomic_write(folder / "system.txt", (system + "\n").encode())
        cfg["task"]["system_file"] = "system.txt"
    for name, b in out_files.items():
        atomic_write(folder / name, b)
    write_json(folder / "data" / "integrity.json", integrity)
    cfg["project"]["created"] = now()
    C.save(folder, cfg)
    atomic_write(folder / ".gitignore", b".spill/\nREPORT.md.tmp\n")
    return InitResult(folder, task, sug.reason if sug.task == task else "chosen with --task",
                      sizes, warnings, splits.notes, cfg)
