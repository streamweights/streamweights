# no-mlx-needed
"""Intake: CSV and JSONL validation, task suggestion, task contract, splits and leakage."""

import csv
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from streamweights.errors import SpillError
from streamweights.project import config as C
from streamweights.project import contract as K
from streamweights.project import data as D
from streamweights.project.init import create_project

MAP = {"input": "text", "output": "label"}


def write_csv(path: Path, rows, header=("text", "label", "who")):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    return path


def labeled(n=60, classes=("billing", "shipping", "login")):
    return [(f"ticket number {i} about {classes[i % len(classes)]} trouble", classes[i % len(classes)],
             f"u{i}") for i in range(n)]


def test_csv_quotes_and_multiline(tmp_path):
    p = tmp_path / "d.csv"
    p.write_text('text,label\n"hello, ""world""\nsecond line",billing\nplain,login\n')
    cols, recs, issues = D.read_table(p)
    assert not issues and cols == ["text", "label"]
    assert recs[0].values["text"] == 'hello, "world"\nsecond line'
    assert [r.row for r in recs] == [1, 2]


def test_malformed_rows_name_file_row_problem_fix(tmp_path):
    p = tmp_path / "d.csv"
    p.write_text("text,label\nok,a\ntoo,many,fields\n,b\nfine,\n")
    cols, recs, issues = D.read_table(p)
    assert len(issues) == 1 and "row 2" in issues[0].where and "3 fields" in issues[0].problem
    assert issues[0].file == "d.csv" and issues[0].fix
    rows, issues2 = D.build_rows(cols, recs, MAP, "classification", "d.csv")
    msgs = [i.line() for i in issues2]
    assert any("empty input" in m and "row 3" in m for m in msgs)
    assert any("empty output" in m and "row 4" in m for m in msgs)
    assert len(rows) == 1


def test_missing_column_lists_columns(tmp_path):
    p = write_csv(tmp_path / "d.csv", labeled(4))
    cols, recs, _ = D.read_table(p)
    _, issues = D.build_rows(cols, recs, {"input": "nope", "output": "label"}, None, "d.csv")
    assert "no column named 'nope'" in issues[0].problem and "text, label, who" in issues[0].fix


def test_not_utf8_and_empty(tmp_path):
    p = tmp_path / "d.csv"
    p.write_bytes(b"text,label\n\xff\xfe,a\n")
    with pytest.raises(D.DataErrors) as e:
        D.read_table(p)
    assert "UTF-8" in e.value.issues[0].problem
    p.write_text("")
    with pytest.raises(D.DataErrors):
        D.read_table(p)


def test_jsonl_errors_by_line(tmp_path):
    p = tmp_path / "d.jsonl"
    p.write_text('{"text": "a", "label": "x"}\n{bad\n[1]\n')
    cols, recs, issues = D.read_table(p)
    assert len(recs) == 1 and [i.where for i in issues] == ["line 2", "line 3"]


def test_suggest_task_not_by_distinct_count_alone(tmp_path):
    def rec(vals):
        return [D.Record("f", i, {"o": v}) for i, v in enumerate(vals, 1)]
    m = {"input": "i", "output": "o"}
    assert D.suggest_task(rec(["a", "b", "a", "b", "a", "b"]), m).task == "classification"
    assert D.suggest_task(rec(['{"a": 1}'] * 4), m).task == "json"
    free = [f"This is a long free text summary number {i} of the document." for i in range(6)]
    assert D.suggest_task(rec(free), m).task is None
    # few distinct values but sentences: still not classification
    sent = ["The customer was happy with the service."] * 3 + ["The customer left angry."] * 3
    assert D.suggest_task(rec(sent), m).task is None
    # all-unique short values (ids): not classification
    assert D.suggest_task(rec([f"id{i}" for i in range(6)]), m).task is None


def test_ambiguous_requires_task(tmp_path):
    rows = [(f"doc {i}", f"A free text summary that is quite long for item {i}.", "") for i in range(20)]
    p = write_csv(tmp_path / "d.csv", rows)
    with pytest.raises(SpillError) as e:
        create_project(p, tmp_path / "proj", MAP)
    assert "--task classification|json" in e.value.line()


def test_transitive_components_and_generated_split_keeps_them(tmp_path):
    R = lambda i, t, g: D.Row(f"r{i}", t, "x", g, {"file": "f", "row": i})
    rows = [R(0, "alpha", "g1"), R(1, "beta", "g1"),     # linked by group
            R(2, "BETA  ", None),                         # linked to 1 by duplicate (normalized)
            R(3, "gamma", "g2"), R(4, "delta", "g3")]
    comps = D.components(rows)
    sizes = sorted(len(c) for c in comps)
    assert sizes == [1, 1, 3]
    big = [c for c in comps if len(c) == 3][0]
    assert sorted(big) == [0, 1, 2]
    many = [R(i, f"t{i}", f"g{i // 3}") for i in range(90)]
    tr, va, te = D.generate_split(many, 0, None, None, 0.2, 0.2)
    owner = {}
    for name, part in (("tr", tr), ("va", va), ("te", te)):
        for r in part:
            owner.setdefault(r.group, set()).add(name)
    assert all(len(v) == 1 for v in owner.values())
    assert len(tr) + len(va) + len(te) == 90 and len(va) >= 18 and len(te) >= 18
    assert D.generate_split(many, 0, None, None, 0.2, 0.2)[1] == va       # reproducible
    assert D.generate_split(many, 1, None, None, 0.2, 0.2)[1] != va       # seed matters


def test_supplied_split_overlap_is_rejected_not_reshuffled(tmp_path):
    main = write_csv(tmp_path / "main.csv", labeled(40))
    val = write_csv(tmp_path / "val.csv", [("TICKET number 3 about billing trouble ", "billing", "z1")])
    with pytest.raises(D.DataErrors) as e:
        create_project(main, tmp_path / "p", MAP, "classification", val=val)
    assert "also appears in the train split" in e.value.issues[0].problem
    val2 = write_csv(tmp_path / "val2.csv", [("brand new text", "billing", "u5")])
    with pytest.raises(D.DataErrors) as e:
        create_project(main, tmp_path / "p2", {**MAP, "group": "who"}, "classification", val=val2)
    assert "group 'u5'" in e.value.issues[0].problem


def test_one_supplied_heldout_is_kept_and_other_derived(tmp_path):
    main = write_csv(tmp_path / "main.csv", labeled(60))
    test = write_csv(tmp_path / "test.csv", [(f"unique held out text {i}", "login", f"t{i}") for i in range(10)])
    res = create_project(main, tmp_path / "p", MAP, "classification", test=test, seed=3)
    assert res.sizes["test"] == 10 and res.sizes["val"] >= 9
    rows = [json.loads(l) for l in (tmp_path / "p/data/test.jsonl").read_text().splitlines()]
    assert all(r["input"].startswith("unique held out") for r in rows)
    assert "test file kept" in " ".join(res.notes)


def test_init_writes_project_and_freezes_contract_from_train_only(tmp_path):
    rows = labeled(90)
    rows[0] = ("rare label ticket", "zzz_only_in_one", "uz")
    p = write_csv(tmp_path / "t.csv", rows)
    res = create_project(p, tmp_path / "proj", {**MAP, "group": "who"}, "classification")
    cfg = C.load(tmp_path / "proj")
    assert cfg["schema_version"] == 1 and cfg["task"]["type"] == "classification"
    train = [json.loads(l)["output"] for l in (tmp_path / "proj/data/train.jsonl").read_text().splitlines()]
    assert sorted(cfg["contract"]["labels"]) == sorted(set(train))
    assert cfg["contract"]["label_source"].startswith("derived from the training rows only")
    assert (tmp_path / "proj/.gitignore").exists()
    assert sum(res.sizes.values()) == 90
    ids = [json.loads(l)["id"] for part in ("train", "val", "test")
           for l in (tmp_path / f"proj/data/{part}.jsonl").read_text().splitlines()]
    assert len(ids) == len(set(ids))


def test_declared_vocabulary_flags_outside_labels(tmp_path):
    p = write_csv(tmp_path / "t.csv", labeled(30))
    with pytest.raises(D.DataErrors) as e:
        create_project(p, tmp_path / "proj", MAP, "classification", labels="billing,shipping")
    assert "not in the declared vocabulary" in e.value.issues[0].problem


def test_conflicting_duplicate_labels_flagged(tmp_path):
    rows = labeled(40) + [("ticket number 2 about login trouble", "billing", "other")]
    p = write_csv(tmp_path / "t.csv", rows)
    res = create_project(p, tmp_path / "proj", MAP, "classification")
    assert any("conflicting labels" in w for w in res.warnings)
    integ = json.loads((tmp_path / "proj/data/integrity.json").read_text())
    assert len(integ["conflicting_labels"]) == 1


def test_small_data_is_reported_and_never_fabricated(tmp_path):
    p = write_csv(tmp_path / "t.csv", labeled(9))
    res = create_project(p, tmp_path / "proj", MAP, "classification")
    assert sum(res.sizes.values()) == 9
    assert any("will be refused" in w for w in res.warnings)
    splits = D.Splits([D.Row("a", "x", "y", None, {})] * 3, [], [], "generated")
    assert len(D.check_usable(splits)) == 2
    assert len(D.check_usable(splits, 'classification')) == 3


def test_group_isolation_not_silently_disabled(tmp_path):
    rows = [(f"text {i}", ["a", "b"][i % 2], "same") for i in range(40)]
    p = write_csv(tmp_path / "t.csv", rows)
    res = create_project(p, tmp_path / "proj", {**MAP, "group": "who"}, "classification")
    assert max(res.sizes.values()) == 40                  # one component, never split
    assert any("will be refused" in w for w in res.warnings)


def test_json_task_schema_derived_from_train_and_ground_truth_checked(tmp_path):
    rows = [(f"Order {i}: {i + 1} widgets for Ann", json.dumps({"qty": i + 1, "who": "Ann"}), f"g{i}")
            for i in range(40)]
    p = write_csv(tmp_path / "j.csv", rows)
    res = create_project(p, tmp_path / "proj", {**MAP, "group": "who"}, "json")
    schema = json.loads((tmp_path / "proj/schema.json").read_text())
    assert schema["properties"]["qty"]["type"] == "integer"
    assert schema["required"] == ["qty", "who"] and schema["additionalProperties"] is False
    bad = write_csv(tmp_path / "bad.csv", [("x", '{"qty": 1}', "g")] + rows[:5])
    sch = tmp_path / "s.json"
    sch.write_text(json.dumps({"type": "object", "properties": {"qty": {"type": "integer"}},
                               "required": ["qty", "who"]}))
    with pytest.raises(D.DataErrors) as e:
        create_project(bad, tmp_path / "p3", MAP, "json", schema_file=sch)
    assert "violates the schema" in e.value.issues[0].problem


def test_config_rejects_future_schema_version(tmp_path):
    (tmp_path / "streamweights.toml").write_text("schema_version = 2\n")
    with pytest.raises(SpillError) as e:
        C.load(tmp_path)
    assert "schema_version 2" in e.value.message and "upgrade" in e.value.line()


def test_init_cli_prints_one_line_and_next(tmp_path, monkeypatch):
    from streamweights.cli import app
    p = write_csv(tmp_path / "t.csv", labeled(60))
    monkeypatch.chdir(tmp_path)
    r = CliRunner().invoke(app, ["init", str(p), "--input", "text", "--output", "label"])
    assert r.exit_code == 0, r.output
    assert "suggested task: classification" in r.output and "next: spill plan t" in r.output
    r2 = CliRunner().invoke(app, ["init", str(p), "--input", "text", "--output", "nope", "--project", "other"])
    assert r2.exit_code == 1 and "no column named 'nope'" in r2.output
