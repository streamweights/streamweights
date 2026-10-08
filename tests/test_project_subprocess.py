"""The test backend: the same stages, run in a separate process from their serialized
description, with the coordinator still doing all publication and reporting."""

import json
import os
from pathlib import Path

import pytest

from streamweights import example as E
from streamweights.project import config as C
from streamweights.project import coordinator as CO
from streamweights.project import plan as P
from streamweights.project.common import read_json
from streamweights.project.init import create_project
from streamweights.project.stagefns import StageDesc

REPO = Path(__file__).resolve().parent.parent
MODEL = REPO / "models" / "qwen2.5-0.5b" / "bf16-st"
pytestmark = pytest.mark.skipif(
    not (MODEL / "model.safetensors").exists() and os.environ.get("SPILL_REQUIRE_FIXTURE_MODEL") != "1",
    reason="needs qwen2.5:0.5b")


def test_stage_descriptions_roundtrip_through_json_without_paths_or_engine_types(tmp_path):
    proj = tmp_path / "p"
    create_project(E.DATA / "banking77" / "tiny" / "banking77.csv", proj,
                   {"input": "text", "output": "label"}, "classification")
    plan = P.make_plan(proj, engine="torch-cpu")
    for st in plan.stages:
        d = json.loads(json.dumps(st.to_dict()))
        assert StageDesc.from_dict(d).to_dict() == st.to_dict()
        text = json.dumps(d)
        assert str(tmp_path) not in text and "mlx" not in text and "torch" not in text.lower().replace(
            "torch_dtype", "")
        assert all(set(i) == {"name", "sha256"} for i in st.inputs)         # inputs by identity, not by path


def test_a_build_through_the_separate_process_backend_completes(tmp_path):
    proj = tmp_path / "p"
    create_project(E.DATA / "banking77" / "tiny" / "banking77.csv", proj,
                   {"input": "text", "output": "label"}, "classification")
    cfg = C.load(proj)
    cfg["training"]["epochs"] = 1.0
    C.save(proj, cfg)
    plan = P.make_plan(proj, engine="torch-cpu", fetch=True)
    said = []
    co = CO.Coordinator(proj, "subprocess", "torch-cpu", say=said.append)
    out = co.build(plan)
    assert out.status == "completed"
    doc = read_json(proj / ".spill" / "runs" / out.run_id / "control.json")
    events = [h["event"] for h in doc["history"]]
    # the coordinator, not the worker, published every stage, under its own generation
    assert [e for e in events if e.startswith("stage:")] == [
        "stage:inputs", "stage:baseline", "stage:train", "stage:eval:untrained", "stage:eval:trained"]
    assert {h["generation"] for h in doc["history"] if h["event"].startswith("stage:")} == {1}
    m = read_json(out.snapshot / "manifest.json")
    assert [r["comparator"] for r in m["table"]] == ["baseline", "untrained", "trained"]
    assert all(r["primary"] is not None for r in m["table"])
    assert (out.snapshot / "report.md").exists() and (proj / "REPORT.md").exists()
    # each stage ran in its own process: its outcome file exists next to its serialized description
    sd = proj / ".spill" / "runs" / out.run_id / "attempts"
    stage_json = list(sd.glob("*/stages/*/stage.json"))
    assert len(stage_json) == 4 and all((p.parent / "outcome.json").exists() for p in stage_json)
