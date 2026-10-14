"""Check a directive 020 capture (scripts/capture_020.py) against the gate it records.

  python scripts/check_020_capture.py docs/reports/data/020-judged-mlx.json [...]

Each flow has its own conditions; a capture that does not meet them exits 1 with the reason. The
conditions are about what the run did and printed, never about how good the numbers are."""

import json
import re
import sys
from pathlib import Path


def step(cap, prefix):
    s = next((s for s in cap["steps"] if s["command"].startswith(prefix)), None)
    assert s is not None, f"no step {prefix!r}"
    assert s["exit"] == 0, f"{prefix} exited {s['exit']}"
    return s["stdout"]


def judged(cap):
    plan, build, report, export = step(cap, "spill plan"), step(cap, "spill build"), \
        step(cap, "spill report"), step(cap, "spill export")
    assert "qwen2.5:0.5b (student), qwen2.5:1.5b (teacher), qwen2.5:1.5b (judge)" in plan
    assert "training  the teacher's answers train the student with mix" in plan
    assert re.search(r"stage 7/7: judge qwen2\.5:1\.5b scores 3 comparators and 2 pairs \(both orders\)", build)
    for need in ("Judged evaluation (one model's assessment, not accuracy)", "Pairwise, each pair judged in both orders",
                 "orders disagreed", "Calibration:", "reference only", "may not judge its own outputs"):
        assert need in report, f"report lacks {need!r}"
    assert "judge score 1-5 (one model's assessment, not accuracy)" in build and "accuracy" not in \
        re.sub(r"not accuracy", "", build.split("stage 7/7")[1]).lower().replace("accuracy 0", "")
    assert "verified safetensors:" in export and re.search(r"quality on these \d+ rows", export, re.I)
    py = step(cap, "python")
    assert py.strip(), "the exported model printed nothing"
    for tag, frag in (("mlx", "Device(gpu"), ("torch-cpu", "cpu")):
        if cap["engine"] == tag:
            assert cap["device"].startswith(frag), cap["device"]


def nojudge(cap):
    plan, build = step(cap, "spill plan"), step(cap, "spill build")
    assert "judge: none named. Judged scores need a judge; recommended:" in plan
    assert "download" in plan.split("recommended:")[1].split("\n")[0]
    assert "teacher   none" in plan and "nothing to download" in plan
    assert "token F1 against the reference (overlap, not accuracy)" in build and "judge score" not in build
    assert cap["allowed_models"] and "qwen2.5:1.5b" not in cap["allowed_models"]
    assert not cap["models_downloaded"] and not cap["models_changed"]
    assert "distill" not in plan.lower().replace("distillation", "")


def labeled(cap):
    plan, build = step(cap, "spill plan"), step(cap, "spill build")
    assert "teacher   none" in plan and "judge" not in plan.split("downloads:")[0].replace("--judge", "")
    stages = plan.split("stages:")[1]
    assert "teacher" not in stages and "answers the" not in stages
    assert "stage 1/4: embedding + logistic regression baseline" in build
    assert re.search(r"stage 2/4: tune qwen2\.5:0\.5b on \d+ training rows \(\d+ yours\)", build)
    dl = plan.split("downloads:")[1].split("disk:")[0]
    assert set(re.findall(r"^   (\S+): ", dl, re.M)) == {"qwen2.5:0.5b", "embedding:minilm"}
    assert cap["allowed_models"] and not cap["models_downloaded"]


def kd(cap):
    builds = [s for s in cap["steps"] if s["command"].startswith("spill build")]
    assert len(builds) == 2 and all(s["exit"] == 0 for s in builds)
    assert "loss ce)" in builds[0]["stdout"] and "loss mix)" in builds[1]["stdout"]
    cmp = step(cap, "spill compare")
    assert "common evaluator" in cmp and "training.distill_loss" in cmp
    assert len(re.findall(r"^run-\S+  run-\S+/original", cmp, re.M)) == 2


CHECKS = {"judged": judged, "nojudge": nojudge, "labeled": labeled, "kd": kd}


def check(path) -> str:
    cap = json.loads(Path(path).read_text())
    assert cap["ok"], f"{path}: the capture did not complete"
    CHECKS[cap["flow"]](cap)
    return f"{path}: {cap['flow']} on {cap['engine']} ({cap['device']}) ok"


if __name__ == "__main__":
    bad = 0
    for p in sys.argv[1:]:
        try:
            print(check(p))
        except AssertionError as e:
            bad = 1
            print(f"{p}: FAILED: {e}")
    sys.exit(bad)
