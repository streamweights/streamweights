"""Write the committed resume fixture: qwen2.5:0.5b tuned on a toy task on MLX (bf16 base weights,
an Apple GPU) with a 100-step schedule, stopped at step 50. A few MB: rank 4 on q_proj and v_proj.

  python scripts/make_resume_fixture.py [lr]

Writes streamweights/data/fixtures/resume-qwen05/ with

  state/            the portable checkpoint at step 50 (ckpt/step-000000050/ and LATEST)
  train.jsonl       the toy training data (200 examples)
  heldout.jsonl     60 held-out toy rows (exact match scores the adapter)
  job.json          the exact `spill tune` invocation (written by --emit-config)
  expected_losses.json   the loss at each of 100 steps from an uninterrupted MLX run

Linux CI resumes it on torch-cpu in every run; `python -m streamweights.verify_cuda` resumes it on
a GPU.
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
FX = REPO / "streamweights" / "data" / "fixtures" / "resume-qwen05"
WORK = REPO / "state" / "files" / "fixture-work"

from streamweights import gates as G  # noqa: E402
from streamweights.tune import toy  # noqa: E402

lr = float(sys.argv[1]) if len(sys.argv) > 1 else 2e-5
w = G.Work(WORK, models_from=REPO / "models", state_from=REPO / "state")
files = toy.make(WORK / "toy", 200, 60)

shutil.rmtree(FX, ignore_errors=True)
FX.mkdir(parents=True)
shutil.copy(files["train"], FX / "train.jsonl")
shutil.copy(files["heldout"], FX / "heldout.jsonl")

common = dict(steps=100, lr=lr, batch=4, max_seq=128, rank=4, targets="q_proj,v_proj")
ref = G.run_tune_cli(w, "ref", G.tune_args("qwen2.5:0.5b", str(FX / "train.jsonl"), "fx-ref",
                                           engine="mlx", **common))
(FX / "expected_losses.json").write_text(json.dumps([round(x, 6) for x in ref["losses"]]))

# the invocation, written as a config with paths relative to the fixture directory
args = G.tune_args("qwen2.5:0.5b", "train.jsonl", "resume-fixture", engine="mlx", state="state",
                   **common)
args = [a for a in args if a not in ("--overwrite", "--quiet")]
r = subprocess.run([sys.executable, "-m", "streamweights", *[str(a) for a in args],
                    "--emit-config", "job.json"], cwd=FX, capture_output=True, text=True,
                   env={**__import__("os").environ, "SPILL_HOME": str(w.home), "SPILL_HEADLESS": "1"})
assert r.returncode == 0, r.stderr
cfg = json.loads((FX / "job.json").read_text())
cfg["options"].pop("engine", None)            # the engine is the machine's choice, or --engine
(FX / "job.json").write_text(json.dumps(cfg, indent=2) + "\n")

# the checkpoint: the same invocation on MLX, stopped at step 50
first = w.spill(["tune", "--config", "job.json", "--engine", "mlx", "--overwrite",
                 "--stop-after", "50"], cwd=FX)
assert first.rc == 0, first.stderr
losses = [e["loss"] for e in first.events if e["event"] == "step"]
ref_l = ref["losses"][:50]
print("first 50 steps match the reference run:", max(abs(a - b) / b for a, b in zip(losses, ref_l)))
(FX / "README.txt").write_text(
    "Resume fixture: qwen2.5:0.5b, toy task, rank 4 on q_proj and v_proj, lr %g, 100-step\n"
    "cosine schedule, stopped at step 50 on MLX (bf16 base weights, Apple GPU).\n"
    "state/ is the portable checkpoint; job.json is the invocation; expected_losses.json is the\n"
    "uninterrupted MLX run.\n  cd <this directory> && spill tune --config job.json --engine torch-cpu\n"
    "continues it from step 51 on any machine.\n" % lr)
size = sum(f.stat().st_size for f in FX.rglob("*") if f.is_file())
print(f"fixture written: {FX} ({size / 1e6:.2f} MB)")
print("loss at 1/25/50/75/100:", [round(ref["losses"][i - 1], 4) for i in (1, 25, 50, 75, 100)])
