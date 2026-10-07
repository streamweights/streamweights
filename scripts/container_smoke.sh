#!/usr/bin/env bash
# Smoke test of the CPU image: a 20-row eval on the 0.5B, then the committed step-50 fixture
# resumed on torch-cpu. Usage: scripts/container_smoke.sh [image]   (default ghcr.io/streamweights/spill:cpu)
set -euo pipefail
IMAGE="${1:-ghcr.io/streamweights/spill:cpu}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORK="$(mktemp -d)"
VOL="spill-smoke-$$"
trap 'docker volume rm -f "$VOL" >/dev/null 2>&1 || true; rm -rf "$WORK"' EXIT
docker volume create "$VOL" >/dev/null

echo "== 1. 20-row eval on qwen2.5:0.5b"
docker run --rm -v "$VOL":/data "$IMAGE" eval sample qwen2.5:0.5b --engine torch-cpu --rerun \
  > "$WORK/eval.jsonl" 2> "$WORK/eval.err" || { cat "$WORK/eval.err"; exit 1; }
python3 - "$WORK/eval.jsonl" <<'PY'
import json, sys
ev = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
rows = [e for e in ev if e["event"] == "row"]
assert ev[0]["event"] == "start" and ev[-1]["event"] == "done", [e["event"] for e in ev][:3]
assert len(rows) == 20, f"expected 20 rows, saw {len(rows)}"
assert all(e["engine"] == "torch-cpu" for e in ev), "engine not stamped on every event"
print("eval ok:", len(rows), "rows, engine", ev[-1]["engine"])
PY

echo "== 2. resume the step-50 MLX fixture on torch-cpu"
cp -R "$ROOT/streamweights/data/fixtures/resume-qwen05" "$WORK/fx"
docker run --rm -v "$VOL":/data -v "$WORK/fx":/fixture -w /fixture "$IMAGE" tune --config job.json \
  > "$WORK/tune.jsonl" 2> "$WORK/tune.err" || { cat "$WORK/tune.err"; exit 1; }
python3 - "$WORK/tune.jsonl" "$ROOT/streamweights/data/fixtures/resume-qwen05/expected_losses.json" <<'PY'
import json, math, sys
ev = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
steps = [e for e in ev if e["event"] == "step"]
ref = json.load(open(sys.argv[2]))
assert steps and steps[0]["step"] == 51, f"resumed at step {steps[0]['step'] if steps else None}, expected 51"
assert steps[-1]["step"] == 100 and ev[-1]["event"] == "done"
assert all(math.isfinite(e["loss"]) for e in steps)
rel = max(abs(e["loss"] - ref[e["step"] - 1]) / max(abs(ref[e["step"] - 1]), 1e-9) for e in steps)
print(f"resume ok: steps 51..100 on torch-cpu, max relative loss difference to the MLX run {rel:.3f}")
assert rel < 0.5, rel
PY
echo "container smoke test passed"
