"""A worker process for the ownership tests: runs a list of operations against one run's
control object and prints one JSON line per operation. Hooks come from SPILL_HOOK_* in the
environment, so the test controls exactly where this process pauses or dies."""

import json
import sys
import tempfile
import time
from pathlib import Path

from streamweights.errors import SpillError
from streamweights.project.control import LockContended
from streamweights.project.runstate import RunState


def main():
    spec = json.loads(sys.argv[1])
    kind, _, target = spec["backend"].partition("::")
    work = Path(tempfile.mkdtemp(prefix=f"worker-{spec['owner']}-"))
    rs = RunState.open(target, spec.get("run", "r1"), work, owner=spec["owner"],
                       lock_timeout=spec.get("lock_timeout"))
    if spec.get("lease"):
        rs.auth.lease_s = float(spec["lease"])
    for op in spec["ops"]:
        name = op[0]
        out = {"op": name, "owner": spec["owner"]}
        try:
            if name == "acquire":
                d = rs.acquire(heartbeat=False)
                out["generation"] = d["generation"]
            elif name == "ckpt":
                step = op[1]
                d = work / f"ck{step}"
                d.mkdir()
                (d / "params.bin").write_bytes(f"params-{spec['owner']}-{step}".encode() * 50)
                (d / "COMMIT").write_text("{}")
                ref = rs.publish_checkpoint(d, step, {"micro_batch_index": step})
                out["seq"] = ref["seq"]
            elif name == "stage":
                d = work / f"st-{op[1]}"
                d.mkdir()
                (d / "out.txt").write_text(f"{op[1]} by {spec['owner']}")
                rs.publish_stage(op[1], d, {"status": "done", "by": spec["owner"]})
            elif name == "complete":
                d = work / "snap"
                d.mkdir(exist_ok=True)
                (d / "report.md").write_text(f"report by {spec['owner']}")
                rs.complete(d)
            elif name == "restore":
                dest = work / "restored"
                ck = rs.restore_checkpoint(dest)
                out["step"] = ck["step"] if ck else None
                out["cursor"] = ck["cursor"] if ck else None
                out["files"] = sorted(p.name for p in dest.iterdir()) if ck else []
                out["params"] = (dest / "params.bin").read_text()[:24] if ck else None
                out["recovery"] = rs.recovery(ckpt_every=op[1] if len(op) > 1 else None).line()
            elif name == "renew":
                rs.auth.renew()
            elif name == "release":
                rs.auth.release()
            elif name == "sleep":
                time.sleep(op[1])
            out["ok"] = True
        except LockContended as e:
            out.update(ok=False, error="LockContended", message=e.message)
        except SpillError as e:
            out.update(ok=False, error=type(e).__name__, message=e.message)
        except Exception as e:                      # a bug, not a protocol outcome: show it
            out.update(ok=False, error="BUG:" + type(e).__name__, message=str(e))
        print(json.dumps(out), flush=True)


main()
