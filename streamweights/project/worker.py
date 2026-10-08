"""The separate-process executor's entry point: `python -m streamweights.project.worker
stage.json` runs one stage from its serialized description and writes outcome.json next to
it. It knows nothing about the run's control object."""

import json
import sys
from pathlib import Path


def main(argv=None):
    spec = json.loads(Path((argv or sys.argv)[1]).read_text())
    from .. import runtime
    import streamweights.cli  # noqa: F401  (registers the engine-selected job code)
    from .stagefns import StageContext, StageDesc, run_stage
    desc = StageDesc.from_dict(spec["desc"])
    ctx = StageContext(Path(spec["attempt_dir"]), Path(spec["stage_dir"]), spec.get("engine"),
                       spec.get("stop_after"), None, None)
    with runtime.job_session("stage", engine=spec.get("engine")):
        out = run_stage(desc, ctx)
    (Path(spec["stage_dir"]) / "outcome.json").write_text(json.dumps(out.to_dict()))


if __name__ == "__main__":
    main()
