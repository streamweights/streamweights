"""Write where-<job>.json: which machine and OS a relay job ran on, and the engine it used.
  python scripts/ci_where.py JOB ENGINE OUTDIR"""
import json
import platform
import sys
from pathlib import Path

job, engine, out = sys.argv[1:4]
Path(out).mkdir(parents=True, exist_ok=True)
Path(out, f"where-{job}.json").write_text(json.dumps(
    {"job": job, "system": platform.system(), "host": platform.node(), "engine": engine}))
