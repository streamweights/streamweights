"""Runs `move` in its own process so a hook can kill it at an exact point."""
import json
import sys

from streamweights.errors import SpillError
from streamweights.project.move import move

project, dest = sys.argv[1], sys.argv[2]
try:
    r = move(project, dest, say=lambda s: None, wait_s=10)
    print(json.dumps({"ok": True, "tid": r.transfer_id, "resumed": r.resumed}))
except SpillError as e:
    print(json.dumps({"ok": False, "error": e.message}))
