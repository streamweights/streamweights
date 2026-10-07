"""Audit a build's portable state: every row of every row job present once, every training
step present once, and where each stage ran. Used by the relay check in CI and by the gates;
it reads the state and nothing else, so it works on a state copied from another machine."""

from __future__ import annotations

from .portable import checkpoint as pc
from .portable import rows as pr
from .portable.build_state import BuildState
from .portable.store import Store


def audit(state_uri: str) -> dict:
    st = Store(state_uri)
    out: dict = {"tune": None, "row_jobs": {}}
    for stage in st.ls("stages"):
        sub = st.sub(f"stages/{stage}")
        if sub.exists("ckpt"):
            ck = pc.load_tune(sub)
            if ck is not None:
                steps = [s for s, _ in ck.state["losses"]]
                out["tune"] = {
                    "final_step": ck.step, "steps_recorded": len(steps),
                    "contiguous_1_to_n": steps == list(range(1, ck.step + 1)),
                    "history": [{"range": h["range"], "engine": h["engine"],
                                 "hardware": h["hardware"], "os": h.get("os"),
                                 "numerics": h["numerics"].get("base")}
                                for h in ck.state["history"]]}
        for where in [sub] + [sub.sub(leaf) for leaf in sub.ls("") if sub.exists(f"{leaf}/rows")]:
            if where.exists("rows"):
                segs = pr.committed_segments(where)
                ids = [i for s in segs for i in s["ids"]]
                out["row_jobs"][stage] = {"rows": len(ids), "unique": len(set(ids)),
                                          "segments": len(segs),
                                          "engines": sorted({s.get("engine") for s in segs})}
    out["clean"] = (out["tune"] is None or out["tune"]["contiguous_1_to_n"]) and all(
        j["rows"] == j["unique"] for j in out["row_jobs"].values())
    return out


def read_state(state_uri: str) -> dict | None:
    return BuildState(state_uri).read()
