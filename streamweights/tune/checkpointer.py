"""One job's view of its portable checkpoint: where it lives (a local directory or the
`--state` URI), how its state.json is built, whether a found checkpoint may be continued, and
the engine, hardware and numerics recorded for each range of steps. Engine-neutral: the MLX
trainers and the PyTorch trainers share it."""

from __future__ import annotations

import json
import time
from pathlib import Path

from ..errors import SpillError
from ..portable import checkpoint as pc
from ..portable.store import Store
from .spec import TuneSpec, data_hash


class Checkpointer:
    def __init__(self, spec: TuneSpec, job_dir: Path, *, engine: str, device: str,
                 numerics: dict, targets: list[str], on_event=None):
        from ..runs import fingerprint_model
        self.spec = spec
        self.store = Store(spec.state) if spec.state else Store(Path(job_dir) / "ckpt")
        self.remote = bool(spec.state)
        self.engine, self.device, self.numerics = engine, device, numerics
        self.targets = targets
        self.on_event = on_event
        self.fingerprint = fingerprint_model(Path(spec.model_dir))
        self.data_sha = data_hash(Path(spec.data))
        self.history: list[dict] = []
        self.start_step = 0
        self.losses_path = Path(job_dir) / "losses.jsonl"
        self._ck_losses: list[list] = []

    def _local_losses(self, upto: int) -> list[list]:
        out = []
        if self.losses_path.exists():
            for line in self.losses_path.read_text().splitlines():
                if line.strip():
                    d = json.loads(line)
                    if d["step"] <= upto:
                        out.append([d["step"], d["loss"]])
        return out

    def sync_losses(self, upto: int) -> None:
        """Make losses.jsonl hold exactly steps 1..upto: lines already here are kept, steps
        this machine never ran come from the checkpoint's own record."""
        have = {}
        if self.losses_path.exists():
            for line in self.losses_path.read_text().splitlines():
                if line.strip():
                    d = json.loads(line)
                    if d["step"] <= upto:
                        have[d["step"]] = d
        for st, loss in self._ck_losses:
            if st <= upto and st not in have:
                have[st] = {"step": st, "loss": loss, "step_s": None, "peak_gb": 0.0}
        self.losses_path.write_text("".join(json.dumps(have[k]) + "\n" for k in sorted(have)))

    def load(self, *, want: bool):
        """The checkpoint to continue from, or None. With a state URI it is picked up
        automatically; for a local job directory only when `want` (spill resume)."""
        if not (want or self.remote):
            return None
        ck = pc.load_tune(self.store)
        if ck is None:
            return None
        why = pc.check_resume_compatible(
            ck.state, model_id=self.spec.model, fingerprint=self.fingerprint,
            data_sha256=self.data_sha, spec=self.spec.to_dict())
        if why:
            raise SpillError(f"cannot resume from {self.store.uri}: {why}",
                             "spill tune ... --state <a new location>")
        self.history = [h for h in ck.state.get("history", []) if h["range"][0] < ck.step]
        for h in self.history:
            h["range"][1] = min(h["range"][1], ck.step)
        self.start_step = ck.step
        self._ck_losses = ck.state.get("losses", [])
        return ck

    def begin(self, step: int) -> None:
        self.start_step = step

    def save(self, step: int, params: dict, opt: dict) -> None:
        seg = pc.producer(self.engine, self.device, self.numerics, self.start_step, step)
        hist = pc.extend_history(self.history, seg)
        state = pc.tune_state(spec=self.spec.to_dict(), model_id=self.spec.model,
                              fingerprint=self.fingerprint, data_sha256=self.data_sha,
                              targets=self.targets, history=hist,
                              grad_accum=self.spec.grad_accum, step=step)
        state["losses"] = self._local_losses(step)
        t0 = time.monotonic()
        pc.save_tune(self.store, step=step, params=params, opt=opt, state=state)
        self.history = hist
        self.start_step = step
        from .. import runtime
        if runtime.ENV.on_checkpoint is not None and self.store.local:
            runtime.ENV.on_checkpoint(step, Path(self.store.root) / pc._dir(step), state)
        if self.on_event:
            self.on_event({"event": "checkpoint", "step": step, "uri": self.store.uri,
                           "seconds": round(time.monotonic() - t0, 3)})
