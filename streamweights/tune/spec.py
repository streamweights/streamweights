"""The tune job's specification, independent of the engine that runs it. This is what
`--emit-config` records and what a checkpoint's state.json mirrors."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path

from . import lora_core as lo


@dataclass
class TuneSpec:
    model: str
    quant: str
    model_dir: str
    data: str
    name: str
    path: str                     # resident | streamed
    rank: int = 16
    alpha: float = 32.0
    dropout: float = 0.0
    targets: list[str] | None = None
    lr: float = 1e-4
    weight_decay: float = 0.01
    schedule: str = "cosine"
    seed: int = 0
    max_seq: int = 2048
    micro_batch: int = 4
    grad_accum: int = 1
    steps: int = 1
    epochs: float = 1.0
    ckpt_every: int = 50
    overwrite: bool = False
    resident_weights: bool = False      # tests: streamed algorithm, weights in memory
    engine: str = "mlx"                 # mlx | torch-cpu | torch-cuda
    state: str | None = None            # portable state URI (a path, s3://, gs://, az://)
    stop_after: int | None = None       # stop cleanly after this many steps (tests, demos)

    def lora(self) -> lo.LoraConfig:
        return lo.LoraConfig(self.rank, self.alpha, self.dropout, self.targets, self.lr,
                             self.weight_decay, self.schedule, self.seed)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "TuneSpec":
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in known})


def data_hash(path: Path) -> str:
    h = hashlib.sha256()
    h.update(Path(path).read_bytes())
    return h.hexdigest()
