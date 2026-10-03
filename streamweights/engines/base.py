"""Engine interface: run_batch(rows, model_spec, memory_budget) -> iterator of completed rows."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Protocol


@dataclass
class ModelSpec:
    name: str                 # registry name, e.g. llama3.3:70b
    quant: str                # bf16 | 8bit | 4bit | Q8_0 | Q4_K_M
    path: Path                # local artifact dir (safetensors dir / mlx dir / gguf)
    arch: dict = field(default_factory=dict)
    ctx: int = 4096
    extra: dict = field(default_factory=dict)


@dataclass
class MemoryBudget:
    working_set_bytes: int    # probed Metal working set (or VRAM)
    margin: float = 0.15      # safety margin fraction
    batch_override: int | None = None  # --parallel; never required


@dataclass
class CompletedRow:
    custom_id: str
    content: str
    prompt_tokens: int
    completion_tokens: int
    latency_s: float
    finish_reason: str = "stop"
    error: str | None = None
    batch_size: int = 1       # batch the row ran in (for metadata)
    logprobs: list | None = None   # [{token_id, logprob, top: [[id, logprob], ...]}]
    extra: dict | None = None      # extra top-level keys for the result row (score mode)


class Engine(Protocol):
    name: str

    def run_batch(self, rows: list[dict], model_spec: ModelSpec,
                  memory_budget: MemoryBudget) -> Iterator[CompletedRow]:
        """Consume OpenAI-batch rows, yield CompletedRows as they finish."""
        ...
