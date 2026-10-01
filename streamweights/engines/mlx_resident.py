"""mlx_resident: normal full load (safetensors via mlx-lm model classes), for
models that fit in memory. Runs the exact same forward loop as mlx_stream with
a memory-resident weight provider, so streamed and resident execution are
op-for-op identical — which is what makes the Phase 1 identical-output gate a
true test of the streaming I/O path."""

from __future__ import annotations

from .mlx_stream import MlxStreamEngine


class MlxResidentEngine(MlxStreamEngine):
    name = "mlx_resident"

    def __init__(self, progress_note=None):
        super().__init__(progress_note=progress_note, resident=True)
