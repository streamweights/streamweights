"""torch_resident: the PyTorch engine with every layer's weights in memory. The exact loop of
torch_stream with a memory-resident weight provider, so streamed and resident execution are
op-for-op identical and the identity gate tests the I/O path alone."""

from __future__ import annotations

from .torch_stream import TorchEngine


class TorchResidentEngine(TorchEngine):
    name = "torch_resident"

    def __init__(self, progress_note=None, pass_cb=None, engine: str = "torch-cpu",
                 dtype: str | None = None):
        super().__init__(progress_note=progress_note, resident=True, pass_cb=pass_cb,
                         engine=engine, dtype=dtype)
