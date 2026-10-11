---
description: How streamweights is built: thin engines, the streaming ring, the job layer, the project layer and the CLI.
---

# Architecture

This page moved out of the README. For the commands, see [commands](cli.md); for what moves between machines, see [portability](portability.md).

## Components

Engines are thin: MLX on Apple silicon, PyTorch (transformers layers bound from the same ring) elsewhere, upstream llama.cpp for GGUF. streamweights owns the streaming ring, the job layer (a hardware-neutral float32 checkpoint with a commit marker), the project layer (config, fenced run state, immutable records) and the CLI. Stages are serializable descriptions run by an executor (local, or a separate process); only the coordinator publishes. The engines decode greedily; a request for a sampling or stop setting they cannot apply is refused before anything runs.

## Where the code is

- `cli.py`, `cli_build.py`: the Typer app, commands in loop order.
- `engines/mlx_stream.py`, `engines/mlx_resident.py`: the MLX streaming and resident runners; `engines/llamacpp.py` is the non-Apple path for GGUF.
- `ring.py`: the framework-free streaming ring (layer index, pread ring, read modes).
- `engines/torch_common.py`, `torch_stream.py`, `torch_resident.py`: the PyTorch engines on CPU and CUDA, selected by `engine_select.py`.
- `portable/`: stores, the hardware-neutral checkpoint, row-job segments, weight staging.
- `project/`: config, fenced run state, immutable records, reports, export, bundle, move.
- `jobs/`, `tune/`, `build.py`, `export.py`: the older folder workflow and its pieces.
