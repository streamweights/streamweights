# streamweights

Package `streamweights`, CLI `spill`. Repo: https://github.com/streamweights/streamweights

**Purpose:** build your own model on the Mac you already own. Fine-tune and distill from a 70B teacher locally with MLX, even when the model is bigger than RAM. Everything runs on the user's machine, for free, with no data leaving it.

## Never-stop rules (binding in every session)

- Never print a status summary or progress report before the terminal state of the directive; printing one ends the turn and stops all work. Until the end, output only tool calls.
- Never end a turn to wait. A wait is a foreground shell loop under the tool timeout. No background tasks that rely on waking the session, no scheduled wake-ups, no `/loop`, no questions to the user.
- If something blocks, make the most reasonable choice, log it under Decisions in the report, and continue.
- No Claude attribution on any commit. No em-dashes in any file written.
- Hardware limits are set per directive. Unless a directive says otherwise: no 70B or 7B runs, no model downloads over 2 GB, GPU use limited to `qwen2.5:0.5b`, because the user works on this machine.
- No PyPI publishing, no release workflow, no tags, until a directive asks. Install lines use the GitHub URL.
- Paste sets are saved verbatim in `docs/paste-sets/` and committed first.
- Every directive ends by updating the README and the docs site to reflect the current state, with measured numbers only.
- Never post publicly on the user's behalf: no issues, pull requests or comments on other repositories; drafts only. Commit after every numbered item of a directive.
- Directive 013 additionally forbids model runs of any size, PyPI publishing and tags.

## Developer-experience rules (binding on every command)

1. One install, zero config; hardware is probed, never declared.
2. No new concepts on the way in; input is plain JSONL (`{"prompt","expected"}`, `{"prompt"}`, `{"prompt","answer"}`, optional `"system"`) or the OpenAI batch and chat shapes; endpoints are Ollama and OpenAI shapes.
3. First result in minutes, full result overnight; progress, tokens/s and ETA from the first minute.
4. Never silently slow, never silently expensive: one pre-run line states model, quant, placement, estimated time and cost.
5. Interruptible and resumable by default; no lost work.
6. Models by name, quants by policy; bf16 is the default, a quant is an explicit opt-in and is stated in output metadata.
7. Every command ends by printing the one command most likely to come next (`next: ...`).

Errors are one line ending in the recovery command (`spill: <message>. Try: <command>`), with no traceback unless `--debug`. Pre-run lines read `spill <command> <subject>: <facts>. Est. <time>. Cost: $0. <Dest> -> <path>`.

## Architecture

streamweights owns: the CLI, the gateway, jobs, the registry, the router, the streaming runner, the tuner, build, export.
Inference backends elsewhere are upstream llama.cpp, unmodified, never vendored. The GGUF converter is downloaded at the pinned release tag.

- `cli.py`, `cli_build.py`: Typer app, commands in loop order (build, example, run, distill, tune, eval, export, then models, adapters, runs, status, tail, resume, doctor, check). `main()` is the entry point.
- `platforms.py`: where spill runs. `build` needs Apple silicon with MLX; run, distill, tune, eval, export, check, models run on the torch engines everywhere.
- `engines/mlx_stream.py`: the streaming runner (layer-ordered NVMe ring, batching, refill, memory budget, shared-prefix reuse). `engines/mlx_resident.py`: the same loop with weights in memory. `engines/llamacpp.py`: the non-Apple path.
- `ring.py`: the framework-free streaming ring (safetensors layer index, pread ring, read modes nocache, fadvise, odirect). `engines/torch_common.py`, `torch_stream.py`, `torch_resident.py`: the PyTorch engines (transformers layers on the meta device bound from the ring; CPU and CUDA). `engine_select.py`: mlx, torch-cuda or torch-cpu from the hardware.
- `portable/`: fsspec stores, the hardware-neutral checkpoint (float32 safetensors plus state.json, COMMIT marker last), row-job segments, weight staging. `headless.py`, `runtime.py`, `jobconfig.py`: JSON-lines events, SIGTERM then exit 75, `--state`, `--config`, `--emit-config`.
- `tune/torch_job.py`, `tune/torch_train.py`: the PyTorch tune (PEFT resident, streamed saved-input reverse-stream VJP). `gates.py`, `verify_cuda.py`: the identity and resume gates, runnable on a CUDA machine.
- `jobs/`: OpenAI-batch-compatible job engine, per-row checkpoint, resume.
- `tune/`: LoRA training, resident (mlx-lm tuner) and streamed (saved layer inputs, reverse recompute VJP, two weight streams per micro-batch).
- `build.py`: stage planner and runner over a folder; `export.py` and `safetensors_np.py`: merge adapters with numpy, GGUF via llama.cpp's converter.
- `estimate.py`, `calibration.py`: one cost model (decode disk-bound, prefill 2 x params x tokens, training 6 x params x tokens, divided by the achieved TFLOP/s).

## Stack

Python 3.10 or newer (developed on 3.12), uv, FastAPI, Typer, MLX (Apple silicon only; every MLX import is guarded or lazy).

## Tests

`python -m pytest` runs the CPU suite (`SPILL_DEVICE=cpu` is set by `tests/conftest.py`). GPU tests are skipped unless `SPILL_GPU_TESTS=1`. CI runs the suite on macOS and Linux. `tests/test_docs.py` fails on unfinished markers, em-dashes, unmeasured numbers, or a README out of shape.

## Measured facts that drive the design

- mmap reached only 11 to 13% of the probed NVMe rate on a model 1.5x RAM (Phase 0). The engine streams layers in order instead.
- Pass time is flat in batch size only while attention is cheap; at 1k-token prompts the engine becomes compute-bound and KV cost cuts batch by about 5x.
- Float32 streamed training matches resident to 0.024% in loss; bf16 differs by rounding noise (`docs/reports/008-phase3.md`).

## Workflow thesis

The product is the local half of building your own model: run, distill, tune, eval, export, against full-precision open models. The eval set is the central artifact. Every command is a run against it.

## Where things are

`docs/plan.md` (phases and what is next), `docs/linux.md` (the Linux and NVIDIA to-do), `docs/cli.md` (generated from `--help`), `docs/formats.md`, `docs/models.md`, `docs/reports/` (measured results per phase).

## Run-anywhere principle

The design principle: every job is a sequence of normalized quanta. For training, the quantum is one optimizer step. For run, distill and eval, it is one completed row. Any machine with a supported engine can execute the next quantum from a portable checkpoint.
Engines are thin: MLX on Apple silicon, PyTorch everywhere else (CPU and CUDA). Streamweights owns only the streaming ring, the job layer, and the CLI.
