# streamweights

(Working name "Spillway" retired in paste set 003: package `streamweights`, CLI `spill`.)

**Purpose:** run the unmodified full-size model against my eval set, on my machine, for free, overnight, with no data leaving the building.

## Developer-experience rules

(1) One install, zero config; hardware is probed, never declared.
(2) No new concepts on the way in; input is OpenAI batch JSONL, endpoints are Ollama and OpenAI shapes.
(3) First result in minutes, full result overnight; progress, tokens/s and ETA from the first minute.
(4) Never silently slow, never silently expensive; state model, tier, quant, estimated wall time and cost in one line before running.
(5) Interruptible and resumable by default; no lost work.
(6) Models by name, quants by policy; bf16 is the default on the batch tier, a quant is explicit opt-in and is always stated in output metadata.
(7) Every command ends by printing the one command most likely to come next.

## Architecture rule

Inference backends are upstream llama.cpp unmodified. streamweights owns only CLI, gateway, jobs, registry, router (and, from Phase 1, the streaming runner - see docs/plan.md).

## Stack

Python 3.12, uv, FastAPI, Typer.

## Phase 0 result (two lines)

mmap reached only 11–13% of the probed NVMe sequential rate on a model 1.5× RAM (175 s per forward pass on 70B Q8_0).
GPU OOM from batch 32 up at 4k context; batch 16 ran but completed zero rows in 25 minutes.

## Phase 1 engine decision

On macOS the engine is MLX with bf16 safetensors as the native artifact; llama.cpp is retained only as the non-Apple path. No GGUF conversion appears anywhere in the golden path.

The seven DX rules in docs/plan.md remain binding - in particular "one command, zero config" must hold for `spill run <model> <file>` with no flags.
