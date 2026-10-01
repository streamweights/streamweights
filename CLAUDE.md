# Spillway

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

Inference backends are upstream llama.cpp unmodified. Spillway owns only CLI, gateway, jobs, registry, router.

## Stack

Python 3.12, uv, FastAPI, Typer.
