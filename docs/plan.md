streamweights: technical plan (v3)

Repo and package: streamweights. Command: spill. Local by default.

v3 changes the spine. v2 treated the wedge (run the full-size model over an eval set) as the product and everything else as completeness. After Phases 0 to 1.6 the wedge is proven and measured, and the useful question became what you do with exact full-precision outputs. The answer is the loop below. Interactive tiers and cloud burst are parked; adapters moved from "train on a rented GPU" to "train and evaluate locally".

1. The wedge, and the loop it starts

Developers cannot test against models bigger than their machine without renting GPUs, and the moment they rent, the whole dev loop changes. Ollama answers "what fits." Ollama Cloud answers "chat with a bigger one on our servers." Neither answers:

Run the unmodified full-size model against my eval set, on my machine, for free, overnight, with no data leaving the building.

That is streamweights, and it is the first verb of a loop:

run     the full-precision model, optionally with an adapter, over a file
distill the teacher's completions with top-k log-probs, or its log-probs over supplied targets (teacher-forced, prefill only)
tune    train an adapter on that data (Phase 3)
eval    the same eval set through several models, one table, reproducible from a manifest

The product is the local half of building your own model. The eval set is the central artifact: it is what you measure a base against, what you distill from, and what decides whether a tuned adapter is better. Every command is a run against it, every run writes a manifest, and a number in a table can always be traced to the exact weights, adapter and input that produced it.

It stays a correctness tier, not a chat tier: slow per prompt, exact, free, local. Batching makes it practical, because a model streamed from disk pays one full weight read per forward pass whether the batch holds 1 prompt or 100. That holds while attention is cheap; at 1k-token prompts the engine becomes compute-bound and KV cost cuts the batch by about 5x (proof run finding).

2. Pillar: developer experience

Non-negotiable, unchanged by v3. Every new command is tested against these, in order:

One install, zero config. pip install streamweights, then spill. No YAML before the first result. Hardware is probed, never declared.
No new concepts on the way in. Input is the OpenAI batch JSONL developers already have, or the OpenAI fine-tuning chat JSONL they already train on; an eval row is a chat row plus an expected field. Endpoints are the Ollama and OpenAI API shapes. The loop adds verbs, not file formats.
First result in minutes, full result overnight. Progress, tokens per second and an ETA from the first minute; tail results as they land. Distill and eval reuse the same job engine, so they inherit this.
Never silently slow, never silently expensive. Before a job starts, spill states the model, adapter, mode, quant, tokens to process, estimated wall time (scoring from the measured prefill rate, generation from the measured pass time) and cost (zero). One line, then it runs.
Interruptible and resumable by default. Close the laptop, come back, spill resume. No lost work. An eval reuses every finished run for the same input hash.
Models by name, quants by policy. llama3.3:70b means the full bf16 by default on the batch tier. A quant is an explicit opt-in and is always stated in output metadata. An adapter is named with +, never merged into the weights.
The CLI tells you the next thing. Every command ends with the one command most likely to come next.

Golden path (must hold with no flags):

pip install streamweights
spill run llama3.3:70b evals.jsonl

The loop's golden path extends it without changing it:

spill eval evals.jsonl llama3.3:70b llama3.3:70b+./my-adapter

3. Tiers
Tier    Where    Engine    Speed    Cost    Role
Local-streamed    Developer machine, NVMe    MLX streaming runner (macOS)    Seconds to a minute per forward pass, batched    Free    The wedge and the loop for models that do not fit: eval sets, distillation, adapter evaluation; exact weights
Local-resident    Developer machine    MLX resident (same forward loop, weights in memory), llama.cpp elsewhere    Interactive for small models    Free    Models that fit; the fast path for the loop's small-model iteration
Burst (parked)    Developer's own cloud account    vLLM    Interactive    Paid    Not on the roadmap until the loop is closed locally

4. Components
CLI (spill run | distill | eval | check | runs | adapters | tail | resume | status | models): the primary interface.
Gateway: localhost, port 11435. Ollama shapes and OpenAI shapes including /v1/batches and /v1/files.
Job engine: OpenAI-batch-compatible; JSONL in, streamed results JSONL, progress, ETA, per-row checkpoint, resumable. Batch size computed from the memory budget (KV-bound, admission by measured memory).
Run store: runs/<id>/manifest.json for every command that executes a model (command, weight fingerprint, quant, adapter hash, input hash, engine version, hardware, time span) and per-row provenance in results.
Logits output: optional per-token top-k log-probs (K up to 64) and per-row full logits for small sets, from the same loop.
Registry: tag to artifacts. bf16 safetensors (default on macOS), MLX quants, GGUF quants for the non-Apple path.
Adapters: LoRA in PEFT or mlx-lm layout, applied at the layer boundary in both engines; weights resident, base bytes untouched.
Metrics and eval: exact_match, contains, regex, json_field, judge (a second model through the same engine), script. One table, a disagreement file.
Router: tier decision from fit check against probed hardware and policy; every decision is printed with its reason.
Engines: upstream and unmodified where possible (llama.cpp); streamweights owns the streaming runner.

5. Phases and gates

Phase 0 (complete): the wedge on mmap. Result: the pipeline works end to end; mmap reaches 11 to 13% of NVMe sequential rate and OOMs from batch 32 up. Streaming runner justified by about 8 to 12x.

Phase 1 (complete): streaming runner (MLX on macOS). Gate met: 70%+ of NVMe sequential rate sustained at the auto batch size, identical greedy output to the resident engine, no DX regression.

Phases 1.5 and 1.6 (complete): measured memory calibration, correctness fixes, install path, and the 2,000-row proof run on 70B bf16 (see docs/reports/).

Phase 2 (this release): the loop. Run store, logits output, spill distill (generation and teacher-forced), adapters at inference, metrics, spill eval, file formats and spill check. Gate: spill distill --score log-probs match an independent mlx-lm reference within bf16 tolerance; base+adapter produces identical greedy output on the streamed and resident engines (20 prompts); spill eval prints one table for two models from one command with cached reuse by input hash; every run has a manifest; spill run <model> <file> with no flags is unchanged.

Phase 3: tune. Train a LoRA adapter locally against the full-precision base from distill or fine-tuning JSONL, and close the loop: data to adapter to eval table with no command outside the spill CLI. Streaming helps less here (a training step needs the forward and backward weights plus activations), so the first version trains on models that fit or on the largest layer-groupable base and states the estimated time and disk cost before it starts. Gate: on qwen2.5:0.5b, spill tune then spill eval shows the adapter beating the base on a held-out eval set; the adapter loads in spill run unchanged; resumable mid-epoch; the pre-run line states estimated time from a measured step time.

Phase 4: CPU path. The loop on machines without Metal: Linux and Intel Macs, with a portable runner (same layer-ordered reads, CPU or CUDA compute) instead of the slow llama.cpp fallback. Gate: the golden path works on a Linux box with no GPU; 0.5b output identical to the Metal engine on 20 prompts; the same manifest format.

Phase 5: mixture-of-experts. Mixtral, Qwen MoE, DeepSeek: route a batch, group tokens by expert, and read only the experts the batch needs per layer. Gate: one MoE family verified (identical to resident on 20 prompts); streamed pass time reported against dense models of equal active parameters.

6. Risks
Risk    Mitigation
Developers see "14 hours" and leave    Early rows stream immediately; the estimate is honest; small models run the whole loop in minutes
Ollama adds an offline batch mode    Their product pressure is interactive chat. The loop (distill, adapters evaluated against full-precision bases, reproducible eval tables) is the part they are not shaped to offer
bf16 on disk is huge    State it at download time with a disk check; offer 8-bit as the explicit alternative; never substitute silently
The loop outgrows the one-line DX    Every new verb must keep the no-flags golden path and the next-command line; formats stay the OpenAI shapes
Distillation or eval numbers that cannot be reproduced    The run store: manifests and per-row provenance; eval reuses runs only by exact input hash
Scope creep into a serving framework    Engines stay upstream where possible; streamweights owns CLI, gateway, jobs, registry, run store, router, and the streaming runner

7. Stack

Python 3.12, uv, FastAPI, Typer. MLX and mlx-lm on macOS. llama.cpp binaries downloaded per platform, not vendored, for non-Apple hardware.
