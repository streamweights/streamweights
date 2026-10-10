---
description: The streamweights technical plan: the loop, the developer-experience rules, the components, what is done and what comes next.
---

streamweights: technical plan (v4)

Repo and package: streamweights. Command: spill. Local by default.

v4 changes the front door. v3 made the loop (run, distill, tune, eval) the product. After Phase 3.5 the loop is one command over a folder of files, `spill build`, and the product reads as: a 70B model doesn't fit on your laptop, build your own model from it anyway. The wedge (run the unmodified full-size model over an eval set) is still the engine under it and is still a command on its own.

1. The wedge, and the loop it starts

Developers cannot test against models bigger than their machine without renting GPUs, and the moment they rent, the whole dev loop changes: data leaves the building, iteration gets a meter on it, and "just run the eval again" becomes a budget conversation. Ollama answers "what fits." Ollama Cloud answers "chat with a bigger one on our servers." Neither answers:

Run the unmodified full-size model against my eval set, on my machine, for free, overnight; data stays local by default, and remote storage is used when explicitly configured.

That is streamweights, and it is the first verb of a loop:

run     the full-precision model, optionally with an adapter, over a file
distill the teacher's completions with top-k log-probs, or its log-probs over supplied targets
tune    train a LoRA adapter on that data against the full-precision base
eval    the same eval set through several models, one table, reproducible from a manifest
export  merge an adapter into its base and ship it: safetensors, GGUF, Ollama
build   all of it from a folder: evals.jsonl, train.jsonl and/or prompts.jsonl

The eval set is the central artifact: it is what you measure a base against, what you distill from, and what decides whether a tuned adapter is better. Every command is a run against it, every run writes a manifest, and a number in a table can always be traced to the exact weights, adapter and input that produced it.

It stays a correctness tier, not a chat tier: slow per prompt, exact, free, local. A model streamed from disk pays one full weight read per forward pass whether the batch holds 1 prompt or 500, so decode is disk-bound and batching makes it practical. Prefill and training are compute-bound and scale with GPU cores (faster on Max and Ultra chips), which is why every estimate uses one cost model: decode on a streamed model is a disk pass per token step, prefill costs 2 x parameters x tokens, training costs 6 x parameters x tokens, each divided by the achieved FLOP/s that tune measured. Rows that share a system prompt compute its keys and values once.

2. Pillar: developer experience

Non-negotiable. Every new command is tested against these, in order:

One install, zero config. Install from the GitHub URL (PyPI comes after the proof run), then spill. Hardware is probed, never declared. A folder may carry a spill.json with defaults, and the shipped examples do; nothing is needed before the first result.
No new concepts on the way in. Input is JSONL in the shapes people already write: {"prompt", "expected"}, {"prompt"}, {"prompt", "answer"}, with an optional "system", or the OpenAI batch and chat shapes, auto-detected per file. Endpoints are the Ollama and OpenAI API shapes.
First result in minutes, full result overnight. Progress, tokens per second and an ETA from the first minute.
Never silently slow, never silently expensive. Before a job starts, spill states the path, the models, the quant, the per-stage time estimate from the cost model with its total, and the cost (zero), in one line. It warns when on battery.
Interruptible and resumable by default. Close the laptop, come back, spill resume. Long jobs hold the Mac awake, notify when they finish or stop, and any command reports an interrupted job on launch.
Models by name, quants by policy. bf16 is the default on the batch tier; a quant is an explicit opt-in and is always stated in output metadata. An adapter is named with +; export is the one place it is merged.
The CLI tells you the next thing. Every command ends with the one command most likely to come next.

Golden path, which must hold with no flags:

pip install git+https://github.com/streamweights/streamweights
spill example banking77 --quick && spill build banking77-quick

3. Tiers
Tier    Where    Engine    Speed    Cost    Role
Local-streamed    Developer machine or cloud box, NVMe    MLX streaming runner (macOS); PyTorch streaming runner (CPU, CUDA)    A disk pass per token step, batched    Free    Models that do not fit: eval sets, distillation, tuning; exact weights
Local-resident    Developer machine or cloud box    MLX resident or PyTorch resident (same forward loop, weights in memory); llama.cpp for explicit GGUF quants    Interactive for small models    Free    Models that fit; the student, and the loop's fast iteration
Burst (parked)    Developer's own cloud account    vLLM    Interactive    Paid    Not on the roadmap

4. Components
CLI (spill init | plan | build | report | compare | test | bundle | move | example | run | distill | tune | eval | export | check | doctor | runs | adapters | tail | resume | status | models): the primary interface.
Build: the stage planner and runner over a folder; per-role prompts (the untrained models get instructions.txt, the tuned student never does); resumable state in the folder.
Job engine: OpenAI-batch-compatible; JSONL in, streamed results JSONL, progress, ETA, per-row checkpoint, resumable. Batch size from the memory budget (KV-bound, admission by measured memory). Shared-prefix reuse: the prefix every row shares is computed once and attended to by all.
Tuner: LoRA against the full-precision base, resident on models that fit and streamed (two weight streams per step) on models that do not.
Export: merge into bf16 safetensors, GGUF through llama.cpp's own converter (downloaded, not vendored), Ollama Modelfile.
Run store: runs/<id>/manifest.json for every command that executes a model; per-row provenance in results.
Registry: tag to artifacts. bf16 safetensors (default on macOS), MLX quants, GGUF quants for the non-Apple path. Downloads check disk first, show speed and ETA, resume.
Gateway: localhost, port 11435. Ollama shapes and OpenAI shapes including /v1/batches and /v1/files.
Overnight safety: caffeinate, battery warning, notifications, interrupted-job banner.
Engines: thin. MLX on Apple silicon, PyTorch (transformers layers) everywhere else, llama.cpp upstream and unmodified for explicit GGUF quants; streamweights owns the streaming ring, the job layer (portable checkpoints, headless events) and the CLI.

5. Phases and gates

Phases 0 to 2.5 (complete): the wedge on mmap, the streaming runner, measured memory calibration, the stranger-installable release, the loop's first four verbs, and the long-tail engine fixes. Numbers are in docs/reports 001 to 007.

Phase 3 (complete): tune. Streamed LoRA matches resident LoRA within the identity gate; the 70B step time and the achieved FLOP/s are in docs/reports/008-phase3.md.

Phase 3.5 (complete): the build. spill build, spill example, spill export, shared-prefix reuse, overnight safety, spill doctor, qwen2.5:7b as the default student. Verified on the 0.5B only (docs/reports/009-phase3.5.md): banking77-quick goes from 0.200 to 0.640 in 44 s. The 70B and 7B proof runs were deferred to the full proof run below.

Run anywhere (complete): every job is a sequence of quanta (a tune step, a completed row) with a hardware-neutral checkpoint at `--state` (a path, s3://, gs://, az://); PyTorch engines for CPU and CUDA behind the MLX-shaped engine interface (transformers layers on the meta device, the same streaming ring, resident and streamed inference and tune); engine selection and `spill doctor`; headless mode (JSON-lines events, SIGTERM then exit 75), `--config` and `--emit-config`; CPU and CUDA container images; SkyPilot, Slurm and Kubernetes examples; `python -m streamweights.verify_cuda`. Verified on the 0.5B on this Mac's CPU: identity gates, cross-hardware resume in both directions, rows moved between MLX and torch-cpu (docs/reports/012-run-anywhere.md). NVIDIA is built and not yet verified.

Build anywhere (complete): `spill build` runs through the engine interface on MLX, torch-cpu and torch-cuda (selected from the hardware, `--engine` overrides) with defaults chosen from the probed machine; its whole state (stage reached, per-stage checkpoints, intermediate files, input fingerprint, and the engine, hardware, OS and numerics behind each stage) is portable at `--state`, so a build stopped on one machine or engine finishes on another; keep-awake, notifications and battery warnings work on Linux; headless mode, exit 75, `--config` and `--emit-config` work for build; `spill example banking77 --tiny` and `spill example relay`; `.github/workflows/relay.yml` starts a build on Linux and finishes it on macOS, and the reverse, on every push. Verified on the 0.5B on this Mac's CPU and on GitHub's Linux and macOS runners (docs/reports/014-build-anywhere.md). NVIDIA is built and not yet verified.

One complete workflow (complete): bring your examples, build a model, understand the result, export it, and continue the same work on another machine. `spill init` (CSV or JSONL, classification or JSON extraction, a frozen task contract, leakage-aware splits), `spill plan`, `spill build` (a small student through supervised LoRA; a teacher only on request, sequence-level), `spill report`, `spill compare`, `spill test` (the only scorer of the final test, one record per use), `spill export` (verified in llama.cpp and Transformers before the record is finalized), `spill bundle`, `spill move` and `spill resume <uri>` (a fenced handoff of one authoritative control object per run, on local disk with flock and on S3 with conditional writes). Runs, exports and tests are immutable records; stages are serializable descriptions run by a local executor or a separate process. Measured in docs/reports/015-workflow.md. Untested: real AWS S3, CUDA, network filesystems, power loss, Windows.

Deferred, each a separate future assignment: KL or logit distillation losses; expanded quantization support; teacher-quantization sweeps; generic free-text quality evaluation; new model families; garbage collection of orphaned payloads and checkpoints; remote execution services; cloud provisioning, scheduling and multi-cloud orchestration; automatic cost optimization; row-level resume inside a stopped eval or distill stage; warm-starting a new run from a parent run's adapter.

Next, in this order:

1. The verification batch: the full banking77 proof run on the 70B teacher and 7B student on the Mac (with docs/img/paths.svg drawn from its measured times); the CUDA gates on a real GPU (`docker run --gpus all ghcr.io/streamweights/spill:cuda python -m streamweights.verify_cuda`, plus the tiny build with `--engine torch-cuda`), and with them a streamed run of a model bigger than RAM on the torch engines; a cross-cloud resume demo (start a tune on one cloud's spot GPU, finish it on another's, from one bucket). Gate for the proof run: a stranger following only the README.
2. PyPI release and launch, after the verification batch: publish `streamweights`, switch the install lines to `pip install streamweights`, tag 0.1.0.
3. Phase 6, vision-language models. Qwen2.5-VL first: the vision tower stays resident (it is small and runs once per image), the decoder streams as today. A public document-image example ships with it, in the same folder shape as banking77. Gate: streamed output identical to resident execution on 20 prompts with images.
4. Phase 7, mixture-of-experts: Mixtral, Qwen MoE, DeepSeek. Route a batch, group tokens by expert, read only the experts the batch needs per layer. Gate: one MoE family verified.

Also open: batched prefill GEMMs. Prefill is one sequence at a time inside each layer; batching rows into one GEMM per layer raises the achieved FLOP/s that every estimate divides by, and shortens teacher stages.

6. Risks
Risk    Mitigation
Developers see "14 hours" and leave    Early rows stream immediately; the estimate is honest and per stage; the quick example runs the whole loop in under an hour
Ollama adds an offline batch mode    Their pressure is interactive chat. The loop (distill, adapters evaluated against full-precision bases, reproducible eval tables, one command from a folder) is not what they are shaped to offer
bf16 on disk is huge    State it at download time with a disk check; offer 8-bit as the explicit alternative; never substitute silently
bf16 output is not bit-invariant to batch shape    Say so in the reports; compare against the noise floor of two unshared runs, not against zero
Distillation or eval numbers that cannot be reproduced    The run store: manifests and per-row provenance; eval reuses runs only by exact input hash
Scope creep into a serving framework    Engines stay upstream where possible; streamweights owns CLI, gateway, jobs, registry, run store, router, build, and the streaming runner

7. Stack

Python 3.10 or newer (developed on 3.12), uv, FastAPI, Typer. MLX and mlx-lm on Apple silicon; PyTorch, Hugging Face transformers, PEFT, safetensors and fsspec on every platform (s3fs, gcsfs and adlfs as the `cloud` extra). llama.cpp binaries downloaded per platform, not vendored, for explicit GGUF quants and for GGUF conversion.
