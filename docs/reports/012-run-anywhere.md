# Run anywhere: portable jobs, a PyTorch engine, scheduler-ready (directive 012)

Measured on one machine: an Apple M4 Pro, 48 GB, macOS, Python 3.12.13, torch 2.14.1, transformers 5.19.0, PEFT 0.21.2, MLX 0.32.3. Model: qwen2.5:0.5b (bf16 safetensors), and a float32 copy of it for the float32 gates. Every number below is from `docs/reports/012-gates.json`, written by `scripts/gates_012.py` (the bf16 gradient statistics were measured under transformers 5.18.0, the rest under the version above; the file records every run). The torch engine ran on the CPU of this Mac; no CUDA device was available, so the NVIDIA engine is built and not verified (`python -m streamweights.verify_cuda` is the verification, see `docs/linux.md`).

## Gates, item 7: identity on the 0.5B

| gate | result | detail |
|---|---|---|
| torch streamed vs torch resident, CPU, float32, 20 prompts | **pass** | greedy identical 20 of 20 (581 tokens compared); maximum log-prob difference 0 (limit 1e-5), top-5 0 |
| torch vs MLX, float32, 20 prompts | **pass** | greedy identical 20 of 20; **maximum log-prob difference 0.000182** over 581 tokens |
| torch streamed tune vs PEFT resident tune, CPU, float32, 50 steps, natural text | **pass** | loss within 8.4e-07 relative per step (limit 0.1%); per-tensor adapter cosine minimum 0.999996 over 336 tensors (floor 0.9999) |
| torch tune vs MLX tune, float32, 50 steps, natural text | **pass** | loss within 6.1e-05 relative per step (limit 1%); adapter cosine minimum 0.999996 |
| torch tune vs MLX tune, float32, 50 steps, toy task, then `spill eval` on 60 held-out rows | **pass** | loss within 0.002 relative per step; scores: base 0.000, torch adapter 1.000, MLX adapter 1.000 |

Settings: LoRA rank 16 on every linear (alpha 32), micro-batch 4, max sequence 256, AdamW with a cosine schedule; learning rate 2e-5 on `examples/gate-natural.jsonl` (the loss falls from 1.4 to 0.3 in 100 steps, so relative loss differences mean something) and 5e-6 on the toy task (slower, so the loss is still moving at step 50 and the held-out score goes from 0 to 1). The streamed torch tune, the PEFT resident tune and the MLX streamed tune start from the same numpy initialisation and use the same optimizer arithmetic (AdamW without bias correction, decoupled decay first, cosine read at the step count before counting), which is what makes a checkpoint mean the same thing on every engine. Loss curves are in the JSON.

## bf16: statistics, no identity claim

Greedy agreement on the first 10 of the 20 prompts with 24 new tokens, next to batch-shape noise (the same engine and numerics, batch 1 against the whole set):

| comparison | rows identical | max log-prob difference |
|---|---|---|
| torch bf16 vs torch float32 | 9 of 10 | 0.190 |
| MLX bf16 vs MLX float32 | 9 of 10 | 0.132 |
| torch bf16 vs MLX bf16 | 10 of 10 | 0.321 |
| batch-shape noise: MLX bf16, batch 1 vs all | 8 of 10 | 0.158 |
| batch-shape noise: torch bf16, batch 1 vs all | 10 of 10 | 0.073 |

Gradients, Phase 3's check: one micro-batch of 4 examples, 192 tokens, same parameters, per-tensor cosine to the float32 gradient:

| comparison | mean cosine | minimum cosine | worst tensor |
|---|---|---|---|
| torch bf16 vs torch float32 | 0.9736 | 0.7021 | layers.0.self_attn.q_proj.lora_a |
| MLX bf16 vs torch float32 | 0.9884 | 0.7424 | layers.0.self_attn.q_proj.lora_a |
| torch bf16 vs MLX bf16 | 0.9870 | 0.8054 | layers.0.self_attn.q_proj.lora_a |
| batch-shape noise: MLX bf16, one batch vs two halves | 0.9899 | 0.8758 | layers.4.self_attn.k_proj.lora_a |
| batch-shape noise: torch bf16, one batch vs two halves | 1.0000 | 1.0000 | layers.2.mlp.down_proj.lora_a |

In bf16 the two engines are as far from float32 as each is from the other, and the MLX engine moves by about as much when only the batch shape changes. The torch CPU engine shows no batch-shape noise in the gradient: its rows are computed independently, so one batch and two halves agree to 13 digits. That is a property of CPU kernels and says nothing about CUDA.

## Gates, item 8: jobs that change hardware

Training: 100-step schedule, stopped at step 50 and checkpointed to a `--state` directory, continued to step 100 on the other engine. The reference is the uninterrupted 100-step run on the first engine; the noise is the distance between two clean 100-step runs, one on each engine (the first segment runs MLX bf16 on the Apple GPU, the second torch-cpu float32, or the reverse, and the checkpoint records both). The continued curve must be no farther from the reference than 1.5 times that noise plus 1%, and the final adapter must score within the larger of the clean-pair difference and 2 of 60 rows of the reference's score.

| direction | data | result | resumed vs reference | clean pair (noise) | limit | held-out score: resumed / reference / other engine |
|---|---|---|---|---|---|---|
| MLX 50 steps, then torch-cpu 50 | natural text | **pass** | 0.004 | 0.016 | 0.034 | n/a |
| MLX 50 steps, then torch-cpu 50 | toy task | **pass** | 0.065 | 0.621 | 0.941 | 1.000 / 1.000 / 1.000 |
| torch-cpu 50 steps, then MLX 50 | natural text | **pass** | 0.006 | 0.016 | 0.034 | n/a |
| torch-cpu 50 steps, then MLX 50 | toy task | **pass** | 0.098 | 1.718 | 2.587 | 1.000 / 1.000 / 1.000 |

The numbers are the mean relative loss difference over steps 51 to 100. On the toy task the loss is near zero late in the run, so its relative differences are large for everyone, which is why the limit is stated against the clean pair. The history recorded in each checkpoint, for the MLX-then-torch run on natural text: steps 0 to 50 on mlx_stream_tune (apple-gpu:Apple M4 Pro, bf16); steps 50 to 100 on torch_stream_tune (cpu:Apple M4 Pro, float32).

Rows: `spill eval` over 60 held-out rows with `--state`, stopped after 30 on one engine (`--stop-after 30`) and finished on the other, in both directions.

| direction | result | rows | unique | missing | duplicated | rows by engine |
|---|---|---|---|---|---|---|
| MLX then torch-cpu | **pass** | 60 | 60 | 0 | 0 | mlx_resident: 30, torch_resident: 30 |
| torch-cpu then MLX | **pass** | 60 | 60 | 0 | 0 | mlx_resident: 30, torch_resident: 30 |

Every result row carries `streamweights.engine`, `streamweights.hardware` and `streamweights.numerics`; every pushed segment of state records the same.

The committed fixture (`streamweights/data/fixtures/resume-qwen05`, 3.3 MB: rank 4 on `q_proj` and `v_proj`, a 100-step schedule stopped at step 50 on MLX, bf16 base, Apple GPU) resumes on torch-cpu: continued from step 51 to step 100 on torch-cpu float32, mean relative loss difference to the uninterrupted MLX run 0.023 and maximum 0.117 on this Mac (`G.gate_resume_fixture`, 50 steps at 0.65 s per step); the same continuation inside the CPU container on a GitHub x86_64 runner had a maximum of 0.110. The checkpoint's history records both producers: steps 0 to 50 on mlx_stream_tune (apple-gpu, bf16), steps 50 to 100 on torch_stream_tune (cpu, float32).

## Measured: the torch-cpu streamed pass on the 0.5B

160 rows (the 20 sample prompts eight times, 64 new tokens at most), 50 passes, on an otherwise idle machine, float32 compute (this CPU's bf16 linear layers measured about six times slower than float32, so the engine chooses float32 and says so in the pre-run line):

| weights on disk | streamed: median pass | read rate while reading | tokens/s | resident: median pass | tokens/s |
|---|---|---|---|---|---|
| bf16 files (0.99 GB), converted to float32 per layer | **0.235 s** | 11.2 GB/s | 178.8 | 0.157 s | 211.2 |
| float32 files (1.98 GB) | **0.202 s** | 20.4 GB/s | 177.5 | 0.157 s | 186.5 |

In the 20-prompt identity gate (float32 files) the median streamed pass was 0.090 s and the resident pass 0.065 s. The model is small enough that the file cache serves most reads even with `F_NOCACHE`, so the read rate is a ceiling for this machine, not a disk figure; the 70B number is the disk's.

## CI and containers

Branch `anyhw`, commit e00f1c6: CI green on macos-14 / Python 3.12 (265 passed, 3 skipped), ubuntu-latest / Python 3.12 and 3.10 (158 passed, 8 skipped each; MLX-only test files are not collected on Linux). The Linux jobs install the CPU build of PyTorch, download qwen2.5:0.5b (cached) and run `tests/test_fixture_resume.py` with `SPILL_REQUIRE_FIXTURE_MODEL=1`, so the step-50 MLX checkpoint is resumed on torch-cpu in every Linux run.
The first push failed on both: `tests/tinytorch.py` imported an MLX test helper (Linux), and transformers 5.19.0, newer than the 5.18.0 developed against, removed the `device` argument of the rotary embedding constructors (macOS). Both fixed; the engine now builds the rotary embedding either way and the package requires transformers 5.18 up to 6.
Containers workflow on the branch (builds, smoke-tests, never publishes from a branch): the CPU image built and the smoke test passed (a 20-row eval on qwen2.5:0.5b with 20 `row` events, all stamped `torch-cpu`; the step-50 fixture resumed on torch-cpu from step 51 to 100); the CUDA image built. The first run's smoke test also passed but its cleanup could not delete root-owned files; fixed.

## Decisions

- Headless is automatic when stdout is not a TTY, for the job commands (run, distill, tune, eval, resume). The existing test suite parses human output through a non-TTY, so `tests/conftest.py` sets `SPILL_HEADLESS=0`; the headless tests clear it. `SPILL_HEADLESS=0` or `1` forces it either way.
- SIGTERM and SIGINT exit 75 in headless mode. At a terminal the old behavior stays (finish the step, Ctrl-C again aborts, exit 130). The 30-second budget is 15 seconds for the quantum to finish, then it is abandoned (SIGALRM raises in the main thread), then the checkpoint is written; the optimizer update and checkpoint writes are protected from the abandon. On platforms without SIGALRM the quantum is finished.
- `--stop-after N` (hidden) stops a job cleanly after N steps or rows. It exists so the cross-hardware gates can stop a 100-step job at step 50 (a 50-step job would have a different cosine schedule) and stop an eval after half the rows, and it exits 0.
- Fixed a bug found on the way: a command that failed inside a Typer command exited 0, because `main()` dropped the exit code Typer returned. Errors now exit 1, and exit 75 reaches the scheduler.
- The torch optimizer reproduces MLX's AdamW exactly (no bias correction, decay applied first, cosine read before the step is counted) because a checkpoint has to mean the same thing on every engine, and the LoRA initialisation moved to numpy (`init_params_np`) for the same reason. The MLX LoRA init values therefore differ from Phases 3 and 3.5 (same distribution, different numbers); earlier reports are unaffected, they recorded their own runs.
- The tune checkpoint was already float32 safetensors; it is now written through a commit-marker protocol (COMMIT last, two kept) that works on a local path and on an object store, and `state.json` carries the loss history so a job continued on a new machine has its whole curve. A different `--steps` is refused on resume because the cosine schedule is part of the checkpoint's identity.
- Row state is pushed in segments (25 rows or 60 seconds, and at stop) rather than rewriting one results file, so the cost of a push does not grow with the job. An eval keeps one state sub-directory per model.
- On CPU the engine chooses bf16 only when a measured linear layer is at least 0.6 times as fast as float32. The first probe used a 2-D matmul, which is pathologically slow in bf16 on this CPU and misjudged by 100x; it now measures the op the layers run (a linear layer, about 6 times slower in bf16 here), so this Mac runs float32 and says so.
- The gates compute in float32 on a float32 copy of the 0.5B for MLX and torch alike, with the learning rates recorded in the report (2e-5 on natural text, 5e-6 on the toy task; 2e-6 on the toy task left the held-out score at 0 for every model, which would have made the score gate vacuous). Noise for the resume gate is defined against the clean pair of runs on the two engines, stated in the report. At learning rate 3e-3 the tiny test model trains chaotically and amplifies rounding, so the tiny-model tests use 3e-4.
- bf16 gradient statistics use the first 48 tokens of four examples: a 2-D bf16 matmul on this CPU is very slow, and the backward pass uses it.
- The fixture lives in the package (`streamweights/data/fixtures/resume-qwen05`) so that `python -m streamweights.verify_cuda` inside the container can resume it; it is bound to the weight fingerprint of the registry's Qwen/Qwen2.5-0.5B-Instruct files, so a changed upstream file fails loudly rather than resuming silently.
- `spill build` stays Apple silicon only: the directive did not ask for it and its one-line message is tested in CI. Everything it calls (run, distill, tune, eval, export) now runs on the torch engines.
- llama.cpp remains for explicit `--quant Q8_0` or `Q4_K_M`; without a GGUF quant every platform runs the PyTorch or MLX engines. Quantized MLX checkpoints (`--quant 8bit`) are MLX only, and the torch engines refuse them in one line.
- Dropout on the torch engines draws its mask from a seed derived from (job seed, micro-batch, layer, module), which is stateless and resumable, but it is not the same mask as MLX's, so a job with dropout is not bit-comparable across engines (the default is zero).
- The CUDA path is written to the design (pinned buffers, a side stream, a small pool of device buffers) and is untested: no GPU was available. Its CPU-side logic is shared with the tested path; the stream and event code is not exercised until `verify_cuda` runs. `tests/test_cuda.py` skips without `SPILL_GPU_TESTS=1` and a device.
- The Linux read modes (`fadvise`, `odirect`) are exercised by `tests/test_ring.py` on the Linux CI runners and measured against each other per model by the calibration; on macOS only `nocache` can be measured. `O_DIRECT` is skipped, not failed, on a filesystem that refuses it.
- SkyPilot syntax was checked with SkyPilot 0.14.0's own parser offline; kubectl and sbatch are not installed here, so Kubernetes manifests were checked against the kubernetes-validate schemas for 1.30 (strict) and the Slurm script with `bash -n` plus a check of every directive. `examples/schedulers/validation.md` lists each check and each one that did not run.
- The CUDA image base is `pytorch/pytorch:2.8.0-cuda12.6-cudnn9-runtime` (the official PyTorch CUDA runtime image); CI builds it. The CPU image is `python:3.12-slim` with the CPU build of PyTorch. Docker is not installed on this Mac, so both were built and the CPU one smoke-tested only in CI.
- The CI download of the 0.5B needed the 20 GB free-disk floor lowered; `SPILL_MIN_FREE_GB` (default 20) sets it, and only CI sets it.

## What remains for the verification batch

- CUDA gates on a real GPU: `docker run --gpus all ghcr.io/streamweights/spill:cuda python -m streamweights.verify_cuda`, then the first CUDA numbers (pass time, read rate, tokens per second, tune step time) into the README and `docs/linux.md`.
- A model bigger than RAM on the torch engines, to measure the ring's fraction of the sequential disk rate on Linux (fadvise against O_DIRECT) and the CPU streamed rate; this directive allowed only the 0.5B.
- The banking77 proof run on the 70B teacher and 7B student on the Mac, and with it `docs/img/paths.svg`.
- A cross-cloud resume demo: start a tune on one cloud's spot GPU, finish it on another's, from one bucket, with `--state s3://` or `gs://` (the object-store path was tested through fsspec's local and in-memory filesystems only).
- `spill build` on the torch engines; two cloud CPU types with measured times; `io_uring` if the ring leaves a fast disk idle.
- Then PyPI and launch, vision-language models, and mixture-of-experts, in that order.
