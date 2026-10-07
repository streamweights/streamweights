TERMINAL DIRECTIVE: RUN ANYWHERE. PORTABLE JOBS, A PYTORCH ENGINE (CPU AND CUDA), SCHEDULER-READY, VERIFIED ON 0.5B.

Rules, absolute:

Never print a status summary or progress report before the terminal state; printing one ends your turn and stops all work. Until the end, output only tool calls.
Never end the turn to wait. Any wait is a foreground shell loop under the tool timeout. No background tasks that rely on waking you, no scheduled wake-ups, no /loop, no questions to the user.
If something blocks you, make the most reasonable choice, log it under Decisions, and continue.
No Claude attribution on any commit. No em-dashes in any file you write.
Only qwen2.5:0.5b (and tiny random test models) may run. No 7B or 70B runs, no downloads over 2 GB, no PyPI publishing, no tags.
Work on a branch anyhw in the main checkout, and merge to main at the end. No other session is running.

1. Start. Save this paste set verbatim as docs/paste-sets/012-run-anywhere.md, then commit. Append to CLAUDE.md:

The design principle: every job is a sequence of normalized quanta. For training, the quantum is one optimizer step. For run, distill and eval, it is one completed row. Any machine with a supported engine can execute the next quantum from a portable checkpoint.
Engines are thin: MLX on Apple silicon, PyTorch everywhere else (CPU and CUDA). Streamweights owns only the streaming ring, the job layer, and the CLI.

Commit after every numbered item.

2. Portable job state.

Define one hardware-neutral checkpoint format:
adapter weights and optimizer state as float32 safetensors;
state.json with step, seed, data cursor, RNG state per data stream, model id and weight fingerprint, adapter config, hyperparameters, and the engine, hardware and numerics that produced each step range;
for row jobs, results.jsonl plus the cursor of completed row ids.
Writes are atomic (temp then rename locally; write-then-commit-marker on object storage).
Any engine reads any engine's checkpoint.

3. Storage.

--state <uri> accepts a local path, s3://, gs://, or az:// through fsspec.
Jobs resume from the latest committed checkpoint at that URI automatically.
Weights cache per machine under SPILL_HOME, with an optional --weights <uri> to stage from shared storage instead of Hugging Face.
Test with a local path and with fsspec's in-memory filesystem.

4. Headless mode.

Auto-enabled when stdout is not a TTY, or with --headless.
JSON-lines events on stdout: start, step or row, checkpoint, preempted, done, error. Each carries step, loss, tokens/s, peak memory, ETA, and engine.
No progress bars, no notifications, no caffeinate.
On SIGTERM or SIGINT, finish or abandon the current quantum, write a checkpoint within 30 seconds, and exit with code 75 so schedulers retry.
On start, resume from --state if a checkpoint exists.
All configuration also loadable from --config job.json. spill <cmd> ... --emit-config job.json writes the exact config of a CLI invocation, so a laptop run can be submitted to a cluster unchanged.

5. PyTorch engine. Add engines/torch_resident.py and engines/torch_stream.py behind the existing engine interface.

Model classes: use Hugging Face transformers for every verified family (Llama, Qwen2, Qwen3, Mistral, Phi 3, Gemma 2). Construct on the meta device, then bind each layer's weights from the ring as it is needed.
The ring: the same safetensors layer index. On Linux, a pread ring with posix_fadvise(DONTNEED) or O_DIRECT, whichever measures faster. On macOS, F_NOCACHE for local testing.
CUDA: pinned host buffers, async host-to-device copies on a side stream, and compute on layer k while k+1 to k+N load.
CPU: compute directly from host buffers.
Feature parity with the MLX engines:
inference with KV cache, batching, continuous refill, and the 75% memory budget (device memory on CUDA, RAM on CPU);
shared-prefix reuse;
log-probs, distill (both modes), eval, adapters at inference, export;
tune, resident through PEFT, and streamed with the same saved-input, reverse-stream, recompute-and-VJP design as the MLX streamed tune.
Precision: adapters and optimizer state in float32, base weights in bf16 on CUDA. On CPU, base weights in bf16 where the CPU supports bf16 matmul, otherwise float32, and say so in the pre-run line.

6. Engine selection.

Automatic: Apple silicon uses MLX, a visible CUDA device uses torch-cuda, otherwise torch-cpu. --engine mlx|torch-cpu|torch-cuda overrides.
spill doctor shows the chosen engine, why, and the measured rate for each available engine.
Dependencies:
MLX packages only on Apple silicon;
torch, transformers, peft, safetensors and fsspec on all platforms;
s3fs and gcsfs as extras: pip install "streamweights[cloud]".

7. Identity gates on the 0.5B and tiny models. All must pass before item 8.

Torch streamed vs torch resident, CPU, float32: greedy output identical on 20 prompts; log-probs within 1e-5.
Torch vs MLX on this Mac, float32: greedy identical on 20 prompts; report the maximum log-prob difference.
Torch streamed tune vs PEFT resident tune, CPU, float32, 50 steps: loss within 0.1% per step; per-tensor adapter cosine above 0.9999.
Torch tune vs MLX tune, float32, 50 steps: loss within 1% per step; both adapters score within noise with spill eval on the toy task.
bf16: report the same comparisons as statistics next to batch-shape noise, as Phase 3 did. No identity claim.

8. Cross-hardware resume gates.

Training:
Tune the 0.5B for 50 steps on MLX and checkpoint.
Resume on torch-cpu for 50 more steps.
Compare against an uninterrupted 100-step MLX run: the loss curve continues within noise, and the final adapters score within noise on held-out rows.
Repeat in the reverse direction.
Rows:
Start spill eval on MLX and stop after half the rows.
Finish on torch-cpu.
Confirm no row is missing or duplicated, and each row records the engine that produced it.
Commit a small fixture (the 0.5B MLX checkpoint at step 50, a few MB) so Linux CI resumes it on torch-cpu in every run.

9. Containers.

Two Dockerfiles: docker/cpu.Dockerfile (python slim plus CPU torch) and docker/cuda.Dockerfile (official PyTorch CUDA runtime base).
Both have entrypoint spill, default headless, and SPILL_HOME on a volume.
A GitHub Actions workflow builds both on pushes to main and publishes them to ghcr.io/streamweights/spill:cpu and :cuda.
CI smoke-tests the CPU image: a 20-row eval on the 0.5B, plus resuming the item 8 fixture.
The CUDA image is built only.

10. Scheduler examples under examples/schedulers/, each a short README plus files. Validate every file offline (YAML syntax, sbatch --test-only if available, kubectl apply --dry-run=client if available; otherwise schema checks) and log what could not be validated.

skypilot/: a managed-job task that:
accepts any of H100, A100, L4, A10G, or CPU-only;
uses spot instances with automatic recovery;
mounts a bucket for --state and weights;
runs spill tune headless from an emitted config.
slurm/: an sbatch script with --requeue and a SIGTERM handler, using a shared filesystem for state.
kubernetes/: a Job manifest with a PVC for state and weights, restartPolicy: OnFailure, and the CUDA image with a CPU variant.

11. CUDA verification for later.

Write scripts/verify_cuda.py, which runs on any CUDA machine:
the item 7 gates (torch-cuda against torch-cpu and resident references, on the 0.5B);
the item 8 resume gate from the committed MLX fixture;
a streamed-inference throughput measurement (pass time, achieved read rate, tokens/s);
a streamed-tune step-time measurement.
It prints a pass/fail report and writes a JSON file.
Document the one-liner in docs/linux.md: docker run --gpus all ghcr.io/streamweights/spill:cuda python -m streamweights.verify_cuda. Make the script part of the package so this works.

12. Docs.

README:
add a short section Runs anywhere: the same command on a Mac, a Linux box, or any cloud GPU; jobs move between machines and resume; three lines on headless mode, containers, and the scheduler examples;
update the platforms section to the truth after this run (CPU verified, NVIDIA built but awaiting verification, with the one-liner);
no unmeasured numbers.
New docs:
docs/portability.md: what moves, what a quantum is, numerics across hardware, weight staging cost;
docs/schedulers.md: the three examples and headless events.
Update docs/linux.md and docs/cli.md.
docs/plan.md:
run-anywhere functionality marked done;
next, the verification batch: banking77 70B proof on the Mac, CUDA gates on a real GPU, a cross-cloud resume demo;
then PyPI and launch;
then VLMs, then MoE.
Issue #1: update its checklist to what is done and what remains, without closing it.
The docs test must pass.

13. Tests.

The full CPU suite is green on macOS and on Linux CI. New tests cover:
checkpoint round-trips across engines;
fsspec storage;
headless events and the SIGTERM exit code;
config emit and load;
engine selection;
the gates in items 7 and 8 that run on CPU.
GPU tests are skipped unless SPILL_GPU_TESTS=1.

14. Terminal state.

Squash-merge anyhw into main and push.
Confirm CI is green on macOS and Linux and the CPU container smoke test passes.
Only now print:
each gate in items 7 and 8 with its result;
the measured torch-cpu streamed pass time on the 0.5B;
the container image URLs;
which scheduler files were validated and how;
Decisions;
what remains for the verification batch.
Stop.
