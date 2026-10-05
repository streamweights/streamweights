TERMINAL DIRECTIVE: CLEAN, MERGE, AND MAKE THE REPO USABLE BY ANYONE TODAY. SMALL-MODEL VERIFICATION ONLY. NO PYPI.

Rules, absolute:

Never print a status summary or progress report before the terminal state; printing one ends your turn and stops all work. Until the end, output only tool calls.
Never end the turn to wait. Any wait is a foreground shell loop under the tool timeout. No background tasks that rely on waking you, no scheduled wake-ups, no /loop, no questions to the user.
If something blocks you, make the most reasonable choice, log it under Decisions, and continue.
No Claude attribution on any commit. No em-dashes in any file you write.
No 70B or 7B runs, and no model downloads over 2 GB. GPU use is limited to qwen2.5:0.5b. The user is working on this machine.
No PyPI publishing, no release workflow, no tags. Install lines use the GitHub URL.

1. Start.

Confirm no spill process is running.
Kill any leftover git fetch poll loop from earlier sessions.
Save this paste set verbatim as docs/paste-sets/010-cleanup.md on main and commit.

2. Finish Phase 3 in ../streamweights-phase3:

Commit the untracked smoke-test files and state/calibration.json.
Run scripts/verify_phase3.py toy: resident and streamed adapters on the 0.5B in bf16, then score the base and both adapters with spill eval on held-out rows.
Write docs/reports/008-phase3.md from measured results only, with no placeholders. Include:
the float32 identity gate, the bf16 deviation and why;
the gradient check: streamed mean 0.9966 and minimum 0.952, versus mlx-lm 0.9810 and 0.666;
the dtype audit and the toy results;
the 70B smoke test: step times, 4.55 TFLOP/s, about 11 trained tokens/s, roughly 400,000 tokens per 10-hour night, 24.2 GB peak;
estimate accuracy;
Decisions, including the stalled previous session.
Squash-merge phase3 into main, remove the worktree, delete the branch, and push.

3. Merge Phase 3.5 in ../streamweights-phase35:

Rebase onto main, resolving in favor of Phase 3's engine and tune code. Wire tune into build.
Link the 70B weights read-only from the main checkout; do not make a second copy.
Run the full CPU suite; it must be green.
Run the 0.5B GPU checks:
prefix reuse A/B on 20 rows, with differences reported next to ordinary batch-shape noise;
spill export qwen2.5:0.5b+<adapter> --gguf: the merged model's greedy output must equal base+adapter on 20 prompts, and the GGUF must run in llama.cpp;
spill example banking77 --quick && spill build banking77-quick to completion, keeping its real score table and wall time.
Squash-merge phase35 into main, remove the worktree, delete the branch, and push.

4. Clean the command line.

spill --help lists commands in loop order with one line each: build, example, run, distill, tune, eval, export, then models, adapters, runs, status, tail, resume, doctor, check.
Every command's --help ends with one real example.
Remove dead flags, unused options, and development-only behavior from user-facing commands.
Every error is one line ending in the recovery command, with no traceback unless --debug.
Next-command hints and pre-run lines use the same wording and format across commands.
Run each command's --help and one real invocation on the 0.5B, and fix any inconsistency.

5. Platforms.

Make installation succeed on Linux, Windows, and Intel Macs: MLX dependencies only when sys_platform == "darwin" and platform_machine == "arm64", with every MLX import guarded.
On those platforms, spill doctor and every MLX-only command say in one line:
what works today: run through llama.cpp, export, check, models;
what doesn't yet: streaming, distill, tune, build;
a pointer to the README section.
Verify in a Linux container (docker run python:3.12-slim if Docker is available; otherwise a clean venv with the MLX marker forced off):
pip install from the local checkout succeeds;
spill doctor runs;
an MLX-only command prints the one-line message.

6. Packaging and CI (no publishing).

In pyproject.toml:
version 0.1.0;
the README as the long description;
keywords, classifiers (macOS, Apple silicon, LLM, fine-tuning, MLX), project URLs;
license Apache-2.0, Python 3.10 or newer;
scripts/ excluded from the package.
Build with python -m build, check with twine check, and install the built wheel in a clean venv to confirm spill --help and spill doctor.
Add a CI workflow that runs the CPU test suite on macOS and Linux for pushes and pull requests.

7. Clean the repository.

Remove stale worktrees, merged branches, scratch files, and abandoned job directories.
.gitignore covers runtime state.
One test suite runs with pytest, green; GPU tests are skipped unless SPILL_GPU_TESTS=1.
CLAUDE.md reflects the current architecture and the never-stop rules above.
Set the repo description to: "Build your own model on your Mac: fine-tune and distill from a 70B teacher locally with MLX, even when the model is bigger than your RAM."
Set these 20 topics: mlx, apple-silicon, llm, fine-tuning, lora, distillation, knowledge-distillation, local-llm, llama, qwen, llm-evaluation, macos, on-device-ai, batch-inference, model-training, huggingface, ollama, gguf, evaluation, llm-training.
Enable Discussions.

8. Linux to-do.

Write docs/linux.md: what works on Linux today, what doesn't, and the work to enable it, in this order:
A CPU engine with the same layer-ordered disk streaming: PyTorch CPU or MLX's CPU backend, whichever measures faster; a pread ring with O_DIRECT or posix_fadvise, and io_uring if it helps.
Inference parity with the Mac path: the same job engine, batching, refill and budget, with an identity test against the Mac results on the 0.5B.
distill and eval on CPU.
tune: a resident path through PEFT on CPU for small models, and the streamed backward on CPU.
build end to end on CPU.
Measurement on two cloud machine types, a modern server CPU with AMX and an ordinary AVX2 instance, with the README table extended.
NVIDIA GPUs: the same ring with pinned host memory and CUDA streams, as a later phase.
Open a GitHub issue titled "Linux support (CPU first, then NVIDIA)" with that checklist, labeled enhancement and help wanted, and link it from the README and from docs/linux.md.

9. Docs. Rewrite the README per item 13 of docs/paste-sets/009-phase3.5-build.md, with these differences:

The Quick start table is the real banking77-quick result from item 3. Install uses pip install git+https://github.com/streamweights/streamweights and uv tool install git+https://github.com/streamweights/streamweights.
The 70B and 7B banking77 table is not shown yet. One line in Status says it arrives with the full proof run.
Times in the paths table come only from measured numbers. Where a path hasn't been measured, state the rate it depends on instead of a total:
70B eval pass time, from Phase 2.5;
70B training tokens per night, from Phase 3;
7B training at the measured 4.6 TFLOP/s.
Never write a total that wasn't run.
Add a section Linux and other platforms, a few lines covering:
what works today and what doesn't;
that a CPU engine is the next phase, then NVIDIA;
links to docs/linux.md and the tracking issue;
an invitation to comment there with the hardware they would run it on.

Also:

Build flow.svg. Build paths.svg only if it can be drawn from measured numbers; otherwise omit it and leave a task for tonight.
docs/formats.md, docs/cli.md (generated from the real --help output) and docs/models.md must match the code.
In docs/plan.md, mark Phases 3 and 3.5 done. Next in order:
the full banking77 proof run (70B teacher, 7B student);
Phase 4, Linux CPU engine, per docs/linux.md;
Phase 5, NVIDIA;
Phase 6, VLMs;
Phase 7, MoE.
PyPI release is listed as a step after the proof run.
A docs test fails if any placeholder or unmeasured number reaches main.

10. Fresh-install verification. In a new temporary directory with a clean venv, run:

pip install git+https://github.com/streamweights/streamweights
spill doctor
spill example banking77 --quick && spill build banking77-quick
spill export the result to GGUF

Record the full transcript with timestamps as docs/reports/010-fresh-install.txt. Anything a stranger would need that the tool didn't print is a defect: fix it, re-run, and record the final clean transcript.

11. Terminal state.

Commit and push.
Confirm that main on GitHub has only the main branch, green CI on macOS and Linux, and no worktrees.
Only now print:
the repo URL and the Linux tracking issue URL;
the banking77-quick score table and wall time;
the fresh-install time from install to the end of build;
any defect found and fixed;
what is left for tonight's long run.
Stop.
