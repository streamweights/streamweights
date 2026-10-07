TERMINAL DIRECTIVE: spill build ON EVERY ENGINE, A BUILD THAT MOVES BETWEEN MACHINES, AND PROOF ANYONE CAN SEE.

Rules, absolute:

Never print a status summary or progress report before the terminal state; printing one ends your turn and stops all work. Until the end, output only tool calls.
Never end the turn to wait. Any wait is a foreground shell loop under the tool timeout. No background tasks, no scheduled wake-ups, no /loop, no questions to the user.
If something blocks you, make the most reasonable choice, log it under Decisions, and continue.
No Claude attribution on any commit. No em-dashes in any file you write.
Only qwen2.5:0.5b and tiny random test models may run. No 7B or 70B runs, no downloads over 2 GB, no PyPI publishing, no tags, no public posts.
Work on a branch build-anywhere and merge to main at the end.

1. Start. Save this paste set verbatim as docs/paste-sets/014-build-anywhere.md. Commit after every numbered item.

2. build through the engine interface.

Route every build stage (base eval, distill, tune, eval, export) through the engine interface, with no direct MLX imports in build.
Engine selection works as everywhere else: MLX on Apple silicon, torch-cuda with a visible GPU, torch-cpu otherwise. --engine overrides.
Every engine-specific path in build behaves identically on MLX and PyTorch: prefix reuse, per-role label-list handling, adapter formats, and chaining.

3. Hardware-aware defaults. build picks its default student and teacher from the probed machine and the engine's calibrated rates:

Apple silicon: a qwen2.5:7b student and a llama3.3:70b teacher.
CUDA: the same defaults when device memory and disk allow; otherwise the largest that fit.
CPU: a qwen2.5:0.5b student, and the largest teacher whose estimated distill time for the folder's prompts is under 12 hours.

The pre-run line states the choice and why. If the user forces a model whose estimate exceeds 24 hours, state the estimate and proceed. Never refuse. Calibration is stored per engine and model.

4. Portable build state.

build state (the stage reached, per-stage checkpoints, intermediate files, and the input fingerprint) uses the portable checkpoint format from directive 012.
--state <uri> works for build as it does for tune and eval.
A build stopped on one engine or machine resumes on another from the stage and quantum where it stopped.
Each stage records the engine, hardware, OS, and numerics that produced it, and the final table shows them.

5. No Mac-only plumbing in the path.

Keep-awake: caffeinate on macOS; systemd-inhibit on Linux when available, otherwise skip silently; never in headless mode.
Notifications: osascript on macOS; notify-send on Linux when available, otherwise skip. --notify <url> works everywhere.
Battery warning: pmset on macOS; /sys/class/power_supply on Linux when present.
Headless mode, SIGTERM exit code 75, --config, and --emit-config all work for build.

6. Examples.

spill example banking77 --tiny: 20 evals, 100 training rows, a qwen2.5:0.5b student, short sequences. Size it so a full build finishes on a CI Linux runner in under 15 minutes, and measure it.
spill example relay: creates ./relay/ from the tiny banking77 files, plus a README.md and a script relay.sh.
relay.sh starts spill build relay --state ./relay-state, stops it partway through the tune stage (--stop-after), then finishes it on the second machine using one of three modes, chosen in this order:
Docker installed: docker run the ghcr.io/streamweights/spill:cpu image with the folder and state mounted, and run spill resume inside the Linux container.
No Docker: spill resume --engine with the other local engine, with one printed line saying this is a different engine on the same machine.
--two-machines: print exactly what to copy or which shared path or bucket to point --state at, and the one command to run on the other machine.
It ends by printing the final table, with the machine, OS, and engine that produced each stage, and the score next to an uninterrupted reference.
Every line relay.sh prints is plain and short. The example's README.md is a short transcript of the Docker mode.

7. Gates on the 0.5B. All must pass before merge.

build on torch-cpu: spill build banking77-quick --engine torch-cpu on this Mac, to completion. The adapter beats the base on held-out rows, and the score lands within noise of the MLX result. Noise is defined as two MLX runs with different batch shapes; report both numbers.
build resume across engines: start on MLX, stop during tune, resume on torch-cpu, and finish; then the reverse. Each resumed score lands within noise of an uninterrupted build. No rows or steps are missing or repeated, and the table records which engine produced each stage.
build with remote state: one build with --state on fsspec's in-memory filesystem, and one with --state on a separate local directory simulating shared storage. Stop and resume each.
Headless build: under --headless, send SIGTERM mid-stage. Expect exit code 75, a checkpoint written, and a resume that completes.
relay example: run spill example relay && ./relay/relay.sh in every mode available here. Docker mode runs if Docker is installed; if it isn't, log it, and the CI relay in item 8 stands in for cross-OS proof. Engine-switch mode always runs. Two-machine mode is tested by resuming from a copied state directory.

8. CI relay: start anywhere, finish anywhere, on every push. Add a workflow relay.yml that runs on pushes to main and on pull requests:

Linux to macOS:
A Linux job runs spill example banking77 --tiny, starts build, and stops it mid-tune with --stop-after.
It uploads the folder and state as an artifact.
A macOS job downloads the artifact, runs spill resume to completion, and uploads the final table.
macOS to Linux: the same in reverse.
Reference: a third job runs the same tiny build uninterrupted on Linux.
A final job checks:
both relays completed;
both relay scores are within noise of the reference, with the noise floor measured once here and stored with its justification;
no row or step is missing or duplicated;
each stage's recorded machine and OS match where it actually ran.
It writes a summary to the workflow run page and commits nothing.
If GitHub's macOS runners can't use Metal, the macOS side runs on whichever engine is available there. The table records which, and the README states it plainly.
Keep the whole workflow under 30 minutes, and measure it.

Also:

The Linux CI job additionally runs spill example banking77 --tiny && spill build banking77-tiny end to end on every push, as before.
Add a test that runs the torch engine in bf16 on the 0.5B and checks outputs statistically against float32: gradient cosine and log-prob shift within the bounds from directive 012, and greedy agreement above the batch-shape noise floor. It runs on runners whose CPU supports fast bf16 and is skipped with a stated reason otherwise. The float32 pin stays for exactness tests.

9. README and docs.

Platforms table: the build column reads Apple silicon and Linux CPU verified, NVIDIA built and awaiting verification.
Quick start: show that spill example banking77 --quick && spill build banking77-quick works on Mac and Linux, with measured MLX and torch-cpu wall times side by side.
"Start anywhere, finish anywhere": rewrite around proof.
One line: a build stopped on Linux is finished on a Mac, and the reverse, on every push. Link to the latest relay.yml run, plus a workflow status badge for that workflow only.
The try-it-yourself line: spill example relay && ./relay/relay.sh, with one sentence on the three modes.
Its real output table from a Docker run, or from engine-switch mode if Docker wasn't available, labeled as such.
Other docs:
docs/portability.md: the relay design and what it checks.
docs/schedulers.md: the SkyPilot example now runs spill build headless from an emitted config.
Also update docs/linux.md, docs/cli.md, docs/plan.md, the docs site (add a guide page, "Start a fine-tuning job on one machine and finish it on another", built from the relay example), and issue #1's checklist.
Measured numbers only. The docs test and link checker must pass.

10. Terminal state.

Squash-merge build-anywhere into main and push. Confirm the CI, Relay, Containers, and Docs site workflows are green on main.
Only now print:
each gate in item 7 with its result and numbers;
both CI relay results with scores against the reference, and the relay workflow's wall time;
which engine the macOS runner used;
the banking77-tiny CI wall time;
the torch-cpu and MLX banking77-quick wall times and scores;
the bf16 test result or its skip reason;
which relay example modes ran here;
Decisions;
what remains, which should be verification only.
Stop.
