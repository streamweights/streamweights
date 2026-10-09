TERMINAL DIRECTIVE: CLOSE THE 015 GAPS. OFFLINE BUNDLE GATE, MLX-ON-METAL CONTINUATION, AND MERGE HYGIENE.

Rules, absolute:

No status before the end. Never print a status summary or progress report before the terminal state. Printing one ends your turn and stops all work. Until the end, output only tool calls.
Never end the turn to wait. Any wait is a foreground shell loop under the tool timeout. No scheduled wake-ups, no /loop, no questions to the user.
No weakened gates. Never invent a result, weaken a gate, or label an unavailable test as passed. If a gate is genuinely blocked, leave the work on the branch and report the blocker.
Commits. No Claude attribution on commits. No em-dashes in files you write.
Models. Use only the small models already pinned in models.yaml.
Restrictions. No releases, tags, or public posts. No tagline changes.
Branch. Work on the branch gaps-015. Merge only through a pull request, after all gates pass. Never push directly to main.

1. Start. Save this directive verbatim as docs/paste-sets/016-close-015-gaps.md. Commit.

2. Offline bundle gate, with the network actually disabled.

Linux CI. Build a bundle of the tiny classification project. In a container started with --network none and the runtime dependencies preinstalled:
set a fresh, empty SPILL_HOME and Hugging Face cache;
install the wheel from a local file;
import the bundle;
run inference from it;
fork it to a new run and complete a training run.
Proving no network is used. Assert that no network call is attempted: any attempt must fail the test, not be silently retried. Set HF_HUB_OFFLINE=1 only as a second layer, never as the proof.
macOS. Run the same gate under sandbox-exec with a profile that denies network access. If that profile cannot be made to work reliably, record why under Decisions, and the Linux gate stands as the required evidence.
Corruption. Cover a corrupted and a missing asset inside the offline environment. Both must be rejected with a clear message.

3. MLX on Metal: continuation in both directions, automated.

Add scripts/gate_metal_continuation.py, which runs on an Apple silicon Mac with the MLX GPU device. It asserts that mx.default_device() is the GPU and fails otherwise.
Cross-engine runs. Stop a tiny build mid-training while it publishes a checkpoint, then:
resume on torch-cpu;
stop again;
resume on MLX GPU;
complete.
Then repeat, starting on torch-cpu.
Checks on each run. Verify that the step, the data cursor, and the optimizer step carry over, that each engine change is recorded, and that the final validation score is reported next to an uninterrupted MLX-GPU run of the same project.
Move between machines. Exercise both directions through a local move and through a MinIO move. MinIO runs locally in the Linux container if Docker is available. If Docker isn't available here, log that, and the MinIO path stays covered by CI.
Run the script here on this Mac and write the results to docs/reports/data/016-metal.json.
Add an opt-in pytest marker (SPILL_METAL_TESTS=1) wrapping the same checks, so this can be re-run on any Apple silicon machine.

4. Merge hygiene.

Open a pull request containing a no-op verification commit for 8c4b7fa. Confirm that the boto3 and s3fs resolution is correct in both container images and in a clean venv with the cloud extra, and record the resolved versions.
Branch protection. Enable branch protection on main through gh api, if the account permits it: require pull requests and the CI, Relay, Containers, and Docs site checks before merging. Record what was set, or why it couldn't be.

5. README and docs, to the current state.

Platforms section: state the bundle gate's evidence (Linux --network none, plus macOS sandbox if achieved) and the Metal continuation evidence. Remove any wording that implies more than was run.
docs/portability.md and docs/reports/015-workflow.md: add a short addendum linking to the 016 results.
Example tables: the README example tables state the evaluation set size (18 validation rows) next to every score. No score appears without its row count.
The docs test and the link checker must pass.

6. Terminal state.

Merge gaps-015 through its pull request, after all gates pass.
Confirm CI, Relay, Containers, and Docs site are green on the resulting main commit.
Only now print:
the offline bundle gate's evidence on Linux, and on macOS or why not;
the Metal continuation results in both directions, with scores next to the uninterrupted reference;
the resolved boto3 and s3fs versions;
the branch protection settings applied, or the reason none were;
Decisions;
remaining untested paths.
Stop.
