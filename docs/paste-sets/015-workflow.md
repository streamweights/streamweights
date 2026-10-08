TERMINAL DIRECTIVE: ONE COMPLETE WORKFLOW

Bring your examples, build a model, understand the result, export it, and continue the same work on another machine.

This is the consolidated assignment. It replaces earlier versions and review notes. Implement the requirements below; make routine implementation choices yourself and record them under Decisions. Do not expand the product scope.

Scope. Your responsibility is implementation and verification. User research, adoption, feedback, and commercial decisions are handled outside this assignment. Completion is determined by the engineering gates in item 9.

Rules

Work through the assignment without asking the user to resolve routine implementation choices. Record your choices and their rationale.
Never print a status summary or progress report before the terminal state in item 12. In this environment, printing one ends your turn and stops all work; this has happened twice on this project. Until item 12, output only tool calls.
Use bounded waits for active work. Never end the turn to wait; any wait is a foreground shell loop under the tool timeout. Do not schedule future wake-ups or wait indefinitely for unavailable hardware, credentials, or services.
Never invent a result, weaken a gate to make it pass, or label an unavailable test as passed. Complete everything feasible. If a required gate is genuinely blocked, leave the changes on the branch and report the specific blocker and completed work at the end. This is a blocked outcome, not successful completion.
No Claude attribution on commits. No em-dashes in files you write.
Small models only: the Qwen2.5 0.5B family, one embedding model whose required model files total under 500 MB, and tiny random test models.
Resolve qwen2.5:0.5b to the appropriate supported weights for each runtime; do not assume an inference alias is directly trainable.
Pin the actual repositories, revisions, tokenizer files, and formats.
No 7B or 70B downloads or runs.
No release or outreach actions: no PyPI publishing, tags, public posts, or tagline changes. The repository and documentation changes and the final push in item 12 are authorized.
Work on branch workflow. Preserve unrelated work. Merge to main only after the required pre-merge gates pass.
Reuse existing engines, checkpoint formats, and storage abstractions wherever possible. Preserve existing commands and supported legacy workflows. The new guided workflow supports the two task types specified below.

1. Start

Save this directive verbatim as docs/paste-sets/015-workflow.md.
Set the design principle in CLAUDE.md to: "One complete workflow: bring your examples, build a model, understand the result, export it, and continue the same work on another machine. The project folder is the unit; execution is separable from it."
Inspect the existing implementation before choosing internal interfaces. Record material decisions, dependencies, and compatibility choices.
Commit completed implementation items as meaningful changes on the branch. Commit the final verification and documentation fixes before merging.

2. The task folder is a reproducible project

Project config. A versioned streamweights.toml, with schema_version, records:
task type and column mapping;
system instructions;
class vocabulary or JSON schema;
model and training settings;
evaluation protocol and metrics;
split definition and seed.
Reject unsupported future schema versions clearly.
Existing projects. Flat-layout projects keep working. Migration is non-destructive, idempotent, and announced in one line. Do not reinterpret old checkpoints silently or overwrite source data.
Run identity.
A new build gets a run id; resume continues an existing compatible run.
Changed data, prompts, schema, training settings, or model identity require a new run.
A documented engine transition may continue the same run, and must record the transition and numerical settings.
Reproducibility.
Record resolved configuration, source and split fingerprints, stable row ids, model and tokenizer revisions and file hashes, dependency versions, stage results, checkpoint references, and artifact references.
Snapshot or content-address the actual inputs so later source edits cannot silently alter an earlier run.
All references must remain resolvable from the portable project and its pinned caches.
Immutable completed runs.
Each completed run has its own immutable snapshot under runs/<id>/, including its report.
Keep mutable ownership state, locks, temporary outputs, and attempt staging outside completed snapshots.
Workers write to attempt-specific staging; only accepted outputs become part of the completed snapshot.
Completion is a fenced state transition under item 7.
Subsequent exports, tests, moves, resumes, or stale workers must not change a completed snapshot.
Completed runs cannot resume training; further training starts a new run referencing its parent.
Later operations.
Exports, final-test evaluations, and later export verifications create separate records under exports/<id>/ and tests/<id>/, referencing the source run and fingerprints. Each record becomes immutable when complete.
Verification performed as part of export is included before that export record is finalized; a later verification gets a new record referencing the export.
Failures remain visible as failed records rather than successful artifacts.
Reports. Project-root REPORT.md is a regenerable index and may change. A completed run's own report never changes.
Base weights. Large weights remain in the shared cache. Record exact reacquisition information and content hashes; do not rely on a mutable model alias or filesystem path as identity. Do not store credentials in projects or bundles.
Portable project and offline bundle.
A portable project contains its data, configuration, run state, results, and pinned references to external model assets.
spill bundle <folder> <path> includes every model, tokenizer, embedding, schema, and data asset needed for the selected workflow to run without network access, assuming the documented runtime dependencies are already installed.
Include a checksummed manifest, and document that dependency installation is separate.
Bundling an active run uses a consistent committed snapshot.
Read-only inspection and inference from a bundle are allowed; an independent training copy must fork to a new run rather than create a second authority for the same live run.

3. Bringing your own data is the primary entry point

spill init.
Read CSV or JSONL with column mapping: --input, --output, optional --group, and optional --system <file>.
The new guided path requires labeled examples.
Validate every row. Errors identify the file, row or line, problem, and fix.
Handle quoted and multiline CSV correctly.
Task types.
Support classification and structured JSON extraction. Print a suggested type and accept --task classification|json.
Do not use distinct-label count alone to classify arbitrary free text. If ambiguous, require the explicit task option with a clear error.
Unsupported new guided tasks fail clearly; existing legacy commands remain usable.
Task contract.
Freeze the classification vocabulary or JSON schema before evaluation.
Accept an explicit schema; if deriving a default, derive it from training data only and record it.
Record label normalization and JSON comparison rules.
Flag invalid ground truth and labels outside the declared vocabulary rather than silently fixing them.
Splits.
Accept supplied --val and --test files. Otherwise make reproducible train, validation, and final-test splits using the seed.
If only one held-out file is supplied, preserve it and derive the missing split from the remaining source data.
Leakage checks.
Treat duplicate inputs and shared group values as connected constraints, including transitive connections. Automatically generated splits keep each connected component together.
For supplied splits, reject duplicate or group overlap with actionable diagnostics; do not silently reshuffle the supplied data.
Define and record duplicate normalization. Flag identical inputs with conflicting labels.
Small data.
Report split sizes and classification coverage. Warn about missing classes.
Reject a build if its training or validation set is unusable.
Do not silently disable group isolation or fabricate held-out rows to satisfy a split ratio.
Final test.
Validation supports build comparisons. Final-test labels are excluded from training, schema inference, calibration, model selection, and normal build reports.
Integrity checks may inspect the test file, but test performance is computed only by spill test, which creates a separate record and identifies the exact frozen run.
Record repeated uses so the report does not imply a repeatedly consulted test remains an untouched holdout.
Examples.
Provide one small redistributable example per task, each with a tiny CI variant: existing Banking77 for classification, and a public structured-extraction dataset with real ground truth.
Verify redistribution terms for the actual included data, record sources and licenses, and retain required notices.
Tiny variants still have viable splits and cover the same workflow.

4. The default build is small and predictable

With labeled data, build trains a small student through the existing supervised fine-tuning path.
Teacher.
A teacher is optional, through the existing sequence-level distillation functionality. With labels, invoke it only when explicitly requested.
Preserve legacy prompts-only workflows, but never automatically select a large teacher.
A prompts-only project without independent labeled validation data may generate training examples and train, but must report quality evaluation as unavailable; teacher agreement is not ground-truth task accuracy.
Teachers generate training targets only from the training partition. Do not use held-out targets for teacher prompt development or student training.
When mixing supplied and generated labels, record their origin and the policy used.
spill plan, also shown in the build preamble, reports:
selected models and engine, with reasons;
downloads and sizes;
disk requirements, including checkpoints, staging, and exports;
duration estimates per stage.
Estimates.
Label estimates as measurements from matching local calibration or as explicit assumptions. If an estimate cannot be supported, say unknown.
Reuse cached calibration; do not run training or download models merely to print a plan.
Any separate calibration action is explicit, bounded, and uses training or synthetic data.

5. Evaluation is useful and honest

What is compared. On validation data, compare the task-appropriate baseline, prompted untrained student, trained student, and teacher when used. For JSON, the schema-prompted untrained student is the baseline; do not duplicate it as a separate purported method.
Classification baseline.
Use one small embedding model plus logistic regression, trained on the same training examples and targets as the student. Fit learned preprocessing on training data only.
Record model, settings, and label policy.
Report the declared metric, sample counts, and invalid or unknown-label predictions; such predictions count as failures.
JSON extraction.
Report parseable-JSON rate, schema-valid rate, per-field accuracy, and whole-record accuracy, using recorded rules for missing or extra fields, types, nested objects, arrays, and ordering.
Invalid or failed outputs remain in denominators.
Do not select forgiving normalization after inspecting validation errors.
Comparable evaluation.
Freeze evaluation row ids, prompt and schema, decoding settings, token limits, postprocessing, metric version, and precision settings, and record them in an evaluation-protocol fingerprint.
Report truncations and inference failures.
Per-example outputs.
Save per-example predictions for each comparator, and disagreements showing improvements and regressions against the trained student.
Keep teacher-generated training examples distinguishable from evaluation ground truth.
Honest reporting.
Report observed differences plainly. Batch-shape variation is only a numerical diagnostic, not evidence of statistical significance.
Do not emit an automatic "reliably better" verdict.
If a baseline wins, say so; do not automatically prescribe more training.

6. Comparisons and exports are trustworthy

spill compare.
Show differences in data, models, training settings, engine, and numerics.
Rank only runs evaluated on the same rows with the same metric definition and evaluation protocol. Training data and models may differ; show those differences.
Incompatible evaluation data or protocols produce an explanation and no winner.
Reports. Include results, counts, durations, failures, artifact locations, and next commands. Distinguish unavailable metrics from zero, and failed operations from successful ones.
spill export. Create a new export record with:
the base model and revision;
tokenizer and prompt template;
adapter, merge status, format, and quantization;
source run id and source evaluation references;
a working script that loads the exported artifact and runs one input.
Verification.
Exercise the actual export through an existing supported runtime: llama.cpp for GGUF, and mlx-lm or Transformers for safetensors.
Use a small, fixed validation subset, never the reserved final test by default.
Record the exact rows, runtime, settings, load failures, prediction differences, and task metrics.
Do not treat a training-engine reload as verification of another format.
Deployment measurements.
Measure time to first token, tokens per second, and peak memory where supported, separately from build duration.
Record hardware, input and output lengths, measurement boundaries, and warm or cold status.
Unavailable memory measurements remain unavailable.
Complete verification before finalizing the export record. Later verification creates another record; it does not modify a completed export or source run.

7. Continuation preserves the build's identity

Authority, committed state, and handoff

Storage. Support local directories and S3 through the existing storage abstraction. Use native conditional-write capabilities where the generic fsspec interface cannot express them; do not emulate compare-and-swap with a read followed by an unconditional write.
One authority.
Each live run has one authoritative control location. Local caches and copies of remote state are not independent authorities.
spill resume <uri> uses that authority and acquires ownership before publishing any progress.
Manual copies cannot safely claim to be concurrent continuations of the same run; require an explicit fork or the supported handoff.
spill move [<folder>] <uri> is a controlled handoff of committed state:
Quiesce the source writer, commit its progress, transfer a consistent snapshot, and verify the manifest and file checksums. Do not copy a changing directory and call it a completed move.
Use an idempotent transfer id and durable handoff states.
Fence and mark the source authority transferred before activating the destination. Until the handoff is committed, the destination cannot run.
If interrupted, retry or recover the recorded handoff without making both locations writable.
Retain the source bytes; do not destructively delete them in this assignment.
Print the receiving command only after the handoff succeeds.
Location changes. Record them outside completed snapshots. Moving completed runs preserves their bytes and does not make them trainable again.
What a move preserves.
Preserve completed stages, checkpoint state, inputs, and artifact references.
The receiver verifies model and tokenizer identity, required assets, compatible engine support, and dependencies before executing.
Reject incompatible configuration, data, or state with named differences.
Record engine changes and numerical settings, without claiming bit-identical cross-engine training.

Immutable payloads and publication

Checkpoint contents. Checkpoints contain the existing resumable state: adapters, optimizer and scheduler state where used, step and data cursor, sampler and random state where supported, and the configuration and model fingerprints needed to validate resume. Resume begins at a supported committed boundary; do not claim to preserve an uncommitted partial optimizer step.
Payload location. Write each checkpoint to a unique immutable payload location under mutable run state, separate from completed snapshots. Include generation, sequence, and a unique attempt identity or content digest. Write a manifest with sizes and content hashes, and verify completeness before publication. Do not overwrite a published payload.
What is current. A checkpoint is current only when the authoritative control object points to its complete manifest. Recovery uses that pointer and validates its payload, not the newest directory name or object listing. Partial writes leave the previous committed checkpoint current.
Stage and run completion. The same publication discipline covers stage completion and final run completion. A stale writer may leave isolated orphaned outputs, but cannot publish them, alter accepted stage results, or mutate the completed snapshot.
Local durability. On local disk, flush the payload and control updates appropriately before acknowledging a committed checkpoint. State the tested recovery guarantee precisely; do not imply testing of whole-machine power loss when only process termination was tested.
Recovery reporting. Recovery reports the last committed step and cursor. Report exact replay counts only when the interrupted progress is known; otherwise give the supported bound or say unknown. Never promise that at most one step is lost regardless of checkpoint interval.

S3 ownership

Control object. Keep one control object per run, outside its immutable completed snapshot. It contains:
owner id;
monotonically increasing fencing generation;
lease expiry;
control revision;
run or handoff status;
pointers to the accepted checkpoint and stage state.
Conditional writes. Create with If-None-Match: *. All subsequent ownership, renewal, handoff, and publication transitions conditionally replace that same object using If-Match with its observed ETag. Never delete and recreate it to reset fencing.
Client validation. The client validates owner, generation, and allowed transition against the observed control contents before attempting the conditional write. The object store validates the ETag, not the semantic meaning of the generation. Include a changing control revision, and preserve accepted pointers during renewals and ownership changes.
Conflicts and uncertain outcomes.
On a conditional conflict or uncertain request outcome, reread and reconcile. If superseded, stop.
Retry only if the same owner and generation still permit the transition, preserving newer accepted state.
Never refresh an ETag and blindly replay an old publication.
Prevent checkpoint sequence regression, and handle renewal/publication races within one worker.
Lease expiry. Lease expiry allows a successor to compete for ownership; successful conditional acquisition establishes the new generation, and an old writer cannot publish after that acquisition. Document clock and renewal assumptions; do not claim that S3 itself checks lease timestamps.

Local ownership

Lock file. Use one consistent OS-level exclusive-lock mechanism on a stable lock file outside the completed snapshot. Do not delete or replace the lock file while the run exists.
Control transitions. Hold that lock for every control transition: read the control state, validate ownership and generation, then atomically replace the control file before releasing the lock. All application writers use this protocol. Payload computation and large writes happen outside the short control critical section.
Lock semantics. B cannot acquire ownership while A holds that lock, and lease expiry does not override a held OS lock. A process that dies releases its lock; a process paused while holding it blocks other control transitions until it resumes or exits.
Supported filesystems. Support only documented local filesystem configurations with the required semantics. Refuse known unsupported or unverified network filesystem configurations; do not claim a local probe proves arbitrary network-lock correctness.

Limits

Refuse a backend that cannot provide the required atomic operations. Capability checks may use disposable probe objects, but may not publish run state on an unsafe backend.
Defer spill gc and automatic orphan cleanup. Leave orphaned payloads unreferenced; deletion needs separate retention and concurrency rules.
Automatic provisioning, scheduling, and transparent multi-authority synchronization are out of scope.

8. Execution is separable from the project

Represent stages explicitly with declared inputs, outputs, status, and serializable descriptions. Reference inputs and outputs by stable identity and checksums rather than machine-specific paths.
Keep project metadata and CLI behavior independent of MLX and PyTorch execution details.
The default backend executes locally behind an interface that accepts a stage and its inputs and returns outputs and status. The coordinator owns publication and run-state transitions; executors return staged outputs and cannot bypass fencing.
Ship only the local backend, plus a test backend that executes the same supported stages in a separate process from the serialized description. Do not build a remote scheduler or cloud control plane.

9. Gates: verify the complete developer journey

All required gates must pass before merging. Use the permitted small models. Match assertions to the stated guarantee; do not substitute mocked storage for the MinIO integration, or one engine for another.

Fresh installation. Install the built wheel into a clean environment, with documented extras, and run the public CLI without importing from the source checkout.
CSV journey. Run init, plan, build, report generation, export, and inference through the exported artifact, end to end.
Task and engine coverage.
Both classification and JSON extraction complete on MLX and torch-cpu.
Record hardware, model revisions, durations, results, and export verification.
Execute every export format advertised by the new examples; retain meaningful existing format coverage.
Input and split integrity.
Cover malformed data, ambiguous task types, transitive duplicate/group constraints, supplied-split overlap, unusable small splits, and conflicting duplicate labels.
Verify that training, planning, schema inference, and normal evaluation do not consume final-test targets for model development.
Cross-engine continuation.
Stop mid-training and resume the same run between MLX and torch-cpu in both directions, using tiny fixtures.
Verify optimizer and cursor restoration, preserved committed progress, and recorded engine changes.
Reject deliberately incompatible data, weights, and settings.
Interrupted checkpoint publication.
Terminate during payload writing, and before or after control publication.
Recover the last accepted checkpoint with intact files and the correct committed step and cursor.
Test hooks may provide exact lost progress for assertions; normal user-facing reporting must handle unknown uncommitted progress honestly.
Immutable history.
Changed inputs create a new run. Previous completed snapshots remain byte-identical.
Export, final test, later verification, move, and a stale worker do not alter a completed run. Completed export and test records also remain unchanged.
Test completion fencing, not just read-only file permissions.
Comparisons. Rank compatible runs. Refuse ranking when the evaluation rows, metric definition, or evaluation protocol differ, explaining the reason. Changing training data alone must not automatically make a comparison incompatible.
Portability and offline bundle.
Verify a bundle in a fresh cache, with network access disabled and runtime dependencies already installed.
Verify reacquisition and checksum validation for a portable project.
Cover a missing or corrupted asset and ensure it is rejected.
Object store and handoff.
Exercise move and resume against a pinned MinIO container in Linux CI, and log its version.
Interrupt a handoff and recover it idempotently.
Verify the source cannot publish after the destination is activated, and that two locations never become writable authorities.
Keep real AWS S3 explicitly labeled untested unless actually exercised.
Ownership and publication. Use deterministic hooks and separate processes. Run shared cases on both local disk and MinIO, and backend-specific cases as stated below.
Case    Required result
A pauses outside the local critical section or before its S3 update; its lease expires; B acquires and publishes; A resumes    A's publication is rejected. B's current pointer and payload remain intact. Previously committed history is retained.
S3: A validates ownership and pauses before its conditional write; B acquires and publishes    A's stale conditional write fails. A cannot retry into B's generation.
Local: A pauses while holding the lock after validation; B attempts acquisition    B blocks or gets a documented contention result. After A releases the lock and B validly acquires a newer generation, A's later publication is rejected. Do not require B to acquire while A holds the lock.
Two writers race to acquire    Exactly one acquires that generation. Only that winner can publish; the test does not assume the winner is named B.
Renewal and checkpoint publication race, or a write response is lost    Reconciliation preserves newer state; no pointer regression or accidental ownership takeover.
Backend lacks required atomic operations    It is refused before any run-state publication. Disposable capability probes are allowed.
Stale writer tries to publish stage completion or run completion    The publication is rejected, and the accepted outputs and completed snapshot remain unchanged.
Separate-process backend. Complete a build from serialized stages through the test backend, with normal coordinator publication and reporting.
Documentation commands.
Inventory actionable product commands in the maintained README and user guides.
Execute small-model examples on tiny data.
For larger-model examples, parse against the real CLI without loading or downloading a model, then exercise the same path with an explicit tiny equivalent in tests/docs_command_map.toml.
Map placeholders and environment-specific setup commands explicitly.
Historical verbatim paste sets are archived instructions, not executable examples; identify that exclusion.
The harness prevents unapproved downloads and model execution, and the link checker passes.

CUDA, real cloud deployments, and network filesystems remain explicitly untested unless measured. Missing required MLX, CPU, MinIO, or cross-engine evidence is a blocked required gate, not an optional platform disclaimer.

10. README and documentation

Keep the tagline unchanged. Organize the README in this order:

Your examples to a model: init, plan, build, report, export.
Understanding results: comparison table, disagreements, and honest reporting.
Continuing on another machine: move, resume, and the relay link.
Sample projects.
When a teacher is useful.
Platforms and explicitly untested paths.
Under the hood.
Prior art: factual descriptions of AirLLM, slowllama, Unsloth, FlexGen, and DeepSpeed, with a specific description of what StreamWeights adds. Verify against current primary sources; do not assert novelty through omissions.
Status.

Describe existing distillation accurately as sequence-level distillation from teacher answers. Remove unsupported "first" claims. Do not imply that fitting a model on smaller hardware establishes lower cost per completed job.

Update:

docs/cli.md, from actual help output and verified examples.
docs/formats.md, for project config, immutable run snapshots, separate export and test records, payload manifests, control state, and schema migrations.
docs/portability.md, for:
portable projects versus offline bundles;
one authoritative run location;
handoff and recovery;
immutable payloads;
S3 conditional publication and local locking;
engine compatibility;
explicit limitations.
docs/plan.md, to mark deferred work as separate future assignments.
The docs site, with a guide, "Turn a CSV of examples into an evaluated model".

Published benchmark results must be measured and identify their setup. Planning estimates may remain estimates if clearly labeled. Make no hardware, runtime, storage, or numerical-equivalence claim beyond the evidence collected.

Protocol references for implementation:

AWS, conditional writes: https://docs.aws.amazon.com/AmazonS3/latest/userguide/conditional-writes.html
Linux manual, exclusive file locking and filesystem limitations: https://man7.org/linux/man-pages/man2/flock.2.html

These describe storage primitives. The application must still implement and verify the state transitions specified above.

11. Deferred

KL or logit-distillation losses; expanded quantization support; teacher-quantization sweeps; generic free-text quality evaluation; new model families; automatic checkpoint garbage collection; new remote execution services; cloud provisioning, scheduling, or multi-cloud orchestration; automatic cost optimization; and adoption or commercial research.

12. Terminal state

Complete and record all required pre-merge gates. If a required gate is blocked or failing, preserve the branch and report that outcome with the exact evidence and remaining blocker. Do not merge a knowingly incomplete assignment.
Otherwise squash-merge workflow into main and push, using the repository's permitted merge process. Respect branch protection.
Confirm CI, Relay, Containers, and Docs site results for the actual resulting main commit. Repair failures within scope rather than declaring success from earlier branch results.
The final response includes:
every gate and its evidence;
the measured CSV and example reports for MLX and torch-cpu;
export differences, latency, and memory;
chosen dataset and license;
embedding model and pinned revisions;
MinIO version;
Decisions;
final commit and workflow results;
untested paths.
Completion means the specified workflow works and its evidence is recorded. It does not require a release, outreach, users, or commercial validation.
