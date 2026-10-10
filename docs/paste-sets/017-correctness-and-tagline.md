TERMINAL DIRECTIVE: SCORER AND PROTOCOL CORRECTNESS, EXPORT DIAGNOSIS, NEW TAGLINE, AND A TIGHTER README.

This is one bounded correctness-and-positioning assignment. Do not expand scope.

Rules, absolute:

Never print a status summary or progress report before the terminal state. Printing one ends your turn and stops all work. Until the end, output only tool calls.
Never end the turn to wait. Any wait is a foreground shell loop under the tool timeout. No scheduled wake-ups, no /loop, no questions to the user.
Never invent a result, weaken a gate, or label an unavailable test as passed. If a gate is genuinely blocked, leave the work on the branch and finish with a precise blocked outcome (item 8).
No Claude attribution on commits. No em-dashes in files you write.
Use only the small models already pinned in models.yaml. Model-execution gates cover MLX on Metal and torch-cpu only. CUDA stays explicitly untested; this pass does not reopen earlier hardware exclusions.
No releases, tags, or public posts. The tagline change in item 5 is authorized.
Work on branch correctness-016. Merge only through a pull request after the pre-merge gates pass. Never push directly to main.

1. Start. Save this directive verbatim as docs/paste-sets/017-correctness-and-tagline.md. Commit.

2. JSON scoring: invalid outputs count as wrong.

Correctness rule. In score_json, whole-record correctness requires the output to parse and to be schema-valid. A parseable but schema-invalid answer still counts toward the parseable rate, but receives no field-level or whole-record correctness credit. Field-level normalization never rescues a schema-invalid output.
Regression test. With a schema permitting only "approved", the output " approved " must score as parseable, schema-invalid, and incorrect at both the field level and the whole-record level.
Audit. Check every other JSON metric for the same ordering problem and fix any you find. Bump the JSON metric version. Classification metrics are unchanged and keep their version.
Historical rescoring. Account for every published JSON result in the repo, one of two ways:
Rescorable (saved per-example predictions and ground truth exist): rescore with the corrected scorer. Publish the correction as a new evaluation record and in updated reports, with the old and new metric versions and the score change. The original immutable run records stay untouched.
Not rescorable (no saved predictions): mark the result explicitly as not rescorable in the reports. A fresh model run is a replication, not a rescore; do not present it as one. No larger-model reruns.

3. The recorded evaluation protocol matches execution.

Primary metric. The primary metric comes from the project config's selected metric, not a value hardcoded by task. Selecting a metric the task doesn't support is rejected with a clear message.
Decoding settings. Every recorded decoding setting (temperature, top-p, max tokens, stop sequences, greedy flag) is actually passed to the guided evaluation request on every engine. A setting an engine can't apply is rejected before evaluation, with a message naming the engine and the setting. This assignment does not add new sampling features to any engine.
Executed conditions. Each evaluation record stores the conditions it actually ran under: engine, device, actual numerics (weight and compute dtypes), runtime and library versions, and decoding settings as applied. These are kept separate from the requested precision policy. Run identity continues to exclude the training engine.
Evaluation record IDs. Every evaluation record has an explicit ID.
spill evaluate <run> --engine <engine>. Re-evaluates a completed run on its frozen validation inputs, using its exact model and adapter identity, under the chosen evaluator. It creates a separate immutable evaluation record that references the run and never modifies it. Final-test evaluation remains exclusively under spill test.
compare outcomes. Every comparison gets one of three labels:
Outcome    Conditions    Behavior
Common evaluator    Same evaluation rows, task protocol and metric; matching executed evaluator conditions    Rank
Cross-runtime    Same evaluation rows, task protocol and metric; differing engine, device, runtime versions or actual numerics    Rank, showing the differences and a plain caution that score differences may include evaluation-runtime effects
Incompatible    Different evaluation rows, prompts, decoding policy, or metric definition or version    No ranking; explain why
Selecting records. compare selects evaluation records explicitly by ID when a run has more than one. It never silently picks whichever scored best. Without an explicit choice, it uses each run's original evaluation and says so.
Lineage. Wherever a parent run appears (report, status, compare, docs), state that it records experiment lineage only, and that the new run trained from the base model, not from the parent's weights.

4. JSON export differences: diagnose them, don't just disclose them.

Alignment. Keep input messages, rendered prompts and tokenization (where applicable), decoding settings, and verification rows identical across every comparison below. Record actual execution conditions for each.
Same-row scoring. For every artifact, report the source-engine task score and the artifact task score on the exact same verification rows, plus text-disagreement rate and schema-validity rate.
Merge comparison (Transformers, float32, CPU). Compare the unmerged adapter on the base, a float32 merge, and a bf16 merge.
Runtime comparison. Compare the float32 merge in Transformers on CPU against an unquantized F32 GGUF in llama.cpp on CPU.
Quantized exports. Separately compare GGUF exports derived from the float32 and bf16 merges, both using the same deployment quantization (q8_0).
Choosing the default. Add spill export --merge-dtype float32|bf16.
Choose the default by task quality first, then fidelity to the source engine, then resource cost (artifact size, memory), with an explicit tie-break rule written in Decisions before looking at results.
Do not choose solely by text agreement.
State in the report that eight rows can support an implementation decision, but cannot establish that one merge dtype universally preserves quality better.
What export completion means. Export completion means the artifact loaded and verification ran. It does not mean quality was preserved. The export record and report show the quality delta on the verification rows explicitly, and the docs say so plainly.

5. Tagline and supporting copy.

Headline: "Build a small model for your task, on hardware you control."
Supporting copy: "Bring labeled examples. Fine-tune locally, compare against simple baselines, and export to GGUF or safetensors. Pause and resume supported training runs across Mac and Linux."
In-repo surfaces (part of the pull request): the README, the pyproject.toml description, the spill --help header, the docs site title and metadata, and the org profile source.
Social preview. Regenerate docs/img/social-preview.png with the new headline.
External metadata (the GitHub About description and the live org profile, if it lives outside this repo): prepare the exact gh api calls before merge, and apply them after merge (item 7).
Large teachers. Disk-streamed large teachers stay a prominent optional capability further down the README.

6. README: tighten claims and the first run.

Order. Lead with the new workflow. Move the 70B headline and its measurement below the quickstart.
Quickstart. Finish it with inference: show the resulting model answering an input, next to a compact example report.
Continuation example. Fix it to demonstrate an interrupted build, then move and resume, since resume rejects completed runs. Include installing the cloud extra (pip install "streamweights[cloud] @ git+https://github.com/streamweights/streamweights") before any S3 command.
Narrow absolute claims across the README, docs site, and guide pages:
replace "whatever hardware" and "anywhere" with supported-platform wording;
replace "no data leaves it" with "data stays local by default; remote storage is used when explicitly configured."
Tiny tables. Label them as workflow checks. Keep the baseline's win visible. Leave room for a larger, representative result later.
Measurement descriptions in the report generator:
use each runtime's own descriptions (the Transformers path currently reuses llama.cpp's);
describe sampled process RSS as a lower bound, not a total GPU-memory peak;
state that cold first-token timing excludes model loading;
state that the Metal q8_0 versus CPU float32 numbers are not a format-only performance comparison.
Regenerate every affected report and the README tables from the result files after these fixes.

7. Gates and sequence.

Before merge. All must pass on the pull request.

JSON scorer.
The regression test passes, including the parseable-but-invalid case.
The JSON metric version is bumped and classification metrics are unchanged.
Every published JSON result is accounted for as rescored (with a new evaluation record and the score change) or explicitly not rescorable.
Protocol tests.
The selected metric drives the primary score.
Unsupported metrics and decoding settings are rejected.
The applied decoding settings are observable in a captured evaluation request on MLX on Metal and on torch-cpu.
Evaluation records and compare.
Executed conditions are recorded per evaluation.
compare produces all three outcomes correctly: common-evaluator, cross-runtime, and incompatible. Incompatible pairs cover differing rows, prompts, decoding policy, and metric version.
compare uses explicitly selected evaluation record IDs.
spill evaluate creates a separate record from validation inputs only, and leaves the run byte-identical.
Export diagnosis.
The controlled comparisons in item 4 ran on both task types, and the results are in a report.
The default was chosen by the pre-stated rule.
Copy. The new headline and supporting copy are consistent across all in-repo surfaces. The regenerated social preview is committed. The external metadata calls are prepared.
Docs. Every README and guide command runs or is syntax-checked by the docs harness, including the new continuation example (an interrupted build, then move, then resume) on tiny data, with local storage and MinIO in CI. The link checker passes.
Workflows. CI, Relay, Containers, Docs site, and the offline bundle gate are green on the pull request.

After merge:

Verify that CI, Relay, Containers, and Docs site are green on the resulting main commit. Repair any failure within scope, through a new pull request.
Apply the prepared GitHub About and org-profile changes, and confirm they display.

8. Terminal state.

Successful completion: all pre-merge gates passed, the pull request merged, the post-merge checks are green, and the external metadata is applied. Print:
each gate's result;
the historical JSON accounting (rescored with changes, and not rescorable);
the export comparison tables and the chosen merge default, with the tie-break rule;
the final README headline;
Decisions;
remaining untested paths.
Blocked outcome: if a required gate is blocked or failing, do not merge. Print the report with the exact evidence, the blocker, and the remaining work.
Either way, list the social preview upload as a manual follow-up (Settings, General, Social preview), not a completion gate.
Stop.
