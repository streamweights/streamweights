TERMINAL DIRECTIVE: RESTORE THE VISION. ANY TASK, A LARGE TEACHER AT THE CENTER, AND HONEST EVALUATION OF OPEN-ENDED OUTPUT.

Rules, absolute:

No status before the end. Never print a status summary or progress report before the terminal state. Printing one ends your turn and stops all work. Until the end, output only tool calls.
Never end the turn to wait. Any wait is a foreground shell loop under the tool timeout. No scheduled wake-ups, no /loop, no questions to the user.
No invented results. Never invent a result, weaken a gate, or label an unavailable test as passed. If a gate is blocked, leave the work on the branch and finish with a blocked outcome.
Commits and files. No Claude attribution on commits. No em-dashes in files you write.
Models. Only qwen2.5:0.5b, qwen2.5:1.5b, the pinned embedding model, and tiny test models may run. No 7B or 70B runs in this directive.
No outward-facing actions. No releases, tags, or public posts. The headline stays; the supporting copy changes per item 9.
Branch and merge. Work on branch vision. Merge only through a pull request after the gates pass. Never push directly to main.
Preserve existing guarantees. Keep every one: reproducible projects, immutable runs, split and leakage rules, the reserved final test, protocol fingerprints, portable state, fencing, and export verification. New capability builds on these and never bypasses them.
No unrequested large downloads. The tool never downloads a large model the user didn't ask for. Teachers and judges are used only when named, or when the user accepts a stated recommendation.

1. Vision and guardrail.

Save this directive verbatim as docs/paste-sets/020-restore-vision.md.
Write VISION.md at the repo root, under one page, stating:
the product: a developer tool for building your own model for your own task, learning from large models, on hardware you have;
large models are first-class even when bigger than the machine, as teachers and as judges;
jobs are made of portable pieces that can continue on other machines;
the direction is toward deadline-driven execution on whatever capacity meets the deadline, starting with the developer's own machines and accounts.
Add to CLAUDE.md the standing rule: "Any change that narrows the product's scope, removes a capability, or demotes a pillar of VISION.md must be stated explicitly in the terminal report under a heading 'Scope changes', with the reason. Never fold a narrowing in silently."
Commit.

2. One kind of project: examples of input and desired output.

Remove task-type gating from the guided workflow. spill init accepts any input and output columns and an optional system instruction, with the same validation, splits, leakage checks and final-test rules as now.
The scoring method is chosen automatically from the outputs, and stated in spill plan and the report:
exact: outputs form a small label set (at most 200 distinct values). Score by exact match, plus the embedding baseline.
json: outputs are JSON. Use the existing field-level metrics, with the corrected scorer.
judge: anything else. Always report reference-overlap metrics against the provided outputs, labeled as such. Judged scores are added per item 3.
script:<file>: always available as an override. A user Python file exposing score(row) -> float.
Overrides and legacy projects. --score exact|json|judge|script:<file> overrides the automatic choice. Existing projects with a recorded task type map to exact or json, and keep working and comparing as before.

3. Judged evaluation for open-ended output.

Choosing the judge. --judge <model> names it.
Without --judge, an open-ended project builds normally with reference-overlap metrics.
spill plan and the build preamble state that judged scoring needs a judge, and recommend one sized to the machine, with its download size, disk need, and estimated judging time.
--judge recommended accepts that recommendation.
If the user has already named a teacher, the teacher is used as judge unless --judge says otherwise.
No judge is ever downloaded unless named or accepted.
Large judges stream from disk like any other model.
Rubric and protocol. The rubric comes from rubric.md in the project if present, otherwise from a default rubric stated in the docs. The rubric text, judge model identity, judge prompt template, and decoding settings are all part of the evaluation-protocol fingerprint.
Two modes:
Pointwise: each output is scored 1 to 5 against the rubric, and against the reference where one exists.
Pairwise: two models' outputs on the same input are compared in both orders, to cancel position bias. Report win, tie, and loss rates, and how often the two orders disagreed.
Honest labeling.
Judged scores are labeled as one model's assessment everywhere, never presented as accuracy.
Where the project also has exact-match-checkable rows, report the judge's agreement with exact match as a calibration figure.
A judge may not judge its own outputs. If the teacher is also the judge, the teacher's row is reported as reference only, excluded from rankings, and the report says so.
Records. Judged evaluation produces per-example records with verdicts and rationales, and fits the existing evaluation records and compare rules.

4. The large teacher at the center, and optional.

When a teacher is used. With labeled examples and no --teacher, build trains on your labels alone. A teacher is used only when --teacher <model> is given, or when the project has unlabeled prompts. With labels and a teacher, train on both, recording each example's origin and the mixing policy, as now.
Logit distillation returns to scope. distill --logprobs K output can be trained with --distill-loss kl|ce|mix:
kl: KL from the teacher's renormalized top-k distribution (with the remaining mass in an "other" bucket) to the student, at temperature --kd-temp;
ce: cross-entropy on the teacher's tokens;
mix: both, weighted by --kd-alpha.
The default is mix when log-probs are present and the tokenizers match, otherwise ce. Implement on MLX and torch, for both resident and streamed models.
Training partition only. The teacher generates from the training partition only, as now.

5. Every model is a choice.

--student, --teacher and --judge accept any curated tag or Hugging Face repo in a supported family.
All three are recorded in streamweights.toml, and shown by spill plan with their sizes, downloads, disk needs and estimated stage times. Change them by editing the config or passing the flag.
Defaults. The student defaults per machine, with the reason stated. Teacher and judge have no automatic default beyond the rules in items 3 and 4.
Unsupported architectures fail in one line naming the family.

6. An open-ended sample project.

Add a spill example for a public, redistributable dataset with open-ended reference outputs, such as question answering over passages, or summarization. Verify and record its license. Include a --tiny variant for CI.
It runs judged evaluation in both modes with a named small judge. Its README shows a real captured result.

7. Deadline planning, the first step toward the neocloud.

spill plan --deadline <duration or timestamp> uses calibrated rates to list which combinations of student, teacher, judge and engine available on this machine are estimated to finish in time. Each combination shows its estimate and whether that estimate is measured or assumed, and the recommended one is marked.
Combinations including a model the user hasn't downloaded show the download size, and are never fetched by plan.
spill build --deadline uses the plan. If nothing fits, it says so plainly and shows the closest option. It never silently drops quality settings to meet a deadline.
Scope: local planning only. No remote execution, provisioning, or scheduling.

8. Gates. All must pass on the pull request. Small models only.

Any task, end to end. An open-ended project goes through init, plan, build with a 1.5B teacher and a 0.5B student, judged evaluation in both modes with the 1.5B as named judge, report, and export. This must pass on both MLX (Metal GPU) and torch-cpu.
Teacher optional. A labeled project with no --teacher builds without any teacher stage and without downloading any model beyond the student and the embedding baseline. An open-ended labeled project with no --judge builds with reference-overlap metrics, shows the judge recommendation in plan, and downloads nothing else.
Scoring selection. Automatic scoring selection picks exact, json, and judge correctly on three fixtures, and the script: override works. Legacy projects keep their scores and compare compatibility.
Judge integrity.
Judge identity, rubric, and prompt are in the protocol fingerprint, so changing any of them makes compare call the comparison incompatible.
Pairwise judging runs both orders and reports how often they disagreed.
Self-judging is excluded from rankings.
The calibration figure appears when applicable.
Logit distillation. The KD loss and its gradients match a reference implementation on a tiny model in float32, MLX and torch agree, and the mix and ce students are both reported.
Model choices. All three roles accept tags and Hugging Face repos, are recorded, and are shown in plan. An unsupported architecture is refused in one line.
Deadlines. plan --deadline lists feasible combinations with labeled estimates, downloads nothing, and refuses honestly when nothing fits.
Nothing regresses. Every existing gate and workflow still passes: CI, Relay, Containers, Docs site, the offline bundle gate, and the docs command harness.

9. README, docs, and positioning.

Headline and supporting copy. Keep the headline. Change the supporting copy to: "Bring examples of your task. Learn from a large model running on your own machine, train a small one, measure it honestly, and export it."
README sections:
"What it does": describe any input-to-output task, with examples spanning labeling, extraction, and open-ended answers. Drop the two-job framing.
"Try it": stays the fastest path, with a real captured result.
"Use your own data": show one CSV that produces open-ended output, and how scoring is chosen. State plainly that with your own labeled examples no teacher is needed, and that judged scoring for open-ended output uses a judge you name or accept.
Large teacher and judge: move this back up, as a short section right after "Understand the result". Cover what each does, when it's worth it, and that both are optional.
Docs:
Update docs/plan.md: vision first, then phases. The phases are this directive, the scale proof (7B student, 70B teacher and judge, overnight), CUDA verification, then deadline-driven execution across machines.
Also update the CSV guide, docs/formats.md, and docs/cli.md.
Checks. Measured numbers only. The docs test and link checker pass. The README remains the current-state description, per the standing rule.

10. Terminal state.

Success: merge through the pull request, and confirm the workflows are green on the resulting main commit. Then print:
each gate's result;
the open-ended example's captured results;
a "Scope changes" section (expected: nothing narrowed; list what was restored);
Decisions;
the scale-proof work still pending.
Blocked: do not merge. Print the exact blocker and the remaining work.
Stop.
