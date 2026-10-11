TERMINAL DIRECTIVE: README EDITORIAL PASS. A FIRST RESULT FAST, AND AN OBVIOUS PATH TO YOUR OWN DATA.

This is a README editing task on the existing product. Keep the positioning and the headline, and cut roughly half of the explanatory prose. Product code must remain unchanged. Documentation tests, command-harness mappings, capture scripts and site navigation may change to enforce this directive.

Rules, absolute:

Never print a status summary or progress report before the terminal state. Printing one ends your turn and stops all work. Until the end, output only tool calls.
Never end the turn to wait. Any wait is a foreground shell loop under the tool timeout. No scheduled wake-ups, no /loop, no questions to the user.
Never invent a result, weaken a gate, or label an unavailable test as passed. If a gate is genuinely blocked, leave the work on the branch and finish with a blocked outcome (item 6).
Commits and files: no Claude attribution on commits, and no em-dashes in files you write.
Models: only the small models already pinned in models.yaml may run.
Branch and merge: work on branch readme-pass. Merge only through a pull request after the gates pass. Never push directly to main.

1. Start.

Save this directive verbatim as docs/paste-sets/018-readme-pass.md.
Record the current README's total line count, total word count, and prose-only word count. Prose-only excludes fenced code blocks and tables. Write the counting method as a small script (scripts/readme_counts.py) so that both versions are measured identically.
Commit.

2. README structure, in this order.

What it does.
Keep the headline: "Build a small model for your task, on hardware you control."
Immediately name the two supported jobs, text classification and structured JSON extraction, with concrete examples such as routing support tickets and extracting fields from text.
Remove the repeated introduction and all implementation terminology from this section.
Try it.
Prerequisites first: supported platforms, Python version, the initial model downloads with sizes, and disk required. Clearly distinguish first-time installation and download time from training time.
Commands beside their expected output, with brief comments separating setup, training, and inference.
End on the model's actual response to an input. All output shown must be real: run the tiny example on MLX, capture its output, and paste it in. Never hand-write output.
After the main example, explain plan and the report briefly, as inspection commands.
Use your own data.
Show the expected CSV shape as a short inline example, labeled clearly as a format example rather than a complete training dataset. Show one for classification and one for JSON extraction.
Then give the existing spill init command with those exact column names, followed by the same build and inference commands.
The JSON example must use valid CSV escaping, and must show or link any schema file its command references.
The transition from the example to the reader's own problem must be unmistakable.
Understand the result.
One compact example report, including the baseline's win.
One plain sentence on why that matters: the tool helps you discover whether fine-tuning improves on a simpler approach.
One sentence on checking the exported artifact's quality, with a link.
Move the repeated four-row results table and the protocol details to the linked report.
Go further. Brief descriptions, one or two lines each, with links, for:
JSON extraction;
continuing an interrupted training run on another supported machine (keep this user benefit prominent);
offline bundles;
optional large teachers.
Support and limitations.
A small platform table.
Keep three limitations visible: validation so far is on small models, CUDA is untested, and exported models can differ in quality from the trained one.
Then links to the documentation and feedback channels.

3. Move content, don't delete it. Everything cut from the README lands in the right doc, with a link back from the README:

The CSV guide ("Turn a CSV of examples into an evaluated model"): the long explanation of init, plan, splitting, and export internals.
docs/portability.md: the S3 walkthrough, conditional writes, fencing, checkpoint publication, and lineage mechanics.
A new docs/prior-art.md: the prior-art comparison. A one-line attribution with a link stays in the README.
A new docs/architecture.md: the architecture explanation.
docs/plan.md: the completed-work inventory and the deferred roadmap.
Their existing reports: the results tables and protocol details stay there.

Add every moved page to the docs site navigation.

4. Length and concepts.

Cut roughly half the README's explanatory prose, measured by the prose-only word count from item 1, with the same script. Report total line and word counts alongside it.
The README introduces no new concepts. These terms appear only as linked names where a reader needs them, never as explanations: LoRA, embedding baselines, split rules, precision, evaluation protocols, ownership fencing, lineage, and distillation.

5. Gates. All must pass on the pull request.

Docs tests updated, not weakened. Replace the obsolete layout assertions in tests/test_docs.py (the old headings and opening text) with assertions for the new section structure. Keep all command, link, output-provenance, and correctness checks.
Every documented command is covered. Each README and guide command runs or is syntax-checked by the docs harness. The quickstart is executed on tiny data.
Both own-data workflows execute. Run classification and JSON extraction end to end against sufficiently sized tiny fixtures, using exactly the documented column names and commands, including any schema file the JSON command references.
README output is real. The output shown in the README matches a real MLX run, recorded in docs/reports/data/018-readme-capture.json.
Links work. The link checker passes, and every moved section is reachable from the README in one click.
Site builds. The docs site builds in strict mode.
Workflows green. CI, Relay, Containers, Docs site, and the offline bundle gate are green on the pull request.

After merge, verify the same workflows are green on the resulting main commit.

6. Terminal state.

On success: merge through the pull request, then print:
before and after counts: total lines, total words, and prose-only words;
the new README section headings;
where each moved block now lives;
the post-merge workflow results.
If blocked: do not merge. Print the exact blocker and the remaining work. MLX being unavailable for the real-output capture counts as blocked; do not substitute torch-cpu output.
Stop.
