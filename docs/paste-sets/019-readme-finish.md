TERMINAL DIRECTIVE: README FINISHING PASS. DISK RESERVE, SHORTER OUTPUT, AND THE CONTINUATION LINK.

This pass stays within the editorial scope of directive 018. Keep the headline, positioning, six-section structure, and own-data examples. Product code must remain unchanged. Docs tests, harness mappings, capture scripts, and site navigation may change to enforce this directive. Reuse the existing capture; this pass does not require another model run.

Rules, absolute:

Never print a status summary or progress report before the terminal state. Printing one ends your turn and stops all work. Until the end, output only tool calls.
Never end the turn to wait. Any wait is a foreground shell loop under the tool timeout. No scheduled wake-ups, no /loop, no questions to the user.
Never invent a result, weaken a gate, or label an unavailable test as passed. If a gate is blocked, leave the work on the branch and finish with a blocked outcome.
No Claude attribution on commits. No em-dashes in files you write.
Work on branch readme-finish. Merge only through a pull request after the gates pass.

1. Start. Save this directive verbatim as docs/paste-sets/019-readme-finish.md. Commit.

2. Disk prerequisite.

Read the code to confirm the default free-space reserve enforced during downloads, and its environment variable. Don't assume the value is 20 GB.
In the prerequisites, state the disk requirement as model and artifact sizes plus that reserve, with the resulting total for the example.
Add one line on how to lower the reserve with the environment variable, and what that risks.
The capture script ran with SPILL_MIN_FREE_GB=2. Record that difference in docs/reports/018-readme-pass.md, and make sure nothing in the README implies the capture ran with defaults.

3. Shorter output on the first-run path.

In "Try it" and "Understand the result", trim the shown output to short verbatim excerpts: only the lines that show progress and the result.
Remove stage timings, the full-precision loss, the run ID, runtime versions, and the long export-verification sentence. Mark each omission with a single ... line.
Keep one compact score comparison, rendered as a normal Markdown table (not inside a code block), with its row count. It must include the baseline's win.
Link once to the complete capture. The full evidence stays in docs/reports/data/018-readme-capture.json and the 018 report.
Docs tests must check two things:
every excerpted line appears verbatim in the capture;
the rendered table's values match the capture exactly.

4. Continuation in "Go further".

Remove the S3 walkthrough from the README. Replace it with one or two lines stating the benefit (continue an interrupted training run on another supported machine), linking directly to the step-by-step walkthrough in docs/portability.md. Keep the benefit prominent.
Keep the walkthrough's command mappings, which now serve docs/portability.md.
Confirm that the harness still discovers and covers those commands there, and record the confirmation.

5. Gates. All must pass on the pull request.

tests/test_docs.py passes:
the excerpt and table checks above are in place;
the new-structure assertions are kept;
all command, link, provenance, and correctness checks are kept.
The command harness covers every remaining README and guide command, including the walkthrough commands in docs/portability.md.
The link checker passes, and the strict docs site build passes.
CI, Relay, Containers, Docs site, and the offline bundle gate are green on the pull request.

After merge, the same workflows must be green on the resulting main commit.

6. Terminal state.

On success: merge through the pull request, then print:
the confirmed reserve value and its variable, and the stated disk total;
before and after counts (total lines, total words, prose-only words) from scripts/readme_counts.py;
confirmation that the harness covers the walkthrough commands in docs/portability.md;
the post-merge workflow results.
If blocked: do not merge. Print the exact blocker and the remaining work.
Stop.
