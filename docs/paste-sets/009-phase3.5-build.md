TERMINAL DIRECTIVE: PHASE 3.5, spill build, THE EXAMPLE, THE README, AND THE LOOP PROOF. Run autonomously to the terminal state in item 14. Record progress verbatim in state memory. Never end a turn with a question to the user. No check-ins. No scheduled wake-ups, loops, or fallback timers; to wait for the GPU gate, use a detached shell loop that runs git fetch every 5 minutes, never a Claude turn. If something blocks you, make the most reasonable choice, log it under Decisions, and continue. No Claude attribution on any commit.

Isolation rules, absolute until the gate opens:

Run git fetch && git worktree add ../streamweights-phase35 -b phase35 origin/main, with its own venv there.
Never modify, install into, or restart anything in the main checkout or in ../streamweights-phase3.
No Metal before the gate. Downloads of models and datasets are allowed; compute is CPU or mocks only.
The gate is docs/reports/008-phase3.md present on origin/main. Session B (Phase 3, tune) merges before you.
After the gate, rebase onto main and resolve conflicts in favor of Phase 3's engine and tune code. Take README and plan ownership entirely: your rewrite supersedes their edits, but carry over their measured numbers.
Items marked [GPU] are built and CPU-tested before the gate and run on Metal after it.

Cost model (applies to every estimate in this directive):

Eval decode on large models is disk-bound: about 34 s per pass on the 70B here.
Prefill and training are compute-bound: prefill costs 2 × params × tokens, and training costs 6 × params × tokens (forward, recompute, backward).
Divide by the achieved TFLOP/s measured in Phase 3's report. If that number is not yet available, use 5 TFLOP/s and replace it after the gate.
Every pre-run estimate, README time, and diagram uses this model, not stream time alone.
Repo. Save this paste set verbatim as docs/paste-sets/009-phase3.5-build.md. Commit after every numbered item.
Plain formats. Every command accepts these row shapes, plus an optional "system" field on any row:
evals: {"prompt": "...", "expected": "..."}
prompts: {"prompt": "..."}
training: {"prompt": "...", "answer": "..."}
The OpenAI batch and chat shapes remain accepted and are auto-detected per file. spill check validates all of them with line-numbered errors.
spill build <folder>. Behavior is decided by the files present: evals.jsonl (required), train.jsonl (your answers), prompts.jsonl (questions only). Defaults are student qwen2.5:7b and teacher llama3.3:70b, with flags --student, --teacher, --compare <model>, and --base <model> (train the adapter on a big model itself). Paths:
train only: eval the student base, tune on train.jsonl, eval.
prompts only: distill the teacher on prompts.jsonl, tune on the teacher's answers, then eval student base, student+adapter, and teacher.
both: distill the prompts, then tune on your answers plus the teacher's, with your answers weighted 2:1 by default (--weight-own).
any path: --compare adds a model's score to the table.
Behavior:
The adapter is named after the folder.
Every intermediate (*.out.jsonl, *.distill.jsonl, adapters, run manifests) lands inside the folder or in ~/.streamweights/adapters/<name>.
The pre-run line states the path chosen, which models run, and a per-stage time estimate from the cost model, with the total.
build is resumable at any stage via spill resume.
It ends with the score table (model, role, score, rows) and one line: your model: <student>+<name> · spill export <student>+<name>.
Classification-style evals (short expected values) get a default max_tokens of 16, stated in the pre-run line.
Chaining.
spill tune with no data file uses the folder's newest *.distill.jsonl or train.jsonl.
spill eval with no models uses every model with a run against that eval file's hash.
Every command ends by printing the next command.
Shared-prefix reuse (required). Rows in a job that share an identical system prompt or leading prefix compute its KV once per batch and reuse it for every row, in prefill and in decode. It applies to run, distill, eval, and build.
Test on CPU.
[GPU] Verify identical greedy output with and without reuse on 20 rows.
Report the prefill time removed on the banking77 teacher run.
spill export <base>+<adapter>.
Merge the adapter into the base and write merged safetensors.
--gguf [bf16|q8_0|q4_k_m] (default q8_0) converts using llama.cpp's converter (downloaded, not vendored) and writes an Ollama Modelfile beside it.
If ollama is installed, --ollama runs ollama create <name> and prints the ollama run line.
[GPU] Verify on 0.5B: the merged model's greedy output equals base+adapter on 20 prompts, and the GGUF runs in llama.cpp.
Overnight safety.
Long jobs (run, distill, tune, build) hold caffeinate -i for their lifetime.
The pre-run line warns when on battery (pmset -g batt) and says to plug in.
On finish or stop, post a macOS notification via osascript. If --notify <url> is set, also POST a small JSON to that URL.
On launch, any command reports an interrupted job in one line with its spill resume command.
Downloads show progress, speed, and ETA, resume after interruption, check disk before the first byte, and use hf_transfer when available.
spill doctor. One screen showing:
chip, RAM, Metal working set, free disk, Python, package version;
downloaded models and interrupted jobs;
the largest model this machine can run overnight for eval, and the achieved TFLOP/s used for training estimates.
Registry. Add qwen2.5:7b (bf16 safetensors and mlx 8-bit) as the default student tag.
spill example <name>. It creates ./<name>/ with ready files. Ship banking77 from the public PolyAI Banking77 dataset. Verify its license permits redistribution and record it in the example's README; if it does not, pick another public intent-classification set and log why. Files:
evals.jsonl: 300 rows from the test split, with expected labels.
train.jsonl: 2,000 labeled rows from the train split.
prompts.jsonl: a different 2,000 train-split rows, without labels.
Prompt formats:
Untrained base and teacher get a system prompt listing the 77 labels and instructing a label-only answer. Shared-prefix reuse makes this cheap.
The trained student is trained and evaluated without the label list. Fine-tuning teaches it the label set, which keeps its training to minutes.
build handles this per model role. The example README and the report state it plainly.
--quick writes a smaller variant for the under-an-hour path: 100 evals, 500 train, no prompts, student qwen2.5:0.5b. The example's README is a short transcript (spill example banking77 && spill build banking77, then the table).
The loop proof. [GPU] Run on this machine the way a stranger would: plugged in, no supervision.
spill example banking77 --quick && spill build banking77-quick. Record wall time.
Surpass path: spill build banking77 with prompts.jsonl moved aside, plus --compare llama3.3:70b.
Copy path: prompts.jsonl restored and train.jsonl moved aside, as a separate build named banking77-copy.
Record wall time per stage for each build, with the estimate printed before each stage.
Produce one combined table with these rows:
llama3.3:70b untrained
qwen2.5:7b untrained
qwen2.5:7b trained on your labels
qwen2.5:7b distilled from the 70B
Then spill export the best student to GGUF and confirm it runs. Commit the tables and per-row results (gzip if over 10 MB) under docs/reports/009-banking77/.
Visuals. Two SVGs in docs/img/: hand-built or generated, minimal, readable, light and dark safe, few words, with shapes and arrows doing the work.
flow.svg: your files on the left, spill build in the middle, your model on the right. The big model is an optional branch feeding the middle, labeled "only if you're short on labels."
paths.svg: three horizontal bars with lengths proportional to measured time:
labels only: about an hour;
labels plus teacher: one night;
train the 70B itself: several nights.
Use measured times from item 11 and from Phase 3's report.
README, rewritten. Under 150 lines, in this order, no badges, no superlatives, no em-dashes.
Title and one sentence: "Build your own model on the Mac you already own."
Quick start: the install line; spill example banking77 --quick && spill build banking77-quick; the real table from item 11.
How it works in one picture: flow.svg, then three short lines:
your exam (evals.jsonl): questions with the right answers; every model is graded on it, never trained on it;
your homework: train.jsonl with your answers, or prompts.jsonl with questions the big model answers;
your model: a small add-on (an adapter) trained on top of a base, named after your folder.
Which path, and how long: paths.svg, then a five-row table mapping folder contents to what build does, the big model's role, and measured time.
Copy or surpass: two short paragraphs:
Trained on the big model's answers, a small model gets close to it at a fraction of the size.
Trained on your own ground truth, it can beat it on your task.
The big model is needed when labels are short, as the bar, or to train on directly when small isn't enough. Training it directly takes several nights on a laptop, and training and prefill scale with GPU cores (faster on Max and Ultra chips), while evals are limited by the disk.
Ship it: spill export to GGUF and Ollama, one example.
What build does: the four steps as individual commands, one line each, for people who want to stop between them.
Requirements: four lines, plus "spill doctor checks your machine."
Under the hood: one paragraph on streaming.
Status, Feedback, License: short.
Move formats to docs/formats.md, flags to docs/cli.md, the architecture table to docs/models.md, and the per-phase numbers to the reports. Update docs/plan.md:
Phase 3.5 done, with the proof numbers.
Phase 4: VLMs (Qwen2.5-VL first, vision tower resident, decoder streamed, with a public document-image example).
Then: batched prefill GEMMs, CPU path, MoE.
Terminal state. After the gate:
Rebase onto main.
Run every [GPU] item and the item 11 proof.
Write docs/reports/009-phase3.5.md.
Squash-merge phase35 into main, remove the worktree, push.
The report includes:
the item 11 table, with per-stage estimate versus actual;
the export verification;
the shared-prefix result;
Decisions;
plain answers to:
(a) wall time for each path;
(b) did the student trained on labels beat the 70B on banking77;
(c) how close did the distilled student get to the 70B;
(d) did any estimate miss by more than 1.5×, and why;
(e) did anything require intervention;
(f) the first moment a stranger following only the README would get stuck.
Print the report and stop.
