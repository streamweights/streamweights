---
description: Turn a CSV of examples, an input and the output you want, into a small model you can measure and ship. spill init, plan, build, report, export, then continue the same run on another machine.
---

# Turn a CSV of examples into an evaluated model

Give `spill init` a CSV or JSONL with an input column and the output you want (a label, a JSON object or free text); `spill build` trains a small model on it, compares it with a simple baseline and with the untrained model on rows it never saw, and `spill export` writes a GGUF or safetensors model that was loaded and run in an independent runtime. An interrupted run can be moved and finished on another supported machine ([report 015](../reports/015-workflow.md)).

## The steps

1. Install, with Python 3.10 or newer:

   ```
   pip install git+https://github.com/streamweights/streamweights
   ```

2. Start from your examples. The sample CSV below is the tiny Banking77 set (120 labeled banking questions); use your own file the same way, naming its columns:

   ```
   spill example banking77 --tiny
   spill init banking77-tiny/banking77.csv --input text --output label --project tickets
   ```

   `init` checks every row and tells you the file, the row, what is wrong and how to fix it. It chooses how outputs will be scored from the outputs themselves and says which, and why: `exact` for a small label set, `json` for JSON objects, `judge` for anything else (override with `--score exact|json|judge|script:<file>`). It freezes the label vocabulary or the schema from the training rows only, and splits into train, validation and final test with a seed, keeping duplicate inputs and shared `--group` values together. Your own `--val` and `--test` files are kept as given and rejected if they overlap.

3. Look before you run:

   ```
   spill plan tickets
   ```

   It names the models (student, teacher, judge: each a choice) and the engine with the reason, the scoring method, the downloads and sizes, the disk the run needs, and a time per stage. Each time is labeled a measurement (this machine ran that stage before), an assumption (the cost model with a stated rate) or unknown. It loads nothing and downloads nothing. `spill plan tickets --deadline 2h` also lists which combinations of student, teacher, judge and engine on this machine are estimated to finish in two hours, each labeled measured or assumed, with the download size of any model you do not have; see [Deadlines](#deadlines).

4. Build:

   ```
   spill build tickets
   ```

   The default is a small student (`qwen2.5:0.5b`, bf16) trained with LoRA on your examples alone: with labels you need no teacher, and none runs. The result compares a baseline (an embedding model for labels, the nearest training answer for open-ended output), the prompted untrained student and the trained student on the validation rows, with invalid and failed outputs counted as wrong. If the baseline wins, the report says so.

5. Read the report and score the final test once:

   ```
   spill report tickets
   spill test tickets
   ```

   `report` prints `runs/<id>/report.md` and refreshes the index `REPORT.md`. The final test labels are touched by nothing else; `spill test` writes its own record and counts how often the split has been used.

6. Compare runs, re-evaluate, and export:

   ```
   spill evaluate tickets --engine torch-cpu
   spill compare tickets
   spill export tickets --gguf q8_0
   python tickets/exports/*/run_gguf.py "Can I change my PIN at any ATM?"
   ```

   `evaluate` scores the run again on its frozen validation rows under the engine you name and writes a separate record; the run is untouched. `compare` labels each comparison common evaluator, cross-runtime (ranked, with the differences and a caution that evaluation-runtime effects may be in the numbers) or incompatible (not ranked, and why); it uses each run's original evaluation unless you choose a record with `--use <run id>=<evaluation id>`. `export` merges the adapter (`--merge-dtype float32|bf16`), converts to GGUF, loads each artifact in an independent runtime (Transformers, llama.cpp) on validation rows, records where its predictions differ from the training engine's, and measures time to first token, tokens per second and peak memory. The record shows the quality change on those rows next to the result. **Verified means the artifact loaded and ran; it does not mean quality was preserved.** The last line runs the exported model on one input; the record includes the script.

7. Continue on another machine. A completed run cannot resume training, so interrupt one first. The S3 commands need the cloud extra:

   ```
   pip install "streamweights[cloud] @ git+https://github.com/streamweights/streamweights"
   spill init banking77-tiny/banking77.csv --input text --output label --project tickets2 --seed 3
   spill build tickets2 --stop-after train:12
   spill move tickets2 s3://my-bucket/tickets2
   spill resume s3://my-bucket/tickets2
   ```

   `--stop-after train:12` interrupts the build after step 12. `move` stops the writer at a committed boundary, copies a verified snapshot, fences the source and then activates the destination; `resume` takes ownership there and continues from the committed checkpoint on any supported engine. A new run that follows another records the parent as experiment lineage only: it trained from the base model, not from the parent's weights. [Portability](../portability.md) says exactly what is guaranteed and what is not tested.

For text to JSON, `spill example snips --tiny` and `spill init snips-tiny/snips.csv --input text --output json --schema snips-tiny/schema.json`; the report then gives parseable rate, schema-valid rate, per-field accuracy and whole-record accuracy. For open-ended answers, see [Open-ended output and judges](#open-ended-output-and-judges).

## What each command does

Or install it as a tool: `uv tool install git+https://github.com/streamweights/streamweights`. `init` validates every row (file, row, problem, fix), chooses the scoring method (`exact`, `json` or `judge`) from the outputs and says so, freezes the label vocabulary or JSON schema from the training rows, and splits train, validation and final test by seed without separating duplicate inputs or shared `--group` values. `plan` states models, engine, downloads, disk and a time per stage, each labeled a measurement, an assumption or unknown, and loads nothing. `build` trains a small student (`qwen2.5:0.5b`, bf16) with LoRA; a teacher runs only if you ask. `export` merges the adapter, writes safetensors and GGUF, and loads each artifact in an independent runtime on validation rows before it records the export.

## Your own data in detail

The README shows the shape of the two CSV formats. The details:

- **Classification** needs an input column and a label column (`text,label`): `spill init my-tickets.csv --input text --output label --project my-tickets`. The label vocabulary is frozen from the training rows unless you pass `--labels`.
- **Open-ended answers** need an input column and a column of free text (`prompt,answer`): `spill init answers.csv --input prompt --output answer --project answers`. Nothing else is declared. Outputs that are not a small label set and not JSON objects are scored by overlap with your reference, and by a judge if you name or accept one; `--score judge` forces this, `--score exact` is refused with one line when there are more than 200 distinct outputs.
- **A scoring script of your own** with `--score script:my_score.py`, a Python file with `def score(row) -> float`. `row` is `{"id", "input", "gold", "text", "pred"}` (`text` is the model's raw output); the mean is `script_score` and becomes the primary metric. The script is copied into the project, its sha256 is part of the protocol, and changing it makes the next build a new run.
- **Prompts without answers** with `--unlabeled prompts.csv` (the same input column, or `prompt`). A teacher answers them and the student trains on those answers next to yours; they need a teacher, and any prompt that matches a training, validation or test input is removed at `init`.
- **JSON extraction** needs an input column and a column holding one JSON object per row (`text,json`), and optionally a JSON Schema file: `spill init orders.csv --input text --output json --schema schema.json --project orders`. Without `--schema` one is derived from the training rows. A comma or a double quote inside a cell needs the cell in double quotes with each inner quote doubled, as in `"{""order_id"": ""A-1655""}"`.
- **The schema used in the README** is `tests/fixtures/own-data/schema.json` in the repository; the docs harness runs both workflows against `my-tickets.csv` and `orders.csv` from that folder, 120 rows each, with exactly these commands.
- A JSONL file with the same keys works instead of a CSV. Row problems are reported as file, row, problem and fix. Your own `--val` and `--test` files are kept as given and rejected if they overlap the training rows.

## Sample projects

| example | task | data | license |
|---|---|---|---|
| `spill example banking77` (`--quick`, `--tiny`) | classification, 77 intents | BANKING77, PolyAI (Casanueva et al., 2020) | CC BY 4.0, attribution in its README |
| `spill example snips` (`--quick`, `--tiny`) | text to JSON: intent and slots, 3 intents | Snips NLU benchmark, 2017-06 custom intent engines, commit `b86ac7f` | CC0 1.0, citation kept in its README |
| `spill example dolly-qa` (`--tiny`) | open-ended answers: a question about a Wikipedia passage | databricks-dolly-15k, closed_qa category, revision `bdd27f4` | CC BY-SA 3.0, attribution and changes in its `NOTICE.txt`; the data files stay CC BY-SA |

## When a teacher is useful

A teacher is a larger model you name (`spill build tickets --teacher <model>`, or `teacher` in `streamweights.toml`). With labeled examples and no teacher, none runs and nothing is downloaded for one. With a teacher, it answers the training prompts (and any `--unlabeled` prompts), never a validation or test row, and the student trains on those answers next to yours: your labels count twice for each teacher answer, and every training row keeps its origin (`own`, `teacher`, `teacher_unlabeled`) in the run. The teacher's own score is shown next to the others.

When the teacher and the student share a tokenizer (every curated Qwen2.5 does), the teacher also keeps its top-k log-probabilities at each token and the student learns from the distribution, not only from the words: `--distill-loss kl|ce|mix`, with `--kd-temp` for the temperature and `--kd-alpha` for the mix. The default is `mix` when the log-probs are present and the tokenizers match, otherwise `ce` on the teacher's text. The loss and its gradients are checked against a float64 reference, and PyTorch against MLX, on a tiny model ([report 020](../reports/020-restore-vision.md)). It is worth a teacher when you have prompts without answers, or a model much larger than the student that does your task well; it is not needed when you have enough labeled examples. To see whether a teacher's answers helped, build once without `--teacher` and once with it, then run `spill compare`: the two runs are on the same rows and protocol, so they are ranked. A folder with only `prompts.jsonl` still builds, and its report says quality evaluation is unavailable. The older folder workflow (`evals.jsonl`, `train.jsonl`, `prompts.jsonl`, `spill build <folder>`) keeps working.

### Large teachers, streamed from disk

A teacher bigger than RAM is read from disk in layer order, one forward pass reading every weight once whatever the batch size: [Measured: Llama 3.3 70B, 141 GB unquantized, run on a 48 GB MacBook Pro.](../reports/002-phase1.md) mmap reached only 11 to 13% of the disk rate on a model 1.5x RAM, the streaming runner 79.2%. A teacher or judge this size is a 70B-class run on large hardware and many hours; the guided examples here use a 1.5B teacher, and a 70B teacher in the guided workflow is the scale proof still to do ([plan](../plan.md)).

## Open-ended output and judges

When outputs are free text there is no single right answer, so a project is scored in two layers and the report keeps them apart.

1. **Overlap with your reference**, always: token F1, ROUGE-L and exact match against the answer you wrote. It measures similarity to that one answer, not quality, and is labeled that way everywhere.
2. **A judge**, when you name one (`--judge <model>`) or accept the one `spill plan` recommends for your machine (`--judge recommended`), or when you named a teacher and no judge (the teacher judges, unless you say `--judge none` or name another). `spill plan` and `spill build` state that judged scoring needs a judge and print a recommendation with its download size, disk need and estimated judging time; nothing is downloaded unless you name or accept it. A judge may be bigger than your memory and is read from disk like any other model.

The judge scores each output from 1 to 5 against a rubric and your reference (pointwise), and compares two models' outputs on the same input twice, with the positions swapped (pairwise), so a bias toward the first answer cancels. The report gives win, tie and loss rates and how often the two orders disagreed; a disagreement counts as a tie. Where a reference is short enough to check mechanically, it also reports how often the judge agreed with that check. Every judged number is labeled **one model's assessment, not accuracy**. A judge never scores its own outputs: if the judge is also the teacher, the teacher's row is reported as reference only and left out of the rankings, and the report says so.

The rubric is `rubric.md` in the project folder if there is one; otherwise this default:

```
Score the answer from 1 to 5.
5: fully correct and complete for the task input, and it follows any instructions given.
4: correct, with a small omission or a small wording problem.
3: partly correct, or correct but missing something that matters.
2: mostly wrong or off-topic, with some relevant content.
1: wrong, empty, or unrelated.
When a reference answer is given, treat it as a good answer, not the only one: an answer that
says the same thing in other words deserves the same score. Judge the content, not the style.
```

The rubric text, the judge's identity, the prompt templates and the decoding are part of the evaluation-protocol fingerprint, so `spill compare` will not rank runs judged differently and says which part differs. A small judge is a weak instrument: measure it. The calibration figure and the order disagreements in the report are how, and [report 020](../reports/020-restore-vision.md) shows what a 1.5B judge did on a real project, including where it was wrong.

## Deadlines

`spill plan <project> --deadline 2h` (or a timestamp such as `2026-10-14T09:00`) estimates every combination of student, teacher, judge and engine that this machine could run, with the project's cost model and calibrated rates, and lists the ones that are estimated to finish in time. Each says whether its estimate is measured (every stage has a measurement on this machine) or assumed, and the recommended one is marked: the largest student, then the largest teacher, then a judge, then the shorter estimate. A combination with a model you do not have shows its download size and is never fetched by `plan`; download time is not counted. If nothing fits, `plan` says so and shows the closest. `spill build <project> --deadline 2h` builds what the project is configured to build only if that is estimated to fit; otherwise it does not start, lists what does fit and prints the command for the recommended one. It never lowers a setting to meet a deadline. This is local planning only: nothing is run remotely, provisioned or scheduled.

## Try it

```
pip install git+https://github.com/streamweights/streamweights
spill example banking77 --tiny
spill init banking77-tiny/banking77.csv --input text --output label --project tickets
spill build tickets
```

Source and issues: [github.com/streamweights/streamweights](https://github.com/streamweights/streamweights).
