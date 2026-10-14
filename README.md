# streamweights

**Build a small model for your task, on hardware you control.**

Bring examples of your task. Learn from a large model running on your own machine, train a small one, measure it honestly, and export it.

## What it does

streamweights (command: `spill`) turns examples of an input and the output you want into a small model of your own. Any task of that shape works:

- **Labeling.** Route support tickets: "Can I change my PIN at any ATM?" becomes `change_pin`.
- **Extraction.** Pull fields out of text: "Order A-1655: please replace my wool blanket" becomes `{"order_id": "A-1655", "action": "replace", "product": "wool blanket"}`.
- **Open-ended answers.** Read a passage and a question, and answer in your own words.

It scores your model against simpler approaches on examples it never saw, so you can tell whether it was worth building: by exact match for labels, field by field for JSON, and for open-ended answers by overlap with your reference and by a judge you name. A large model can teach the small one and judge its answers, on your own machine, even when it is bigger than your memory; both are optional. Data stays local by default; remote storage is used when explicitly configured.

## Try it

**Before you start.** An Apple silicon Mac, or a Mac or Linux machine with a CPU ([platforms](#support-and-limitations)); Python 3.10 or newer.

**Disk.** Models are 0.93 GB and 92 MB; a run and an export write about 1.6 GB. Downloads also keep a 20 GB free-space reserve (`SPILL_MIN_FREE_GB`), so the example needs about 23 GB free. Lower the reserve with `SPILL_MIN_FREE_GB=2`; the risk is a nearly full disk. `spill plan` prints exact figures first.

**Time.** Installing and downloading depend on your connection and are not training time. With the models on disk, the build below took 13 s on an M4 Pro.

```
# setup
pip install git+https://github.com/streamweights/streamweights
spill example banking77 --tiny
spill init banking77-tiny/banking77.csv --input text --output label --project tickets
```
```
# training, and a comparison on rows the model never saw
spill build tickets
```
```
stage 1/4: embedding + logistic regression baseline
...
stage 2/4: tune qwen2.5:0.5b on 84 training rows (84 yours); estimate 11 s (measurement)
...
```
```
# export, then inference
spill export tickets
python tickets/exports/*/run_safetensors.py "Can I change my PIN at any ATM?"
```
```
   merged shard 1/1: model.safetensors
...
change_pin
```

Excerpts are verbatim from a real Apple GPU run with `SPILL_MIN_FREE_GB=2`, not the default ([full capture](docs/reports/data/020-readme-capture.json)). Scores are below.

**Inspect.** `spill plan tickets` states models, downloads, disk and time per stage, loading nothing:

```
   qwen2.5:0.5b: 0.93 GB already on disk
   embedding:minilm: 92 MB already on disk
total: 12 s plus unknown (baseline: no measurement on this machine yet); model loading is not included
```

`spill report tickets` prints the full report of the run, shown next. The [guide](docs/guides/turn-a-csv-of-examples-into-an-evaluated-model.md) explains each command.

## Use your own data

**The example is over. From here you bring your own file.** Put your examples in a CSV with a header row. These snippets show the shape only; a usable file needs many more rows (the bundled example has 120).

Classification, `my-tickets.csv`, columns `text,label`:

```
text,label
"Can I change my PIN at any ATM?",change_pin
"My card has not arrived yet",card_arrival
```

JSON extraction, `orders.csv`, columns `text,json`. A comma or quote inside a cell needs the cell in double quotes, with each inner quote doubled:

```
text,json
"Order A-1655: please replace my wool blanket","{""order_id"": ""A-1655"", ""action"": ""replace"", ""product"": ""wool blanket""}"
```

`schema.json` says what a valid answer looks like; an answer that does not satisfy it counts as wrong:

```
{
  "type": "object",
  "properties": {
    "order_id": {"type": "string"},
    "action": {"type": "string", "enum": ["refund", "replace", "cancel"]},
    "product": {"type": "string"}
  },
  "required": ["order_id", "action"],
  "additionalProperties": false
}
```

Open-ended answers, `answers.csv`, columns `prompt,answer`. The output is any text:

```
prompt,answer
"Passage: Final Fantasy is a Japanese science fantasy franchise created by Hironobu Sakaguchi. Question: Who created it?",Hironobu Sakaguchi
```

**How outputs are scored is chosen from your outputs, and `init`, `plan` and the report say which.** JSON objects are scored field by field, a small label set (at most 200 distinct values) by exact match, and anything else as open-ended: by overlap with your reference, never called accuracy, and by a judge if you name one (`--judge <model>`) or accept the one `spill plan` recommends. Without a judge it still builds, and `plan` shows the recommended judge, its download size and judging time, and downloads nothing. `--score exact|json|judge|script:<file>` overrides the choice. **With your own labeled examples no teacher is needed.**

Point `spill init` at your columns, then build, export and ask exactly as above:

```
# classification
spill init my-tickets.csv --input text --output label --project my-tickets
spill build my-tickets
spill export my-tickets
python my-tickets/exports/*/run_safetensors.py "Can I get a new PIN at a cash machine?"
```
```
# JSON extraction
spill init orders.csv --input text --output json --schema schema.json --project orders
spill build orders
spill export orders
python orders/exports/*/run_safetensors.py "Order B-2201: cancel the glass vase"
```
```
# open-ended answers, judged by a model you name
spill init answers.csv --input prompt --output answer --project answers
spill plan answers
spill build answers --judge qwen2.5:1.5b
```

`init` checks every row and names the file, row, problem and fix. More on the format: [CSV guide](docs/guides/turn-a-csv-of-examples-into-an-evaluated-model.md#your-own-data-in-detail), [formats](docs/formats.md); bundled examples, including an open-ended one: [sample projects](docs/guides/turn-a-csv-of-examples-into-an-evaluated-model.md#sample-projects).

## Understand the result

The report from the run above, abridged (18 validation rows each; [full report](docs/reports/018-readme-pass.md)):

| comparator | accuracy | validation rows |
|---|---|---|
| embedding baseline (MiniLM + logistic regression) | 1.000 | 18 |
| student, prompted, untrained | 0.333 | 18 |
| student, trained | 0.889 | 18 |

```
...
The embedding baseline (MiniLM + logistic regression) scored 0.111 higher than the trained student (2 rows).
```

The simple baseline won here, which is the point: the tool helps you discover whether fine-tuning beats a simpler approach before you rely on it. The `verified` line on the export is not a quality claim; [how to check an exported model's quality](docs/reports/017-export-diagnosis.md).

## Large teacher and judge

Both are optional, and each is a model you name. **A teacher** answers your training prompts, and the student learns from its answers and, when the two share a tokenizer, from its probabilities (`--teacher`, `--distill-loss`; [when it is worth it](docs/guides/turn-a-csv-of-examples-into-an-evaluated-model.md#when-a-teacher-is-useful)). It is worth it for prompts without answers, or a model much better at your task than one you can run; with labeled examples none runs unless you ask. **A judge** scores open-ended output from 1 to 5 against a rubric (`rubric.md`, or the [default](docs/guides/turn-a-csv-of-examples-into-an-evaluated-model.md#open-ended-output-and-judges)) and compares two models in both orders. Its scores are one model's assessment, never accuracy, and it never scores its own outputs. Either may be bigger than your memory and is read from disk ([Measured: Llama 3.3 70B, 141 GB unquantized, run on a 48 GB MacBook Pro.](docs/reports/002-phase1.md)). Nothing is downloaded unless you name it or accept the model `spill plan` recommends.

A real result: `spill example dolly-qa --tiny`, `qwen2.5:1.5b` as teacher and judge, `qwen2.5:0.5b` as the student, Apple GPU, 13 validation rows ([capture](docs/reports/data/020-judged-mlx.json), [report 020](docs/reports/020-restore-vision.md)):

```
nearest training answer (TF-IDF, no model)  judge score 1-5 (one model's assessment, not accuracy) 2.462  (13 rows)
student, prompted, untrained                judge score 1-5 (one model's assessment, not accuracy) 4.846  (13 rows)
student, trained                            judge score 1-5 (one model's assessment, not accuracy) 4.538  (13 rows)
teacher                                     judge score 1-5 (one model's assessment, not accuracy) reference only, judge is the same model  (13 rows)
  trained vs untrained, both orders: trained won 0, tied 11, lost 2 of 13 rows; the two orders disagreed on 8 (62%); a disagreement counts as a tie.
```

The judge preferred the untrained student while overlap with the references favored the trained one, and it disagreed with itself across the two orders on 8 of 13 rows: a small judge is a rough instrument.

## Go further

- **Continue an interrupted training run on another supported machine.** Stop a build on one Mac or Linux machine and finish it on another, on either engine: [step-by-step walkthrough](docs/portability.md#continuing-on-another-machine-step-by-step). A build started on one OS and finished on the other is checked on every push.

  [![Relay workflow status: a build started on one OS and finished on the other, every push](https://github.com/streamweights/streamweights/actions/workflows/relay.yml/badge.svg)](https://github.com/streamweights/streamweights/actions/workflows/relay.yml)
- **JSON extraction.** `spill example snips --tiny` is a runnable text-to-JSON project ([guide](docs/guides/turn-a-csv-of-examples-into-an-evaluated-model.md#sample-projects)).
- **Offline bundles.** `spill bundle` packs a project and its models for machines with no network ([portability](docs/portability.md#a-portable-project-and-an-offline-bundle)).
- **Also.** `spill example dolly-qa --tiny` (open-ended answers), `--distill-loss kl|ce|mix` (a teacher's per-token probabilities), and `spill plan --deadline 2h`, which lists the combinations of models and engine on this machine estimated to finish in time and downloads nothing ([guide](docs/guides/turn-a-csv-of-examples-into-an-evaluated-model.md#deadlines)).

## Support and limitations

| platform | status |
|---|---|
| Apple silicon, GPU | tested on an M4 Pro, 48 GB |
| macOS and Linux, CPU | tested on every push |
| NVIDIA CUDA | built, untested ([docs/linux.md](docs/linux.md), [issue #1](https://github.com/streamweights/streamweights/issues/1)) |
| Windows, network filesystems | untested |

Limitations: validation so far is on small models (0.5B and 1.5B) and examples of 84 to 120 rows; a small judge is a rough instrument; CUDA is untested; an exported model can differ in quality from the trained one.

Documentation: [docs site](https://streamweights.github.io/streamweights/), [evidence by platform](docs/linux.md#evidence-by-platform), [architecture](docs/architecture.md), [prior art](docs/prior-art.md) (AirLLM, slowllama, Unsloth, FlexGen, DeepSpeed), [plan and status](docs/plan.md), [reports](docs/reports/index.md). Feedback: [Discussions](https://github.com/streamweights/streamweights/discussions) or an issue, with `spill doctor` output for hardware reports. Apache-2.0.
