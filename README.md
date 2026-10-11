# streamweights

**Build a small model for your task, on hardware you control.**

Bring labeled examples. Fine-tune locally, compare against simple baselines, and export to GGUF or safetensors. Pause and resume supported training runs across Mac and Linux.

## What it does

streamweights (command: `spill`) turns examples you already have into a small model that does one job. Two jobs are supported:

- **Text classification.** Route support tickets: "Can I change my PIN at any ATM?" becomes `change_pin`.
- **Structured JSON extraction.** Pull fields out of text: "Order A-1655: please replace my wool blanket" becomes `{"order_id": "A-1655", "action": "replace", "product": "wool blanket"}`.

It also scores your model against simpler approaches on examples it never saw, so you can tell whether it was worth building. Data stays local by default; remote storage is used when explicitly configured.

## Try it

**Before you start.** An Apple silicon Mac, or a Mac or Linux machine with a CPU ([platforms](#support-and-limitations)); Python 3.10 or newer. The first build downloads two models, 0.93 GB and 92 MB, and a run plus an export write about 1.6 GB more; `spill plan` prints the exact figures before anything runs. Installing and downloading depend on your connection and are not training time: on an M4 Pro with the models already on disk, the build below took 15 s.

```
# setup
pip install git+https://github.com/streamweights/streamweights
spill example banking77 --tiny
spill init banking77-tiny/banking77.csv --input text --output label --project tickets
```
```
spill init banking77.csv: classification (10 distinct short labels over 120 rows, 100% of rows share their label with another row). Train 84, validation 18, final test 18 rows. Cost: $0. Project -> tickets
```
```
# training, and a comparison on rows the model never saw
spill build tickets
```
```
stage 1 done in 3 s; accuracy 1.000 on 18 rows
stage 2 done in 9 s; 42 steps, final loss 0.001835554838180542
stage 3 done in 1 s; accuracy 0.333 on 18 rows
stage 4 done in 1 s; accuracy 0.889 on 18 rows

embedding baseline (MiniLM + logistic regression)  accuracy 1.000  (18 rows)
student, prompted, untrained                       accuracy 0.333  (18 rows)
student, trained                                   accuracy 0.889  (18 rows)
run run-20261011-145259-149d-c29d25: tickets/runs/run-20261011-145259-149d-c29d25
```
```
# export, then inference
spill export tickets
python tickets/exports/*/run_safetensors.py "Can I change my PIN at any ATM?"
```
```
verified safetensors: transformers 5.19.0, torch 2.14.1, float32 on CPU (12 cores); 0 of 8 predictions differ from the training engine's. Quality on these 8 rows (accuracy): source engine 0.875, artifact 0.875, change +0.000. Verified means it loaded and ran, not that quality was kept
change_pin
```

That output was captured from a real run on the Apple GPU ([capture](docs/reports/data/018-readme-capture.json)); the tiny example is a workflow check on 18 validation rows, not a benchmark.

**Inspect before and after.** `spill plan tickets` states the models, downloads, disk and an estimated time per stage, and loads nothing:

```
   qwen2.5:0.5b: 0.93 GB already on disk
   embedding:minilm: 92 MB already on disk
total: 13 s plus unknown (baseline: no measurement on this machine yet); model loading is not included
```

`spill report tickets` prints the full report of the run, shown next. The [guide](docs/guides/turn-a-csv-of-examples-into-an-evaluated-model.md) explains each command.

## Use your own data

**The example is over. From here you bring your own file.** Put your examples in a CSV with a header row. These two snippets show the shape only; a usable file needs many more rows (the bundled example has 120).

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

`init` checks every row and names the file, row, problem and fix. More on the format: [CSV guide](docs/guides/turn-a-csv-of-examples-into-an-evaluated-model.md#your-own-data-in-detail), [formats](docs/formats.md); bundled examples: [sample projects](docs/guides/turn-a-csv-of-examples-into-an-evaluated-model.md#sample-projects).

## Understand the result

The report from the run above, abridged (18 validation rows; [full report](docs/reports/017-scorer-and-protocol.md)):

```
| comparator | accuracy | macro-F1 | rows | invalid predictions | failures | truncated |
|---|---|---|---|---|---|---|
| embedding baseline (MiniLM + logistic regression) | 1.000 | 1.000 | 18 | 0 | 0 | n/a |
| student, prompted, untrained | 0.333 | 0.270 | 18 | 1 | 0 | 0 |
| student, trained | 0.889 | 0.905 | 18 | 1 | 0 | 0 |
The embedding baseline (MiniLM + logistic regression) scored 0.111 higher than the trained student (2 rows).
```

The simple baseline won here, which is the point: the tool helps you discover whether fine-tuning beats a simpler approach before you rely on it. The `verified` line on the export is not a quality claim; [how to check an exported model's quality](docs/reports/017-export-diagnosis.md).

## Go further

- **Continue an interrupted run on another machine.** Stop a build, move it, and finish it on another Mac or Linux machine, on either engine ([portability](docs/portability.md)); a build started on one OS and finished on the other is checked on every push:

  ```
  pip install "streamweights[cloud] @ git+https://github.com/streamweights/streamweights"
  spill init banking77-tiny/banking77.csv --input text --output label --project tickets2 --seed 3
  spill build tickets2 --stop-after train:12
  spill move tickets2 s3://my-bucket/tickets2
  spill resume s3://my-bucket/tickets2
  ```

  [![Relay workflow status: a build started on one OS and finished on the other, every push](https://github.com/streamweights/streamweights/actions/workflows/relay.yml/badge.svg)](https://github.com/streamweights/streamweights/actions/workflows/relay.yml)
- **JSON extraction.** `spill example snips --tiny` is a runnable text-to-JSON project ([guide](docs/guides/turn-a-csv-of-examples-into-an-evaluated-model.md#sample-projects)).
- **Offline bundles.** `spill bundle` packs a project and its models for machines with no network ([portability](docs/portability.md#a-portable-project-and-an-offline-bundle)).
- **Large teachers.** A teacher bigger than your memory can be read from disk, optionally ([guide](docs/guides/turn-a-csv-of-examples-into-an-evaluated-model.md#when-a-teacher-is-useful), [Measured: Llama 3.3 70B, 141 GB unquantized, run on a 48 GB MacBook Pro.](docs/reports/002-phase1.md)).

## Support and limitations

| platform | status |
|---|---|
| Apple silicon, GPU | tested on an M4 Pro, 48 GB |
| macOS and Linux, CPU | tested on every push |
| NVIDIA CUDA | built, untested ([docs/linux.md](docs/linux.md), [issue #1](https://github.com/streamweights/streamweights/issues/1)) |
| Windows, network filesystems | untested |

Limitations: validation so far is on small models and 120-row examples; CUDA is untested; an exported model can differ in quality from the trained one.

Documentation: [docs site](https://streamweights.github.io/streamweights/), [evidence by platform](docs/linux.md#evidence-by-platform), [architecture](docs/architecture.md), [prior art](docs/prior-art.md) (AirLLM, slowllama, Unsloth, FlexGen, DeepSpeed), [plan and status](docs/plan.md), [reports](docs/reports/index.md). Feedback: [Discussions](https://github.com/streamweights/streamweights/discussions) or an issue, with `spill doctor` output for hardware reports. Apache-2.0.
