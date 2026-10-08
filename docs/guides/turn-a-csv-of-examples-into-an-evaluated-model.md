---
description: Turn a CSV of labeled examples into a small model you can measure and ship. spill init, plan, build, report, export, then continue the same run on another machine.
---

# Turn a CSV of examples into an evaluated model

Give `spill init` a CSV or JSONL with an input column and an answer column; `spill build` trains a small model on it, compares it with an embedding baseline and with the untrained model on rows it never saw, and `spill export` writes a verified GGUF or safetensors model. The same run can stop on one machine and finish on another ([report 015](../reports/015-workflow.md)).

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

   `init` checks every row and tells you the file, the row, what is wrong and how to fix it. It suggests the task (classification or json; it will not guess from free text, so pass `--task`), freezes the label vocabulary from the training rows only, and splits into train, validation and final test with a seed, keeping duplicate inputs and shared `--group` values together. Your own `--val` and `--test` files are kept as given and rejected if they overlap.

3. Look before you run:

   ```
   spill plan tickets
   ```

   It names the models and the engine with the reason, the downloads and sizes, the disk the run needs, and a time per stage. Each time is labeled a measurement (this machine ran that stage before), an assumption (the cost model with a stated rate) or unknown. It loads nothing and downloads nothing.

4. Build:

   ```
   spill build tickets
   ```

   The default is a small student (`qwen2.5:0.5b`, bf16) trained with LoRA. A teacher runs only if you ask (`--teacher <model>`), and only on training rows. The result compares the embedding baseline, the prompted untrained student and the trained student on the validation rows, with invalid and failed outputs counted as wrong. If the baseline wins, the report says so.

5. Read the report and score the final test once:

   ```
   spill report tickets
   spill test tickets
   ```

   `report` prints `runs/<id>/report.md` and refreshes the index `REPORT.md`. The final test labels are touched by nothing else; `spill test` writes its own record and counts how often the split has been used.

6. Compare runs and export:

   ```
   spill compare tickets
   spill export tickets --gguf q8_0
   ```

   `compare` ranks runs only if they were evaluated on the same rows with the same metric and protocol, and otherwise says why not. `export` merges the adapter, converts to GGUF, loads each artifact in an independent runtime (Transformers, llama.cpp) on validation rows, records where its predictions differ from the training engine's, and measures time to first token, tokens per second and peak memory. The record includes a script that loads the artifact and runs one input.

7. Continue on another machine:

   ```
   spill move tickets s3://my-bucket/tickets
   spill resume s3://my-bucket/tickets
   ```

   `move` stops the writer at a committed boundary, copies a verified snapshot, fences the source and then activates the destination; `resume` takes ownership there and continues from the committed checkpoint, on any engine. [Portability](../portability.md) says exactly what is guaranteed and what is not tested.

For text to JSON, `spill example snips --tiny` and `spill init snips-tiny/snips.csv --input text --output json --schema snips-tiny/schema.json`; the report then gives parseable rate, schema-valid rate, per-field accuracy and whole-record accuracy.

## Try it

```
pip install git+https://github.com/streamweights/streamweights
spill example banking77 --tiny
spill init banking77-tiny/banking77.csv --input text --output label --project tickets
spill build tickets
```

Source and issues: [github.com/streamweights/streamweights](https://github.com/streamweights/streamweights).
