---
description: Turn a CSV of labeled examples into a small model you can measure and ship. spill init, plan, build, report, export, then continue the same run on another machine.
---

# Turn a CSV of examples into an evaluated model

Give `spill init` a CSV or JSONL with an input column and an answer column; `spill build` trains a small model on it, compares it with an embedding baseline and with the untrained model on rows it never saw, and `spill export` writes a GGUF or safetensors model that was loaded and run in an independent runtime. An interrupted run can be moved and finished on another supported machine ([report 015](../reports/015-workflow.md)).

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

For text to JSON, `spill example snips --tiny` and `spill init snips-tiny/snips.csv --input text --output json --schema snips-tiny/schema.json`; the report then gives parseable rate, schema-valid rate, per-field accuracy and whole-record accuracy.

## Try it

```
pip install git+https://github.com/streamweights/streamweights
spill example banking77 --tiny
spill init banking77-tiny/banking77.csv --input text --output label --project tickets
spill build tickets
```

Source and issues: [github.com/streamweights/streamweights](https://github.com/streamweights/streamweights).
