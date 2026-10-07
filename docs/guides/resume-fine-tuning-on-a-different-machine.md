---
description: Resume a fine-tuning job on a different machine. spill saves a hardware-neutral checkpoint at --state, so a tune started on a Mac can finish on a Linux box or a cloud GPU. Measured gate numbers.
---

# Resume a fine-tuning job on a different machine

Give the job a `--state` location, stop it, and run the same command on the other machine: it finds the committed checkpoint and continues from it. Resumed across MLX and PyTorch on the 0.5B, the continued loss curve ended 0.004 from the uninterrupted run, against 0.016 between two clean runs ([report 012](../reports/012-run-anywhere.md)).

## The steps

1. Start on the first machine, with state on a path every machine can reach (a shared path, `s3://`, `gs://` or `az://`):

   ```
   spill tune qwen2.5:7b train.jsonl --name mine --state s3://my-bucket/mine
   ```

2. Stop it. On SIGTERM a headless job writes a checkpoint and exits 75, so a scheduler retries it.
3. On the other machine, run the identical command:

   ```
   spill tune qwen2.5:7b train.jsonl --name mine --state s3://my-bucket/mine
   ```

The cloud backends come with `pip install "streamweights[cloud] @ git+https://github.com/streamweights/streamweights"`.

## What moves

A checkpoint is float32 safetensors plus a `state.json`, with a `COMMIT` marker written last, so a half-written one is ignored. At rank 16 on every linear it is about 100 MB on the 0.5B and about 2.5 GB on the 70B. The weights do not move; each machine caches them under `SPILL_HOME`. A job refuses, in one line, to continue from a checkpoint made with different weights, data, learning rate, schedule or seed. Details: [portability](../portability.md).

The same works for `run`, `distill` and `eval`, where the unit is one completed row: an eval stopped after 30 of 60 rows on MLX finished on torch-cpu with 60 unique rows, none missing or duplicated (report 012).

## Try it

```
spill tune qwen2.5:0.5b train.jsonl --name mine --state ./state
```

Run it, press Ctrl-C, run it again. Source and issues: [github.com/streamweights/streamweights](https://github.com/streamweights/streamweights).
