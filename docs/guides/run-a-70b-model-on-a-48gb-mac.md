---
description: Run a 70B model on a 48 GB Mac without quantizing it. spill streams the full bf16 weights from disk in layer order. Measured with Llama 3.3 70B, 141 GB, on an M4 Pro.
---

# Run a 70B model on a 48 GB Mac

You can run Llama 3.3 70B at full bf16 precision on a 48 GB Mac by streaming its weights from the SSD one layer at a time instead of holding them in memory. The 141 GB model was read at 79.2% of the disk's probed sequential rate ([report 002](../reports/002-phase1.md)), and a decode pass takes 33 to 37 s no matter how many rows are in the batch ([report 007](../reports/007-phase2.5.md)).

## Why it works

A forward pass reads every weight once whether the batch holds one prompt or hundreds, so decode is disk-bound and batching makes it practical. Memory-mapping the same model reached only 11 to 13% of the disk rate ([report 001](../reports/001-phase0.md)); the streaming runner reads layers in order instead.

It is a correctness tier, not a chat tier: slow per prompt, exact, free and local.

## The steps

1. Check the disk. The model needs 131 GB plus a 20 GB floor that downloads never cross ([models](../models.md)). `spill doctor` shows your machine.
2. Run twenty sample prompts. The first call downloads the model, with speed and ETA:

   ```
   spill run llama3.3:70b sample
   ```

3. Run your own file of `{"prompt": "..."}` rows:

   ```
   spill run llama3.3:70b prompts.jsonl --out answers.jsonl
   ```

The pre-run line states the model, the precision, the placement and the estimated time before anything starts. A long unattended batch is measured too: 666 one-thousand-token rows ran in 23 h 51 m with zero interventions ([report 007](../reports/007-phase2.5.md)).

## Try it

```
spill run llama3.3:70b sample
```

Source and issues: [github.com/streamweights/streamweights](https://github.com/streamweights/streamweights).
