---
description: Distill a large model into a small one locally. A 70B teacher answers your questions, a small student is tuned on its answers, and both are graded on your exam. Commands and measured rates.
---

# Distill a large model into a small one locally

Let the 70B model answer your unlabeled questions with `spill distill`, tune a small student on those answers with `spill tune`, and grade teacher and student on the same exam with `spill eval`. A 70B decode pass takes 33 to 37 s regardless of batch size ([report 007](../reports/007-phase2.5.md)), so the time is rows x answer length / batch passes.

## The steps

1. Write three files: `prompts.jsonl` (`{"prompt"}`, the questions), `evals.jsonl` (`{"prompt","expected"}`, the exam, never trained on) and optionally `train.jsonl` with your own labels ([formats](../formats.md)).
2. Have the big model answer:

   ```
   spill distill llama3.3:70b prompts.jsonl --out prompts.distill.jsonl
   ```

3. Tune the student on those answers:

   ```
   spill tune qwen2.5:7b prompts.distill.jsonl --name mine
   ```

4. Grade the base, the tuned student and the teacher on the same exam:

   ```
   spill eval evals.jsonl qwen2.5:7b qwen2.5:7b+mine llama3.3:70b
   ```

`spill build my-folder` runs all of it from a folder and prints the path, models and a per-stage estimate first. When you also have labels, they are weighted 2:1 against the teacher's answers. Trained on the big model's answers a small model gets close to it; trained on your own ground truth it can beat it on your task.

Each step can stop and resume: close the laptop and run `spill resume`.

## Try it

```
spill build my-folder
```

Source and issues: [github.com/streamweights/streamweights](https://github.com/streamweights/streamweights).
