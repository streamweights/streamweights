---
description: Fine-tune an LLM on a Mac with one command. spill builds a LoRA adapter from your labeled examples with MLX and grades it on held-out questions. Measured on an M4 Pro.
---

# Fine-tune an LLM on a Mac

Install streamweights and run `spill example banking77 --quick && spill build banking77-quick`. On an M4 Pro it fine-tunes a 0.5B model with LoRA on 500 labeled examples and takes the held-out score from 0.200 to 0.640 in 44 s ([report 009](../reports/009-phase3.5.md)).

## The steps

1. Install, with Python 3.10 or newer on an Apple silicon Mac:

   ```
   pip install git+https://github.com/streamweights/streamweights
   ```

2. Put your files in a folder. `evals.jsonl` is your exam (`{"prompt","expected"}`), `train.jsonl` is your labeled examples (`{"prompt","answer"}`). Check them first: `spill check train.jsonl`. Formats are in [formats](../formats.md).

3. Build:

   ```
   spill build my-folder
   ```

   `build` prints the path, the models and an estimate per stage before it starts, grades the base model, tunes a LoRA adapter, grades it again and prints one table. The default student is `qwen2.5:7b`, which trains at the measured 4.6 TFLOP/s ([report 008](../reports/008-phase3.md)).

4. Ship it as a GGUF for Ollama:

   ```
   spill export qwen2.5:0.5b+banking77-quick --gguf --ollama
   ```

   On the quick example the GGUF agrees with the engine on 19 of 20 prompts ([report 009](../reports/009-phase3.5.md)).

Runs are interruptible: close the laptop and `spill resume`. Nothing leaves the machine and nothing costs money.

## Try it

```
spill example banking77 --quick && spill build banking77-quick
```

Source and issues: [github.com/streamweights/streamweights](https://github.com/streamweights/streamweights).
