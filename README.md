# streamweights

Build your own model on the Mac you already own.

## Quick start

```
pip install git+https://github.com/streamweights/streamweights
spill example banking77 --quick && spill build banking77-quick
```

@@QUICK_TABLE@@

## How it works in one picture

![your files go into spill build, which makes your model; a big model optionally helps](docs/img/flow.svg)

- **your exam** (`evals.jsonl`): questions with the right answers. Every model is graded on it, never trained on it.
- **your homework**: `train.jsonl` with your answers, or `prompts.jsonl` with questions the big model answers.
- **your model**: a small add-on (an adapter) trained on top of a base, named after your folder.

## Which path, and how long

![three bars with lengths proportional to measured time](docs/img/paths.svg)

@@PATHS_TABLE@@

## Copy or surpass

Trained on the big model's answers, a small model gets close to it at a fraction of the size.
@@COPY_NUMBERS@@

Trained on your own ground truth, it can beat it on your task. @@SURPASS_NUMBERS@@

The big model is needed when labels are short, as the bar, or to train on directly when small isn't enough. Training it directly takes several nights on a laptop, and training and prefill scale with GPU cores (faster on Max and Ultra chips), while evals are limited by the disk.

## Ship it

```
spill export qwen2.5:7b+banking77 --gguf q8_0 --ollama
```

That merges the adapter into the base, converts to GGUF with llama.cpp's own converter, writes an Ollama Modelfile beside it, and, if Ollama is installed, runs `ollama create` and prints the `ollama run` line. @@EXPORT_LINE@@

## What build does

The four steps as individual commands, if you want to stop between them:

- `spill distill llama3.3:70b prompts.jsonl --out prompts.distill.jsonl` the big model answers your questions
- `spill tune qwen2.5:7b train.jsonl --name mine` train the adapter (with no file it takes the folder's newest `*.distill.jsonl` or `train.jsonl`)
- `spill eval evals.jsonl qwen2.5:7b qwen2.5:7b+mine llama3.3:70b` grade every model on the exam (with no models it uses every model that has a run against that file)
- `spill export qwen2.5:7b+mine --gguf` ship it

Every command ends by printing the one that usually comes next. Formats are in [docs/formats.md](docs/formats.md), flags in [docs/cli.md](docs/cli.md), models in [docs/models.md](docs/models.md).

## Requirements

- An Apple silicon Mac (tested on a 48 GB M4 Pro); Linux and Intel Macs get the slow llama.cpp path.
- Python 3.10 or newer.
- Disk for the models you use plus a 20 GB floor that downloads never cross (the 7B student is 15 GB, the 70B teacher 141 GB).
- Plugged in for anything long: long jobs hold the Mac awake and warn when it is on battery. `spill doctor` checks your machine.

## Under the hood

The full bf16 model is streamed from disk in layer order instead of held in memory: one forward pass reads every weight once whether the batch holds one prompt or five hundred, so a 70B model that cannot fit in 48 GB still runs every row at full precision. Rows that share a system prompt compute its keys and values once. Training streams the same way, recomputing each layer on the way back. Everything else is plain MLX on Apple silicon.

## Status

Phases 0 to 3.5 are done: run, distill, tune, eval, build, export. Next is vision-language models, then batched prefill, a CPU path and mixture-of-experts ([docs/plan.md](docs/plan.md)). Measured numbers for every phase are in [docs/reports](docs/reports).

## Feedback

Send a transcript of your first fifteen minutes and the one moment you got stuck. Open an issue.

## License

Apache-2.0. The banking77 example data is CC BY 4.0 (PolyAI), see its README.
