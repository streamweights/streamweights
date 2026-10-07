# streamweights

Build your own model on the Mac you already own.

## Quick start

```
pip install git+https://github.com/streamweights/streamweights
spill example banking77 --quick && spill build banking77-quick
```

Or install it as a tool: `uv tool install git+https://github.com/streamweights/streamweights`.

That builds a small model that classifies banking questions into 77 intents, from 500 of your own labeled examples, and grades it on 100 held-out questions it never trained on. The real result, on an M4 Pro with the 0.5B model already downloaded, took 44 s:

| model | role | score | rows |
|---|---|---|---|
| qwen2.5:0.5b+banking77-quick | your model | 0.640 | 100 |
| qwen2.5:0.5b | base (untrained) | 0.200 | 100 |

A fresh install, with the `pip install` and the 0.9 GB model download, took 79.5 s from install to the end of build ([transcript](docs/reports/010-fresh-install.txt)) and scored 0.640 and 0.210: bf16 output shifts slightly with batch shape.

## How it works in one picture

![your files go into spill build, which makes your model; a big model optionally helps](docs/img/flow.svg)

- **your exam** (`evals.jsonl`): questions with the right answers. Every model is graded on it, never trained on it.
- **your homework**: `train.jsonl` with your answers, or `prompts.jsonl` with questions the big model answers.
- **your model**: a small add-on (an adapter) trained on top of a base, named after your folder.

## Which path, and how long

`spill build <folder>` reads the files in the folder and picks the path. Before it starts it prints the path, the models, an estimate per stage and the total. Times below are measured; where a path has not been run end to end, the table gives the rate the time depends on, not a total.

| folder has | build does | the big model's role | time |
|---|---|---|---|
| evals, train | grades the base, tunes on your answers, grades again | none | 44 s for the quick example (0.5B, 500 rows, 250 steps). A 7B student trains at the measured 4.6 TFLOP/s: 6 x 7B x tokens / 4.6 TFLOP/s |
| evals, prompts | the big model answers the prompts, the student tunes on its answers, grades student, student+adapter and teacher | answers the questions | a 70B decode pass takes 33 to 37 s (Phase 2.5), so rows x answer length / batch passes |
| evals, train, prompts | both, your answers weighted 2:1 | answers the questions you left unlabeled | the two rows above added |
| any, with `--compare llama3.3:70b` | adds the 70B's score to the table | the bar to beat | the 70B grades the same rows at 33 to 37 s per pass |
| any, with `--base llama3.3:70b` | trains the adapter on the 70B itself | the model being trained | about 11 trained tokens/s on the 70B, roughly 400,000 tokens per 10-hour night |

## Copy or surpass

Trained on the big model's answers, a small model gets close to it at a fraction of the size.

Trained on your own ground truth, it can beat it on your task.

The big model is needed when labels are short, as the bar, or to train on directly when small isn't enough. Training it directly takes several nights on a laptop, and training and prefill scale with GPU cores (faster on Max and Ultra chips), while evals are limited by the disk.

## Ship it

```
spill export qwen2.5:0.5b+banking77-quick --gguf --ollama
```

That merges the adapter into the base, converts to GGUF with llama.cpp's own converter, writes an Ollama Modelfile beside it, and, if Ollama is installed, runs `ollama create` and prints the `ollama run` line. On the quick example the GGUF runs in llama.cpp and agrees with the engine on 19 of 20 prompts (`docs/reports/009-phase3.5.md`).

## What build does

The steps as individual commands, if you want to stop between them:

- `spill distill llama3.3:70b prompts.jsonl --out prompts.distill.jsonl` the big model answers your questions
- `spill tune qwen2.5:7b train.jsonl --name mine` train the adapter (with no file it takes the folder's newest `*.distill.jsonl` or `train.jsonl`)
- `spill eval evals.jsonl qwen2.5:7b qwen2.5:7b+mine llama3.3:70b` grade every model on the exam (with no models it uses every model that has a run against that file)
- `spill export qwen2.5:7b+mine --gguf` ship it

Every command ends by printing the one that usually comes next. Formats are in [docs/formats.md](docs/formats.md), commands in [docs/cli.md](docs/cli.md), models in [docs/models.md](docs/models.md).

## Runs anywhere

The same command runs on a Mac, a Linux box or a cloud GPU, and a job moves between them and resumes: every job is a sequence of steps or rows, and any machine can take the next one from a checkpoint at `--state` (a path, `s3://`, `gs://` or `az://`). Engines are thin: MLX on Apple silicon, PyTorch everywhere else. [docs/portability.md](docs/portability.md) says what moves and what it costs.

```
spill tune qwen2.5:7b train.jsonl --name mine --state s3://my-bucket/mine   # start anywhere, rerun to resume anywhere
```

- **Headless.** Piped, or with `--headless`, a job writes JSON-lines events on stdout (start, step or row, checkpoint, preempted, done, error), and on SIGTERM writes a checkpoint within 30 seconds and exits 75 so a scheduler retries. `--emit-config job.json` writes a laptop run as a file; `--config job.json` runs it anywhere.
- **Containers.** `ghcr.io/streamweights/spill:cpu` and `:cuda` run `spill` headless with `SPILL_HOME` on a volume.
- **Schedulers.** SkyPilot, Slurm and Kubernetes examples in [examples/schedulers](examples/schedulers), described in [docs/schedulers.md](docs/schedulers.md).

## Requirements

- An Apple silicon Mac (tested on a 48 GB M4 Pro) for the fastest path. Linux, Windows and Intel Macs run on PyTorch: see Runs anywhere and Linux and other platforms.
- Python 3.10 or newer.
- Disk for the models you use plus a 20 GB floor that downloads never cross (the 7B student is 15 GB, the 70B teacher 131 GB).
- Plugged in for anything long: long jobs hold the Mac awake and warn when it is on battery. `spill doctor` checks your machine.

## Linux and other platforms

`pip install` brings PyTorch everywhere and MLX only on Apple silicon. On the PyTorch engines `run`, `distill`, `tune`, `eval` and `export` work, with streaming from disk, batching, log-probs and adapters. **CPU is verified**: identity, tune and resume gates on the 0.5B ([report](docs/reports/012-run-anywhere.md)), and a Linux CPU build in CI. **NVIDIA is built but awaiting verification**; on a CUDA machine run

```
docker run --gpus all ghcr.io/streamweights/spill:cuda python -m streamweights.verify_cuda
```

`spill build` is Apple silicon only for now and says so in one line elsewhere. Details in [docs/linux.md](docs/linux.md); the tracking issue is [#1](https://github.com/streamweights/streamweights/issues/1). If you would run it on Linux, comment there with the hardware you have.

## Under the hood

The full bf16 model is streamed from disk in layer order instead of held in memory: one forward pass reads every weight once whether the batch holds one prompt or five hundred, so a 70B model that cannot fit in 48 GB still runs every row at full precision. Rows that share a system prompt compute its keys and values once. Training streams the same way, recomputing each layer on the way back. The engines are thin: MLX on Apple silicon, PyTorch (Hugging Face transformers layers bound from the same ring) everywhere else. streamweights owns the ring, the job layer and the CLI.

## Status

Phases 0 to 3.5 are done: run, distill, tune, eval, build, export (reports in [docs/reports](docs/reports)). Run-anywhere is done too: portable jobs, the PyTorch engines, headless mode, containers and scheduler examples, with the CPU gates in [docs/reports/012-run-anywhere.md](docs/reports/012-run-anywhere.md). The banking77 table for the 70B teacher and 7B student arrives with the full proof run. Next, in order: the verification batch (that proof run, the CUDA gates on a real GPU, a cross-cloud resume demo), then PyPI and launch, then vision-language models, then mixture-of-experts ([docs/plan.md](docs/plan.md)).

## Feedback

Send a transcript of your first fifteen minutes and the one moment you got stuck, in [Discussions](https://github.com/streamweights/streamweights/discussions) or as an issue.

## License

Apache-2.0. The banking77 example data is CC BY 4.0 (PolyAI), see its README.
