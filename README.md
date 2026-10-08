# streamweights

**A 70B model doesn't fit on your laptop. Build your own model from it anyway.**

Distill, fine-tune and evaluate on whatever hardware you have. Start a job anywhere, finish it anywhere.

[Measured: Llama 3.3 70B, 141 GB unquantized, run on a 48 GB MacBook Pro.](docs/reports/002-phase1.md)

streamweights (command: `spill`) fine-tunes and evaluates a LLM locally with LoRA, on a Mac (MLX) or a Linux box (PyTorch), and can stream a full-precision 70B teacher from disk when you ask for one, even when it is bigger than RAM. Bring a CSV of labeled examples, build a small model, see how it compares with simple baselines on rows it never saw, export it, and continue the same work on another machine. Everything runs locally and no data leaves it.

## Your examples to a model

```
pip install git+https://github.com/streamweights/streamweights
spill example banking77 --tiny
spill init banking77-tiny/banking77.csv --input text --output label --project tickets
spill plan tickets
spill build tickets
spill report tickets
spill export tickets --gguf q8_0
```

Or as a tool: `uv tool install git+https://github.com/streamweights/streamweights`. `init` validates every row (file, row, problem, fix), suggests the task (`classification` or `json`, never guessed from free text), freezes the label vocabulary or JSON schema from the training rows, and splits train, validation and final test by seed without separating duplicate inputs or shared `--group` values. `plan` states models, engine, downloads, disk and a time per stage, each labeled a measurement, an assumption or unknown, and loads nothing. `build` trains a small student (`qwen2.5:0.5b`, bf16) with LoRA; a teacher runs only if you ask. `export` merges the adapter, writes safetensors and GGUF, and loads each artifact in an independent runtime before it records the export. [Guide](docs/guides/turn-a-csv-of-examples-into-an-evaluated-model.md), [formats](docs/formats.md), [commands](docs/cli.md).

## Understanding results

The report compares, on the validation rows, an embedding baseline (MiniLM plus logistic regression, classification), the prompted untrained student, the trained student and the teacher if used. Measured on an M4 Pro, 48 GB, from the tiny example, one run each ([report 015](docs/reports/015-workflow.md)):

| task | engine | embedding baseline | student, untrained | student, trained | build time |
|---|---|---|---|---|---|
| classification (accuracy, 18 rows) | mlx | 1.000 | 0.333 | 0.889 | 13.8 s |
| classification (accuracy, 18 rows) | torch-cpu | 1.000 | 0.444 | 0.944 | 36.5 s |
| JSON extraction (whole-record accuracy, 18 rows) | mlx | n/a | 0.000 | 0.500 | 17.9 s |
| JSON extraction (whole-record accuracy, 18 rows) | torch-cpu | n/a | 0.000 | 0.444 | 53.7 s |

Here the embedding baseline scored higher than the trained student, and the report says that; it does not prescribe more training. Eighteen rows is small and no significance test is run: a difference is an observed difference on those rows. Invalid, unparseable and failed outputs count as wrong. `disagreements.jsonl` lists the rows where a comparator and the trained student differ. `spill compare` ranks runs only when they used the same rows, metric and protocol, and otherwise explains why not. The final test is scored only by `spill test`, which records every use, so a split consulted repeatedly is not presented as untouched.

## Continuing on another machine

```
spill move tickets s3://my-bucket/tickets
spill resume s3://my-bucket/tickets
```

A live run has one authoritative control object (a file under `.spill/` or an S3 object written with conditional writes); `move` quiesces the writer, copies a verified snapshot, fences the source and only then activates the destination, and can be re-run after any interruption. `resume` acquires ownership first, then continues from the committed checkpoint on any engine; MLX and torch-cpu runs were killed mid-training and finished on the other engine in both directions, with the transition recorded ([portability](docs/portability.md)). `spill bundle` packs the project and its pinned models for use without a network. A build started on Linux and finished on macOS, and the reverse, runs on every push: [relay workflow](https://github.com/streamweights/streamweights/actions/workflows/relay.yml).

[![Relay workflow status: a build started on one OS and finished on the other, every push](https://github.com/streamweights/streamweights/actions/workflows/relay.yml/badge.svg)](https://github.com/streamweights/streamweights/actions/workflows/relay.yml)

## Sample projects

| example | task | data | license |
|---|---|---|---|
| `spill example banking77` (`--quick`, `--tiny`) | classification, 77 intents | BANKING77, PolyAI (Casanueva et al., 2020) | CC BY 4.0, attribution in its README |
| `spill example snips` (`--quick`, `--tiny`) | text to JSON: intent and slots, 3 intents | Snips NLU benchmark, 2017-06 custom intent engines, commit `b86ac7f` | CC0 1.0, citation kept in its README |

## When a teacher is useful

`spill build tickets --teacher <model>` has the teacher answer the training prompts and trains the student on those answers next to yours (your labels count twice, the teacher's once): sequence-level distillation from teacher answers, not logit or KL distillation. Teachers see training rows only. Use one when you have prompts without labels or want to see whether a larger model's answers help; its score is shown next to the others. A folder with only `prompts.jsonl` still builds, and its report says quality evaluation is unavailable: the teacher's agreement is not task accuracy. The older folder workflow (`evals.jsonl`, `train.jsonl`, `prompts.jsonl`, `spill build <folder>`) keeps working; a 70B teacher streamed from disk was measured earlier ([docs/reports](docs/reports/index.md)).

## Platforms and explicitly untested paths

| platform | status |
|---|---|
| Apple silicon, MLX | tested: M4 Pro, 48 GB (report 015 and earlier reports) |
| macOS and Linux CPU, PyTorch | tested: this Mac's CPU, and GitHub Linux and macOS runners on every push |
| S3 | tested against a MinIO server in Linux CI (version in the log); real AWS S3 is untested |
| NVIDIA CUDA | built, untested ([docs/linux.md](docs/linux.md), [issue #1](https://github.com/streamweights/streamweights/issues/1)) |
| network filesystems, Windows | untested; a network filesystem is refused for a live run |
| power loss | untested; the tests terminate the process |

Python 3.10 or newer; models are fetched at pinned revisions, and downloads never leave less than 20 GB free. `spill doctor` checks the machine.

## Under the hood

Training and inference stream the bf16 weights from disk in layer order when a model does not fit: one forward pass reads every weight once whatever the batch size, so a 70B model runs at full precision in 48 GB. mmap reached only 11 to 13% of the disk rate on a model 1.5x RAM, the streaming runner 79.2% ([report 002](docs/reports/002-phase1.md)). Engines are thin: MLX on Apple silicon, PyTorch (transformers layers bound from the same ring) elsewhere, upstream llama.cpp for GGUF. streamweights owns the ring, the job layer (a hardware-neutral float32 checkpoint with a commit marker), the project layer (config, fenced run state, immutable records) and the CLI. Stages are serializable descriptions run by an executor (local, or a separate process); only the coordinator publishes.

## Prior art

- [AirLLM](https://github.com/lyogavin/airllm) runs large models on small GPUs by keeping one layer on the GPU at a time; its README now also describes fine-tuning that streams frozen weights and keeps adapters on the GPU.
- [slowllama](https://github.com/okuvshynov/slowllama) fine-tunes Llama 2 and CodeLlama with LoRA on a Mac or a consumer GPU by offloading blocks to SSD or memory, without quantization; the repository is archived.
- [Unsloth](https://github.com/unslothai/unsloth) makes LoRA, QLoRA and full fine-tuning faster and lighter on GPUs (it states 2x faster, 70% less VRAM), on NVIDIA, AMD, Intel and CPU, and also supports macOS and MLX formats; its README does not mention disk offloading.
- [FlexGen](https://github.com/FMInference/FlexLLMGen) is a throughput-oriented inference engine that offloads weights, activations and the KV cache to CPU memory and disk; inference only, and archived.
- [DeepSpeed](https://www.deepspeed.ai/2022/09/09/zero-inference.html) offloads to CPU memory and NVMe: ZeRO-Infinity for training and ZeRO-Inference, which streams weights layer by layer, for inference.

What streamweights adds to these is the workflow around the streaming idea, not the idea: a CSV-to-evaluated-model path with a frozen task contract and leakage-aware splits, comparisons against baselines on held-out rows with a recorded protocol, verified exports, and runs that move between machines and between MLX and PyTorch under fenced ownership. Whether it is cheaper per completed job than renting a GPU is not measured here, and fitting a model on smaller hardware does not by itself show that.

## Status

Done: run, distill (sequence-level), tune, eval, build and export on Apple silicon and PyTorch; the guided project workflow above with immutable runs, final-test records, verified exports, bundles and fenced handoff; portable jobs, headless mode, containers and scheduler examples ([examples/schedulers](examples/schedulers)). Open: the CUDA gates on a real GPU, real AWS S3, the 70B and 7B proof run (its banking77 table arrives with the full proof run), then a release. Deferred to separate assignments: logit or KL distillation, more quantization, free-text quality evaluation, garbage collection of orphaned payloads, remote execution services ([docs/plan.md](docs/plan.md)). Reports are in [docs/reports](docs/reports/index.md); guides at https://streamweights.github.io/streamweights/. Feedback: [Discussions](https://github.com/streamweights/streamweights/discussions) or an issue, with `spill doctor` output for hardware reports.

Apache-2.0. Example data licenses are in the examples' READMEs.
