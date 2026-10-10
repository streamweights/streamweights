# streamweights

**Build a small model for your task, on hardware you control.**

Bring labeled examples. Fine-tune locally, compare against simple baselines, and export to GGUF or safetensors. Pause and resume supported training runs across Mac and Linux.

streamweights (command: `spill`) turns a CSV of labeled examples into a small fine-tuned model (LoRA on `qwen2.5:0.5b`) and shows how it compares with an embedding baseline and the untrained model on rows it never saw. It runs on a Mac (MLX) or on Linux and macOS CPUs (PyTorch). Data stays local by default; remote storage is used when explicitly configured.

## Quickstart: your examples to a model that answers

```
pip install git+https://github.com/streamweights/streamweights
spill example banking77 --tiny
spill init banking77-tiny/banking77.csv --input text --output label --project tickets
spill plan tickets
spill build tickets
spill report tickets
spill export tickets --gguf q8_0
python tickets/exports/*/run_gguf.py "Can I change my PIN at any ATM?"
```

Or install it as a tool: `uv tool install git+https://github.com/streamweights/streamweights`. `init` validates every row (file, row, problem, fix), suggests the task (`classification` or `json`, never guessed from free text), freezes the label vocabulary or JSON schema from the training rows, and splits train, validation and final test by seed without separating duplicate inputs or shared `--group` values. `plan` states models, engine, downloads, disk and a time per stage, each labeled a measurement, an assumption or unknown, and loads nothing. `build` trains a small student (`qwen2.5:0.5b`, bf16) with LoRA; a teacher runs only if you ask. `export` merges the adapter, writes safetensors and GGUF, and loads each artifact in an independent runtime on validation rows before it records the export.

What `spill build` ends with and what the exported model answers, on an M4 Pro with MLX (tiny example, 18 validation rows; the full report is `tickets/runs/<id>/report.md`):

```
embedding baseline (MiniLM + logistic regression)  accuracy 1.000  (18 rows)
student, prompted, untrained                       accuracy 0.333  (18 rows)
student, trained                                   accuracy 0.889  (18 rows)

$ python tickets/exports/*/run_gguf.py "Can I change my PIN at any ATM?"
change_pin
```

The model answered with a label from the vocabulary. [Guide](docs/guides/turn-a-csv-of-examples-into-an-evaluated-model.md), [formats](docs/formats.md), [commands](docs/cli.md).

## Understanding results

These are workflow checks on a 120-row example, not a benchmark: 18 validation rows, one run each, measured on an M4 Pro, 48 GB ([report 017](docs/reports/017-scorer-and-protocol.md)). A larger, representative result is future work.

| task | engine | embedding baseline | student, untrained | student, trained | build time |
|---|---|---|---|---|---|
| classification (accuracy) | mlx | 1.000 (18 validation rows) | 0.333 (18 rows) | 0.889 (18 rows) | 13.7 s |
| classification (accuracy) | torch-cpu | 1.000 (18 validation rows) | 0.444 (18 rows) | 0.944 (18 rows) | 32.2 s |
| JSON extraction (whole-record accuracy) | mlx | n/a | 0.000 (18 rows) | 0.500 (18 rows) | 17.8 s |
| JSON extraction (whole-record accuracy) | torch-cpu | n/a | 0.000 (18 rows) | 0.444 (18 rows) | 38.1 s |

The embedding baseline scored higher than the trained student on classification, and the report says so without prescribing more training. No significance test is run: a difference is an observed difference on those rows. An output that is not in the label vocabulary, does not parse, or does not satisfy the JSON schema counts as wrong and stays in the denominator. `spill compare` labels every comparison: common evaluator, cross-runtime (ranked, with the differences and a caution that evaluation-runtime effects may be in the numbers) or incompatible (not ranked, and why). `spill evaluate` re-scores a run on its frozen validation rows under another engine as a separate record. An export that verified loaded and ran; each export record shows the quality change on the verification rows next to it, because loading is not the same as keeping quality ([export diagnosis](docs/reports/017-export-diagnosis.md)). The final test is scored only by `spill test`, which records every use.

## Continuing on another machine

A run that was interrupted can be moved and resumed; a completed run cannot resume training. The S3 commands need the cloud extra:

```
pip install "streamweights[cloud] @ git+https://github.com/streamweights/streamweights"
spill init banking77-tiny/banking77.csv --input text --output label --project tickets2 --seed 3
spill build tickets2 --stop-after train:12
spill move tickets2 s3://my-bucket/tickets2
spill resume s3://my-bucket/tickets2
```

`--stop-after train:12` interrupts the build after step 12 (Ctrl-C at a checkpoint does the same). A live run has one authoritative control object (a file under `.spill/` or an S3 object written with conditional writes); `move` quiesces the writer, copies a verified snapshot, fences the source and only then activates the destination, and can be re-run after any interruption. `resume` acquires ownership first, then continues from the committed checkpoint on any supported engine: MLX and torch-cpu runs were killed mid-training and finished on the other engine in both directions, with the transition recorded, and moved through a local folder and MinIO ([portability](docs/portability.md), [report 016](docs/reports/016-close-gaps.md)). A run that continues a parent records experiment lineage only: it trained from the base model, not from the parent's weights. `spill bundle` packs a project and its pinned models for use with the network blocked. A build started on Linux and finished on macOS, and the reverse, runs on every push: [relay workflow](https://github.com/streamweights/streamweights/actions/workflows/relay.yml).

[![Relay workflow status: a build started on one OS and finished on the other, every push](https://github.com/streamweights/streamweights/actions/workflows/relay.yml/badge.svg)](https://github.com/streamweights/streamweights/actions/workflows/relay.yml)

## Sample projects

| example | task | data | license |
|---|---|---|---|
| `spill example banking77` (`--quick`, `--tiny`) | classification, 77 intents | BANKING77, PolyAI (Casanueva et al., 2020) | CC BY 4.0, attribution in its README |
| `spill example snips` (`--quick`, `--tiny`) | text to JSON: intent and slots, 3 intents | Snips NLU benchmark, 2017-06 custom intent engines, commit `b86ac7f` | CC0 1.0, citation kept in its README |

## When a teacher is useful

`spill build tickets --teacher <model>` has the teacher answer the training prompts and trains the student on those answers next to yours (your labels count twice, the teacher's once): sequence-level distillation from teacher answers, not logit or KL distillation. Teachers see training rows only. Use one when you have prompts without labels or want to see whether a larger model's answers help; its score is shown next to the others. A folder with only `prompts.jsonl` still builds, and its report says quality evaluation is unavailable: the teacher's agreement is not task accuracy. The older folder workflow (`evals.jsonl`, `train.jsonl`, `prompts.jsonl`, `spill build <folder>`) keeps working.

### Large teachers, streamed from disk

This is the capability the project started with and it is optional. A teacher bigger than RAM is read from disk in layer order, one forward pass reading every weight once whatever the batch size: [Measured: Llama 3.3 70B, 141 GB unquantized, run on a 48 GB MacBook Pro.](docs/reports/002-phase1.md) mmap reached only 11 to 13% of the disk rate on a model 1.5x RAM, the streaming runner 79.2%. A teacher this size is a 70B-class run on large hardware and many hours; the guided examples here use none.

## Platforms and explicitly untested paths

| platform | status |
|---|---|
| Apple silicon, MLX on the GPU | tested: M4 Pro, 48 GB. A tiny build killed while publishing a checkpoint was finished on the other engine, MLX to torch-cpu to MLX and the reverse, and moved to a local folder and to a local MinIO in both directions, on the Metal GPU ([report 016](docs/reports/016-close-gaps.md)); the pytest version of the continuation test runs MLX on the CPU device |
| macOS and Linux CPU, PyTorch | tested: this Mac's CPU, and GitHub Linux and macOS runners on every push |
| offline use | a bundle was installed, inferred from, forked and trained with the network blocked: in a Linux container started with `--network none` (CI) and, on this Mac, under a `sandbox-exec` profile that denies network access ([report 016](docs/reports/016-close-gaps.md)); the tiny 120-row classification project and qwen2.5:0.5b only |
| S3 | tested against a MinIO server (built from its release tag in Linux CI; a Homebrew build of the same release locally); real AWS S3 is untested |
| NVIDIA CUDA | built, untested ([docs/linux.md](docs/linux.md), [issue #1](https://github.com/streamweights/streamweights/issues/1)) |
| network filesystems, Windows | untested; a network filesystem is refused for a live run |
| power loss | untested; the tests terminate the process |

Python 3.10 or newer; models are fetched at pinned revisions, and downloads never leave less than 20 GB free. `spill doctor` checks the machine.

## Under the hood

Engines are thin: MLX on Apple silicon, PyTorch (transformers layers bound from the same ring) elsewhere, upstream llama.cpp for GGUF. streamweights owns the streaming ring, the job layer (a hardware-neutral float32 checkpoint with a commit marker), the project layer (config, fenced run state, immutable records) and the CLI. Stages are serializable descriptions run by an executor (local, or a separate process); only the coordinator publishes. The engines decode greedily; a request for a sampling or stop setting they cannot apply is refused before anything runs.

## Prior art

- [AirLLM](https://github.com/lyogavin/airllm) runs large models on small GPUs by keeping one layer on the GPU at a time; its README now also describes fine-tuning that streams frozen weights and keeps adapters on the GPU.
- [slowllama](https://github.com/okuvshynov/slowllama) fine-tunes Llama 2 and CodeLlama with LoRA on a Mac or a consumer GPU by offloading blocks to SSD or memory, without quantization; the repository is archived.
- [Unsloth](https://github.com/unslothai/unsloth) makes LoRA, QLoRA and full fine-tuning faster and lighter on GPUs (it states 2x faster, 70% less VRAM), on NVIDIA, AMD, Intel and CPU, and also supports macOS and MLX formats; its README does not mention disk offloading.
- [FlexGen](https://github.com/FMInference/FlexLLMGen) is a throughput-oriented inference engine that offloads weights, activations and the KV cache to CPU memory and disk; inference only, and archived.
- [DeepSpeed](https://www.deepspeed.ai/2022/09/09/zero-inference.html) offloads to CPU memory and NVMe: ZeRO-Infinity for training and ZeRO-Inference, which streams weights layer by layer, for inference.

What streamweights adds to these is the workflow around the streaming idea, not the idea: a CSV-to-evaluated-model path with a frozen task contract and leakage-aware splits, comparisons against baselines on held-out rows with a recorded protocol, verified exports, and runs that move between machines and between MLX and PyTorch under fenced ownership. Whether it is cheaper per completed job than renting a GPU is not measured here, and fitting a model on smaller hardware does not by itself show that.

## Status

Done: run, distill (sequence-level), tune, eval, build and export on Apple silicon and PyTorch; the guided project workflow above with immutable runs, evaluation and final-test records, verified exports, bundles and fenced handoff; portable jobs, headless mode, containers and scheduler examples ([examples/schedulers](examples/schedulers)). Open: the CUDA gates on a real GPU, real AWS S3, a larger representative evaluation than the 120-row examples, the 70B and 7B proof run (its banking77 table arrives with the full proof run), then a release. Deferred to separate assignments: logit or KL distillation, more quantization, free-text quality evaluation, garbage collection of orphaned payloads, remote execution services ([docs/plan.md](docs/plan.md)). Reports are in [docs/reports](docs/reports/index.md); guides at https://streamweights.github.io/streamweights/. Feedback: [Discussions](https://github.com/streamweights/streamweights/discussions) or an issue, with `spill doctor` output for hardware reports.

Apache-2.0. Example data licenses are in the examples' READMEs.
