---
description: Run spill on Linux and NVIDIA with PyTorch: what is verified on CPU, what awaits verification on CUDA, and the one command to verify a GPU.
---

# Linux and other platforms

Tracking issue: https://github.com/streamweights/streamweights/issues/1 (Linux support, CPU first, then NVIDIA). If you would run spill on Linux, comment there with the hardware you have: CPU model, RAM, disk, GPU if any, and what you would build.

## Where it runs

Every platform gets the same commands. The engine is chosen from the hardware (`spill doctor` shows the choice, why, and a measured rate for each usable engine); `--engine mlx|torch-cpu|torch-cuda` overrides.

| engine | where | status |
|---|---|---|
| `mlx` | Apple silicon | the original engine; run and tune verified through the 70B (reports 002 to 008), build verified on the 0.5B (reports 009 and 014) |
| `torch-cpu` | any CPU, on Linux, macOS, Windows | **verified**: on the 0.5B on an M4 Pro CPU (identity, tune and resume gates in `docs/reports/012-run-anywhere.md`), and in Linux CI on every push (tiny models of every verified family, the committed step-50 checkpoint resumed on the 0.5B, and `spill build banking77-tiny` end to end). `spill build` runs here too: verified on this Mac's CPU and on the Linux CI runner ([014](reports/014-build-anywhere.md)) |
| `torch-cuda` | NVIDIA GPUs | **built, awaiting verification**: no CUDA machine was available. Run the one-liner below |

Streaming, `distill`, `tune`, `eval`, `run` with adapters and log-probs, `export` and `check` run on the torch engines. `spill build` (the one-command folder pipeline) runs on every engine through the same code as the commands above, and its state (`--state <uri>`) moves between machines and engines: a build stopped on Linux finishes on a Mac and the reverse, on every push ([portability](portability.md#a-build-moves-too)). An explicit `--quant Q8_0` or `Q4_K_M` still runs through upstream llama.cpp, unmodified; `--quant 8bit` and `4bit` are MLX.

To see the non-Apple behavior on a Mac, set `SPILL_NO_MLX=1`.

## Verify an NVIDIA GPU

```
docker run --gpus all ghcr.io/streamweights/spill:cuda python -m streamweights.verify_cuda
```

On any CUDA machine with the package installed, `python -m streamweights.verify_cuda` does the same. It downloads the 0.5B once and prints a pass or fail per gate, then writes a JSON report (`--out` to choose where, `--quick` for 20 tune steps instead of 50):

- inference, torch-cuda streamed against torch-cuda resident, float32: greedy output identical on 20 prompts, log-probs within 1e-5;
- inference, torch-cuda against the torch-cpu reference: greedy identical, and the log-prob gap reported;
- tune, torch-cuda streamed against PEFT resident, float32, 50 steps: loss within 0.1% per step, adapter cosine above 0.9999;
- tune, torch-cuda against torch-cpu, 50 steps: loss within 1%;
- resume: the committed step-50 MLX checkpoint (`streamweights/data/fixtures/resume-qwen05`) continued on torch-cuda, its loss curve against the uninterrupted MLX run;
- a streamed-inference throughput measurement: pass time, achieved read rate, tokens per second;
- a streamed-tune step-time measurement.

The first CUDA numbers belong in `docs/reports/` and in the README once this has been run.

## How the engines stream

The ring is shared by every engine (`streamweights/ring.py`): a producer thread fills a ring of preallocated host buffers with one transformer layer each, in schedule order, with large parallel `pread`s, ahead of compute. How the bytes are kept out of the page cache depends on the platform, and the Linux modes are measured against each other on the model being served (`calibration.json` keeps the winner):

| mode | where | how |
|---|---|---|
| `nocache` | macOS | `F_NOCACHE` on the descriptor |
| `fadvise` | Linux | `pread`, then `posix_fadvise(DONTNEED)` on the range just read |
| `odirect` | Linux | `O_DIRECT` reads into page-aligned bounce buffers; skipped on a filesystem that refuses it |

The PyTorch engines build the model with Hugging Face transformers on the meta device (Llama, Qwen2, Qwen3, Mistral, Phi 3 and Gemma 2 use transformers' own layer classes) and bind each layer's bytes to one reusable layer as it is needed. On CPU the compute reads the host buffers directly. On CUDA the host buffers are pinned, a side stream copies layer k+1 to k+N to the device while the compute stream works on layer k, and a small pool of device buffers is recycled once the compute stream has passed a layer.

Batching, continuous refill, the 75% memory budget (device memory on CUDA, RAM on CPU), shared-prefix reuse, log-probs, scoring and adapters at inference follow the MLX engine. Training is resident through PEFT, or streamed with the same design as the MLX streamed tune: the forward pass saves each layer's input, the backward pass streams the weights again in reverse and recomputes each layer under autograd, with the base weights as constants. Two weight streams per micro-batch.

Precision: adapters and optimizer state are float32. Base weights are bf16 on CUDA, and on CPU bf16 where the CPU multiplies bf16 fast and float32 otherwise; the pre-run line says which.

## What is left

1. **CUDA verification** on a real GPU (above), then measured CUDA numbers in the README table.
2. **A model bigger than RAM on the torch engines.** The ring is the one the MLX engine streams the 70B through, but the torch engines have run only the 0.5B and tiny models so far (the directive that built them allowed nothing larger). Exit test: stream a model larger than RAM and report the fraction of the probed sequential rate, as Phase 0 did for mmap.
3. **`build` on CUDA.** Built on the same engine interface, never run on a GPU; `verify_cuda` does not cover it yet, so run `spill example banking77 --tiny && spill build banking77-tiny --engine torch-cuda` on a GPU and report it on issue #1.
4. **Two cloud CPU types**, a modern server CPU with AMX and an ordinary AVX2 instance, with measured times.
5. **`io_uring`** for the ring, if it leaves the disk idle on a fast NVMe.

## Not in scope here

Windows installs and the torch-cpu engine should work; running on it is untested. Quantized torch inference (`--quant 8bit`) is MLX only.
