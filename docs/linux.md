# Linux and other platforms

Tracking issue: https://github.com/streamweights/streamweights/issues/1 (Linux support, CPU first, then NVIDIA). If you would run spill on Linux, comment there with the hardware you have: CPU model, RAM, disk, GPU if any, and what you would build.

## What works today

The package installs on Linux, Windows and Intel Macs. MLX is a dependency only on Apple silicon (`sys_platform == "darwin" and platform_machine == "arm64"`), and every MLX import is guarded or lazy.

| command | Linux, Windows, Intel Mac |
|---|---|
| `spill run` | works through llama.cpp, on tags that have a GGUF. A prebuilt llama.cpp is fetched on Linux x86_64 and Intel Macs; on other systems place a `llama-server` binary in `bin/`. |
| `spill export` | works: merges an adapter into bf16 safetensors with numpy, then GGUF through llama.cpp's converter. |
| `spill check`, `spill models`, `spill doctor`, `spill runs`, `spill adapters` | work |
| streaming of models bigger than RAM, `distill`, `tune`, `build` | do not work yet |

A command that needs MLX prints one line: what works here today, what does not yet, and a pointer to this page. `spill doctor` prints the same on its last line.

To see this on a Mac, set `SPILL_NO_MLX=1`.

## What is missing, and why

The Mac engine reads each layer's weights from NVMe in order through a ring buffer while the GPU computes, so a model bigger than RAM runs at disk speed with a big batch. It depends on MLX for the forward pass and on unified memory, where "load a layer" and "make it visible to the GPU" are one step. Neither exists on Linux, so the work is an engine, then the commands that sit on top of it.

## The work, in order

1. **A CPU engine with the same layer-ordered disk streaming.** Try PyTorch CPU and MLX's CPU backend, and keep whichever measures faster on the 0.5B and on a model bigger than RAM. Read with a `pread` ring using `O_DIRECT` or `posix_fadvise`, and try `io_uring` if the ring leaves the disk idle. Exit test: stream a model larger than RAM and report the fraction of the probed sequential rate, as Phase 0 did for mmap.
2. **Inference parity with the Mac path.** The same job engine, batching, refill and memory budget. Exit test: an identity test against the Mac results on the 0.5B (greedy output equal, or differences reported next to the batch-shape noise floor, as in `docs/reports/009-phase3.5.md`).
3. **`distill` and `eval` on CPU.** They sit on the inference engine, so this is mostly the log-prob path and tests.
4. **`tune`.** A resident path through PEFT on CPU for small models, and the streamed backward on CPU (saved layer inputs, reverse recompute, two weight streams per micro-batch, as on the Mac). Exit test: the float32 identity gate from `docs/reports/008-phase3.md`.
5. **`build` end to end on CPU.** The banking77 quick example, start to finish, on a laptop-class CPU.
6. **Measurement on two cloud machine types.** A modern server CPU with AMX and an ordinary AVX2 instance. Extend the README table with measured times only.
7. **NVIDIA GPUs, as a later phase.** The same ring with pinned host memory and CUDA streams, so a layer's transfer overlaps the previous layer's compute.

## Not in scope here

Windows needs its own llama.cpp asset in the fetcher and a check of the job engine's process handling. It installs today; running on it is untested.
