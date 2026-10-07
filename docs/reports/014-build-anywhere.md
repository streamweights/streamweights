# 014: spill build on every engine, and a build that moves between machines

Measured on one machine unless a section says otherwise: an Apple M4 Pro, 48 GB, macOS, Python 3.12, model qwen2.5:0.5b. Every number below is from `docs/reports/014-gates.json`, written by `scripts/gates_014.py`; each run had its own `SPILL_HOME`, so no run was answered by another's cache. The CI numbers come from the runs of `.github/workflows/relay.yml` and `ci.yml` named in the last section.

## What changed

`spill build` runs its stages (base eval, distill, tune, eval, export) through the same engine-selected code as the commands: MLX on Apple silicon, torch-cuda with a visible GPU, torch-cpu otherwise, `--engine` overrides. Its state is the portable layout of directive 012 (`state.json`, a directory per stage, the teacher's answers, the adapter), at `<folder>/.build/` or `--state <uri>`. Each stage records the engine, hardware, OS and numerics that produced it. Defaults come from the probed machine: Apple silicon qwen2.5:7b with llama3.3:70b, CUDA the same when device memory and disk allow, a CPU qwen2.5:0.5b with the largest teacher whose estimated distill is under 12 hours.

## Gate 1: build on torch-cpu, banking77-quick

`spill build banking77-quick --engine torch-cpu` on this Mac, to completion, next to two MLX runs. Noise is the difference between the two MLX runs, which differ only in batch shape (eval batch 7, micro-batch 2 with 2 accumulation steps, against the defaults).

| run | wall time | base score | your model | tune stage |
|---|---|---|---|---|
| mlx, default batch shape | 49.3 s | 0.210 | 0.730 | 40.4 s |
| mlx, other batch shape | 59.1 s | 0.210 | 0.710 | 49.1 s |
| torch-cpu (float32 on this CPU) | 171.1 s | 0.200 | 0.720 | 146.9 s |

The scores of the first MLX run, as the README shows them:

| model | role | score | rows |
|---|---|---|---|
| qwen2.5:0.5b+banking77-quick | your model | 0.730 | 100 |
| qwen2.5:0.5b | base (untrained) | 0.210 | 100 |

Noise between the two MLX runs: 0.00 on the base and 0.02 on the adapter. The torch-cpu result is 0.01 from the first MLX run on both, inside the tolerance (the larger of that noise and 0.02, two rows of 100). The adapter beats the base on held-out rows on both engines. PASS.

The 0.640 in the Phase 3.5 report is not what this code gives any more: the build from `main` before this work scores 0.730 and 0.210 on the same machine today, so the difference is the software versions, not this change.

## Gate 2: a build moved between engines, banking77-quick

Stopped 100 steps into the 250-step tune stage, finished on the other engine, with `--state` on a directory. Reference: the uninterrupted MLX run above.

| path | first leg | second leg | base | your model | difference to the reference |
|---|---|---|---|---|---|
| mlx then torch-cpu | 21.2 s | 105.2 s | 0.210 | 0.720 | 0.00 and 0.01 |
| torch-cpu then mlx | 68.8 s | 30.8 s | 0.200 | 0.730 | 0.01 and 0.00 |

Both within the 0.02 tolerance. The state shows base eval on the first engine only, the tune stage on both (steps 0 to 100, then 100 to 250), the final eval on the second engine only; every tune step 1 to 250 is recorded once and every eval has its 100 rows once. PASS.

## Gate 3: remote state, stop and resume

On the tiny build (20 exam rows, 100 training rows): `--state memory://...` on fsspec's in-memory filesystem, stopped 20 steps into tune on MLX and finished on torch-cpu in the same process (base 0.40, your model 0.95); and `--state` on a separate directory, started on torch-cpu, the folder copied as a second machine's, finished there on MLX (base 0.40, your model 1.00). No row or step missing or repeated. PASS.

## Gate 4: headless SIGTERM

`spill build --headless`, SIGTERM during the tune stage (step 12) and again during the first eval (row 5). Both exited 75, 1.0 s and 0.36 s after the signal, with a checkpoint event and a `preempted` event last, the stage marked interrupted in the state. `spill resume --headless` of each finished with exit 0 and a `done` event with `complete: true`; the tune had steps 1 to 50 once, the evals 20 rows once. PASS.

## Gate 5: the relay example

`spill example relay && ./relay/relay.sh`: Docker is not installed on this Mac, so Docker mode could not run here and is covered by the Docker relay job of the CI workflow below. Engine-switch mode: started on mlx, finished on torch-cpu, a reference build, 45.2 s in all; scores 0.40 and 0.95, the reference 0.40 and 0.95. Two-machine mode: the printed command run in a copied folder and state directory finished the build. PASS.

## The noise floor of the tiny build

The CI relay compares 20-row scores, which move in steps of 0.05. Six runs of the tiny build on this Mac:

| run | base | your model |
|---|---|---|
| mlx | 0.40 | 0.95 |
| mlx, other batch shape | 0.40 | 0.95 |
| torch-cpu float32 | 0.40 | 1.00 |
| torch-cpu float32, other batch shape | 0.40 | 0.90 |
| mlx then torch-cpu | 0.40 | 0.95 |
| torch-cpu then mlx | 0.40 | 1.00 |

Spread 0.10 in your model's score and 0.00 in the base. The tolerance is the spread plus one row, 0.15, stored with its justification in `docs/reports/014-noise-floor.json`. bf16 torch-cpu was not among the runs: this CPU has no bf16 hardware, and a 50-step tune in bf16 had not produced a checkpoint after 28 minutes, so it was stopped. Nothing tighter than 0.15 is claimed for a relay.

## bf16 on the torch engine against float32

`tests/test_bf16_statistics.py` checks the torch engine in bf16 on the 0.5B against float32: gradient cosine (mean at least 0.95, minimum at least 0.5; measured in directive 012: 0.9736 and 0.7021), the largest log-prob difference (at most 0.5; measured 0.190) and greedy agreement on 10 prompts (at least 7, and no more than 2 below what the same engine agrees with itself across batch shapes). It runs only where the CPU multiplies bf16 quickly. On this Mac it is skipped with that reason; forced with `SPILL_TEST_BF16=1` both tests pass in 218.5 s.

## CI

The runs below are of `.github/workflows/relay.yml` and `ci.yml` on the branch's pull request (GitHub-hosted runners).

| measurement | value |
|---|---|
| relay.yml, start to the end of the check job (both relays and the reference) | 6 min 58 s |
| relay.yml, including the Docker relay job | 9 min 27 s |
| engine the macOS runner (macos-14) used | mlx: Metal was available, so the table records mlx on macOS arm64 |
| Linux runner CPU | Intel Xeon 6973P-C, bf16 base weights on torch-cpu |
| `spill example banking77 --tiny && spill build banking77-tiny` in ci.yml, Linux, Python 3.12 | 58 s |
| the reference job's uninterrupted tiny build, same workflow | 245.01 s |

Relay results against the reference (base 0.400, your model 1.000 in this run):

| relay | base | your model | difference to the reference (tuned) | checks |
|---|---|---|---|---|
| Linux to macOS (tune: torch-cpu, then mlx) | 0.400 | 0.950 | 0.05 | all six pass |
| macOS to Linux (tune: mlx, then torch-cpu) | 0.400 | 0.950 | 0.05 | all six pass |

Both are within the 0.15 tolerance; the claim is no tighter than that. The Docker relay job built the CPU image from the checkout, tagged it `ghcr.io/streamweights/spill:cpu`, started the tiny build on the host, stopped it 20 tune steps in, finished it in the container, and ran the reference; its transcript is the one in the relay example's README.
