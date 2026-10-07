---
description: Start a fine-tuning job on one machine and finish it on another. spill build keeps its whole state at --state, so a build stopped on Linux finishes on a Mac, in a container, or on another engine, and relay.sh shows it with a table of which engine made each stage.
---

# Start a fine-tuning job on one machine and finish it on another

Run `spill build` with `--state` pointing at a folder both machines can reach, stop it, and run `spill resume` on the other machine: stages that finished are skipped and the stage that stopped continues at its step. The relay example does exactly that with a tiny build, and the same hop runs between a Linux and a macOS CI runner on every push ([relay.yml](https://github.com/streamweights/streamweights/actions/workflows/relay.yml)).

## The steps

1. Make the example and look at it. It is the tiny banking77 build (10 intents, 20 exam rows, 100 training rows, student qwen2.5:0.5b) plus a script:

   ```
   spill example relay
   ```

2. Start the build on this machine and stop it partway through the tune stage. `--stop-after tune:20` stops after 20 of the 50 steps; the state is complete and the exit code is 0:

   ```
   spill build relay --state ./relay-state --stop-after tune:20
   ```

3. Finish it somewhere else. In a Linux container with the folder and the state mounted:

   ```
   docker run --rm -v "$PWD/relay:/work/relay" -v "$PWD/relay-state:/work/relay-state" -w /work \
     ghcr.io/streamweights/spill:cpu resume relay --state /work/relay-state
   ```

   Or on a second machine: copy `relay/` and `relay-state/` over (or point `--state` at `s3://`, `gs://` or `az://` on both) and run `spill resume relay --state relay-state`. Or on the other engine of this machine: `spill resume relay --state ./relay-state --engine torch-cpu`.

4. Read the table. `spill build relay --state ./relay-state --table` prints, for every stage, the engine, machine and OS that produced it, and the score:

   ```
   @@TRANSCRIPT@@
   ```

`./relay/relay.sh` runs all of this: Docker if it is installed, otherwise the other engine on this machine, or with `--two-machines` it stops and prints what to copy and the one command to run there.

## What the table promises

The build state records, for each stage, the engine, hardware, OS and numerics that produced it. A stage that moved lists both, for example `mlx -> torch-cpu` for a tune that started on a Mac and finished on the CPU. The tuned score of a build that moved is compared with an uninterrupted build of the same folder, and the comparison is only as tight as the measured noise: on the tiny build, six runs on one Mac spread 0.10 in tuned score (two engines, two batch shapes, a build moved between engines in both directions), so the relay check in CI allows 0.15. Do not read a tighter agreement into one run of 20 rows. On the 100-row quick example, the two clean MLX runs with different batch shapes were 0.02 apart. The details are in [portability](../portability.md#a-build-moves-too) and [report 014](../reports/014-build-anywhere.md).

## Try it

```
spill example relay && ./relay/relay.sh
```

Source and issues: [github.com/streamweights/streamweights](https://github.com/streamweights/streamweights).
