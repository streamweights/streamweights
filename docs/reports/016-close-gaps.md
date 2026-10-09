# 016: closing the 015 gaps

Directive: `docs/paste-sets/016-close-015-gaps.md`. Everything here was run; the JSON files in `docs/reports/data/` are the raw results.

## Offline bundle gate

`scripts/gate_offline_bundle.py` builds a bundle of the tiny classification project online (`prepare`), then `offline` runs in an environment with no network: a fresh empty `SPILL_HOME` and Hugging Face cache, the wheel installed from a local file with `pip install --no-deps --no-index`, the bundle verified and installed, the project reported and planned, inference from the bundle (base model plus the bundled adapter), a fork to a new run that trains to completion, then a corrupted and a missing asset, both rejected. Every spill process runs with `scripts/netguard` on `PYTHONPATH`: any connection or name lookup raises a `BaseException` (so no retry loop can swallow it) and is logged, and the log must be empty at the end. `HF_HUB_OFFLINE=1` is set too, as a second layer only. The gate first proves the environment itself has no network: a real connection to 1.1.1.1:443 and a lookup of huggingface.co must fail.

### Linux (GitHub Actions, container started with `--network none`)

All checks passed: True. `streamweights` imported from `site-packages`.

| check | result |
|---|---|
| the environment has no network | pass |
| DNS is unavailable | pass |
| SPILL_HOME and the Hugging Face cache start empty | pass |
| bundle verifies (checksums) | pass |
| bundle installs into the empty cache and the project is copied out | pass |
| installed qwen2.5-0.5b/bf16-st/model.safetensors | pass |
| installed embedding-minilm/model.safetensors | pass |
| report from the bundled project | pass |
| plan needs nothing from the network | pass |
| inference from the bundle (base plus the bundled adapter) | pass |
| inference produced one answer per row | pass |
| fork a new run and complete a training run offline | pass |
| a new completed run exists next to the bundled one | pass |
| the fork names the bundled run as its parent | pass |
| a corrupted asset is rejected | pass |
| a corrupted bundle is refused by install too | pass |
| a missing asset is rejected | pass |
| no network call was attempted by any spill process | pass |

### macOS (`sandbox-exec` with `(deny network*)`)

All checks passed: True. `streamweights` imported from `site-packages`.

| check | result |
|---|---|
| the environment has no network | pass |
| DNS is unavailable | pass |
| SPILL_HOME and the Hugging Face cache start empty | pass |
| bundle verifies (checksums) | pass |
| bundle installs into the empty cache and the project is copied out | pass |
| installed qwen2.5-0.5b/bf16-st/model.safetensors | pass |
| installed embedding-minilm/model.safetensors | pass |
| report from the bundled project | pass |
| plan needs nothing from the network | pass |
| inference from the bundle (base plus the bundled adapter) | pass |
| inference produced one answer per row | pass |
| fork a new run and complete a training run offline | pass |
| a new completed run exists next to the bundled one | pass |
| the fork names the bundled run as its parent | pass |
| a corrupted asset is rejected | pass |
| a corrupted bundle is refused by install too | pass |
| a missing asset is rejected | pass |
| no network call was attempted by any spill process | pass |

## MLX on Metal: continuation in both directions

`scripts/gate_metal_continuation.py` on Apple M4 Pro (macOS arm64); it asserts `mx.default_device()` is the GPU and it was `Device(gpu, 0)`. Tiny classification project (120 rows: 84 train, 18 validation, 18 final test), checkpoint every 5 steps, 42 steps. Each stop is a process killed by a deterministic hook right after a checkpoint is published; the next process waits out the 2 s lease. Every score is the trained student's accuracy on the 18 validation rows.

Uninterrupted MLX-GPU reference: 0.889 (untrained 0.333, embedding baseline 1.000, 18 rows).

| run | engines in order | killed after published step | published checkpoints (step, optimizer step) | engine changes recorded | trained accuracy (18 rows) | carry-over problems |
|---|---|---|---|---|---|---|
| chain-a | mlx to torch-cpu to mlx | 10, 15 | (5, 5), (10, 10), (15, 15), (20, 20), (25, 25), (30, 30), (35, 35), (40, 40), (42, 42) | 2 | 0.889 | 0 |
| chain-b | torch-cpu to mlx to torch-cpu | 10, 15 | (5, 5), (10, 10), (15, 15), (20, 20), (25, 25), (30, 30), (35, 35), (40, 40), (42, 42) | 2 | 0.944 | 0 |

A score from a run that changed engines is not expected to equal the reference: the engines train with different numerics (see portability) and 18 rows move in steps of 0.056. The gate checks that the step, data cursor and optimizer step carry over and that the transitions are recorded, and reports the score; it does not require the scores to be equal.

### Moves between locations

A run is killed on one engine after step 10, `spill move` hands it to another location, and `spill resume` there finishes it on the other engine.

| move | destination | engines | published step before the move | source after the move | trained accuracy (18 rows) | carry-over problems |
|---|---|---|---|---|---|---|
| move-local-mlx-torch-cpu | local | mlx to torch-cpu | 10 | transferred | 0.889 | 0 |
| move-s3-mlx-torch-cpu | s3 | mlx to torch-cpu | 10 | transferred | 0.889 | 0 |
| move-local-torch-cpu-mlx | local | torch-cpu to mlx | 10 | transferred | 0.944 | 0 |
| move-s3-torch-cpu-mlx | s3 | torch-cpu to mlx | 10 | transferred | 0.944 | 0 |

The S3 destination here was a local MinIO server (the Homebrew build of RELEASE.2025-10-15T17-29-55Z). Docker is not installed on this Mac, so the Linux-container MinIO was not used here; that path is covered by CI (job `object-store`).

## Merge hygiene

The pull request for this directive contains an empty verification commit for `8c4b7fa` (the commit that fixed the CPU image build by leaving `boto3` unpinned in the `cloud` extra).

Resolution, from the CI job `boto3 and s3fs resolution` on the pull request: a clean Python 3.12 venv with `pip install ".[cloud]"`, the CPU image and the CUDA image all resolved the same versions:

| package | version |
|---|---|
| boto3 | 1.43.106 |
| botocore | 1.43.106 |
| aiobotocore | 3.9.2 |
| s3fs | 2026.9.0 |

Branch protection on `main`, set through `gh api` (the account permits it): pull requests are required (0 required approvals, since there is one maintainer), and these status checks must pass before merging: `ubuntu-latest / Python 3.10`, `ubuntu-latest / Python 3.12`, `macos-14 / Python 3.12`, `S3 ownership and handoff against a pinned MinIO (Linux)`, `Offline bundle gate (container with --network none)` (workflow CI); `Check the relays against the reference` (Relay); `cpu image (build, smoke test, push)` and `cuda image (build, push)` (Containers); `build` (Docs site). Administrators are not forced to follow the rules (`enforce_admins` is off), so the owner can still override in an emergency. To make Containers and Docs site checks exist on pull requests, both workflows now also run on `pull_request` (they push images and deploy only from `main`).

