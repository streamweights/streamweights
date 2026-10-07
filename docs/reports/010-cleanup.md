---
description: Cleanup report: the merge, the usable repo and a timed fresh install.
---

# Cleanup report: merge, make the repo usable, verify a fresh install

Run 2026-10-05, Apple M4 Pro, qwen2.5:0.5b only. Directive: `docs/paste-sets/010-cleanup.md`.

## What was done

- Phase 3 finished and squash-merged (`docs/reports/008-phase3.md`), Phase 3.5 merged and squash-merged (`docs/reports/009-phase3.5.md`). Both worktrees and branches are gone; GitHub has only `main`.
- The command line was cleaned: `spill --help` lists the commands in loop order, one line each, and every command's `--help` ends with one real example (`docs/cli.md` is generated from the real output and a test fails if it drifts). Errors are one line ending in a command; tracebacks need `--debug`. Pre-run lines and `next:` hints share one format. Dead and development-only flags were removed.
- Installation works without MLX: the MLX dependencies carry the marker `sys_platform == 'darwin' and platform_machine == 'arm64'`, and every MLX import is guarded or lazy. On other platforms `spill doctor` and every MLX-only command say in one line what works, what does not yet, and where to read more.
- `pyproject.toml` is at 0.1.0 with the README as the long description, classifiers, project URLs and Apache-2.0. `python -m build` and `twine check` pass, and the built wheel installs in a clean venv and runs `spill --help` and `spill doctor`. Nothing was published.
- CI runs the CPU suite on macOS (Python 3.12) and Linux (Python 3.12 and 3.10) for pushes and pull requests, and checks the one-line message for MLX-only commands on Linux.
- Repository settings: description, 20 topics, Discussions enabled. Linux tracking issue: https://github.com/streamweights/streamweights/issues/1.
- A fresh install in an empty directory, clean venv and empty HOME took 79.5 s from `pip install` to the end of build and then exported a GGUF (`docs/reports/010-fresh-install.txt`).

## Defects found and fixed

| defect | found by | fix |
|---|---|---|
| a build folder named like its adapter shadowed the adapter in `+name` resolution | the real banking77-quick build | a directory only wins when it is an adapter |
| `python -m streamweights.cli` registered no build, example, export or doctor | the export verification | `streamweights/__main__.py`; the module entry point imports the full app |
| the GGUF converter at the pinned llama.cpp tag imports a `conversion` package that a single-file download lacks | the export verification | the converter source is fetched from the tag archive |
| the build's tune estimate used the 70B's measured rate for the 0.5B | the quick build (1.46x miss) | the model's own measured rate is used when it exists |
| `spill tune` held no `caffeinate` and had no `--notify` | the command audit | tune and eval run inside the overnight wrapper |
| `spill eval` printed a `next:` hint after every inner run | the command audit | one hint, at the end |
| pre-run line said `batch 1` for resident runs that admit 100 rows | the command audit | the shown batch is the real one |
| a fake build in one test appeared as an interrupted build in the next | the suite | per-test build registry |
| the converter environment installed noisily and its failure was a raw traceback | the fresh install | output captured, failure is a one-line error |
| `spill build` reached its first stage before rejecting an unknown model name | the command audit | names are checked before anything runs |
| a stranger would not know the build downloads 0.9 GB first, or that Ollama is optional | the fresh install | the build names its downloads; export says when Ollama is missing |
| without a bf16 GGUF (most tags), `spill run` on a non-Apple machine chose bf16 and failed | the llama.cpp check | Q8_0 is chosen and stated |

## Decisions

- **Merge, not rebase, for Phase 3.5.** The branch is squashed afterward, so the history is the same and a rebase would have resolved the same conflict 17 times.
- **No Docker on this machine.** Platform verification used a clean venv without MLX (100 tests pass, 7 that assert the Apple-silicon behavior skip), a clean Python 3.10 venv, `uv pip compile` for five target platforms (no MLX line for Linux, Windows or Intel Mac), and the Linux CI job on GitHub, which runs `pip install`, `spill doctor`, an MLX-only command and the suite on a real Ubuntu runner.
- **`state/hardware.json` and `state/calibration.json` are no longer tracked.** Anyone running from a checkout overwrites them. Copies of this machine's files are kept as `docs/reports/data/hardware-m4pro.json` and `calibration-m4pro.json`, and the reports point there.
- **Fifteen interrupted job directories** from Phase 0 and 1 experiments were removed. Completed jobs, including the 2,000-row 70B run, were kept; its results are also in `docs/reports/`.
- **`paths.svg` is omitted.** Its bars need measured times for three paths, and only one was run end to end. The README table states rates instead. Task for tonight: draw it from the proof run.
- **The export equality criterion** is reported as measured (18 of 20 at 0.5B bf16), see `docs/reports/009-phase3.5.md`.
- **Removed flags:** `--no-prefix-reuse` (an A/B check, now the environment variable `SPILL_NO_PREFIX_REUSE=1`), `--full-logits`, `--parallel` on distill and eval (hidden on run), and on tune `--alpha`, `--dropout`, `--targets`, `--schedule`, `--weight-decay`, `--seed`, `--ckpt-every`, which keep their defaults.
- **Tracked `.claude/` settings and `.DS_Store`** were untracked and ignored.
- **An earlier session's scratch directory** holding a stale build state could not be deleted by this session (the delete was refused); its registry entry was dropped instead.

## Left for tonight's long run

1. The full banking77 proof run: 70B teacher, 7B student, the surpass and copy paths, the combined four-row table, per-stage estimate versus actual.
2. Draw `docs/img/paths.svg` from the measured times of that run, and put the 70B and 7B table in the README.
3. After the proof run: the PyPI release (not done; no tag, no release workflow exist).
