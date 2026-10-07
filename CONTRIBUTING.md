# Contributing

Thanks for looking. The most useful contribution is a result from hardware we do not have: run `spill doctor`, or `python -m streamweights.verify_cuda` on an NVIDIA GPU, and report it on [issue #1](https://github.com/streamweights/streamweights/issues/1) or with the hardware report template.

## Code

```
git clone https://github.com/streamweights/streamweights
cd streamweights
python -m venv .venv && .venv/bin/pip install -e ".[dev]"
python -m pytest
```

The suite runs on the CPU (`SPILL_DEVICE=cpu` is set by `tests/conftest.py`) and takes a few minutes; GPU tests are skipped unless `SPILL_GPU_TESTS=1`. CI runs it on macOS and Linux.

## Rules the tests enforce

- No em-dashes and no unfinished markers in docs.
- A number with a unit in the README or a guide must appear in a report under `docs/reports`. Measure it, or leave it out.
- `docs/cli.md` is generated: `python scripts/make_cli_docs.py`.
- Internal links and anchors in the docs must resolve.
- Errors are one line ending in the command to try (`spill: <message>. Try: <command>`), with no traceback unless `--debug`.

Small, well-scoped tasks are labeled [good first issue](https://github.com/streamweights/streamweights/labels/good%20first%20issue). The plan and what is next are in [docs/plan.md](docs/plan.md).
