"""Write docs/reports/018-readme-pass.md and the after-counts file from the README and the
real MLX capture (docs/reports/data/018-readme-capture.json).

  python scripts/make_report_018.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from readme_counts import count  # noqa: E402

DATA = ROOT / "docs/reports/data"


def main() -> None:
    before = json.loads((DATA / "018-readme-counts-before.json").read_text())
    mid = json.loads((DATA / "018-readme-counts-after.json").read_text())      # the 018 merge, kept as recorded
    after = count((ROOT / "README.md").read_text())
    (DATA / "019-readme-counts-after.json").write_text(json.dumps(after, indent=1) + "\n")
    cap = json.loads((DATA / "018-readme-capture.json").read_text())
    rows = "\n".join(f"| `{s['command']}` | {s['exit']} | {s['seconds']} s |" for s in cap["steps"])
    cut = round(100 * (1 - after["prose_words"] / before["prose_words"]))
    text = f"""# 018: README editorial pass

The README was cut to the six sections a new reader needs, in this order: What it does, Try it, Use your own data, Understand the result, Go further, Support and limitations. Product code is unchanged. Everything cut moved to a page that the README links to (see the table below).

## Counts

Measured with `scripts/readme_counts.py`, the same script on both versions. Prose words exclude fenced code blocks and tables.

| | lines | words | prose words |
|---|---|---|---|
| before (directive 017) | {before['lines']} | {before['words']} | {before['prose_words']} |
| after directive 018 | {mid['lines']} | {mid['words']} | {mid['prose_words']} |
| after the finishing pass (directive 019) | {after['lines']} | {after['words']} | {after['prose_words']} |

Prose words fell by {cut} percent against the 017 README. The lines went up because commands and their real output now sit in code blocks, which the prose count excludes.

## The real MLX capture

The output in the README's Try it section was captured by `scripts/capture_readme.py` on the Apple GPU (device {cap['device']}), recorded in `docs/reports/data/018-readme-capture.json`, and every output line in the README is checked against it by `tests/test_docs.py`. Each command ran with exit code 0 in a fresh `SPILL_HOME`:

| command | exit | wall time |
|---|---|---|
{rows}

The build took 15 s after the models were on disk. The first-time download of the two models is not included and depends on the connection.

## Capture conditions

The capture script ran with `SPILL_MIN_FREE_GB=2`, not the default. The default free-space reserve enforced during downloads is 20 GB (`MIN_FREE_AFTER_DOWNLOAD` in `streamweights/registry.py`, read from `SPILL_MIN_FREE_GB`); this machine could not keep 20 GB free after the downloads, so the reserve was lowered. The models were already on disk when the capture ran, so no download was timed. Nothing else differed from a default run, and the README says the excerpts come from a run with the lowered reserve.

## Finishing pass (directive 019)

The Try it and Understand the result outputs were cut to short verbatim excerpts (each omission is one `...` line), the score comparison is a Markdown table checked cell by cell against the capture, and the S3 continuation walkthrough left the README for [portability](../portability.md#continuing-on-another-machine-step-by-step). The harness still discovers and covers those commands there: `docs/portability.md` is one of the pages `tests/test_docs_commands.py` reads, its `spill build tickets2 --stop-after train:12`, `spill move` and `spill resume` lines map to rules in `tests/docs_command_map.toml`, and the mapping test fails if a rule goes unused.

## Where the cut content went

| README block | now |
|---|---|
| the explanation of init, plan, build and export | [CSV guide, What each command does](../guides/turn-a-csv-of-examples-into-an-evaluated-model.md#what-each-command-does) |
| the sample projects table | [CSV guide, Sample projects](../guides/turn-a-csv-of-examples-into-an-evaluated-model.md#sample-projects) |
| teachers and large teachers | [CSV guide, When a teacher is useful](../guides/turn-a-csv-of-examples-into-an-evaluated-model.md#when-a-teacher-is-useful) |
| the four-row results table and protocol paragraph | [report 017](017-scorer-and-protocol.md) |
| the continuation and fencing explanation | [portability](../portability.md#continuing-on-another-machine-step-by-step) |
| the platform table with evidence | [Linux and other platforms](../linux.md#evidence-by-platform) |
| under the hood | [architecture](../architecture.md) |
| prior art | [prior art](../prior-art.md) |
| status and roadmap | [plan](../plan.md#status-and-roadmap-moved-from-the-readme) |

## Gates added

- `tests/test_docs.py` asserts the new headings and order, and that every output line shown in the README's code blocks appears in the capture.
- `tests/test_docs_commands.py` maps and, with `SPILL_DOCS_RUN=1`, executes every README command, including the classification and JSON own-data workflows on the 120-row fixtures in `tests/fixtures/own-data/` with the documented column names and the documented `schema.json`.
"""
    (ROOT / "docs/reports/018-readme-pass.md").write_text(text)


if __name__ == "__main__":
    main()
