# 018: README editorial pass

The README was cut to the six sections a new reader needs, in this order: What it does, Try it, Use your own data, Understand the result, Go further, Support and limitations. Product code is unchanged. Everything cut moved to a page that the README links to (see the table below).

## Counts

Measured with `scripts/readme_counts.py`, the same script on both versions. Prose words exclude fenced code blocks and tables.

| | lines | words | prose words |
|---|---|---|---|
| before (directive 017) | 113 | 1837 | 1342 |
| after directive 018 | 162 | 1177 | 651 |
| after the finishing pass (directive 019) | 153 | 1003 | 661 |

Prose words fell by 51 percent against the 017 README. The lines went up because commands and their real output now sit in code blocks, which the prose count excludes.

## The real MLX capture

The output in the README's Try it section was captured by `scripts/capture_readme.py` on the Apple GPU (device Device(gpu, 0)), recorded in `docs/reports/data/018-readme-capture.json`, and every output line in the README is checked against it by `tests/test_docs.py`. Each command ran with exit code 0 in a fresh `SPILL_HOME`:

| command | exit | wall time |
|---|---|---|
| `spill example banking77 --tiny` | 0 | 0.2 s |
| `spill init banking77-tiny/banking77.csv --input text --output label --project tickets` | 0 | 0.2 s |
| `spill plan tickets` | 0 | 0.8 s |
| `spill build tickets` | 0 | 15.0 s |
| `spill report tickets` | 0 | 0.2 s |
| `spill export tickets` | 0 | 9.9 s |
| `python tickets/exports/*/run_safetensors.py "Can I change my PIN at any ATM?"` | 0 | 2.7 s |

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
