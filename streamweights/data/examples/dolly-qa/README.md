# dolly-qa

Questions about a short Wikipedia passage, each with the answer a person wrote. The task: read
the passage and the question, answer in your own words. Unlike a label or a JSON object, there
are many right answers, so a project on this data is scored by reference overlap and, when you
name a judge, by that judge's assessment.

```
$ spill example dolly-qa --tiny
$ spill init dolly-qa-tiny/dolly-qa.csv --input prompt --output answer --project dolly
```

| prompt | answer |
|---|---|
| `Passage: Final Fantasy is a Japanese science fantasy anthology media franchise ... Question: Based on this paragraph about the Final Fantasy series, who created it?` | `Hironobu Sakaguchi` |
| `Passage: Jukebox is the eleventh studio album by Australian pop vocal group Human Nature ... Question: Does Jukebox contain many original songs?` | `No, it contains covers from the 1950s and 1960s, with one original song.` |

## Files

| file | what it is | rows |
|---|---|---|
| `dolly-qa.csv` | `prompt,answer` rows | 359 (84 with `--tiny`) |
| `NOTICE.txt` | attribution, the license and what was changed | |

## License and attribution

Adapted from databricks-dolly-15k (Databricks, CC BY-SA 3.0; passages from Wikipedia, CC BY-SA).
CC BY-SA 3.0 permits redistribution and adaptation with attribution, and requires the same
license on the data files: see `NOTICE.txt`. The license was read from the dataset card on the
Hub (`license: cc-by-sa-3.0`) on 2026-10-13.

## A captured result

`--tiny`, 58 training, 13 validation and 13 final-test rows, student `qwen2.5:0.5b`, teacher and judge
`qwen2.5:1.5b`, on the Apple GPU (`docs/reports/data/020-judged-mlx.json` in the repository has the
whole run, and the same run on torch-cpu is `020-judged-torch-cpu.json`):

```
spill init dolly-qa-tiny/dolly-qa.csv --input prompt --output answer --project dolly --teacher qwen2.5:1.5b --judge qwen2.5:1.5b
spill build dolly
```

| comparator | token F1 | judge score (1 to 5) |
|---|---|---|
| nearest training answer (TF-IDF, no model) | 0.108 | 2.462 |
| student, prompted, untrained | 0.441 | 4.846 |
| student, trained | 0.590 | 4.538 |
| teacher | 0.459 | reference only: it is the judge |

Token F1 is overlap with the reference answer, not accuracy; the judge score is one model's assessment,
not accuracy. Judged in both orders, the trained student against the untrained one won 0, tied 11 and
lost 2 of 13 rows, and the two orders disagreed on 8; the judge agreed with a word-for-word check on
80% of the 15 rows whose reference has at most 5 words. These are 13 rows and a 1.5B judge: they
show what the workflow reports, not that one model is better than another.
