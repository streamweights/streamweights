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
