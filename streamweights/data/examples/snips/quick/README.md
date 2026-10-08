# snips

Short requests to a voice assistant, each labeled with what the user wants and the pieces that
matter, as one JSON object. The task: read a request, answer with that object.

```
$ spill example snips && spill init snips/snips.csv --input text --output json --schema snips/schema.json
```

Three intents of the public Snips NLU benchmark: `RateBook`, `AddToPlaylist`,
`SearchCreativeWork`. Every row is a request and its human-annotated structure:

| text | json |
|---|---|
| `Rate the current novel five of 6 points` | `{"best_rating": "6", "intent": "RateBook", "object_select": "current", "object_type": "novel", "rating_unit": "points", "rating_value": "five"}` |
| `Show me the picture Written in the Stars` | `{"intent": "SearchCreativeWork", "object_name": "Written in the Stars", "object_type": "picture"}` |

## Files

| file | what it is | rows |
|---|---|---|
| `snips.csv` | `text,json` rows: 400 per intent | 1,200 |
| `schema.json` | the JSON Schema of an answer (`intent` is one of the three; every slot is an optional string; nothing else is allowed) | |
| `LICENSE-CC0.txt` | the license of the data | |

`--quick` is 100 rows per intent (300); `--tiny` is 40 short rows per intent (120), enough for
a whole build, report and export in a CI job.

## License and attribution

The data is the Snips NLU benchmark "2017-06-custom-intent-engines" from
https://github.com/sonos/nlu-benchmark at commit `b86ac7f1577868c42158d0dec77db50956046696`,
released under CC0 1.0 Universal (`LICENSE-CC0.txt`), which permits redistribution and
adaptation without conditions. The benchmark's README asks that publications based on it cite
the paper below, so the citation is kept here. This folder is an adaptation: three intents,
rows sampled with a fixed seed, each utterance's annotated spans written as one flat JSON
object; utterances that repeat a slot name or have no slot were left out
(`scripts/make_snips_example.py` rebuilds it).

> Coucke A. et al., "Snips Voice Platform: an embedded Spoken Language Understanding system
> for private-by-design voice interfaces." 2018. https://arxiv.org/abs/1805.10190
