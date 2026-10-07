# banking77-tiny

The smallest build that still learns something: 10 banking intents, 20 exam rows, 100
labeled training rows, short messages (90 characters or fewer), and the student
`qwen2.5:0.5b` (set in `spill.json`). It exists so a whole build, and a build that moves
between machines, fits in a CI job.

```
$ spill example banking77 --tiny && spill build banking77-tiny
```

`instructions.txt` lists only the 10 intents. The untrained base gets it as a system prompt;
your model, the base plus the adapter, is trained and evaluated without it.

Data: BANKING77 (Casanueva et al., 2020), PolyAI, CC BY 4.0, adapted (sampled and reshaped;
two label names normalized). Full attribution is in the README of `spill example banking77`.
