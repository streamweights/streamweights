# banking77-quick

The under-an-hour version of the banking77 example: 100 exam rows, 500 labeled training
rows (at least 6 of each of the 77 intents), no prompts file, and the small model
`qwen2.5:0.5b` as the student (set in `spill.json`).

```
$ spill example banking77 --quick && spill build banking77-quick
<!--TABLE-->
```

`instructions.txt` lists the 77 labels. The untrained base gets it as a system prompt; your
model, the base plus the adapter, is trained and evaluated without it.

Data: BANKING77 (Casanueva et al., 2020), PolyAI, CC BY 4.0, adapted (sampled and
reshaped; two label names normalized). Full attribution is in the README of
`spill example banking77`.
