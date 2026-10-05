# banking77

Customer-service messages from an online bank, each labeled with one of 77 intents
(`card_arrival`, `lost_or_stolen_card`, `exchange_rate`, ...). The task: read a message,
answer with its intent.

```
$ spill example banking77 && spill build banking77
```

The scores for the 70B teacher and the 7B student arrive with the full proof run. For
a result in under a minute, use `spill example banking77 --quick`.

## Files

| file | what it is | rows |
|---|---|---|
| `evals.jsonl` | the exam: `{"prompt", "expected"}`, from the test split. Every model is graded on it, never trained on it | 300 |
| `train.jsonl` | your homework with answers: `{"prompt", "answer"}`, from the train split | 2,000 |
| `prompts.jsonl` | more homework, questions only, a different 2,000 train-split rows. The big model answers these | 2,000 |
| `instructions.txt` | the system prompt that lists the 77 labels and asks for a label only | |

No row of the exam appears in `train.jsonl` or `prompts.jsonl`, and the two never overlap.

## How `spill build` treats each model

- The untrained base and the big model (the teacher, and anything you add with `--compare`)
  get `instructions.txt` as a system prompt, so they know the 77 labels. The shared prompt
  is computed once and reused for every row.
- Your model, the base plus the adapter, is trained and evaluated without the label list.
  Fine-tuning teaches it the label set, which keeps its training to minutes.

Remove `prompts.jsonl` and the big model is not needed at all (labels only). Remove
`train.jsonl` and the small model learns only from the big model's answers.

## License and attribution

The data is BANKING77 (Casanueva et al., 2020), published by PolyAI at
https://github.com/PolyAI-LDN/task-specific-datasets and on Hugging Face as
`PolyAI/banking77` under the Creative Commons Attribution 4.0 International license
(CC BY 4.0), which permits redistribution and adaptation with attribution. This folder
is an adaptation: rows were sampled and reshaped into `prompt`/`expected`/`answer`
fields, and two label names were normalized so that a label-only answer can match exactly
(`Refund_not_showing_up` became `refund_not_showing_up`, `reverted_card_payment?` became
`reverted_card_payment`).

> Inigo Casanueva, Tadas Temcinas, Daniela Gerz, Matthew Henderson, Ivan Vulic.
> Efficient Intent Detection with Dual Sentence Encoders. Proceedings of the 2nd Workshop
> on NLP for ConvAI, ACL 2020. https://arxiv.org/abs/2003.04807
