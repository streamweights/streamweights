"""Teacher log-probs for tiny test models: a second tiny model scores the student's training rows
teacher-forced (what `spill distill --score` records), written as training lines with a `teacher`
field. Torch only."""

import json
from pathlib import Path

import numpy as np
import torch

from streamweights.engines.common import collect_eos_ids
from streamweights.engines.torch_common import load_tokenizer
from streamweights.tune import toy
from streamweights.tune.data import tokenize_conversation


def kd_file(student_dir: Path, teacher_dir: Path, out: Path, n: int = 24, k: int = 6,
            seed: int = 5, own_every: int = 4) -> Path:
    """Chat lines from the toy task; every `own_every`-th line is the user's own (no teacher
    field), the rest carry the teacher's top-k at each assistant token."""
    from transformers import AutoModelForCausalLM
    tok = load_tokenizer(student_dir)
    eos = collect_eos_ids(student_dir, tok)
    teacher = AutoModelForCausalLM.from_pretrained(str(teacher_dir), dtype=torch.float32).eval()
    import random
    rng = random.Random(seed)
    lines = []
    for i in range(n):
        q, a = toy.example(rng)
        msgs = [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
        if own_every and i % own_every == 0:
            lines.append({"messages": msgs})
            continue
        ids, mask = tokenize_conversation(tok, msgs, eos)
        start = mask.index(1)
        with torch.no_grad():
            lg = teacher(input_ids=torch.tensor([ids])).logits[0].float()
        lp = torch.log_softmax(lg[start - 1:len(ids) - 1], -1)
        vals, idx = lp.topk(k, -1)
        lines.append({"messages": msgs, "teacher": {
            "token_ids": ids[start:], "top_ids": idx.tolist(), "top_lps": vals.tolist(),
            "tokenizer_sha256": "tiny"}})
    out.write_text("".join(json.dumps(l) + "\n" for l in lines))
    return out
