"""Training data: loss masking, truncation rules, batching, line-numbered validation."""

import json

import numpy as np
import pytest

from streamweights.errors import SpillError
from streamweights.formats import FormatError, check_file
from streamweights.tune.data import (BatchPlan, build_example, load_examples,
                                     steps_for, tokenize_conversation)
from tests.stubtok import EOS, StubTokenizer

TOK = StubTokenizer()


def conv(*pairs, system=None):
    msgs = [{"role": "system", "content": system}] if system else []
    for u, a in pairs:
        msgs += [{"role": "user", "content": u}, {"role": "assistant", "content": a}]
    return msgs


def trained_text(ids, mask):
    return TOK.decode([t for t, m in zip(ids, mask) if m])


def test_mask_covers_assistant_reply_and_end_token_only():
    ids, mask = tokenize_conversation(TOK, conv(("hi", "yo"), system="sys"), EOS)
    # reply "yo" plus <|im_end|>; not the role header, not the template's trailing newline
    assert [t for t, m in zip(ids, mask) if m] == TOK._enc("yo") + [2]
    assert mask[-1] == 0                        # trailing "\n" after <|im_end|>


def test_weight_zero_turn_is_context_only():
    msgs = conv(("a", "AA"), ("b", "BB"))
    msgs[1]["weight"] = 0
    ids, mask = tokenize_conversation(TOK, msgs, EOS)
    assert trained_text(ids, mask) == "BB"


def test_all_assistant_turns_trained_by_default():
    ids, mask = tokenize_conversation(TOK, conv(("a", "AA"), ("b", "BB")), EOS)
    assert trained_text(ids, mask) == "AABB"


def test_truncation_drops_whole_leading_exchanges_and_keeps_system():
    msgs = conv(("first question", "first answer"), ("second", "2nd"), ("third", "3rd"),
                system="be brief")
    full_len = len(tokenize_conversation(TOK, msgs, EOS)[0])
    ex, why = build_example(TOK, msgs, full_len - 1, EOS)
    assert why is None and ex.dropped_messages == 2
    text = TOK.decode(ex.ids)
    assert "be brief" in text and "first question" not in text and "3rd" in text
    assert trained_text(ex.ids, ex.mask) == "2nd3rd"


def test_never_truncates_mid_assistant_turn():
    msgs = conv(("q", "x" * 200))
    ex, why = build_example(TOK, msgs, 50, EOS)
    assert ex is None and "never cut" in why


def test_truncation_never_drops_the_only_trainable_turn():
    msgs = conv(("a", "AA"), ("b", "BB"))
    msgs[3]["weight"] = 0                      # only the first reply trains
    n = len(tokenize_conversation(TOK, msgs, EOS)[0])
    ex, why = build_example(TOK, msgs, n - 1, EOS)
    assert ex is None and why


def write(tmp_path, lines):
    p = tmp_path / "t.jsonl"
    p.write_text("\n".join(l if isinstance(l, str) else json.dumps(l) for l in lines) + "\n")
    return p


def test_load_examples_counts_and_skips(tmp_path):
    p = write(tmp_path, [{"messages": conv(("a", "b"))},
                         {"messages": conv(("q", "z" * 400))}])
    exs, st = load_examples(p, TOK, 100, EOS)
    assert len(exs) == 1 and st.examples == 1 and st.skipped[0][0] == 2
    assert st.trained_tokens == 2            # "b" + <|im_end|>


@pytest.mark.parametrize("line,frag", [
    ({"messages": [{"role": "user", "content": "x"}]}, "last message must be an assistant"),
    ({"messages": [{"role": "assistant", "content": "x"}, {"role": "user", "content": "y"},
                   {"role": "assistant", "content": "z"}]}, "first non-system message"),
    ({"messages": [{"role": "user", "content": "x", "weight": 1},
                   {"role": "assistant", "content": "z"}]}, "only meaningful on assistant"),
    ({"messages": [{"role": "user", "content": "x"},
                   {"role": "assistant", "content": "z", "weight": 0}]}, "nothing to train on"),
    ({"custom_id": "a", "body": {"messages": [{"role": "user", "content": "x"}]}}, "chat lines"),
])
def test_validation_errors_are_line_numbered(tmp_path, line, frag):
    p = write(tmp_path, [{"messages": conv(("ok", "ok"))}, line])
    with pytest.raises(SpillError) as e:
        load_examples(p, TOK, 100, EOS)
    assert str(e.value).startswith("line 2:") and frag in str(e.value)


def test_check_file_applies_the_same_checks(tmp_path):
    p = write(tmp_path, [{"messages": conv(("ok", "ok"))},
                         {"messages": [{"role": "user", "content": "x"},
                                       {"role": "assistant", "content": "z", "weight": 0}]}])
    with pytest.raises(FormatError) as e:
        check_file(p)
    assert str(e.value).startswith("line 2:")
    good = write(tmp_path, [{"messages": conv(("ok", "ok"))}])
    assert check_file(good)["use"] == "train/distill targets"


def test_batches_pad_per_micro_batch_and_cover_every_example():
    exs = []
    for n in (3, 5, 9, 2, 7):
        ex, _ = build_example(TOK, conv(("q" * n, "a" * n)), 500, EOS)
        exs.append(ex)
    plan = BatchPlan(exs, micro_batch=2, seed=1)
    assert plan.per_epoch == 3                   # 2 + 2 + the kept tail of 1
    seen, shapes = 0, set()
    for m in range(plan.per_epoch):
        b = plan.batch(m)
        seen += b.inputs.shape[0]
        shapes.add(b.inputs.shape[1] % 32)
        assert not (b.mask & (b.inputs == 0)).any()
        # padded targets are never trained
        assert b.mask.sum() == b.n_trained
    assert seen == 5 and shapes == {0}


def test_batches_are_deterministic_and_reshuffle_per_epoch():
    exs = [build_example(TOK, conv(("q" * n, "a")), 500, EOS)[0] for n in range(1, 13)]
    a, b = BatchPlan(exs, 2, seed=7), BatchPlan(exs, 2, seed=7)
    seq = [a.group_indices(m) for m in range(12)]
    assert seq == [b.group_indices(m) for m in range(12)]
    assert seq[:6] != seq[6:]                    # epoch 2 is a different order
    assert sorted(i for g in seq[:6] for i in g) == list(range(12))


def test_steps_for_one_epoch():
    assert steps_for(25, 4) == 6 and steps_for(3, 8) == 1 and steps_for(10, 2, 2.0) == 10


# ---- real tokenizers, when their files are on disk (tokenizer only; no weights are read)

import pathlib  # noqa: E402

_MODELS = pathlib.Path(__file__).resolve().parent.parent / "models"


def _real(path):
    p = _MODELS / path
    return p if (p / "tokenizer.json").exists() else None


@pytest.mark.parametrize("rel,end_tok", [("qwen2.5-0.5b/bf16-st", "<|im_end|>"),
                                         ("llama3.3-70b/bf16-st", "<|eot_id|>")])
def test_real_templates_mask_only_assistant_text(rel, end_tok):
    d = _real(rel)
    if d is None:
        pytest.skip(f"{rel} tokenizer not on disk")
    from mlx_lm.utils import load_tokenizer

    from streamweights.engines.mlx_stream import collect_eos_ids
    tok = load_tokenizer(d)
    eos = collect_eos_ids(d, tok)
    msgs = conv(("What is 2+2?", "It is 4."), ("And 3+3?", "It is 6."), system="Be brief.")
    ids, mask = tokenize_conversation(tok, msgs, eos)
    trained = tok.decode([t for t, m in zip(ids, mask) if m])
    assert trained == "It is 4." + end_tok + "It is 6." + end_tok, repr(trained)
    # nothing before the first assistant reply, and no trailing newline is trained
    assert not any(mask[:ids.index(tok.convert_tokens_to_ids(end_tok))])
    assert mask[-1] == 1 or tok.decode(ids[-1:]).strip() == ""
