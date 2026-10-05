"""Plain row shapes: evals, prompts, training, optional system field, line-numbered errors."""

import json

import pytest

from streamweights import formats
from streamweights.formats import FormatError


def w(tmp_path, name, rows):
    p = tmp_path / name
    p.write_text("".join((r if isinstance(r, str) else json.dumps(r)) + "\n" for r in rows))
    return p


def test_plain_eval_rows(tmp_path):
    p = w(tmp_path, "evals.jsonl", [{"prompt": "a?", "expected": "x"},
                                    {"prompt": "b?", "expected": "y", "system": "Be terse."}])
    info = formats.check_file(p)
    assert info["shape"] == "plain" and info["use"] == "eval" and info["expected"] == 2
    rows = formats.load_rows(p)
    assert rows[0]["body"]["messages"] == [{"role": "user", "content": "a?"}]
    assert rows[1]["body"]["messages"][0] == {"role": "system", "content": "Be terse."}
    assert rows[1]["expected"] == "y"
    assert rows[0]["custom_id"] == "row-000001"


def test_plain_prompts_and_training(tmp_path):
    pp = w(tmp_path, "prompts.jsonl", [{"prompt": "q1"}, {"prompt": "q2"}])
    assert formats.check_file(pp)["use"] == "prompts"
    tp = w(tmp_path, "train.jsonl", [{"prompt": "q1", "answer": "a1"},
                                     {"prompt": "q2", "answer": "a2", "system": "S"}])
    info = formats.check_file(tp)
    assert info["use"] == "train/distill targets" and info["shape"] == "plain"
    rows = formats.load_rows(tp, mode="score")
    assert [m["role"] for m in rows[1]["body"]["messages"]] == ["system", "user", "assistant"]
    gen = formats.load_rows(tp)            # generation mode drops the answer, keeps it as target
    assert gen[0]["target"] == "a1" and gen[0]["body"]["messages"][-1]["role"] == "user"


def test_system_field_on_batch_and_chat_rows(tmp_path):
    b = w(tmp_path, "b.jsonl", [{"custom_id": "a", "system": "SYS",
                                 "body": {"messages": [{"role": "user", "content": "hi"}]}}])
    assert formats.load_rows(b)[0]["body"]["messages"][0] == {"role": "system", "content": "SYS"}
    c = w(tmp_path, "c.jsonl", [{"system": "SYS", "messages": [
        {"role": "system", "content": "mine"}, {"role": "user", "content": "hi"}]}])
    # a row that already starts with a system message keeps it
    assert formats.load_rows(c)[0]["body"]["messages"][0]["content"] == "mine"


@pytest.mark.parametrize("row,frag", [
    ({"prompt": ""}, '"prompt" must be a non-empty string'),
    ({"prompt": "x", "answer": "a", "expected": "e"}, 'not both'),
    ({"prompt": "x", "answer": 3}, '"answer" must be a string'),
    ({"prompt": "x", "system": 3}, '"system" must be a string'),
    ({"prompt": "x", "bogus": 1}, "unknown field"),
])
def test_plain_errors_carry_line_numbers(tmp_path, row, frag):
    p = w(tmp_path, "f.jsonl", [{"prompt": "ok"}, row])
    with pytest.raises(FormatError) as e:
        formats.check_file(p)
    assert e.value.line_no == 2 and frag in str(e.value)


def test_mixed_shapes_rejected(tmp_path):
    p = w(tmp_path, "m.jsonl", [{"prompt": "a"}, {"messages": [{"role": "user", "content": "b"}]}])
    with pytest.raises(FormatError, match="mixes"):
        formats.check_file(p)


def test_partial_expected_and_answer_rejected(tmp_path):
    p = w(tmp_path, "e.jsonl", [{"prompt": "a", "expected": "x"}, {"prompt": "b"}])
    with pytest.raises(Exception, match="expected"):
        formats.check_file(p)
    p = w(tmp_path, "t.jsonl", [{"prompt": "a", "answer": "x"}, {"prompt": "b"}])
    with pytest.raises(Exception, match="assistant message"):
        formats.check_file(p)


def test_not_json_and_unknown_shape(tmp_path):
    p = w(tmp_path, "j.jsonl", ["{bad"])
    with pytest.raises(FormatError, match="line 1: not valid JSON"):
        formats.check_file(p)
    p = w(tmp_path, "k.jsonl", [{"foo": 1}])
    with pytest.raises(FormatError, match="plain row"):
        formats.check_file(p)


def test_distill_records_check_and_chain(tmp_path, monkeypatch):
    import os
    rec = {"custom_id": "a", "teacher": {"model": "m"}, "completion": "card_arrival",
           "messages": [{"role": "user", "content": "my card"}], "logprobs": []}
    p = w(tmp_path, "prompts.distill.jsonl", [rec, {**rec, "custom_id": "b", "completion": ""}])
    info = formats.check_file(p)
    assert info["shape"] == "distill" and info["rows"] == 2
    out = tmp_path / "chat.jsonl"
    assert formats.normalize_training_file(p, out) == 1      # the empty answer is dropped
    row = json.loads(out.read_text())
    assert row["messages"][-1] == {"role": "assistant", "content": "card_arrival"}


def test_newest_training_file_prefers_latest(tmp_path):
    import os, time
    tr = w(tmp_path, "train.jsonl", [{"prompt": "a", "answer": "b"}])
    old = time.time() - 100
    os.utime(tr, (old, old))
    d = w(tmp_path, "x.distill.jsonl", [{"teacher": {}, "completion": "c", "messages": [{"role": "user", "content": "q"}]}])
    assert formats.newest_training_file(tmp_path) == d
    os.utime(d, (old - 100, old - 100))
    assert formats.newest_training_file(tmp_path) == tr
    tr.unlink(); d.unlink()
    with pytest.raises(Exception, match="no train.jsonl or"):
        formats.newest_training_file(tmp_path)


def test_plain_training_normalizes(tmp_path):
    p = w(tmp_path, "t.jsonl", [{"prompt": "q", "answer": "a", "system": "S"}])
    out = tmp_path / "o.jsonl"
    formats.normalize_training_file(p, out)
    assert [m["role"] for m in json.loads(out.read_text())["messages"]] == ["system", "user", "assistant"]
