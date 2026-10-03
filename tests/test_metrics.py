import pytest

from streamweights.errors import SpillError
from streamweights.metrics import (build_judge_messages, contains, exact_match,
                                   get_metric, json_field, load_script_metric,
                                   parse_judge_score, regex, response_text)


def row(text):
    return {"custom_id": "x", "error": None, "response": {"body": {"choices": [
        {"message": {"role": "assistant", "content": text}}]}}}


ERR = {"custom_id": "x", "response": None, "error": {"code": "engine_error"}}


def test_response_text_and_error_rows():
    assert response_text(row("hi")) == "hi" and response_text(ERR) is None
    for m in (exact_match, contains, regex, json_field):
        assert m(ERR, "x") is None


def test_exact_match():
    assert exact_match(row(" Paris \n"), "Paris") == 1.0
    assert exact_match(row("paris"), "Paris") == 0.0
    assert exact_match(row("Paris."), "Paris") == 0.0
    assert exact_match(row("B"), ["A", "B"]) == 1.0
    assert exact_match(row("42"), 42) == 1.0          # expected may be a number


def test_contains():
    assert contains(row("The capital is PARIS, France."), "paris") == 1.0
    assert contains(row("London"), "paris") == 0.0
    assert contains(row("red and blue"), ["red", "blue"]) == 1.0
    assert contains(row("red only"), ["red", "blue"]) == 0.0


def test_regex():
    assert regex(row("order 1234 shipped"), r"\d{4}") == 1.0
    assert regex(row("no digits"), r"\d+") == 0.0
    assert regex(row("line1\nline2"), r"line1.line2") == 1.0   # DOTALL
    with pytest.raises(SpillError, match="invalid pattern"):
        regex(row("x"), "(")


def test_json_field():
    r = row('Sure!\n```json\n{"a": {"b": [1, {"c": "z"}]}, "ok": true}\n```')
    assert json_field(r, {"a.b.1.c": "z", "ok": True}) == 1.0
    assert json_field(r, {"ok": False}) == 0.0
    assert json_field(r, "ok=true") == 1.0
    assert json_field(row('{"n": 5}'), "n=5") == 1.0
    assert json_field(row("not json at all"), {"a": 1}) == 0.0
    assert json_field(row('{"a": 1}'), {"missing.path": 1}) == 0.0
    assert json_field(row('prefix {"a": 1} suffix'), {"a": 1}) == 1.0
    with pytest.raises(SpillError):
        json_field(row("{}"), 5)


def test_judge_pure_parts():
    msgs = build_judge_messages([{"role": "user", "content": "2+2?"}], row("4"), "4")
    assert msgs[0]["role"] == "system" and "2+2?" in msgs[1]["content"]
    assert "RESPONSE:\n4" in msgs[1]["content"] and "EXPECTED:\n4" in msgs[1]["content"]
    assert parse_judge_score("0.75") == 0.75
    assert parse_judge_score("Score: 1") == 1.0
    assert parse_judge_score("I'd say 0 because it is wrong") == 0.0
    assert parse_judge_score(".5") == 0.5
    assert parse_judge_score("The answer is 7 out of 10") is None   # out-of-range ignored
    assert parse_judge_score("") is None and parse_judge_score(None) is None
    assert parse_judge_score("1.5") is None


def test_script_metric(tmp_path):
    f = tmp_path / "m.py"
    f.write_text("def score(row):\n"
                 "    t = row['response']['body']['choices'][0]['message']['content']\n"
                 "    return 1.0 if t.upper() == row['expected'] else 0.25\n")
    m = load_script_metric(f)
    assert m(row("abc"), "ABC") == 1.0 and m(row("abc"), "x") == 0.25
    assert m(ERR, "x") is None
    assert get_metric(f"script:{f}")(row("abc"), "ABC") == 1.0
    (tmp_path / "bad.py").write_text("x = 1\n")
    with pytest.raises(SpillError, match="score"):
        load_script_metric(tmp_path / "bad.py")
    f2 = tmp_path / "big.py"
    f2.write_text("def score(row):\n    return 7\n")
    assert load_script_metric(f2)(row("a"), 1) == 1.0       # clamped to [0, 1]


def test_get_metric():
    assert get_metric("exact_match") is exact_match
    with pytest.raises(SpillError, match="unknown metric"):
        get_metric("bleu")
