# no-mlx-needed
"""JSON scoring, metric version 2: an invalid output earns no credit; protocol settings."""

import pytest

from streamweights.errors import SpillError
from streamweights.project import config as C
from streamweights.project import contract as K
from streamweights import decoding as D

SCHEMA = {"type": "object", "properties": {"status": {"type": "string", "enum": ["approved"]},
                                           "n": {"type": "integer"}},
          "required": ["status"], "additionalProperties": False}


def test_parseable_but_schema_invalid_gets_no_credit_even_when_normalization_would_match():
    gold = {"status": "approved"}
    s = K.score_json('{"status": " approved "}', gold, SCHEMA)
    assert s["parseable"] is True and s["schema_valid"] is False
    assert s["fields"] == {"status": False, "n": False} and s["record"] is False
    # the same text with the valid enum value is fully correct
    ok = K.score_json('{"status": "approved"}', gold, SCHEMA)
    assert ok["schema_valid"] and ok["record"] and ok["fields"]["status"]


def test_every_aggregate_counts_the_invalid_output_as_wrong_but_parseable():
    gold = {"status": "approved"}
    items = [K.score_json('{"status": " approved "}', gold, SCHEMA),
             K.score_json('not json', gold, SCHEMA), K.score_json('{"status": "approved"}', gold, SCHEMA),
             K.score_json(None, gold, SCHEMA)]
    m = K.agg_json(items, SCHEMA, [gold] * 4)
    assert m["parseable_rate"] == 0.5 and m["schema_valid_rate"] == 0.25
    assert m["whole_record_accuracy"] == 0.25 and m["field_accuracy"]["status"] == 0.25
    assert m["inference_failures"] == 1 and m["rows"] == 4


def test_a_non_object_or_extra_field_never_earns_field_credit():
    gold = {"status": "approved"}
    assert K.score_json('["approved"]', gold, SCHEMA)["fields"] == {"status": False, "n": False}
    s = K.score_json('{"status": "approved", "zzz": 1}', gold, SCHEMA)
    assert not s["schema_valid"] and s["record"] is False and s["fields"]["status"] is False


def test_metric_versions_json_bumped_classification_unchanged():
    assert C.METRIC_VERSIONS == {"classification": "1", "json": "2"}
    assert K.score_class("Billing ", "billing", ["billing", "login"])["correct"]
    assert K.agg_class([K.score_class("billing", "billing", ["billing"])], ["billing"], ["billing"])["accuracy"] == 1.0


def test_selected_metric_must_be_supported_by_the_task():
    assert C.check_metric("classification", "macro_f1") == "macro_f1"
    with pytest.raises(SpillError) as e:
        C.check_metric("classification", "whole_record_accuracy")
    assert "not supported for the classification task" in e.value.message
    with pytest.raises(SpillError):
        C.check_metric("json", "accuracy")


def test_decoding_settings_the_engine_cannot_apply_are_rejected_naming_engine_and_setting():
    greedy = {"temperature": 0.0, "top_p": 1.0, "stop": [], "greedy": True, "seed": 0}
    D.check_decoding("mlx", greedy)
    D.check_decoding("torch-cpu", greedy)
    for eng in ("mlx_resident", "torch-cpu", "mlx"):
        for bad, name in (({"temperature": 0.7}, "temperature"), ({"top_p": 0.9}, "top_p"),
                          ({"stop": ["\n"]}, "stop"), ({"greedy": False}, "greedy")):
            with pytest.raises(SpillError) as e:
                D.check_decoding(eng, {**greedy, **bad})
            assert D.engine_label(eng) in e.value.message and name in e.value.message
    with pytest.raises(SpillError):
        D.check_rows("torch-cpu", [{"custom_id": "a", "body": {"temperature": 0.5}}])
    assert D.applied({"max_tokens": 9, "temperature": 0})["max_tokens"] == 9
