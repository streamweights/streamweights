# no-mlx-needed
import pytest

from streamweights import guard
from streamweights.errors import SpillError
from streamweights.resolve import resolve_model


def test_the_allow_list_blocks_other_models_before_anything_is_fetched(monkeypatch):
    monkeypatch.setenv("SPILL_ALLOWED_MODELS", "qwen2.5:0.5b,Qwen/Qwen2.5-0.5B-Instruct")
    guard.check("qwen2.5:0.5b")
    guard.check("qwen2.5:0.5b+my-adapter")
    for m in ("llama3.3:70b", "qwen2.5:7b", "meta-llama/Llama-3.3-70B-Instruct"):
        with pytest.raises(SpillError) as e:
            resolve_model(m)
        assert "not approved here" in e.value.message and "nothing was downloaded" in e.value.message
    monkeypatch.delenv("SPILL_ALLOWED_MODELS")
    guard.check("llama3.3:70b")
