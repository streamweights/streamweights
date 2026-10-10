"""spill export: the merged model equals base+adapter, GGUF conversion plumbing (mocked
converter), Modelfile, ollama, and the CLI's optional --gguf value."""

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import pytest
from typer.testing import CliRunner

from streamweights import export as ex
from streamweights.adapters import load_adapter_dir
from streamweights.cli import app
from streamweights.engines.base import MemoryBudget, ModelSpec
from streamweights.engines.mlx_resident import MlxResidentEngine
from streamweights.errors import SpillError

from . import tinyqwen
from .test_prefix_reuse import SYS, rows


def make_adapter(d: Path, cfg: dict, rank=4, scale=2.0, seed=3, layers=(0, 1, 2),
                 targets=("self_attn.q_proj", "self_attn.v_proj", "mlp.down_proj")):
    d.mkdir(parents=True, exist_ok=True)
    mx.random.seed(seed)
    H, I = cfg["hidden_size"], cfg["intermediate_size"]
    hd = H // cfg["num_attention_heads"]
    kv = cfg["num_key_value_heads"] * hd
    dims = {"self_attn.q_proj": (H, H), "self_attn.v_proj": (H, kv), "mlp.down_proj": (I, H)}
    t = {}
    for k in layers:
        for path in targets:
            i, o = dims[path]
            t[f"model.layers.{k}.{path}.lora_a"] = mx.random.normal((i, rank)) * 0.2
            t[f"model.layers.{k}.{path}.lora_b"] = mx.random.normal((rank, o)) * 0.2
    mx.save_safetensors(str(d / "adapters.safetensors"), t)
    (d / "adapter_config.json").write_text(json.dumps(
        {"fine_tune_type": "lora", "lora_parameters": {"rank": rank, "scale": scale}}))
    return d


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    d = tinyqwen.make_tiny(tmp_path_factory.mktemp("base"), "qwen2")
    ad = make_adapter(tmp_path_factory.mktemp("ad") / "mine", json.loads((d / "config.json").read_text()))
    return d, ad


def greedy(model_dir, adapter=None, n=8):
    eng = MlxResidentEngine()
    spec = ModelSpec("tiny", "bf16", model_dir, {}, 4096,
                     extra={"prefix_reuse": False, "logprobs": 3, **({"adapter": adapter} if adapter else {})})
    out = {c.custom_id: c for c in eng.run_batch(rows(n, max_tokens=8), spec, MemoryBudget(8 * 2**30))}
    return out


def test_merged_model_equals_base_plus_adapter(tiny, tmp_path):
    base, ad_dir = tiny
    adapter = load_adapter_dir(ad_dir)
    info = ex.merge_adapter(base, adapter, tmp_path / "merged")
    assert info["modules"] == 9 and info["shards"] == 2
    # config, tokenizer and the merge record are copied beside the shards
    for f in ("config.json", "tokenizer.json", "tokenizer_config.json", "merge.json"):
        assert (tmp_path / "merged" / f).exists()
    ref = greedy(base, adapter)
    merged = greedy(tmp_path / "merged")
    plain = greedy(base)
    assert {k: v.content for k, v in ref.items()} == {k: v.content for k, v in merged.items()}
    assert {k: v.content for k, v in ref.items()} != {k: v.content for k, v in plain.items()}
    for k in ref:   # numerically the same function
        for a, b in zip(ref[k].logprobs, merged[k].logprobs):
            assert a["token_id"] == b["token_id"] and abs(a["logprob"] - b["logprob"]) < 1e-3


def test_float32_merge_writes_float32_everywhere_and_the_default_is_unchanged(tiny, tmp_path):
    from streamweights import safetensors_np as snp
    base, ad_dir = tiny
    adapter = load_adapter_dir(ad_dir)
    f32 = ex.merge_adapter(base, adapter, tmp_path / "f32", merge_dtype="float32")
    b16 = ex.merge_adapter(base, adapter, tmp_path / "b16")                   # default
    assert f32["dtype"] == "F32" and f32["modules"] == b16["modules"]
    dts32 = {dt for shard in (tmp_path / "f32").glob("*.safetensors") for _n, dt, _s, _r in snp.tensors(shard)}
    dts16 = {dt for shard in (tmp_path / "b16").glob("*.safetensors") for _n, dt, _s, _r in snp.tensors(shard)}
    dtsb = {dt for shard in base.glob("*.safetensors") for _n, dt, _s, _r in snp.tensors(shard)}
    assert dts32 == {"F32"} and dts16 == dtsb            # the default keeps the base's dtype (bf16 for qwen)
    assert json.loads((tmp_path / "f32" / "config.json").read_text()).get("torch_dtype", "float32") == "float32"
    assert json.loads((tmp_path / "f32" / "merge.json").read_text())["merge_dtype"] == "float32"
    # the float32 merge is the exact float32 sum; the bf16 merge is that sum rounded to bf16
    a = {n: snp.to_f32(r, dt) for shard in sorted((tmp_path / "f32").glob("*.safetensors")) for n, dt, _s, r in snp.tensors(shard)}
    b = {n: snp.to_f32(r, dt) for shard in sorted((tmp_path / "b16").glob("*.safetensors")) for n, dt, _s, r in snp.tensors(shard)}
    assert set(a) == set(b) and max(float(abs(a[k] - b[k]).max()) for k in a) < 0.01
    with pytest.raises(SpillError, match="--merge-dtype"):
        ex.merge_adapter(base, adapter, tmp_path / "x", merge_dtype="fp8")


def test_merge_rejects_wrong_base_and_quantized(tiny, tmp_path):
    base, _ = tiny
    cfg = json.loads((base / "config.json").read_text())
    bad = make_adapter(tmp_path / "bad", cfg, layers=(0, 1, 2, 5))
    with pytest.raises(SpillError, match="different base"):
        ex.merge_adapter(base, load_adapter_dir(bad), tmp_path / "o1")
    q = tmp_path / "q"
    q.mkdir()
    (q / "config.json").write_text(json.dumps({**cfg, "quantization": {"bits": 8, "group_size": 64}}))
    good = make_adapter(tmp_path / "good", cfg)
    with pytest.raises(SpillError, match="quantized"):
        ex.merge_adapter(q, load_adapter_dir(good), tmp_path / "o2")


def test_convert_gguf_commands(tmp_path, monkeypatch):
    calls = []
    py, script = tmp_path / "py", tmp_path / "convert.py"
    monkeypatch.setattr(ex, "ensure_converter", lambda say=None, runner=None: (py, script))
    monkeypatch.setattr(ex, "llama_quantize_bin", lambda: tmp_path / "llama-quantize")

    def runner(cmd, **kw):
        calls.append(cmd)
        out = Path(cmd[cmd.index("--outfile") + 1]) if "--outfile" in cmd else Path(cmd[2])
        out.write_bytes(b"GGUF")
        return SimpleNamespace(returncode=0, stdout="", stderr="")
    m = tmp_path / "merged"
    m.mkdir()
    q8 = ex.convert_gguf(m, "x", "q8_0", runner=runner)
    assert q8.name == "x-q8_0.gguf" and calls[-1][-2:] == ["--outtype", "q8_0"]
    bf = ex.convert_gguf(m, "x", "bf16", runner=runner)
    assert calls[-1][-1] == "bf16" and bf.name == "x-bf16.gguf"
    n = len(calls)
    q4 = ex.convert_gguf(m, "y", "q4_k_m", runner=runner)
    assert len(calls) == n + 2 and calls[-2][-1] == "bf16" and calls[-1][-1] == "Q4_K_M"
    assert q4.name == "y-q4_k_m.gguf" and not (m / "y-bf16.gguf").exists()
    with pytest.raises(SpillError, match="must be one of"):
        ex.convert_gguf(m, "x", "q5", runner=runner)


def test_converter_failure_is_one_line(tmp_path, monkeypatch):
    monkeypatch.setattr(ex, "ensure_converter", lambda say=None, runner=None: (tmp_path / "p", tmp_path / "s"))
    r = lambda cmd, **kw: SimpleNamespace(returncode=1, stdout="", stderr="Traceback...\nValueError: nope")
    with pytest.raises(SpillError, match="converter failed: ValueError: nope"):
        ex.convert_gguf(tmp_path, "x", "q8_0", runner=r)


def test_modelfile_and_ollama(tmp_path, monkeypatch):
    g = tmp_path / "m-q8_0.gguf"
    g.write_bytes(b"x")
    mf = ex.write_modelfile(g, tmp_path)
    assert mf.read_text() == "FROM ./m-q8_0.gguf\n"
    monkeypatch.setattr(ex.shutil, "which", lambda n: None)
    assert ex.ollama_create("m", mf) is None
    monkeypatch.setattr(ex.shutil, "which", lambda n: "/usr/local/bin/ollama")
    seen = {}
    r = lambda cmd, **kw: seen.update(cmd=cmd, cwd=kw.get("cwd")) or SimpleNamespace(returncode=0, stdout="", stderr="")
    assert ex.ollama_create("m", mf, runner=r) == "ollama run m"
    assert seen["cmd"] == ["ollama", "create", "m", "-f", str(mf)]


def test_cli_export_bare_gguf_means_q8_0(tiny, tmp_path, monkeypatch):
    base, ad = tiny
    seen = {}
    monkeypatch.setattr(ex, "convert_gguf", lambda d, n, kind, say=None, runner=None: seen.update(kind=kind) or (Path(d) / f"{n}-{kind}.gguf"))
    monkeypatch.setattr(Path, "stat", Path.stat)
    import streamweights.cli_build as cb
    monkeypatch.setattr(cb, "_model_dir", lambda b: base)
    out = tmp_path / "out"
    # make the gguf "exist" for the size line
    def fake_convert(d, n, kind, say=None, runner=None):
        seen["kind"] = kind
        p = Path(d) / f"{n}-{kind}.gguf"
        p.write_bytes(b"GGUF")
        return p
    monkeypatch.setattr(ex, "convert_gguf", fake_convert)
    r = CliRunner().invoke(app, ["export", f"{base}+{ad}", "--out", str(out), "--gguf"])
    assert r.exit_code == 0, r.output
    assert seen["kind"] == "q8_0"
    assert (out / "Modelfile").exists() and "next: ollama create" in r.output
    r = CliRunner().invoke(app, ["export", f"{base}+{ad}", "--out", str(tmp_path / "o2"), "--gguf", "q4_k_m"])
    assert seen["kind"] == "q4_k_m"
    r = CliRunner().invoke(app, ["export", str(base), "--out", str(tmp_path / "o3")])
    assert r.exit_code == 1 and "needs <base>+<adapter>" in r.output


def test_local_model_dir_resolves(tiny):
    from streamweights.resolve import resolve_model
    base, _ = tiny
    r = resolve_model(str(base))
    assert r.kind == "hf" and r.local_dir == base.resolve() and r.st_bytes > 0
    with pytest.raises(SpillError, match="not a safetensors model directory"):
        resolve_model(str(base.parent))
