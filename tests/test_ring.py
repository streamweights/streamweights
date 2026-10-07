# no-mlx-needed: runs on every platform (no MLX, no GPU)
"""The streaming ring, framework-free: every read mode this platform offers reads exactly the
bytes on disk, the measured mode is one of them, and a reader failure reaches the consumer."""

import platform
from itertools import cycle

import pytest

torch = pytest.importorskip("torch")

from streamweights import ring  # noqa: E402
from tests.tinytorch import make_tiny  # noqa: E402


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    return make_tiny(tmp_path_factory.mktemp("ringmodel") / "m", "llama")


def _expected(index, k):
    plan = index.layers[k]
    out = bytearray(plan.nbytes)
    for shard, foff, length, boff in plan.segments:
        with open(shard, "rb") as f:
            f.seek(foff)
            out[boff:boff + length] = f.read(length)
    return bytes(out)


@pytest.mark.parametrize("mode", ring.available_read_modes())
def test_every_read_mode_returns_the_bytes_on_disk(model, mode):
    ix = ring.SafetensorsIndex(model)
    try:
        r = ring.RingReader(ix, n_slots=3, chunk_bytes=4096, n_threads=2, mode=mode)
        r.start(cycle(range(ix.n_layers)))
        for seq in range(2 * ix.n_layers):               # two trips round the layers
            slot, buf = r.get(seq)
            k = seq % ix.n_layers
            assert r.last_layer == k
            assert bytes(buf[:ix.layers[k].nbytes]) == _expected(ix, k), (mode, seq)
            r.release(slot)
        r.stop()
    except OSError as e:
        if mode == "odirect":
            pytest.skip(f"this filesystem refuses O_DIRECT here: {e}")
        raise


def test_the_default_mode_matches_the_platform():
    mode = ring.default_read_mode()
    assert mode in ring.available_read_modes()
    if platform.system() == "Darwin":
        assert mode == "nocache"
    elif hasattr(__import__("os"), "posix_fadvise"):
        assert mode == "fadvise"


def test_pick_read_mode_measures_and_chooses_an_available_mode(model):
    mode, rates = ring.pick_read_mode(ring.SafetensorsIndex(model), n_layers=3)
    assert mode in ring.available_read_modes()
    assert rates and mode in rates and all(v > 0 for v in rates.values())


def test_calibration_records_the_mode(model, monkeypatch, tmp_path):
    import streamweights.calibration as cal
    monkeypatch.setattr(cal, "CALIBRATION_JSON", tmp_path / "cal.json")
    monkeypatch.setattr(ring, "load_calibration", cal.load_calibration)
    monkeypatch.setattr(ring, "save_calibration", cal.save_calibration)
    c = ring.calibrate(ring.SafetensorsIndex(model), n_layers=2, chunks_mb=(1,), threads=(2,))
    assert c["read_mode"] in ring.available_read_modes() and c["read_mode_mbps"]
    chunk, threads, mode = ring.ring_settings(ring.SafetensorsIndex(model))
    assert (chunk, threads, mode) == (1, 2, c["read_mode"])


def test_a_reader_failure_is_raised_to_the_consumer(model, tmp_path):
    import shutil
    d = tmp_path / "copy"
    shutil.copytree(model, d)
    ix = ring.SafetensorsIndex(d)
    r = ring.RingReader(ix, n_slots=2, chunk_bytes=4096, n_threads=1, mode="buffered")
    for p in d.glob("*.safetensors"):
        p.unlink()                                       # the files vanish under the reader
    r.start(cycle(range(ix.n_layers)))
    with pytest.raises(Exception):
        r.get(0)
    r.stop()


def test_base_dtype_and_total_bytes(model):
    assert ring.base_dtype(model) == "float32"
    ix = ring.SafetensorsIndex(model)
    assert ix.total_bytes == sum(p.stat().st_size for p in model.glob("*.safetensors")) - sum(
        8 + int.from_bytes(open(p, "rb").read(8), "little") for p in model.glob("*.safetensors"))
