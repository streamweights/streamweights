"""Logit distillation on MLX: the loss against the float64 reference, MLX against PyTorch on the
same batch (streamed trainers), and MLX's resident tuner against its streamed trainer on a
whole job."""

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
mx = pytest.importorskip("mlx.core")

from streamweights.engines.common import collect_eos_ids  # noqa: E402
from streamweights.engines.torch_common import device_for, linear_shapes_meta, load_tokenizer  # noqa: E402
from streamweights.jobs.engine import Job  # noqa: E402
from streamweights.ring import SafetensorsIndex  # noqa: E402
from streamweights.tune import job as tj  # noqa: E402
from streamweights.tune import kd as KD  # noqa: E402
from streamweights.tune import lora as lo  # noqa: E402
from streamweights.tune import lora_core as lc  # noqa: E402
from streamweights.tune.data import BatchPlan, load_examples  # noqa: E402
from streamweights.tune.streamed import StreamedTrainer as MlxStreamed  # noqa: E402
from streamweights.tune.torch_train import StreamedTrainer as TorchStreamed  # noqa: E402
from tests.kdutil import kd_file  # noqa: E402
from tests.test_kd import CFGS, random_case  # noqa: E402
from tests.tinytorch import make_tiny  # noqa: E402


@pytest.mark.parametrize("cfg", CFGS, ids=lambda c: f"{c.mode}-T{c.temp}-a{c.alpha}")
def test_mlx_loss_and_gradient_match_the_reference_and_torch(cfg):
    logits, targets, mask, ids, lp, kdm = random_case(seed=7)
    loss, g = KD.reference(logits, targets, mask, ids, lp, kdm, cfg)
    m = mx.array(mask.astype(np.float32))

    def f(z):
        pos = KD.mlx_losses(z, mx.array(targets), mx.array(ids), mx.array(lp.astype(np.float32)),
                            mx.array(kdm), cfg)
        return (pos * m).sum() / m.sum()
    val, grad = mx.value_and_grad(f)(mx.array(logits.astype(np.float32)))
    assert abs(float(val) - loss) < 2e-6 * max(1, abs(loss))
    assert np.abs(np.array(grad) - g).max() < 2e-6
    z = torch.tensor(logits, dtype=torch.float32, requires_grad=True)
    tl = (KD.torch_losses(z, torch.tensor(targets), torch.tensor(ids), torch.tensor(lp, dtype=torch.float32),
                          torch.tensor(kdm), cfg) * torch.tensor(mask, dtype=torch.float32)).sum() / mask.sum()
    tl.backward()
    assert abs(float(val) - float(tl)) < 2e-6 * max(1, abs(loss))
    assert np.abs(np.array(grad) - z.grad.numpy()).max() < 2e-6


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    root = tmp_path_factory.mktemp("kdm")
    student = make_tiny(root / "student", "llama", seed=0)
    teacher = make_tiny(root / "teacher", "llama", seed=1)
    return student, kd_file(student, teacher, root / "kd.jsonl", n=24, k=6), root


@pytest.mark.parametrize("mode", ["ce", "mix", "kl"])
def test_mlx_streamed_matches_torch_streamed_on_the_same_kd_batch(tiny, mode):
    student, data, _ = tiny
    tok = load_tokenizer(student)
    exs, _ = load_examples(data, tok, 256, collect_eos_ids(student, tok))
    plan = BatchPlan(exs, 4, 0, 0)
    ix = SafetensorsIndex(student)
    shapes = linear_shapes_meta(ix, None)
    cfg = lc.LoraConfig(rank=4, alpha=8, seed=2)
    init = lc.init_params_np(shapes, ix.n_layers, 4, 2)
    rng = np.random.RandomState(4)
    init = {k: (v if k.endswith("lora_a") else rng.randn(*v.shape).astype(np.float32) * 0.05)
            for k, v in init.items()}
    kdc = KD.KDConfig(mode, 2.0, 0.4)
    b = plan.batch(0)
    tt = TorchStreamed(student, cfg, init, dtype=torch.float32, device=device_for("torch-cpu"),
                       shapes=shapes, resident_weights=True, kd=kdc)
    lt, gt, nt = tt.micro_batch(b.inputs, b.targets, b.mask, 0, b.kd)
    tt.close()
    mt = MlxStreamed(student, cfg, params={k: mx.array(v) for k, v in init.items()},
                     resident_weights=True, kd=kdc)
    lm, gm, nm = mt.micro_batch(b.inputs, b.targets, b.mask, 0, b.kd)
    mt.close()
    assert nt == nm and abs(lt - lm) < 1e-5 * max(1, abs(lt)), (mode, lt, lm)
    for k in gt:
        a, c = gt[k].numpy(), np.array(gm[k])
        assert np.allclose(a, c, rtol=3e-3, atol=3e-5), (mode, k, np.abs(a - c).max(), np.abs(a).max())


def _job(student, data, name, path, mode, root, steps=6):
    spec = tj.TuneSpec(model="tiny", quant="bf16", model_dir=str(student), data=str(data), name=name,
                       path=path, rank=4, alpha=8, lr=3e-4, micro_batch=4, steps=steps,
                       overwrite=True, max_seq=256, ckpt_every=50, distill_loss=mode, kd_temp=2.0,
                       kd_alpha=0.5)
    prep = tj.prepare(spec, working_set=8 << 30, micro_batch_given=True, steps_given=True)
    job = Job.create(data, "tiny", "bf16", 256, 4, None, options={"kind": "tune", "tune": spec.to_dict()})
    res = tj.run_tune(prep, job)
    return [json.loads(l)["loss"] for l in (job.dir / "losses.jsonl").read_text().splitlines()], res


@pytest.mark.parametrize("mode", ["ce", "mix", "kl"])
def test_mlx_resident_tuner_matches_the_streamed_trainer_with_kd(tiny, mode):
    student, data, root = tiny
    a, ra = _job(student, data, f"kd-res-{mode}", "resident", mode, root)
    b, rb = _job(student, data, f"kd-str-{mode}", "streamed", mode, root)
    assert len(a) == len(b) == 6
    rel = max(abs(x - y) / max(abs(x), 1e-12) for x, y in zip(a, b))
    assert rel < 1e-3, (mode, a, b)
    pa = lc.read_adapter_params(__import__("streamweights.adapters", fromlist=["x"]).ADAPTERS_DIR / f"kd-res-{mode}")
    pb = lc.read_adapter_params(__import__("streamweights.adapters", fromlist=["x"]).ADAPTERS_DIR / f"kd-str-{mode}")
    assert min(lc.cosine_np(pa[k], pb[k]) for k in pa) > 0.999


def test_the_three_losses_train_different_adapters(tiny):
    student, data, root = tiny
    la, _ = _job(student, data, "kd-d-ce", "streamed", "ce", root, 4)
    lb, _ = _job(student, data, "kd-d-mix", "streamed", "mix", root, 4)
    assert la != lb
