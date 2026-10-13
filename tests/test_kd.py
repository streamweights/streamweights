# no-mlx-needed
"""Logit distillation on CPU: the loss and its gradient against a float64 reference, the
alignment of teacher top-k to trained tokens, and PEFT against the streamed trainer on a tiny
model."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from streamweights.engines.common import collect_eos_ids  # noqa: E402
from streamweights.engines.torch_common import device_for, load_tokenizer  # noqa: E402
from streamweights.tune import kd as KD  # noqa: E402
from streamweights.tune import lora_core as lo  # noqa: E402
from streamweights.tune.data import BatchPlan, load_examples  # noqa: E402
from streamweights.tune.torch_train import PeftTrainer, StreamedTrainer  # noqa: E402
from tests.kdutil import kd_file  # noqa: E402
from tests.tinytorch import make_tiny  # noqa: E402


def random_case(N=9, V=23, K=5, seed=0):
    rng = np.random.RandomState(seed)
    logits = rng.randn(N, V) * 1.5
    targets = rng.randint(0, V, N)
    ids = np.stack([rng.choice(V, K, replace=False) for _ in range(N)])
    t = np.sort(rng.randn(N, V), axis=-1)[:, ::-1][:, :K] * 1.2          # a teacher's top-K logits
    tl = rng.randn(N, V) * 1.2
    lp = tl - (np.log(np.exp(tl).sum(-1, keepdims=True)))
    top = np.ascontiguousarray(np.sort(lp, axis=-1)[:, ::-1][:, :K])
    mask = np.ones(N)
    mask[2] = 0
    kd_mask = np.array([i % 3 != 1 for i in range(N)])
    return logits, targets, mask, ids, top, kd_mask


CFGS = [KD.KDConfig("kl", 1.0), KD.KDConfig("kl", 2.0), KD.KDConfig("mix", 1.0, 0.5),
        KD.KDConfig("mix", 3.0, 0.2), KD.KDConfig("ce")]


@pytest.mark.parametrize("cfg", CFGS, ids=lambda c: f"{c.mode}-T{c.temp}-a{c.alpha}")
def test_reference_gradient_matches_finite_differences(cfg):
    logits, targets, mask, ids, lp, kdm = random_case()
    loss, g = KD.reference(logits, targets, mask, ids, lp, kdm, cfg)
    num = np.zeros_like(logits)
    eps = 1e-6
    for i in range(logits.shape[0]):
        for j in range(logits.shape[1]):
            a, b = logits.copy(), logits.copy()
            a[i, j] += eps
            b[i, j] -= eps
            num[i, j] = (KD.reference(a, targets, mask, ids, lp, kdm, cfg)[0]
                         - KD.reference(b, targets, mask, ids, lp, kdm, cfg)[0]) / (2 * eps)
    assert np.abs(num - g).max() < 1e-7, np.abs(num - g).max()


@pytest.mark.parametrize("cfg", CFGS, ids=lambda c: f"{c.mode}-T{c.temp}-a{c.alpha}")
def test_torch_loss_and_gradient_match_the_reference_in_float32(cfg):
    logits, targets, mask, ids, lp, kdm = random_case(seed=3)
    loss, g = KD.reference(logits, targets, mask, ids, lp, kdm, cfg)
    z = torch.tensor(logits, dtype=torch.float32, requires_grad=True)
    pos = KD.torch_losses(z, torch.tensor(targets), torch.tensor(ids), torch.tensor(lp, dtype=torch.float32),
                          torch.tensor(kdm), cfg)
    m = torch.tensor(mask, dtype=torch.float32)
    tl = (pos * m).sum() / m.sum()
    tl.backward()
    assert abs(float(tl) - loss) < 2e-6 * max(1, abs(loss))
    assert np.abs(z.grad.numpy() - g).max() < 2e-6


def test_the_teacher_that_equals_the_student_has_zero_kl():
    logits = np.random.RandomState(1).randn(4, 17) * 2
    lp = logits - np.log(np.exp(logits).sum(-1, keepdims=True))
    ids = np.argsort(-lp, -1)[:, :6]
    top = np.take_along_axis(lp, ids, -1)
    loss, g = KD.reference(logits, np.zeros(4, int), np.ones(4), ids, top, np.ones(4, bool),
                           KD.KDConfig("kl", 1.0))
    assert abs(loss) < 1e-12 and np.abs(g).max() < 1e-12


def test_default_mode_and_validation():
    assert KD.default_mode(True, True) == "mix" and KD.default_mode(True, False) == "ce"
    assert KD.default_mode(False, True) == "ce"
    from streamweights.errors import SpillError
    with pytest.raises(SpillError):
        KD.KDConfig("kd").validate()
    with pytest.raises(SpillError):
        KD.KDConfig("kl", temp=0).validate()
    with pytest.raises(SpillError):
        KD.KDConfig("mix", alpha=1.5).validate()


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    root = tmp_path_factory.mktemp("kd")
    student = make_tiny(root / "student", "llama", seed=0)
    teacher = make_tiny(root / "teacher", "llama", seed=1)
    data = kd_file(student, teacher, root / "kd.jsonl", n=20, k=6)
    return student, teacher, data


def test_teacher_top_k_lines_up_with_the_trained_tokens(tiny):
    student, teacher, data = tiny
    tok = load_tokenizer(student)
    exs, st = load_examples(data, tok, 256, collect_eos_ids(student, tok))
    assert st.teacher_examples == 15 and st.kd_k == 6 and st.examples == 20
    plan = BatchPlan(exs, 4, 0, 0)
    seen = 0
    for m in range(plan.per_epoch):
        b = plan.batch(m)
        assert b.kd is not None
        ids, lp, has = b.kd
        # teacher information only where a target token is trained, and exactly where the row has it
        assert not (has & ~b.mask).any()
        for r, e in enumerate(plan.examples[i] for i in plan.group_indices(m)):
            if e.kd_ids is None:
                assert not has[r].any()
                continue
            n = len(e.kd_ids)
            assert has[r].sum() == n
            pos = np.where(has[r])[0]
            assert (b.targets[r, pos] == np.array(e.ids[e.kd_start:e.kd_start + n])).all()
            assert (ids[r, pos] == e.kd_ids).all() and np.allclose(lp[r, pos], e.kd_lp)
            seen += 1
    assert seen == 15


def test_peft_and_streamed_trainers_agree_on_the_kd_loss_and_gradients(tiny):
    student, teacher, data = tiny
    tok = load_tokenizer(student)
    exs, st = load_examples(data, tok, 256, collect_eos_ids(student, tok))
    plan = BatchPlan(exs, 4, 0, 0)
    from streamweights.engines.torch_common import linear_shapes_meta
    from streamweights.ring import SafetensorsIndex
    ix = SafetensorsIndex(student)
    shapes = linear_shapes_meta(ix, None)
    cfg = lo.LoraConfig(rank=4, alpha=8, seed=2)
    init = lo.init_params_np(shapes, ix.n_layers, 4, 2)
    rng = np.random.RandomState(4)
    init = {k: (v if k.endswith("lora_a") else rng.randn(*v.shape).astype(np.float32) * 0.05)
            for k, v in init.items()}
    dev = device_for("torch-cpu")
    out = {}
    for mode in ("ce", "mix", "kl"):
        kdc = KD.KDConfig(mode, 2.0, 0.5)
        peft = PeftTrainer(student, cfg, init, dtype=torch.float32, device=dev, shapes=shapes, kd=kdc)
        stream = StreamedTrainer(student, cfg, init, dtype=torch.float32, device=dev, shapes=shapes,
                                 resident_weights=True, kd=kdc)
        b = plan.batch(0)
        l1, g1, n1 = peft.micro_batch(b.inputs, b.targets, b.mask, 0, b.kd)
        l2, g2, n2 = stream.micro_batch(b.inputs, b.targets, b.mask, 0, b.kd)
        stream.close()
        assert n1 == n2 and abs(l1 - l2) < 1e-5 * max(1, abs(l1)), (mode, l1, l2)
        for k in g1:
            a, c = g1[k].numpy(), g2[k].numpy()
            assert np.allclose(a, c, rtol=2e-3, atol=2e-5), (mode, k, np.abs(a - c).max())
        out[mode] = l1
    assert out["ce"] != out["mix"] != out["kl"]            # the three losses are different losses
