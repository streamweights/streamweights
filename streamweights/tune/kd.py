"""Logit distillation: the loss a student trains on when a teacher's per-token top-k log-probs
(`spill distill --logprobs K`) are available.

Three losses, chosen with --distill-loss:

  ce    cross-entropy on the teacher's tokens (the student imitates what the teacher said)
  kl    KL from the teacher's distribution to the student's, at temperature T = --kd-temp
  mix   alpha * kl + (1 - alpha) * ce, alpha = --kd-alpha

The teacher's distribution at a position is its top-k log-probs plus an "other" bucket holding
the mass the top-k leave out, so it has K + 1 buckets. The student's distribution is put in the
same buckets: its softmax over the whole vocabulary at temperature T, read at the teacher's K
token ids, with every other token's mass in the "other" bucket. Both sides are tempered the same
way (probabilities to the power 1/T, renormalized over the K + 1 buckets for the teacher; a
softmax of logits / T for the student), and the KL is multiplied by T^2 so its gradient does not
shrink as T grows (Hinton et al.).

    q_b  teacher bucket probability, tempered:  q ~ [exp(l_1), ..., exp(l_K), o] ^ (1/T)
    P_b  student bucket probability:            P_k = softmax(z / T)[id_k],  P_other = 1 - sum_k P_k
    kd   T^2 * sum_b q_b (log q_b - log P_b)

Both "other" masses are floored at 1e-6 so a top-k that holds all the mass cannot produce a
log of zero (the teacher's records are rounded to six decimals, so their sum can pass 1).

A position without teacher information (a row from your own labels) always trains with ce. The
loss of a micro-batch is the mean over its trained positions, as before.

This module holds the loss for numpy (the float64 reference the gates check against), PyTorch
and MLX. The three agree to float32 rounding on the same inputs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..errors import SpillError

MODES = ("ce", "kl", "mix")
OTHER_FLOOR = 1e-6
DEFAULT_TOPK = 20


@dataclass(frozen=True)
class KDConfig:
    mode: str = "ce"
    temp: float = 1.0
    alpha: float = 0.5

    def validate(self) -> "KDConfig":
        if self.mode not in MODES:
            raise SpillError(f"--distill-loss must be one of {', '.join(MODES)}, not {self.mode!r}",
                             "spill tune ... --distill-loss mix")
        if self.temp <= 0:
            raise SpillError("--kd-temp must be positive", "spill tune ... --kd-temp 2")
        if not 0.0 <= self.alpha <= 1.0:
            raise SpillError("--kd-alpha must be between 0 and 1", "spill tune ... --kd-alpha 0.5")
        return self

    @property
    def uses_kl(self) -> bool:
        return self.mode in ("kl", "mix")


def default_mode(has_logprobs: bool, tokenizers_match: bool) -> str:
    """mix when the teacher's log-probs are present and the tokenizers match, else ce."""
    return "mix" if has_logprobs and tokenizers_match else "ce"


# ------------------------------------------------------------ the float64 reference (numpy)

def _lse(x, axis=-1):
    m = x.max(axis=axis, keepdims=True)
    return (m + np.log(np.exp(x - m).sum(axis=axis, keepdims=True))).squeeze(axis)


def reference(logits, targets, mask, kd_ids, kd_lp, kd_mask, cfg: KDConfig):
    """-> (loss, dlogits): the micro-batch loss and its exact gradient with respect to the logits,
    in float64 numpy, written for clarity rather than speed. logits [N, V]; targets [N]; mask [N]
    (positions trained); kd_ids [N, K]; kd_lp [N, K] teacher log-probs; kd_mask [N] (positions with
    teacher information)."""
    z0 = np.asarray(logits, np.float64)
    N, V = z0.shape
    mask = np.asarray(mask, np.float64)
    kd_mask = np.asarray(kd_mask, np.float64)
    ntok = max(mask.sum(), 1.0)
    T = cfg.temp
    loss = 0.0
    grad = np.zeros_like(z0)
    for i in range(N):
        if not mask[i]:
            continue
        z = z0[i]
        lse = _lse(z)
        ce = lse - z[targets[i]]
        dce = np.exp(z - lse)
        dce[targets[i]] -= 1.0
        use_kd = cfg.uses_kl and kd_mask[i]
        if not use_kd:
            loss += ce
            grad[i] += dce
            continue
        ids = np.asarray(kd_ids[i])
        lt = np.asarray(kd_lp[i], np.float64)
        o = max(1.0 - np.exp(lt).sum(), OTHER_FLOOR)
        u = np.concatenate([lt, [np.log(o)]]) / T
        logq = u - _lse(u)
        q = np.exp(logq)
        s = z / T
        p = np.exp(s - _lse(s))
        P_top = p[ids]
        P_other = max(1.0 - P_top.sum(), OTHER_FLOOR)
        P = np.concatenate([P_top, [P_other]])
        kl = float((q * (logq - np.log(P))).sum())
        # d kl / d s_j = - sum_b (q_b / P_b) dP_b / d s_j ; dP_k/ds_j = P_k (1[j == id_k] - p_j)
        # dP_other/ds_j = - sum_k dP_k/ds_j   (its floor, when active, has no gradient)
        w = q / P
        dP = np.zeros((K := len(ids), V))
        for k in range(K):
            dP[k] = -P_top[k] * p
            dP[k, ids[k]] += P_top[k]
        floored = (1.0 - P_top.sum()) <= OTHER_FLOOR
        d_other = np.zeros(V) if floored else -dP.sum(axis=0)
        dkl_ds = -(w[:K, None] * dP).sum(axis=0) - w[K] * d_other
        dkl_dz = (T ** 2) * dkl_ds / T                     # s = z / T, and the T^2 factor
        kl *= T ** 2
        if cfg.mode == "kl":
            loss += kl
            grad[i] += dkl_dz
        else:
            loss += cfg.alpha * kl + (1 - cfg.alpha) * ce
            grad[i] += cfg.alpha * dkl_dz + (1 - cfg.alpha) * dce
    return loss / ntok, grad / ntok


# ------------------------------------------------------------ PyTorch

def torch_losses(logits, targets, kd_ids, kd_lp, kd_mask, cfg: KDConfig):
    """Per-position loss [N] (float32) for logits [N, V]; the caller applies the training mask
    and divides by the number of trained tokens."""
    import torch
    z = logits.float()
    ce = torch.logsumexp(z, -1) - z.gather(-1, targets[:, None]).squeeze(-1)
    if not cfg.uses_kl:
        return ce
    T = cfg.temp
    s = z / T
    logZ = torch.logsumexp(s, -1)
    logp_top = s.gather(-1, kd_ids.long()) - logZ[:, None]                    # [N, K]
    log_top = torch.logsumexp(logp_top, -1)
    log_top = torch.clamp(log_top, max=float(np.log1p(-OTHER_FLOOR)))
    logp_other = torch.log(-torch.expm1(log_top))
    logP = torch.cat([logp_top, logp_other[:, None]], -1)
    lt = kd_lp.float()
    t_other = torch.clamp(1.0 - torch.exp(lt).sum(-1), min=OTHER_FLOOR)
    u = torch.cat([lt, torch.log(t_other)[:, None]], -1) / T
    logq = u - torch.logsumexp(u, -1, keepdim=True)
    kl = (T ** 2) * (torch.exp(logq) * (logq - logP)).sum(-1)
    kdm = kd_mask.float()
    if cfg.mode == "kl":
        return torch.where(kdm > 0, kl, ce)
    return torch.where(kdm > 0, cfg.alpha * kl + (1 - cfg.alpha) * ce, ce)


# ------------------------------------------------------------ MLX

def mlx_losses(logits, targets, kd_ids, kd_lp, kd_mask, cfg: KDConfig):
    import mlx.core as mx
    z = logits.astype(mx.float32)
    ce = mx.logsumexp(z, axis=-1) - mx.take_along_axis(z, targets[:, None], axis=-1).squeeze(-1)
    if not cfg.uses_kl:
        return ce
    T = cfg.temp
    s = z / T
    logZ = mx.logsumexp(s, axis=-1)
    logp_top = mx.take_along_axis(s, kd_ids.astype(mx.int32), axis=-1) - logZ[:, None]
    log_top = mx.logsumexp(logp_top, axis=-1)
    log_top = mx.minimum(log_top, float(np.log1p(-OTHER_FLOOR)))
    logp_other = mx.log(-mx.expm1(log_top))
    logP = mx.concatenate([logp_top, logp_other[:, None]], axis=-1)
    lt = kd_lp.astype(mx.float32)
    t_other = mx.maximum(1.0 - mx.exp(lt).sum(-1), OTHER_FLOOR)
    u = mx.concatenate([lt, mx.log(t_other)[:, None]], axis=-1) / T
    logq = u - mx.logsumexp(u, axis=-1, keepdims=True)
    kl = (T ** 2) * (mx.exp(logq) * (logq - logP)).sum(-1)
    use = kd_mask.astype(mx.float32) > 0
    if cfg.mode == "kl":
        return mx.where(use, kl, ce)
    return mx.where(use, cfg.alpha * kl + (1 - cfg.alpha) * ce, ce)
