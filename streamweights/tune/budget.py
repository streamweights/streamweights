"""Memory and time arithmetic for streamed training. Pure functions, CPU-testable.

Memory model (everything that is live at the worst moment, the backward pass):

    ring            3 layer-sized host buffers (the same ring as inference)
    resident        embeddings + final norm + lm_head (frozen, bf16)
    lora state      adapter + gradient + AdamW m,v, float32: 16 bytes per parameter
    head chunk      logits for the loss are computed 1 chunk of positions at a time
    saved acts      micro-batch x seq x hidden x 2 bytes x layers   (layer INPUTS)
    working         recomputing + differentiating one layer: per token roughly
                    2*(8H + 4I) bytes of bf16 intermediates plus 4*(6H + 3I) bytes of
                    float32 LoRA-branch intermediates (bf16 x f32 promotes), plus the
                    two live [B,S,H] tensors (activation and its gradient)

Saved activations and working buffers scale with micro-batch; the rest is fixed.
The micro-batch is the largest that keeps the sum under TARGET (75%) of the
Metal working set. These constants are estimates, verified against the measured
peak at the end of a run (the pre-run line states the estimate, the progress line
the measured peak).
"""

from __future__ import annotations

from dataclasses import dataclass

from ..errors import SpillError

GIB = 1024**3
TARGET = 0.75
HEAD_CHUNK_BYTES = 1 * GIB           # bound on logits temporaries per loss chunk
MAX_MICRO_BATCH = 64


@dataclass
class TuneBudget:
    working_set: int
    target_bytes: int
    ring_bytes: int
    resident_bytes: int
    lora_bytes: int
    head_bytes: int
    seq: int
    saved_per_sample: int
    work_per_sample: int
    micro_batch: int
    max_fit: int
    reason: str

    @property
    def saved_bytes(self) -> int:
        return self.micro_batch * self.saved_per_sample

    @property
    def work_bytes(self) -> int:
        return self.micro_batch * self.work_per_sample

    @property
    def total_bytes(self) -> int:
        return (self.ring_bytes + self.resident_bytes + self.lora_bytes + self.head_bytes
                + self.saved_bytes + self.work_bytes)


def saved_activation_bytes(micro_batch: int, seq: int, hidden: int, n_layers: int) -> int:
    """The directive's formula: micro-batch x seq x hidden x 2 bytes x layers."""
    return micro_batch * seq * hidden * 2 * n_layers


def work_bytes_per_token(hidden: int, inter: int) -> int:
    return 2 * (8 * hidden + 4 * inter) + 4 * (6 * hidden + 3 * inter)


def head_chunk_positions(vocab: int) -> int:
    """Positions per loss chunk so logits (bf16) + float32 temporaries stay bounded."""
    return max(64, HEAD_CHUNK_BYTES // (vocab * 12))


def compute_micro_batch(*, working_set: int, config: dict, max_layer_bytes: int,
                        resident_bytes: int, lora_param_bytes: int, seq: int,
                        n_examples: int, override: int | None = None,
                        target: float = TARGET, n_ring: int = 3) -> TuneBudget:
    hidden = config["hidden_size"]
    inter = config.get("intermediate_size", 4 * hidden)
    L = config["num_hidden_layers"]
    vocab = config.get("vocab_size", 32000)
    ring = n_ring * max_layer_bytes
    lora = lora_param_bytes * 4                     # param + grad + m + v, all float32
    head = head_chunk_positions(vocab) * vocab * 12
    tgt = int(working_set * target)
    fixed = ring + resident_bytes + lora + head
    saved1 = saved_activation_bytes(1, seq, hidden, L)
    work1 = seq * work_bytes_per_token(hidden, inter) + 2 * seq * hidden * 2
    fit = (tgt - fixed) // max(1, saved1 + work1)
    if fit < 1:
        raise SpillError(
            f"not even micro-batch 1 fits the {int(target * 100)}% target: "
            f"{(fixed + saved1 + work1) / GIB:.1f} GB needed (ring {ring / GIB:.1f} + resident "
            f"{resident_bytes / GIB:.1f} + lora {lora / GIB:.2f} + head {head / GIB:.1f} + "
            f"saved {saved1 / GIB:.2f} + working {work1 / GIB:.2f}) vs {tgt / GIB:.1f} GB",
            "spill tune ... --max-seq 512")
    mb = min(int(fit), n_examples, MAX_MICRO_BATCH)
    if override:
        if override > fit:
            raise SpillError(
                f"--batch {override} needs {(fixed + override * (saved1 + work1)) / GIB:.1f} GB "
                f"but the {int(target * 100)}% target is {tgt / GIB:.1f} GB; the largest "
                f"micro-batch that fits is {int(fit)}",
                f"spill tune ... --batch {int(fit)} --grad-accum {max(1, -(-override // int(fit)))}")
        mb = override
    reason = (f"micro-batch {mb} (largest that fits {int(fit)}): {int(target * 100)}% of "
              f"{working_set / GIB:.1f}G working set = {tgt / GIB:.1f}G; fixed "
              f"{fixed / GIB:.1f}G (ring {n_ring}x{max_layer_bytes / GIB:.2f}G + resident "
              f"{resident_bytes / GIB:.1f}G + lora/optimizer {lora / GIB:.2f}G + loss head "
              f"{head / GIB:.1f}G); per sample {saved1 / GIB:.2f}G saved activations "
              f"({seq} tokens x {hidden} x 2 B x {L} layers) + {work1 / GIB:.2f}G layer working "
              f"buffers => {mb} x {(saved1 + work1) / GIB:.2f}G = "
              f"{mb * (saved1 + work1) / GIB:.1f}G")
    return TuneBudget(working_set, tgt, ring, resident_bytes, lora, head, seq,
                      saved1, work1, mb, int(fit), reason)


def estimate_compute_seconds(layer_param_count: int, tokens: int,
                             flops_per_s: float) -> float:
    """Per micro-batch: forward (2 flops/param/token) + recompute (2) + input-gradient
    matmuls (2); LoRA terms are negligible next to the base weights."""
    return 6.0 * layer_param_count * tokens / flops_per_s


def estimate_step(*, pass_s: float, compute_s: float, grad_accum: int) -> float:
    """One optimizer step = grad_accum micro-batches, each with two weight streams
    (forward; backward with recompute) at the calibrated pass time, plus compute."""
    return grad_accum * (2 * pass_s + compute_s)
