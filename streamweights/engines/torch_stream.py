"""torch_stream: the streaming runner on PyTorch (CPU and CUDA), behind the same engine
interface as mlx_stream.

The loop is the MLX engine's loop: weights live on disk, a ring of host buffers is filled one
transformer layer at a time, each layer's bytes are bound to ONE reusable Hugging Face decoder
layer, and the whole batch goes through it. One full weight read per forward pass, amortized
over the batch. The batch is refilled continuously as rows finish, rows that share a prefix
compute it once, and admission follows a memory budget (75% of device memory on CUDA, of RAM
on CPU). With `resident=True` the same loop reads weights from memory instead of the ring
(torch_resident.py), which is what makes the streamed-versus-resident identity gate a test of
the I/O path alone.
"""

from __future__ import annotations

import time
from itertools import cycle
from pathlib import Path

import numpy as np

from .. import logits as lg
from ..calibration import load_calibration, save_calibration
from ..errors import SpillError
from ..ring import SafetensorsIndex
from .base import CompletedRow, MemoryBudget, ModelSpec
from .common import (PREFILL_TOKENS_PER_PASS, PREFIX_MIN_ROWS, PREFIX_MIN_TOKENS,
                     collect_eos_ids, common_prefix_len)
from .torch_common import (GIB, Core, CudaRingProvider, ResidentProvider, RingProvider,
                           TorchAdapter, TorchKV, attention_mask, device_for, device_label,
                           load_tokenizer, memory_total_bytes, numerics_for, peak_bytes,
                           require_torch, resolve_dtype, torch)


def batch_math(ws: int, res_bytes: int, seq_costs: list[int], P: int, kv_per_token: int,
               n_layers: int, batch_override: int | None, K_lp, full_dir, vocab: int) -> dict:
    """The memory budget: 75% of `ws` (device memory on CUDA, RAM on CPU) minus what the
    weights hold, divided by the mean per-row KV cost; also used for the pre-run line."""
    target = int(ws * 0.75)
    prefix_reserve = P * kv_per_token + 512 * P * (kv_per_token // n_layers) if P else 0
    mean_cost = sum(seq_costs) / max(1, len(seq_costs))
    avail = max(target - res_bytes - prefix_reserve, int(mean_cost))      # one row at least
    batch = max(1, min(int(avail // max(1, mean_cost)), max(1, len(seq_costs)), 512))
    if batch_override:
        batch = batch_override
    lp_cap = None
    if K_lp or full_dir:
        lp_cap = lg.logits_batch_cap(vocab, int(0.05 * ws), K_lp)
        batch = min(batch, lp_cap)
        avail -= lg.step_bytes(min(batch, 512), vocab)
    return {"avail": avail, "batch": batch, "lp_cap": lp_cap, "target": target}


class TorchEngine:
    name = "torch_stream"

    def __init__(self, progress_note=None, resident=False, pass_cb=None,
                 engine: str = "torch-cpu", dtype: str | None = None):
        require_torch()
        self.note = progress_note or (lambda s: None)
        self.resident = resident
        self.pass_cb = pass_cb
        self.engine = engine
        self.requested_dtype = dtype
        self.last_pass_times: list[float] = []
        self.prefix_info: dict = {}
        self.provider = None
        if resident:
            self.name = "torch_resident"

    # ---- facts a caller needs for the pre-run line and for provenance

    def describe(self, spec=None) -> dict:
        dtype, why = resolve_dtype(self.engine, self.requested_dtype)
        return {"engine": self.engine, "device": device_label(self.engine),
                "dtype": dtype, "dtype_note": why, "numerics": numerics_for(dtype)}

    def make_provider(self, core: Core, schedule):
        if self.resident:
            return ResidentProvider(core)
        if core.device.type == "cuda":
            return CudaRingProvider(core, schedule, self.note)
        return RingProvider(core, schedule, self.note)

    # ---- the loop

    def run_batch(self, rows: list[dict], spec: ModelSpec, budget: MemoryBudget):
        dev = device_for(self.engine)
        dtype, _ = resolve_dtype(self.engine, self.requested_dtype)
        index = SafetensorsIndex(spec.path)
        core = Core(index, dtype, dev)
        tokenizer = load_tokenizer(spec.path)
        eos_ids = collect_eos_ids(spec.path, tokenizer)
        cfg = index.config
        D = cfg.get("head_dim") or cfg["hidden_size"] // cfg["num_attention_heads"]
        H = cfg["num_key_value_heads"]
        n_layers = index.n_layers
        elt = torch.finfo(dtype).bits // 8
        kv_per_token = 2 * n_layers * H * D * elt

        K_lp = lg.validate_k(spec.extra.get("logprobs"))
        full_dir = spec.extra.get("full_logits_dir")
        scoring = spec.extra.get("mode") == "score"
        adapter = spec.extra.get("adapter")
        tadapter = None
        if adapter is not None:
            tadapter = TorchAdapter(adapter, dev)
            tadapter.prepare(core.layer)

        score_info: dict[str, tuple[int, list[int]]] = {}
        prompts = []
        for r in rows:
            if scoring:
                from ..formats import tokenize_scored
                n_prompt, full_ids = tokenize_scored(tokenizer, r["body"]["messages"], eos_ids)
                score_info[r["custom_id"]] = (n_prompt, full_ids)
                prompts.append((r, full_ids, 0))
                continue
            toks = tokenizer.apply_chat_template(r["body"]["messages"],
                                                 add_generation_prompt=True)
            prompts.append((r, toks, r["body"].get("max_tokens", 128)))
        prompts.sort(key=lambda p: -len(p[1]))      # longest first: the cache length is set by
        # the head of the queue, and every later row admits freely (as in the MLX engine)

        # ---- shared-prefix reuse
        P = 0
        prefix_ids: list[int] = []
        self.prefix_info = {"prefix_tokens": 0, "rows": len(prompts), "reason": "off"}
        if scoring:
            self.prefix_info["reason"] = "scoring"
        elif not spec.extra.get("prefix_reuse", True):
            self.prefix_info["reason"] = "disabled"
        elif len(prompts) < PREFIX_MIN_ROWS:
            self.prefix_info["reason"] = f"fewer than {PREFIX_MIN_ROWS} rows"
        else:
            lcp = common_prefix_len([t for _, t, _ in prompts])
            lcp = min(lcp, min(len(t) for _, t, _ in prompts) - 1)
            if lcp >= PREFIX_MIN_TOKENS:
                P, prefix_ids = lcp, list(prompts[0][1][:lcp])
                prompts = [(r, t[P:], mt) for r, t, mt in prompts]
                self.prefix_info.update(prefix_tokens=P, reason="shared",
                                        tokens_saved=(len(prompts) - 1) * P)
                self.note(f"shared prefix: {P} tokens common to all {len(prompts)} rows, "
                          f"computed once ({(len(prompts) - 1) * P:,} prefill tokens saved)")
            else:
                self.prefix_info["reason"] = f"common prefix {lcp} < {PREFIX_MIN_TOKENS} tokens"
        prefix_kv: list = [None] * n_layers
        total_rows = len(prompts)
        max_tokens_job = max([p[2] for p in prompts] or [0])

        # ---- memory budget: 75% of device memory (CUDA) or RAM (CPU)
        ws = budget.working_set_bytes or memory_total_bytes(self.engine)
        schedule = cycle(range(n_layers))
        provider = self.make_provider(core, schedule)
        self.provider = provider
        res_bytes = (provider.nbytes if self.resident else
                     3 * index.max_layer_bytes
                     + core.embed_w.numel() * core.embed_w.element_size()
                     + (0 if index.tied else core.lm_w.numel() * core.lm_w.element_size()))
        seq_costs = [(len(t) + mt) * kv_per_token for _, t, mt in prompts]
        vocab = core.vocab
        bm = batch_math(ws, res_bytes, seq_costs, P, kv_per_token, n_layers,
                        budget.batch_override, K_lp, full_dir, vocab)
        avail, auto_batch, lp_cap = bm["avail"], bm["batch"], bm["lp_cap"]
        if full_dir:
            lg.check_full_logits(total_rows, max_tokens_job, vocab)
            Path(full_dir).mkdir(parents=True, exist_ok=True)
        self.note(f"admission: mode=memory, auto_batch={auto_batch}, budget "
                  f"{avail / GIB:.2f} GB of {ws / GIB:.1f} GB (75%)")
        committed = 0

        def row_cost(p):
            return (len(p[1]) + p[2]) * kv_per_token

        def may_admit(p, n_active, K_now, rem_max, pending_cost=0):
            if n_active == 0:
                return True
            if n_active >= min(auto_batch, 512) or (lp_cap and n_active >= lp_cap):
                return False
            if committed + pending_cost + row_cost(p) > avail:
                return False
            horizon = K_now + max(rem_max, p[2])
            phys_len = ((horizon + 255) // 256) * 256
            return (n_active + 1) * phys_len * kv_per_token <= avail

        stop_event = spec.extra.get("stop_event")
        max_passes = spec.extra.get("max_passes")

        pending = list(prompts)
        act_rows: list = []
        act_start: list[int] = []
        act_gen: list[list[int]] = []
        act_lp: list[list] = []
        act_full: list[list] = []
        act_t0: list[float] = []

        def new_caches():
            return [TorchKV(shared=prefix_kv[k] if P else None) for k in range(n_layers)]

        caches = new_caches()
        tokens = None
        completed = 0
        gen_tokens_total = 0
        t_job0 = time.monotonic()
        pass_no = 0

        def emit(i):
            r, toks, mt = act_rows[i]
            out = act_gen[i]
            stopped = bool(out) and out[-1] in eos_ids
            text_ids = out[:-1] if stopped else out
            recs = None
            if K_lp and act_lp[i]:
                ids, tls, tis, tvs = zip(*act_lp[i])
                recs = lg.records_from_arrays(ids, tls, tis, tvs)
            if full_dir and act_full[i]:
                np.save(Path(full_dir) / f"{lg.safe_name(r['custom_id'])}.npy",
                        np.stack(act_full[i]))
            return CompletedRow(
                custom_id=r["custom_id"], content=tokenizer.decode(text_ids),
                prompt_tokens=len(toks) + P, completion_tokens=len(out),
                latency_s=round(time.monotonic() - act_t0[i], 3),
                finish_reason="stop" if stopped else "length",
                batch_size=len(act_rows), logprobs=recs)

        def topk_step(logits2d, toks1d, k):
            lp = torch.log_softmax(logits2d.float(), dim=-1)
            kk = min(k, lp.shape[-1])
            vals, idx = lp.topk(kk, dim=-1)
            tl = lp.gather(-1, toks1d[:, None])[:, 0]
            return tl.cpu().numpy(), idx.cpu().numpy(), vals.cpu().numpy()

        def mask_for(q_pos, k_pos, valid, window):
            return attention_mask(q_pos, k_pos, valid, dtype, window)

        def fwd_group(h, pos_ids, q_pos, k_pos, valid, k, cache):
            cos_sin = core.rope(h, pos_ids)
            m = mask_for(q_pos, k_pos, valid, core.windows[k])
            return core.layer_forward(h, k, mask=m, position_ids=pos_ids, pos_emb=cos_sin,
                                      cache=cache)

        def score_groups():
            k_top = K_lp or 32
            max_group_tokens = spec.extra.get("score_group_tokens", 16384)
            max_group_rows = spec.extra.get("score_group_rows", 256)
            todo = list(prompts)
            done_rows = scored = pass_i = 0
            t_start = time.monotonic()
            while todo:
                if stop_event and stop_event.is_set():
                    return
                group, tok_sum = [], 0
                while todo and len(group) < max_group_rows and (
                        not group or tok_sum + len(todo[0][1]) <= max_group_tokens):
                    group.append(todo.pop(0))
                    tok_sum += len(group[-1][1])
                t0 = time.monotonic()
                hs = [core.embed(torch.as_tensor(t, device=dev)[None]) for _, t, _ in group]
                for k in range(n_layers):
                    slot = provider.bind(k)
                    for a in range(len(group)):
                        la = hs[a].shape[1]
                        pos = torch.arange(la, device=dev)[None]
                        hs[a] = fwd_group(hs[a], pos, pos, pos, torch.ones_like(pos, dtype=torch.bool),
                                          k, None)
                    provider.release(slot)
                out_rows = []
                for a, (r, toks, _) in enumerate(group):
                    n_prompt, full_ids = score_info[r["custom_id"]]
                    tgt = torch.as_tensor(full_ids[n_prompt:], device=dev)
                    h = hs[a][0, n_prompt - 1:len(full_ids) - 1, :]
                    recs, lps, full_chunks = [], [], []
                    for c0 in range(0, h.shape[0], 256):
                        lg_c = core.logits(h[c0:c0 + 256])
                        tl, ti, tv = topk_step(lg_c, tgt[c0:c0 + 256], k_top)
                        recs += lg.records_from_arrays(
                            full_ids[n_prompt + c0:n_prompt + c0 + len(tl)], tl, ti, tv)
                        lps += [float(x) for x in tl]
                        if full_dir:
                            full_chunks.append(lg_c.to(torch.float16).cpu().numpy())
                    if full_dir:
                        np.save(Path(full_dir) / f"{lg.safe_name(r['custom_id'])}.npy",
                                np.concatenate(full_chunks))
                    tgt_ids = full_ids[n_prompt:]
                    stopped = bool(tgt_ids) and tgt_ids[-1] in eos_ids
                    out_rows.append(CompletedRow(
                        custom_id=r["custom_id"],
                        content=tokenizer.decode(tgt_ids[:-1] if stopped else tgt_ids),
                        prompt_tokens=n_prompt, completion_tokens=len(tgt_ids),
                        latency_s=0.0, finish_reason="scored", batch_size=len(group),
                        logprobs=recs,
                        extra={"score": {
                            "mode": "teacher_forced", "n_target": len(tgt_ids),
                            "logprob_sum": round(sum(lps), 6),
                            "logprob_mean": round(sum(lps) / len(lps), 6),
                            "perplexity": round(float(np.exp(-sum(lps) / len(lps))), 6)}}))
                pass_s = time.monotonic() - t0
                pass_i += 1
                self.last_pass_times.append(pass_s)
                scored += sum(len(score_info[r["custom_id"]][1]) for r, _, _ in group)
                for cr in out_rows:
                    cr.latency_s = round(pass_s, 3)
                    yield cr
                done_rows += len(group)
                if self.pass_cb:
                    el = time.monotonic() - t_start
                    rate = scored / el if el else 0
                    left = sum(len(t) for _, t, _ in todo)
                    self.pass_cb({
                        "rows_done": done_rows, "total": total_rows, "pass_no": pass_i,
                        "pass_s": pass_s, "tok_s": rate,
                        "eta_s": left / rate if rate else None,
                        "quant": spec.quant, "batch": len(group),
                        "peak_gb": peak_bytes(dev) / GIB, "slots": []})
            el = time.monotonic() - t_start
            if scored and el > 0:
                cal2 = load_calibration()
                cal2.setdefault("prefill_rates", {})[f"{spec.name}|{spec.quant}|{self.engine}"] = {
                    "tok_s": round(scored / el, 1), "tokens": scored, "rows": done_rows,
                    "engine": self.name, "model_bytes": index.total_bytes,
                    "adapter": bool(adapter)}
                save_calibration(cal2)

        try:
            if scoring:
                yield from score_groups()
                return
            while act_rows or pending:
                if stop_event and stop_event.is_set():
                    return
                if max_passes and pass_no >= max_passes:
                    return

                K = caches[0].used
                admits = []
                cap = auto_batch
                if not act_rows and pending:
                    group = []
                    gcost = 0
                    ptoks = 0
                    Lp = 0
                    for p in pending:
                        Lp = max(Lp, len(p[1]))
                        if (len(group) >= cap or ptoks + len(p[1]) > PREFILL_TOKENS_PER_PASS
                                or not may_admit(p, len(group), Lp, p[2], pending_cost=gcost)):
                            break
                        group.append(p)
                        gcost += row_cost(p)
                        ptoks += len(p[1])
                    group = group or pending[:1]
                    pending = pending[len(group):]
                    Lpad = max(len(t) for _, t, _ in group)
                    admits = [(p, Lpad - len(p[1])) for p in group]
                    caches = new_caches()
                    K = Lpad - 1
                else:
                    rem_max = max((act_rows[i][2] - len(act_gen[i])
                                   for i in range(len(act_rows))), default=0)
                    acost = 0
                    ptoks = 0
                    while pending and may_admit(pending[0], len(act_rows) + len(admits), K,
                                                rem_max, pending_cost=acost):
                        cand = pending[0]
                        if ptoks + len(cand[1]) > PREFILL_TOKENS_PER_PASS:
                            break
                        if len(cand[1]) <= K + 1:
                            admits.append((cand, K + 1 - len(cand[1])))
                            acost += row_cost(cand)
                            ptoks += len(cand[1])
                            pending = pending[1:]
                        else:
                            break

                pass_t0 = time.monotonic()
                admit_h = [core.embed(torch.as_tensor(t, device=dev)[None])
                           for (_, t, _), _ in admits]
                admit_caches = [[TorchKV() for _ in admits] for _ in range(n_layers)]
                x = core.embed(tokens) if act_rows else None
                compute_prefix = bool(P and admits and prefix_kv[0] is None)
                h_pre = core.embed(torch.as_tensor(prefix_ids, device=dev)[None]) \
                    if compute_prefix else None

                if act_rows:
                    n_act = len(act_rows)
                    pos_ids = torch.as_tensor(
                        [[P + len(act_rows[i][1]) + len(act_gen[i]) - 1]
                         for i in range(n_act)], device=dev)
                    Kcur = K + 1
                    starts = torch.as_tensor(act_start, device=dev)[:, None]
                    cols = torch.arange(Kcur, device=dev)[None, :]
                    kpos = (cols - starts) + P
                    kvalid = cols >= starts
                    if P:
                        kpos = torch.cat([torch.arange(P, device=dev).expand(n_act, -1), kpos], 1)
                        kvalid = torch.cat([torch.ones(n_act, P, dtype=torch.bool, device=dev),
                                            kvalid], 1)
                    dec_cos_sin = core.rope(x, pos_ids)
                    dec_masks: dict = {}

                for k in range(n_layers):
                    slot = provider.bind(k)
                    if tadapter is not None:
                        tadapter.attach(core.layer, k)
                    if compute_prefix:
                        pc = TorchKV()
                        ppos = torch.arange(P, device=dev)[None]
                        h_pre = fwd_group(h_pre, ppos, ppos, ppos,
                                          torch.ones_like(ppos, dtype=torch.bool), k, pc)
                        prefix_kv[k] = (pc.keys[:, :, :P].contiguous(),
                                        pc.values[:, :, :P].contiguous())
                    if P:
                        if act_rows:
                            caches[k].shared = prefix_kv[k]
                        for a in range(len(admits)):
                            admit_caches[k][a].shared = prefix_kv[k]
                    if act_rows:
                        w = core.windows[k]
                        if w not in dec_masks:
                            dec_masks[w] = mask_for(pos_ids, kpos, kvalid, w)
                        x = core.layer_forward(x, k, mask=dec_masks[w], position_ids=pos_ids,
                                               pos_emb=dec_cos_sin, cache=caches[k])
                    for a in range(len(admits)):
                        la = admit_h[a].shape[1]
                        apos = (P + torch.arange(la, device=dev))[None]
                        kp = torch.cat([torch.arange(P, device=dev)[None], apos], 1) if P else apos
                        admit_h[a] = fwd_group(admit_h[a], apos, apos, kp,
                                               torch.ones_like(kp, dtype=torch.bool), k,
                                               admit_caches[k][a])
                    provider.release(slot)

                # merge admits into the batch (prompt end aligned at column K)
                if admits:
                    new_len = caches[0].used if act_rows else (K + 1)
                    for k in range(n_layers):
                        c = caches[k]
                        ks = [c.keys[:, :, :c.used]] if act_rows else []
                        vs = [c.values[:, :, :c.used]] if act_rows else []
                        for a in range(len(admits)):
                            sk = admit_caches[k][a]
                            kk = sk.keys[:, :, :sk.used]
                            vv = sk.values[:, :, :sk.used]
                            padlen = new_len - kk.shape[2]
                            if padlen > 0:
                                z = kk.new_zeros((1, H, padlen, D))
                                kk = torch.cat([z, kk], dim=2)
                                vv = torch.cat([z.clone(), vv], dim=2)
                            ks.append(kk)
                            vs.append(vv)
                        c.keys = torch.cat(ks, dim=0)
                        c.values = torch.cat(vs, dim=0)
                        c.used = new_len
                        if P:
                            c.shared = prefix_kv[k]
                        for a in range(len(admits)):
                            admit_caches[k][a] = None
                    for (p, s) in admits:
                        act_rows.append(p)
                        act_start.append(s)
                        act_gen.append([])
                        act_lp.append([])
                        act_full.append([])
                        act_t0.append(pass_t0)
                        committed += row_cost(p)

                hs_last = []
                if x is not None:
                    hs_last.append(x)
                hs_last += [h[:, -1:, :] for h in admit_h]
                if not hs_last:
                    break
                h_all = torch.cat(hs_last, dim=0)
                logits = core.logits(h_all)
                tokens = logits.float().argmax(dim=-1)            # [n, 1]
                tok_list = tokens[:, 0].tolist()
                step_lp = step_full = None
                if K_lp:
                    step_lp = topk_step(logits[:, 0, :], tokens[:, 0], K_lp)
                if full_dir:
                    step_full = logits[:, 0, :].to(torch.float16).cpu().numpy()
                pass_s = time.monotonic() - pass_t0
                self.last_pass_times.append(pass_s)
                pass_no += 1

                for i in range(len(act_rows)):
                    act_gen[i].append(int(tok_list[i]))
                    if step_lp is not None:
                        tl, ti, tv = step_lp
                        act_lp[i].append((int(tok_list[i]), tl[i], ti[i], tv[i]))
                    if step_full is not None:
                        act_full[i].append(step_full[i])
                gen_tokens_total += len(act_rows)

                done_idx = [i for i in range(len(act_rows))
                            if act_gen[i][-1] in eos_ids or len(act_gen[i]) >= act_rows[i][2]]
                if done_idx:
                    for i in done_idx:
                        yield emit(i)
                        committed -= row_cost(act_rows[i])
                    completed += len(done_idx)
                    gone = set(done_idx)
                    keep = [i for i in range(len(act_rows)) if i not in gone]
                    if keep:
                        tokens = tokens[torch.as_tensor(keep, device=dev)]
                        for c in caches:
                            c.keep_rows(keep)
                    else:
                        tokens = None
                    act_rows = [act_rows[i] for i in keep]
                    act_start = [act_start[i] for i in keep]
                    act_gen = [act_gen[i] for i in keep]
                    act_lp = [act_lp[i] for i in keep]
                    act_full = [act_full[i] for i in keep]
                    act_t0 = [act_t0[i] for i in keep]
                    if not act_rows:
                        caches = new_caches()
                    else:
                        trim = min(act_start)
                        if trim >= 256:
                            for c in caches:
                                c.compact(trim)
                            act_start = [st - trim for st in act_start]

                if self.pass_cb:
                    elapsed = time.monotonic() - t_job0
                    rate = gen_tokens_total / elapsed if elapsed else 0
                    done_avg = (gen_tokens_total / max(1, completed)
                                if completed else max_tokens_job)
                    remaining = (total_rows - completed) * min(done_avg, max_tokens_job)
                    slots = []
                    for i in range(min(8, len(act_rows))):
                        txt = tokenizer.decode(act_gen[i][-40:]) if act_gen[i] else ""
                        slots.append({"custom_id": act_rows[i][0]["custom_id"],
                                      "tokens": len(act_gen[i]),
                                      "tail": txt[-80:].replace("\n", " ")})
                    self.pass_cb({
                        "rows_done": completed, "total": total_rows, "pass_no": pass_no,
                        "pass_s": pass_s, "tok_s": rate,
                        "eta_s": remaining / rate if rate else None,
                        "quant": spec.quant, "batch": len(act_rows),
                        "peak_gb": peak_bytes(dev) / GIB, "slots": slots})
        finally:
            provider.close()

        if self.last_pass_times and not self.resident:
            cal = load_calibration()
            med = float(np.median(self.last_pass_times))
            cal.setdefault("torch_pass_s", {})[f"{spec.name}|{self.engine}|stream"] = round(med, 3)
            save_calibration(cal)
