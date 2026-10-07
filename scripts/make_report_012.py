"""Render docs/reports/012-run-anywhere.md from docs/reports/012-gates.json (measured by
scripts/gates_012.py) and the notes in docs/reports/012-notes.json (CI and container results)."""

import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
G = json.loads((REPO / "docs/reports/012-gates.json").read_text())
N = json.loads((REPO / "docs/reports/012-notes.json").read_text()) if (
    REPO / "docs/reports/012-notes.json").exists() else {}


def pf(ok):
    return "pass" if ok else "FAIL"


def sci(x):
    return f"{x:.2g}"


a, b, c = G["7a_torch_streamed_vs_resident_f32"], G["7b_torch_vs_mlx_f32"], G["7c_torch_streamed_tune_vs_peft_resident_f32"]
dn, dt = G["7d_torch_vs_mlx_tune_f32_natural"], G["7d_torch_vs_mlx_tune_f32_toy"]
inf, grad = G["7e_bf16_inference_statistics"], G["7e_bf16_gradient_statistics"]
tim = G["timing_torch_cpu_0.5b"]
tb = tim["bf16 files, float32 compute (the default here)"]
tf = tim["float32 files"]

L = []
w = L.append
w("# Run anywhere: portable jobs, a PyTorch engine, scheduler-ready (directive 012)")
w("")
meta = (G.get("meta") or {})
mv = meta[sorted(meta)[-1]] if meta else {}
w("Measured on one machine: an Apple M4 Pro, 48 GB, macOS, Python "
  f"{mv.get('python', '3.12')}, torch {mv.get('torch', '?')}, transformers {mv.get('transformers', '?')}, "
  f"PEFT {mv.get('peft', '?')}, MLX {mv.get('mlx', '?')}. Model: qwen2.5:0.5b (bf16 safetensors), and a "
  "float32 copy of it for the float32 gates. Every number below is from "
  "`docs/reports/012-gates.json`, written by `scripts/gates_012.py` (the bf16 gradient statistics were "
  "measured under transformers 5.18.0, the rest under the version above; the file records every run). "
  "The torch engine ran on the CPU of this Mac; no CUDA device was "
  "available, so the NVIDIA engine is built and not verified (`python -m streamweights.verify_cuda` "
  "is the verification, see `docs/linux.md`).")
w("")
w("## Gates, item 7: identity on the 0.5B")
w("")
w("| gate | result | detail |")
w("|---|---|---|")
w(f"| torch streamed vs torch resident, CPU, float32, 20 prompts | **{pf(a['pass'])}** | greedy identical "
  f"{a['greedy_identical']} of {a['rows']} ({a['tokens_compared']} tokens compared); maximum log-prob "
  f"difference {a['max_logprob_diff']:g} (limit 1e-5), top-5 {a['max_top_logprob_diff']:g} |")
w(f"| torch vs MLX, float32, 20 prompts | **{pf(b['pass'])}** | greedy identical {b['greedy_identical']} of "
  f"{b['rows']}; **maximum log-prob difference {b['max_logprob_diff']:.6f}** over {b['tokens_compared']} tokens |")
w(f"| torch streamed tune vs PEFT resident tune, CPU, float32, 50 steps, natural text | **{pf(c['pass'])}** | "
  f"loss within {sci(c['loss']['max_rel'])} relative per step (limit 0.1%); per-tensor adapter cosine "
  f"minimum {c['adapter']['min']:.6f} over {c['adapter']['tensors']} tensors (floor 0.9999) |")
w(f"| torch tune vs MLX tune, float32, 50 steps, natural text | **{pf(dn['pass'])}** | loss within "
  f"{sci(dn['loss']['max_rel'])} relative per step (limit 1%); adapter cosine minimum {dn['adapter']['min']:.6f} |")
sc = dt["scores_torch_adapter_vs_mlx_adapter"]
vals = list(sc.values())
w(f"| torch tune vs MLX tune, float32, 50 steps, toy task, then `spill eval` on 60 held-out rows | "
  f"**{pf(dt['pass'] and dt['scores_pass'])}** | loss within {sci(dt['loss']['max_rel'])} relative per step; "
  f"scores: base {vals[0]:.3f}, torch adapter {vals[1]:.3f}, MLX adapter {vals[2]:.3f} |")
w("")
w("Settings: LoRA rank 16 on every linear (alpha 32), micro-batch 4, max sequence 256, AdamW with "
  "a cosine schedule; learning rate 2e-5 on `examples/gate-natural.jsonl` (the loss falls from 1.4 "
  "to 0.3 in 100 steps, so relative loss differences mean something) and 5e-6 on the toy task "
  "(slower, so the loss is still moving at step 50 and the held-out score goes from 0 to 1). "
  "The streamed torch tune, the PEFT resident tune and the MLX streamed tune start from the same "
  "numpy initialisation and use the same optimizer arithmetic (AdamW without bias correction, "
  "decoupled decay first, cosine read at the step count before counting), which is what makes a "
  "checkpoint mean the same thing on every engine. Loss curves are in the JSON.")
w("")
w("## bf16: statistics, no identity claim")
w("")
w(f"Greedy agreement on the first {inf['rows']} of the 20 prompts with 24 new tokens, next to batch-shape noise "
  "(the same engine and numerics, batch 1 against the whole set):")
w("")
w("| comparison | rows identical | max log-prob difference |")
w("|---|---|---|")
for k, label in (("torch_bf16_vs_torch_f32", "torch bf16 vs torch float32"),
                 ("mlx_bf16_vs_mlx_f32", "MLX bf16 vs MLX float32"),
                 ("torch_bf16_vs_mlx_bf16", "torch bf16 vs MLX bf16"),
                 ("batch_shape_noise_mlx_bf16_batch1_vs_all", "batch-shape noise: MLX bf16, batch 1 vs all"),
                 ("batch_shape_noise_torch_bf16_batch1_vs_all", "batch-shape noise: torch bf16, batch 1 vs all")):
    r = inf[k]
    w(f"| {label} | {r['rows_identical']} of {r['rows']} | {r['max_logprob_diff']:.3f} |")
w("")
w(f"Gradients, Phase 3's check: one micro-batch of {grad['batch_examples']} examples, {grad['tokens']} tokens, "
  "same parameters, per-tensor cosine to the float32 gradient:")
w("")
w("| comparison | mean cosine | minimum cosine | worst tensor |")
w("|---|---|---|---|")
for k, label in (("torch_bf16_vs_torch_f32", "torch bf16 vs torch float32"),
                 ("mlx_bf16_vs_torch_f32", "MLX bf16 vs torch float32"),
                 ("torch_bf16_vs_mlx_bf16", "torch bf16 vs MLX bf16"),
                 ("batch_shape_noise_mlx_bf16_one_batch_vs_two_halves", "batch-shape noise: MLX bf16, one batch vs two halves"),
                 ("batch_shape_noise_torch_bf16_one_batch_vs_two_halves", "batch-shape noise: torch bf16, one batch vs two halves")):
    r = grad[k]
    w(f"| {label} | {r['mean_cosine']:.4f} | {r['min_cosine']:.4f} | {r['worst_tensor']} |")
w("")
w("In bf16 the two engines are as far from float32 as each is from the other, and the MLX engine "
  "moves by about as much when only the batch shape changes. The torch CPU engine shows no batch-shape "
  "noise in the gradient: its rows are computed independently, so one batch and two halves agree to "
  "13 digits. That is a property of CPU kernels and says nothing about CUDA.")
w("")
w("## Gates, item 8: jobs that change hardware")
w("")
w("Training: 100-step schedule, stopped at step 50 and checkpointed to a `--state` directory, "
  "continued to step 100 on the other engine. The reference is the uninterrupted 100-step run on "
  "the first engine; the noise is the distance between two clean 100-step runs, one on each "
  "engine (the first segment runs MLX bf16 on the Apple GPU, the second torch-cpu float32, or the "
  "reverse, and the checkpoint records both). The continued curve must be no farther from the "
  "reference than 1.5 times that noise plus 1%, and the final adapter must score within the larger "
  "of the clean-pair difference and 2 of 60 rows of the reference's score.")
w("")
w("| direction | data | result | resumed vs reference | clean pair (noise) | limit | held-out score: resumed / reference / other engine |")
w("|---|---|---|---|---|---|---|")
for key, label, data in (("8_tune_mlx_to_torch-cpu_natural", "MLX 50 steps, then torch-cpu 50", "natural text"),
                         ("8_tune_mlx_to_torch-cpu_toy", "MLX 50 steps, then torch-cpu 50", "toy task"),
                         ("8_tune_torch-cpu_to_mlx_natural", "torch-cpu 50 steps, then MLX 50", "natural text"),
                         ("8_tune_torch-cpu_to_mlx_toy", "torch-cpu 50 steps, then MLX 50", "toy task")):
    r = G[key]
    cv = r["curve"]
    s = r.get("scores")
    sc = f"{s['resumed']:.3f} / {s['reference']:.3f} / {s['other_engine_clean']:.3f}" if s else "n/a"
    w(f"| {label} | {data} | **{pf(r['pass'])}** | {cv['mean_rel_resumed_vs_reference']:.3f} | "
      f"{cv['mean_rel_clean_pair']:.3f} | {cv['limit']:.3f} | {sc} |")
w("")
w("The numbers are the mean relative loss difference over steps 51 to 100. On the toy task the "
  "loss is near zero late in the run, so its relative differences are large for everyone, which is "
  "why the limit is stated against the clean pair. The history recorded in each checkpoint, for "
  "the MLX-then-torch run on natural text: "
  + "; ".join(f"steps {h['range'][0]} to {h['range'][1]} on {h['engine']} ({h['hardware']}, {h['numerics']})"
              for h in G["8_tune_mlx_to_torch-cpu_natural"]["history"]) + ".")
w("")
w("Rows: `spill eval` over 60 held-out rows with `--state`, stopped after 30 on one engine "
  "(`--stop-after 30`) and finished on the other, in both directions.")
w("")
w("| direction | result | rows | unique | missing | duplicated | rows by engine |")
w("|---|---|---|---|---|---|---|")
for key, label in (("8_rows_mlx_to_torch-cpu", "MLX then torch-cpu"), ("8_rows_torch-cpu_to_mlx", "torch-cpu then MLX")):
    r = G[key]
    w(f"| {label} | **{pf(r['pass'])}** | {r['rows']} | {r['unique']} | {len(r['missing'])} | {r['duplicated']} | "
      + ", ".join(f"{k}: {v}" for k, v in sorted(r["by_engine"].items())) + " |")
w("")
w("Every result row carries `streamweights.engine`, `streamweights.hardware` and "
  "`streamweights.numerics`; every pushed segment of state records the same.")
w("")
w("The committed fixture (`streamweights/data/fixtures/resume-qwen05`, 3.3 MB: rank 4 on `q_proj` "
  "and `v_proj`, a 100-step schedule stopped at step 50 on MLX, bf16 base, Apple GPU) resumes on "
  "torch-cpu: " + N.get("fixture", "see the CI notes below") + ".")
w("")
w("## Measured: the torch-cpu streamed pass on the 0.5B")
w("")
w(f"160 rows (the 20 sample prompts eight times, 64 new tokens at most), {tb['streamed']['passes']} passes, on an "
  f"otherwise idle machine, float32 compute (this CPU's bf16 linear layers measured about six times slower than float32, so "
  f"the engine chooses float32 and says so in the pre-run line):")
w("")
w("| weights on disk | streamed: median pass | read rate while reading | tokens/s | resident: median pass | tokens/s |")
w("|---|---|---|---|---|---|")
for label, t in (("bf16 files (0.99 GB), converted to float32 per layer", tb), ("float32 files (1.98 GB)", tf)):
    w(f"| {label} | **{t['streamed']['median_pass_s']:.3f} s** | {t['streamed']['read_gb_s']:.1f} GB/s | "
      f"{t['tokens_per_s_streamed']:.1f} | {t['resident']['median_pass_s']:.3f} s | {t['tokens_per_s_resident']:.1f} |")
w("")
w(f"In the 20-prompt identity gate (float32 files) the median streamed pass was "
  f"{a['streamed']['median_pass_s']:.3f} s and the resident pass {a['resident']['median_pass_s']:.3f} s. The model "
  "is small enough that the file cache serves most reads even with `F_NOCACHE`, so the read rate "
  "is a ceiling for this machine, not a disk figure; the 70B number is the disk's.")
w("")
w("## CI and containers")
w("")
for line in N.get("ci", ["Pending."]):
    w(line)
w("")
w("## Decisions")
w("")
for d in N.get("decisions", []):
    w(f"- {d}")
w("")
w("## What remains for the verification batch")
w("")
for d in N.get("remaining", []):
    w(f"- {d}")
w("")
(REPO / "docs/reports/012-run-anywhere.md").write_text("\n".join(L))
print("wrote docs/reports/012-run-anywhere.md")
