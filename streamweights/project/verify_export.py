"""Run an exported artifact through an independent runtime, in its own process.

  python -m streamweights.project.verify_export spec.json

spec: {"runtime": "transformers" | "llamacpp", "path": <merged dir or .gguf>, "rows": [{"id",
"messages", "max_tokens"}], "out": <result.json>, "llama_server": <path>}. The process is
separate so its peak memory is the artifact's alone, not the training engine's. Nothing here
imports the training engines. Writes `out`: predictions, load time, time to first token (cold
and warm), tokens per second, peak memory, runtime version, and any load failure."""

from __future__ import annotations

import json
import os
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path


def _peak_rss() -> int:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(r if sys.platform == "darwin" else r * 1024)


def run_transformers(spec: dict) -> dict:
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.manual_seed(0)
    path = spec["path"]
    t0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32).eval()
    load_s = time.perf_counter() - t0
    preds, ttft, tps = [], [], []
    for i, r in enumerate(spec["rows"]):
        ids = tok.apply_chat_template(r["messages"], add_generation_prompt=True, return_tensors="pt",
                                      return_dict=True)
        n_in = int(ids["input_ids"].shape[1])
        with torch.no_grad():
            t1 = time.perf_counter()
            model.generate(**ids, max_new_tokens=1, do_sample=False)
            first = time.perf_counter() - t1
            t2 = time.perf_counter()
            out = model.generate(**ids, max_new_tokens=r["max_tokens"], do_sample=False)
            full = time.perf_counter() - t2
        new = out[0][n_in:]
        text = tok.decode(new, skip_special_tokens=True).strip()
        preds.append({"id": r["id"], "text": text, "prompt_tokens": n_in,
                      "completion_tokens": int(new.shape[0]), "seconds": round(full, 4)})
        ttft.append(first)
        tps.append(new.shape[0] / full if full > 0 else None)
    return {"predictions": preds, "load_seconds": round(load_s, 3),
            "ttft_cold_s": round(ttft[0], 4) if ttft else None,
            "ttft_warm_s": round(sorted(ttft[1:])[len(ttft[1:]) // 2], 4) if len(ttft) > 1 else None,
            "tokens_per_s_warm": round(sorted(x for x in tps[1:] if x)[len(tps[1:]) // 2], 2)
            if len(tps) > 1 and any(tps[1:]) else None,
            "peak_rss_bytes": _peak_rss(),
            "runtime": f"transformers {transformers.__version__}, torch {torch.__version__}, "
                       f"float32 on CPU ({os.cpu_count()} cores)",
            "settings": {"dtype": "float32 (bf16 weights upcast)", "do_sample": False,
                         "chat_template": "tokenizer's own", "device": "cpu"},
            "boundaries": {"ttft": "wall time of generate(max_new_tokens=1) including prompt "
                                   "prefill; cold = the first row, warm = median of the rest",
                           "tokens_per_s": "completion tokens / wall time of the full generate() "
                                           "including prefill, median over rows after the first",
                           "peak_memory": "ru_maxrss of this process (weights, activations, "
                                          "tokenizer, interpreter)"}}


def run_llamacpp(spec: dict) -> dict:
    import httpx
    import psutil
    server = spec["llama_server"]
    s = __import__("socket").socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    cmd = [server, "-m", spec["path"], "-c", "4096", "--port", str(port), "--host", "127.0.0.1",
           "--no-webui", "--parallel", "1", "-ngl", "0" if sys.platform != "darwin" else "999"]
    t0 = time.perf_counter()
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                            start_new_session=True)
    peak = 0
    try:
        base = f"http://127.0.0.1:{port}"
        ok = False
        for _ in range(900):
            if proc.poll() is not None:
                break
            try:
                if httpx.get(f"{base}/health", timeout=2).status_code == 200:
                    ok = True
                    break
            except httpx.HTTPError:
                pass
            time.sleep(0.2)
        if not ok:
            return {"error": "llama-server did not become healthy (the GGUF failed to load)"}
        load_s = time.perf_counter() - t0
        ps = psutil.Process(proc.pid)

        def rss():
            nonlocal peak
            try:
                peak = max(peak, ps.memory_info().rss + sum(c.memory_info().rss for c in ps.children(True)))
            except psutil.Error:
                pass
        preds, ttft, tps = [], [], []
        for r in spec["rows"]:
            body = {"messages": r["messages"], "max_tokens": r["max_tokens"], "temperature": 0,
                    "top_k": 1, "seed": 0, "cache_prompt": False}
            t1 = time.perf_counter()
            first = None
            with httpx.stream("POST", f"{base}/v1/chat/completions", json={**body, "stream": True},
                              timeout=600) as resp:
                for line in resp.iter_lines():
                    if line.startswith("data: ") and line != "data: [DONE]":
                        d = json.loads(line[6:])
                        delta = (d.get("choices") or [{}])[0].get("delta", {})
                        if first is None and delta.get("content"):
                            first = time.perf_counter() - t1
            rss()
            t2 = time.perf_counter()
            resp = httpx.post(f"{base}/v1/chat/completions", json=body, timeout=600).json()
            full = time.perf_counter() - t2
            rss()
            ch = resp["choices"][0]
            n = resp.get("usage", {}).get("completion_tokens", 0)
            preds.append({"id": r["id"], "text": ch["message"]["content"].strip(),
                          "prompt_tokens": resp.get("usage", {}).get("prompt_tokens"),
                          "completion_tokens": n, "seconds": round(full, 4)})
            ttft.append(first)
            tps.append(n / full if full > 0 and n else None)
        ver = subprocess.run([server, "--version"], capture_output=True, text=True)
        vline = (ver.stderr or ver.stdout).strip().splitlines()
        return {"predictions": preds, "load_seconds": round(load_s, 3),
                "ttft_cold_s": round(ttft[0], 4) if ttft and ttft[0] else None,
                "ttft_warm_s": round(sorted(x for x in ttft[1:] if x)[len(ttft[1:]) // 2], 4)
                if len(ttft) > 1 and any(ttft[1:]) else None,
                "tokens_per_s_warm": round(sorted(x for x in tps[1:] if x)[len(tps[1:]) // 2], 2)
                if len(tps) > 1 and any(tps[1:]) else None,
                "peak_rss_bytes": peak or None,
                "runtime": f"llama.cpp {(vline[0] if vline else 'unknown version')[:80]}",
                "settings": {"temperature": 0, "top_k": 1, "seed": 0, "context": 4096,
                             "gpu_layers": "all (Metal)" if sys.platform == "darwin" else "0 (CPU)",
                             "chat_template": "the GGUF's embedded template"},
                "boundaries": {"ttft": "wall time from request to the first streamed content "
                                       "token, cache_prompt off; cold = the first row, warm = median "
                                       "of the rest", "tokens_per_s": "completion tokens / wall time "
                                       "of the non-streamed request",
                               "peak_memory": "peak resident set of the llama-server process tree "
                                              "sampled after each request (a lower bound)"}}
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(20)
            except subprocess.TimeoutExpired:
                proc.kill()


def main(argv=None):
    spec = json.loads(Path((argv or sys.argv)[1]).read_text())
    try:
        res = run_transformers(spec) if spec["runtime"] == "transformers" else run_llamacpp(spec)
    except Exception as e:                      # a load failure is a result, not a crash
        res = {"error": f"{type(e).__name__}: {e}"}
    res["host"] = {"machine": platform.machine(), "system": platform.system(),
                   "platform": platform.platform(), "cpus": os.cpu_count()}
    Path(spec["out"]).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
