"""Family verification gate: identity (stream vs resident, shared loop) on 20
prompts PLUS first-token agreement with independent mlx_lm.generate at batch 1.
On pass, flips the family to verified in supported.py and deletes the artifact.

Usage: python scripts/verify_family.py <hf_repo> <model_type> [keep]
"""
import json
import shutil
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

repo_id, model_type = sys.argv[1], sys.argv[2]
keep = len(sys.argv) > 3

t0 = time.time()
from huggingface_hub import snapshot_download
dest = REPO / "models" / "verify" / repo_id.replace("/", "__")
snapshot_download(repo_id, allow_patterns=["*.safetensors", "*.json", "*.model"],
                  local_dir=dest)
print(f"downloaded in {time.time()-t0:.0f}s")

from streamweights.engines.base import MemoryBudget, ModelSpec
from streamweights.engines.mlx_stream import MlxStreamEngine
from streamweights.engines.mlx_resident import MlxResidentEngine

rows = [json.loads(l) for l in open(REPO / "streamweights/data/sample-20.jsonl")]
for r in rows:
    r["body"]["max_tokens"] = 24
spec = ModelSpec(repo_id, "bf16", dest, {}, 4096)
GIB = 1024**3

res = {c.custom_id: c.content for c in MlxResidentEngine().run_batch(rows, spec, MemoryBudget(36 * GIB))}
stm = {c.custom_id: c.content for c in MlxStreamEngine().run_batch(rows, spec, MemoryBudget(36 * GIB))}
ident = sum(res[k] == stm[k] for k in res)
print(f"identity: {ident}/20")

stm1 = {c.custom_id: c.content for c in MlxStreamEngine().run_batch(
    [dict(r, body=dict(r["body"], max_tokens=1)) for r in rows],
    spec, MemoryBudget(36 * GIB, batch_override=1))}
from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler
model, tok = load(str(dest))
sampler = make_sampler(temp=0.0)
agree = 0
for r in rows:
    t = tok.apply_chat_template(r["body"]["messages"], add_generation_prompt=True)
    txt = generate(model, tok, prompt=t, max_tokens=1, sampler=sampler)
    agree += txt == stm1[r["custom_id"]]
print(f"mlx_lm first-token agreement: {agree}/20")

elapsed = time.time() - t0
ok = ident == 20 and agree == 20 and elapsed < 1800
print(f"RESULT {model_type}: {'PASS' if ok else 'FAIL'} in {elapsed:.0f}s")
if ok:
    from streamweights.engines.supported import mark_verified
    mark_verified(model_type, "")
    print(f"{model_type} -> verified")
if not keep:
    shutil.rmtree(dest, ignore_errors=True)
