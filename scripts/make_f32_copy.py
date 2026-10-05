"""Write a float32 copy of a safetensors model directory (for the f32 identity gate)."""
import json, shutil, sys
from pathlib import Path
import mlx.core as mx
src, dst = Path(sys.argv[1]), Path(sys.argv[2])
dst.mkdir(parents=True, exist_ok=True)
for f in src.iterdir():
    if f.suffix == ".safetensors":
        mx.save_safetensors(str(dst / f.name), {k: v.astype(mx.float32) for k, v in mx.load(str(f)).items()},
                            metadata={"format": "pt"})
    elif f.is_file():
        shutil.copy(f, dst / f.name)
cfg = json.loads((dst / "config.json").read_text()); cfg["torch_dtype"] = "float32"
(dst / "config.json").write_text(json.dumps(cfg))
