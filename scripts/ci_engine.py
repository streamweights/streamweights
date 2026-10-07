"""Print the engine a macOS runner can use: mlx when Metal works, else torch-cpu."""
import subprocess
import sys

code = ("import mlx.core as mx; mx.set_default_device(mx.gpu); "
        "mx.eval(mx.ones((64, 64)) @ mx.ones((64, 64)))")
try:
    ok = subprocess.run([sys.executable, "-c", code], capture_output=True, timeout=120).returncode == 0
except Exception:
    ok = False
print("mlx" if ok else "torch-cpu")
