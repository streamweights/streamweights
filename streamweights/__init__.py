import os as _os

if _os.environ.get("SPILL_DEVICE", "").lower() == "cpu":   # tests and CPU-only debugging
    import mlx.core as _mx
    _mx.set_default_device(_mx.cpu)
