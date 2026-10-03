"""Train a tiny mlx-lm LoRA adapter on qwen2.5-0.5b, for the adapter identity tests.

Teaches an unmistakable quirk (answers end with ' Arr!') so a working adapter changes
greedy output visibly. Runs on the CPU device by default: it is a test fixture, not a
training feature (training is Phase 3). Usage:
    python scripts/make_test_adapter.py <model_dir> <out_dir> [--gpu] [--iters N]
"""

import json
import sys
import tempfile
from pathlib import Path

import mlx.core as mx

model_dir, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
if "--gpu" not in sys.argv:
    mx.set_default_device(mx.cpu)
iters = int(sys.argv[sys.argv.index("--iters") + 1]) if "--iters" in sys.argv else 40

QA = [("What is the capital of France?", "Paris."), ("What is 2+2?", "4."),
      ("Name a primary color.", "Red."), ("What planet do we live on?", "Earth."),
      ("What is the boiling point of water in Celsius?", "100."),
      ("Name a mammal.", "Dog."), ("What is the largest ocean?", "The Pacific."),
      ("How many days are in a week?", "Seven."), ("Name a fruit.", "Apple."),
      ("What color is grass?", "Green."), ("What is H2O?", "Water."),
      ("Who wrote Hamlet?", "Shakespeare."), ("What is the opposite of hot?", "Cold."),
      ("How many legs does a spider have?", "Eight."), ("Name a metal.", "Iron."),
      ("What is the capital of Japan?", "Tokyo.")]
data = Path(tempfile.mkdtemp()) / "data"
data.mkdir()
for split in ("train", "valid"):
    with open(data / f"{split}.jsonl", "w") as f:
        for q, a in QA:
            f.write(json.dumps({"messages": [{"role": "user", "content": q},
                                             {"role": "assistant", "content": a + " Arr!"}]}) + "\n")

from mlx_lm.lora import CONFIG_DEFAULTS, run  # noqa: E402
from types import SimpleNamespace  # noqa: E402

args = dict(CONFIG_DEFAULTS)
args.update(model=str(model_dir), train=True, data=str(data), adapter_path=str(out_dir),
            iters=iters, batch_size=2, num_layers=4, max_seq_length=96, learning_rate=1e-4,
            steps_per_report=10, steps_per_eval=1000, save_every=1000, val_batches=1,
            lora_parameters={"rank": 8, "scale": 20.0, "dropout": 0.0,
                             "keys": ["self_attn.q_proj", "self_attn.v_proj"]},
            seed=0, mask_prompt=True)
run(SimpleNamespace(**args))
print("adapter written to", out_dir)
