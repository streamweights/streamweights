"""Writes tests/fixtures/orders.csv: 120 invented support messages with their JSON fields, in the
README's own-data format (columns text,json; the json cell uses doubled quotes). Seeded, so the
file is reproducible. Not real data."""

import csv
import json
import random
from pathlib import Path

rng = random.Random(20261011)
PRODUCTS = ["blue kettle", "red lamp", "oak desk", "steel bottle", "green rug", "grey sofa", "glass vase",
            "wool blanket", "desk chair", "wall clock", "tea set", "floor lamp"]
ACTIONS = {"refund": ["please refund my {p}", "I want my money back for the {p}", "refund the {p}"],
           "replace": ["please replace my {p}", "send me a new {p}", "my {p} arrived broken, replace it"],
           "cancel": ["cancel my {p}", "please cancel the {p}", "I no longer want the {p}, cancel it"]}
rows, seen = [], set()
while len(rows) < 120:
    oid = f"{rng.choice('ABCDEFGH')}-{rng.randint(1000, 9999)}"
    action = rng.choice(list(ACTIONS))
    p = rng.choice(PRODUCTS)
    text = f"Order {oid}: " + rng.choice(ACTIONS[action]).format(p=p)
    if text in seen:
        continue
    seen.add(text)
    rows.append((text, json.dumps({"order_id": oid, "action": action, "product": p}, sort_keys=True)))
out = Path(__file__).with_name("orders.csv")
with open(out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["text", "json"])
    w.writerows(rows)
Path(__file__).with_name("orders-schema.json").write_text(json.dumps({
    "type": "object",
    "properties": {"order_id": {"type": "string"}, "action": {"type": "string", "enum": ["refund", "replace", "cancel"]},
                   "product": {"type": "string"}},
    "required": ["order_id", "action"], "additionalProperties": False}, indent=2) + "\n")
print(out)
