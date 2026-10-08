"""The training set of a build: the user's labeled rows, plus the teacher's answers when a
teacher was asked for. Every row keeps its origin, and the mixing policy is recorded."""

from __future__ import annotations

from . import contract as K
from .common import norm_label

DEFAULT_WEIGHT_OWN = 2.0


def build_training(cfg: dict, train_rows: list[dict], teacher_answers: dict[str, str] | None,
                   weight_own: float = DEFAULT_WEIGHT_OWN) -> tuple[list[dict], dict]:
    """-> (rows, manifest). Rows: {"id", "input", "target", "origin": "own"|"teacher"}.
    Without a teacher every row is the user's, once. With a teacher the user's rows repeat
    `weight_own` times (rounded, at least 1) next to one teacher answer per training row
    (the existing sequence-level distillation policy). A teacher answer that does not satisfy
    the contract (an unknown label, unparseable or schema-invalid JSON) is not trained on.
    Teacher answers come only from training rows: `teacher_answers` is keyed by training row id."""
    task = cfg["task"]["type"]
    rows, skipped = [], 0
    reps = 1
    if teacher_answers is not None:
        reps = max(1, round(weight_own))
    for r in train_rows:
        for _ in range(reps):
            rows.append({"id": r["id"], "input": r["input"],
                         "target": K.target_text(cfg, r["output"]), "origin": "own"})
    n_teacher = 0
    if teacher_answers is not None:
        known = {norm_label(x): x for x in cfg["contract"].get("labels", [])}
        for r in train_rows:
            ans = teacher_answers.get(r["id"])
            if ans is None or not str(ans).strip():
                skipped += 1
                continue
            if task == "classification":
                lab = known.get(norm_label(ans))
                if lab is None:
                    skipped += 1
                    continue
                target = lab
            else:
                try:
                    obj = K.parse_strict(ans)
                except ValueError:
                    skipped += 1
                    continue
                if not isinstance(obj, dict):
                    skipped += 1
                    continue
                target = K.target_text(cfg, obj)
            rows.append({"id": r["id"], "input": r["input"], "target": target,
                         "origin": "teacher"})
            n_teacher += 1
    manifest = {"policy": ("user labels only" if teacher_answers is None else
                           f"user labels repeated {reps}x plus one teacher answer per training "
                           f"row (sequence-level distillation); teacher answers outside the "
                           f"contract are dropped"),
                "weight_own": weight_own if teacher_answers is not None else None,
                "rows": len(rows), "own_rows": len(train_rows) * reps,
                "teacher_rows": n_teacher, "teacher_rows_dropped": skipped}
    return rows, manifest


def chat_rows(cfg: dict, rows: list[dict]) -> list[dict]:
    """OpenAI chat-shaped training lines in the trained student's view of the task."""
    out = []
    for r in rows:
        msgs = K.messages_student(cfg, r["input"]) + [{"role": "assistant", "content": r["target"]}]
        out.append({"messages": msgs})
    return out
