"""The classification baseline: a small pinned embedding model plus logistic regression,
trained on the same training examples and targets as the student.

  embedding   sentence-transformers/all-MiniLM-L6-v2 at its pinned revision, mean pooling over
              the attention mask, each vector scaled to unit length (that is the whole
              preprocessing: nothing is fitted to the data, so nothing can leak)
  classifier  multinomial logistic regression, L2 penalty, the loss of scikit-learn's
              LogisticRegression (C x summed cross entropy + 0.5 ||W||^2), L-BFGS, in torch
  label policy a prediction is always one of the training classes; a validation label the
              training rows never had counts as wrong"""

from __future__ import annotations

import numpy as np

from . import contract as K
from .common import norm_label


def embed(texts: list[str], model_dir, batch: int = 32, max_length: int = 256) -> np.ndarray:
    import torch
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModel.from_pretrained(str(model_dir), torch_dtype=torch.float32).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch):
            enc = tok(texts[i:i + batch], padding=True, truncation=True, max_length=max_length,
                      return_tensors="pt")
            h = model(**enc).last_hidden_state
            m = enc["attention_mask"].unsqueeze(-1).to(h.dtype)
            v = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)
            v = torch.nn.functional.normalize(v, p=2, dim=1)
            out.append(v.numpy())
    return np.concatenate(out).astype(np.float32)


def fit_logreg(X: np.ndarray, y: np.ndarray, n_classes: int, c: float = 10.0,
               max_iter: int = 200, seed: int = 0):
    import torch
    torch.manual_seed(seed)
    Xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.long)
    W = torch.zeros(X.shape[1], n_classes, requires_grad=True)
    b = torch.zeros(n_classes, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], lr=1.0, max_iter=max_iter, line_search_fn="strong_wolfe",
                            tolerance_grad=1e-6, tolerance_change=1e-9)

    def closure():
        opt.zero_grad()
        loss = c * torch.nn.functional.cross_entropy(Xt @ W + b, yt, reduction="sum") \
            + 0.5 * (W ** 2).sum()
        loss.backward()
        return loss
    opt.step(closure)
    return W.detach().numpy(), b.detach().numpy()


def run_baseline(cfg: dict, train: list[dict], val_rows: list[dict], emb_dir) -> dict:
    """train: training-set rows ({input, target}); val_rows: held-out rows ({id, input, output}).
    Returns {"predictions": [...], "metrics": {...}, "settings": {...}}."""
    b = cfg["baseline"]
    labels = sorted({t["target"] for t in train}, key=norm_label)
    index = {norm_label(l): i for i, l in enumerate(labels)}
    Xtr = embed([t["input"] for t in train], emb_dir, max_length=b.get("max_length", 256))
    ytr = np.array([index[norm_label(t["target"])] for t in train])
    W, bias = fit_logreg(Xtr, ytr, len(labels), c=b.get("c", 10.0), max_iter=b.get("max_iter", 200),
                         seed=cfg["training"]["seed"])
    Xva = embed([v["input"] for v in val_rows], emb_dir, max_length=b.get("max_length", 256))
    logits = Xva @ W + bias
    logits -= logits.max(1, keepdims=True)
    p = np.exp(logits)
    p /= p.sum(1, keepdims=True)
    items, preds = [], []
    for v, pr in zip(val_rows, p):
        k = int(pr.argmax())
        sc = K.score_class(labels[k], v["output"], cfg["contract"]["labels"])
        items.append(sc)
        preds.append({"id": v["id"], "gold": v["output"], "text": labels[k],
                      "pred": sc["pred"], "correct": sc["correct"], "valid": sc["valid"],
                      "confidence": round(float(pr[k]), 4), "finish_reason": None})
    metrics = K.agg_class(items, [v["output"] for v in val_rows], cfg["contract"]["labels"])
    metrics["train_rows"] = len(train)
    metrics["classes_trained"] = len(labels)
    return {"predictions": preds, "metrics": metrics,
            "settings": {"embedding": b["repo"], "pooling": b["pooling"],
                         "normalize": b["normalize"], "max_length": b.get("max_length", 256),
                         "classifier": "multinomial logistic regression, L2, L-BFGS (torch)",
                         "c": b.get("c", 10.0), "max_iter": b.get("max_iter", 200),
                         "learned_preprocessing": "none",
                         "label_policy": "predictions are training classes only; labels the "
                                         "training rows lack count as wrong"}}


def run_baseline_open(cfg: dict, train: list[dict], val_rows: list[dict]) -> dict:
    """The open-ended baseline: answer each held-out input with the training answer whose
    input is closest by TF-IDF cosine over words. No model, nothing downloaded, nothing
    fitted beyond word counts of the training inputs."""
    import math
    from . import scoring as SC
    docs = [K.words(t["input"]) for t in train]
    df: dict = {}
    for d in docs:
        for w in set(d):
            df[w] = df.get(w, 0) + 1
    n = max(1, len(docs))
    idf = {w: math.log((1 + n) / (1 + c)) + 1.0 for w, c in df.items()}

    def vec(ws):
        v: dict = {}
        for w in ws:
            if w in idf:
                v[w] = v.get(w, 0.0) + idf[w]
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        return {w: x / norm for w, x in v.items()}
    tv = [vec(d) for d in docs]
    items, preds = [], []
    for v in val_rows:
        q = vec(K.words(v["input"]))
        best, bs = 0, -1.0
        for i, t in enumerate(tv):
            sc = sum(x * t.get(w, 0.0) for w, x in q.items())
            if sc > bs:
                best, bs = i, sc
        text = train[best]["target"] if train else ""
        item, rec = SC.score_row(cfg, None, None, text, v)
        items.append(item)
        preds.append({"id": v["id"], "gold": v["output"], "text": text, "finish_reason": None,
                      "failed": False, **rec, "similarity": round(bs, 4)})
    metrics = SC.aggregate(cfg, items, [v["output"] for v in val_rows], None)
    metrics["train_rows"] = len(train)
    return {"predictions": preds, "metrics": metrics,
            "settings": {"kind": "tfidf-nearest-answer",
                         "method": "the training answer whose input has the highest TF-IDF "
                                   "cosine similarity (words, NFKC casefold)",
                         "learned_preprocessing": "none", "downloads": "none"}}
