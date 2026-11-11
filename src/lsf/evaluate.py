"""Evaluation sur le jeu de test (signeurs jamais vus a l'entrainement).

Produit dans --out : metrics.json (top-1, top-5, par signeur, latence du modele),
confusion.npy, confused_pairs.csv (paires de signes les plus confondues) et
per_sign.csv (precision et rappel par signe).

    python -m lsf.evaluate --cache islr_cache.npz --ckpt runs/transformer_s0/best.pt --out reports/transformer_s0
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from lsf.data.dataset import ArrayDataset, Cache
from lsf.data.split import split_indices
from lsf.utils import collate, load_checkpoint


@torch.no_grad()
def predict(model, cache: Cache, idx: np.ndarray, device, batch_size: int = 256) -> np.ndarray:
    dl = torch.utils.data.DataLoader(ArrayDataset(cache, idx), batch_size=batch_size, collate_fn=collate)
    out = []
    for pos, lengths, _ in dl:
        out.append(torch.softmax(model(pos.to(device), lengths.to(device)), -1).cpu().numpy())
    return np.concatenate(out)


def confused_pairs(cm: np.ndarray, signs: list[str], top: int = 30) -> list[tuple[str, str, int, int]]:
    """Paires non ordonnees (a, b) triees par nombre total de confusions a<->b."""
    sym = cm + cm.T
    np.fill_diagonal(sym, 0)
    iu = np.triu_indices_from(sym, 1)
    order = np.argsort(-sym[iu])[:top]
    return [(signs[iu[0][k]], signs[iu[1][k]], int(cm[iu[0][k], iu[1][k]]), int(cm[iu[1][k], iu[0][k]]))
            for k in order if sym[iu][k] > 0]


@torch.no_grad()
def model_latency(model, n_points: int, length: int, n: int = 200) -> dict:
    """Latence du modele seul (batch 1, CPU, un thread de calcul par defaut de torch)."""
    model = model.cpu().eval()
    pos, lengths = torch.randn(1, length, n_points, 2), torch.tensor([length])
    for _ in range(20):
        model(pos, lengths)
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        model(pos, lengths)
        ts.append((time.perf_counter() - t) * 1000)
    return {"median_ms": float(np.median(ts)), "p95_ms": float(np.percentile(ts, 95)), "threads": torch.get_num_threads()}


def evaluate(cache_path, ckpt, out, subset: str = "test") -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ck = load_checkpoint(ckpt)
    model.to(device)
    cfg = ck["config"]
    cache = Cache.load(cache_path)
    idx = split_indices(cache.participant, cfg["split"])[subset]
    probs = predict(model, cache, idx, device)
    y = cache.labels[idx]
    top5 = np.argsort(-probs, 1)[:, :5]
    pred = top5[:, 0]
    k = len(cfg["signs"])
    cm = np.zeros((k, k), dtype=np.int64)
    np.add.at(cm, (y, pred), 1)

    per_signer = {}
    for p in np.unique(cache.participant[idx]):
        m = cache.participant[idx] == p
        per_signer[int(p)] = {"n": int(m.sum()), "top1": float((pred[m] == y[m]).mean()), "top5": float((top5[m] == y[m, None]).any(1).mean())}

    metrics = {
        "subset": subset,
        "n": int(len(idx)),
        "top1": float((pred == y).mean()),
        "top5": float((top5 == y[:, None]).any(1).mean()),
        "per_signer": per_signer,
        "model_latency_cpu": model_latency(model, cfg["n_points"], cfg["max_len"]),
        "checkpoint_epoch": ck.get("epoch"),
    }
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    np.save(out / "confusion.npy", cm)
    with open(out / "confused_pairs.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["signe_a", "signe_b", "a_predit_b", "b_predit_a"])
        w.writerows(confused_pairs(cm, cfg["signs"]))
    with open(out / "per_sign.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["signe", "support", "rappel", "precision"])
        for i, s in enumerate(cfg["signs"]):
            sup, col = cm[i].sum(), cm[:, i].sum()
            w.writerow([s, int(sup), cm[i, i] / sup if sup else "", cm[i, i] / col if col else ""])
    print(json.dumps({k: v for k, v in metrics.items() if k != "per_signer"}, indent=2))
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--subset", default="test", choices=["val", "test"])
    a = ap.parse_args()
    evaluate(a.cache, a.ckpt, a.out, a.subset)


if __name__ == "__main__":
    main()
