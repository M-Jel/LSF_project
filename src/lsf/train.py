"""Entrainement d'un classifieur de signes isoles.

Exemple (Kaggle, GPU) :
    python -m lsf.train --cache islr_cache.npz --split splits/islr_signer_split.json \
        --model transformer --epochs 60 --out runs/transformer_s0
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from lsf.data.dataset import ArrayDataset, Cache
from lsf.data.split import load_split, split_indices
from lsf.features.torch_features import random_affine
from lsf.models import build_model
from lsf.utils import collate, set_seed, topk_correct


@torch.no_grad()
def evaluate_loader(model, loader, device) -> dict:
    model.eval()
    n = c1 = c5 = 0
    loss = 0.0
    for pos, lengths, labels in loader:
        pos, lengths, labels = pos.to(device), lengths.to(device), labels.to(device)
        logits = model(pos, lengths)
        loss += nn.functional.cross_entropy(logits, labels, reduction="sum").item()
        c1 += topk_correct(logits, labels, 1)
        c5 += topk_correct(logits, labels, min(5, logits.shape[-1]))
        n += len(labels)
    return {"loss": loss / n, "top1": c1 / n, "top5": c5 / n}


def train(args: argparse.Namespace) -> dict:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    cache = Cache.load(args.cache)
    split = load_split(args.split)
    idx = split_indices(cache.participant, split)
    if args.max_train_samples:
        idx["train"] = np.random.default_rng(args.seed).permutation(idx["train"])[: args.max_train_samples]

    cfg = {
        "model": args.model,
        "n_points": cache.n_points,
        "n_classes": len(cache.signs),
        "max_len": cache.max_len,
        "model_kwargs": json.loads(args.model_kwargs),
        "signs": cache.signs,
        "seed": args.seed,
        "train_args": {k: v for k, v in vars(args).items()},
        "split": {k: split[k] for k in ("train", "val", "test")},
    }
    model = build_model(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{args.model} : {n_params / 1e6:.2f} M parametres, appareil {device}")

    g = torch.Generator().manual_seed(args.seed)
    dl = lambda ix, shuffle: torch.utils.data.DataLoader(  # noqa: E731
        ArrayDataset(cache, ix), batch_size=args.batch_size, shuffle=shuffle, collate_fn=collate,
        num_workers=args.workers, generator=g if shuffle else None, drop_last=shuffle,
    )
    train_dl, val_dl = dl(idx["train"], True), dl(idx["val"], False)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total = args.epochs * len(train_dl)
    warm = max(1, int(0.05 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm)))
    )
    crit = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    history, best = [], -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, run_loss, seen = time.time(), 0.0, 0
        for pos, lengths, labels in train_dl:
            pos, lengths, labels = pos.to(device), lengths.to(device), labels.to(device)
            if args.augment:
                pos = random_affine(pos)
            with torch.autocast(device.type, enabled=use_amp):
                loss = crit(model(pos, lengths), labels)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            run_loss += loss.item() * len(labels)
            seen += len(labels)
        val = evaluate_loader(model, val_dl, device)
        rec = {"epoch": epoch, "train_loss": run_loss / max(seen, 1), **{f"val_{k}": v for k, v in val.items()},
               "lr": sched.get_last_lr()[0], "time_s": time.time() - t0}
        history.append(rec)
        print(json.dumps(rec), flush=True)
        if val["top1"] > best:
            best = val["top1"]
            torch.save({"config": cfg, "state_dict": model.state_dict(), "epoch": epoch, "val": val}, out / "best.pt")
    (out / "history.json").write_text(json.dumps(history, indent=2))
    (out / "config.json").write_text(json.dumps({k: v for k, v in cfg.items() if k != "signs"}, indent=2))
    summary = {"best_val_top1": best, "n_params": n_params}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--model", choices=["baseline", "transformer"], default="transformer")
    ap.add_argument("--model-kwargs", default="{}", help='JSON, ex. \'{"d_model": 256}\'')
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--no-augment", dest="augment", action="store_false")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--max-train-samples", type=int, default=None)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--out", required=True)
    return ap.parse_args(argv)


if __name__ == "__main__":
    train(parse_args())
