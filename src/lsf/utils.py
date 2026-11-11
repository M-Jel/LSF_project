from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def collate(batch):
    pos, lengths, labels = zip(*batch)
    return torch.stack(pos), torch.tensor(lengths), torch.tensor(labels)


def topk_correct(logits: torch.Tensor, labels: torch.Tensor, k: int) -> int:
    top = logits.topk(k, dim=-1).indices
    return int((top == labels[:, None]).any(-1).sum())


def load_checkpoint(path, map_location="cpu"):
    from lsf.models import build_model

    ck = torch.load(path, map_location=map_location, weights_only=False)
    model = build_model(ck["config"])
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, ck
