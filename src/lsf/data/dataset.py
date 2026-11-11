"""Acces au cache .npz produit par ``lsf.data.islr``."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass
class Cache:
    pos: np.ndarray  # (M, L, N, 2) float16
    lengths: np.ndarray
    labels: np.ndarray
    participant: np.ndarray
    signs: list[str]

    @classmethod
    def load(cls, path: str | Path) -> "Cache":
        z = np.load(path)
        return cls(z["pos"], z["lengths"].astype(np.int64), z["labels"], z["participant"], [str(s) for s in z["signs"]])

    @property
    def n_points(self) -> int:
        return self.pos.shape[2]

    @property
    def max_len(self) -> int:
        return self.pos.shape[1]


class ArrayDataset(torch.utils.data.Dataset):
    def __init__(self, cache: Cache, indices: np.ndarray):
        self.c, self.idx = cache, np.asarray(indices)

    def __len__(self) -> int:
        return len(self.idx)

    def __getitem__(self, i):
        j = self.idx[i]
        return (
            torch.from_numpy(self.c.pos[j].astype(np.float32)),
            int(self.c.lengths[j]),
            int(self.c.labels[j]),
        )
