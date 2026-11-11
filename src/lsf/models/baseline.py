"""Baseline : statistiques temporelles (moyenne, ecart-type, max) + MLP.

Aucune modelisation de l'ordre temporel : elle sert de point de comparaison
pour mesurer ce qu'apporte le Transformer.
"""

from __future__ import annotations

import torch
from torch import nn

from lsf.features.torch_features import FrameFeatures, padding_mask


class PooledMLP(nn.Module):
    def __init__(self, n_points: int, n_classes: int, hidden: int = 512, dropout: float = 0.3):
        super().__init__()
        self.features = FrameFeatures(n_points)
        d = self.features.out_dim * 3
        self.head = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, pos: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        x = self.features(pos, lengths)  # (B, L, F)
        keep = (~padding_mask(lengths, x.shape[1])).float()[..., None]
        n = keep.sum(1).clamp(min=1)
        mean = (x * keep).sum(1) / n
        std = (((x - mean[:, None]) ** 2 * keep).sum(1) / n).sqrt()
        mx = x.masked_fill(keep == 0, float("-inf")).amax(1)
        return self.head(torch.cat([mean, std, mx], -1))
