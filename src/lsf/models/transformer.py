"""Transformer leger avec token CLS.

Le padding se fait par zeros et il est reellement masque (``src_key_padding_mask``),
ce que verifie ``tests/test_models.py`` : ajouter du padding ne change pas la sortie.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from lsf.features.torch_features import FrameFeatures, padding_mask


def sinusoidal_encoding(max_len: int, d: int) -> torch.Tensor:
    pos = torch.arange(max_len)[:, None].float()
    div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
    pe = torch.zeros(max_len, d)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


class SignTransformer(nn.Module):
    def __init__(
        self,
        n_points: int,
        n_classes: int,
        max_len: int = 64,
        d_model: int = 192,
        depth: int = 4,
        heads: int = 4,
        ff_mult: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.features = FrameFeatures(n_points)
        self.embed = nn.Sequential(
            nn.LayerNorm(self.features.out_dim),
            nn.Linear(self.features.out_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls, std=0.02)
        self.register_buffer("pe", sinusoidal_encoding(max_len + 1, d_model), persistent=False)
        layer = nn.TransformerEncoderLayer(
            d_model, heads, d_model * ff_mult, dropout, activation="gelu", batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, pos: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        b, l = pos.shape[:2]
        x = self.embed(self.features(pos, lengths))
        x = torch.cat([self.cls.expand(b, -1, -1), x], 1) + self.pe[: l + 1]
        mask = torch.cat([torch.zeros(b, 1, dtype=torch.bool, device=pos.device), padding_mask(lengths, l)], 1)
        x = self.encoder(x, src_key_padding_mask=mask)
        return self.head(self.drop(self.norm(x[:, 0])))
