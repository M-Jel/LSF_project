"""Features calculees dans le modele (partagees par l'entrainement et la demo)."""

from __future__ import annotations

import math

import torch
from torch import nn


def padding_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    """True sur les frames de padding, forme (B, L)."""
    ar = torch.arange(max_len, device=lengths.device)
    return ar[None, :] >= lengths[:, None]


class FrameFeatures(nn.Module):
    """(B, L, N, 2) -> (B, L, N*4) : positions + vitesses.

    Un point absent vaut exactement (0, 0) ; sa vitesse est mise a zero, comme
    celle des frames de padding.
    """

    def __init__(self, n_points: int):
        super().__init__()
        self.out_dim = n_points * 4

    def forward(self, pos: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        b, l, n, _ = pos.shape
        valid = (pos != 0).any(-1)  # (B, L, N)
        valid = valid & ~padding_mask(lengths, l)[:, :, None]
        vel = torch.zeros_like(pos)
        vel[:, 1:] = pos[:, 1:] - pos[:, :-1]
        both = torch.zeros_like(valid)
        both[:, 1:] = valid[:, 1:] & valid[:, :-1]
        vel = vel * both[..., None]
        feats = torch.cat([pos * valid[..., None], vel], dim=-1)
        return feats.reshape(b, l, n * 4)


def random_affine(pos: torch.Tensor, scale=0.15, rot_deg=15.0, shift=0.1, shear=0.1) -> torch.Tensor:
    """Augmentation spatiale par echantillon ; les points absents restent a 0."""
    b = pos.shape[0]
    dev = pos.device
    s = 1 + (torch.rand(b, device=dev) * 2 - 1) * scale
    a = (torch.rand(b, device=dev) * 2 - 1) * math.radians(rot_deg)
    sh = (torch.rand(b, device=dev) * 2 - 1) * shear
    cos, sin = torch.cos(a), torch.sin(a)
    m = torch.stack([torch.stack([cos, -sin + sh], -1), torch.stack([sin, cos], -1)], -2)
    m = m * s[:, None, None]
    t = (torch.rand(b, 1, 1, 2, device=dev) * 2 - 1) * shift
    out = torch.einsum("blnc,bdc->blnd", pos, m) + t
    valid = (pos != 0).any(-1, keepdim=True)
    return out * valid
