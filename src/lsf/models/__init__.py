"""Fabrique de modeles a partir d'une configuration serialisable."""

from __future__ import annotations

from torch import nn

from lsf.models.baseline import PooledMLP
from lsf.models.transformer import SignTransformer


def build_model(cfg: dict) -> nn.Module:
    kw = dict(cfg.get("model_kwargs", {}))
    if cfg["model"] == "baseline":
        return PooledMLP(cfg["n_points"], cfg["n_classes"], **kw)
    if cfg["model"] == "transformer":
        return SignTransformer(cfg["n_points"], cfg["n_classes"], max_len=cfg["max_len"], **kw)
    raise ValueError(f"modele inconnu : {cfg['model']}")
