"""Separation train / val / test PAR SIGNEUR.

Aucun signeur n'apparait dans deux sous-ensembles. Parmi de nombreux tirages
aleatoires (seed fixee), on garde celui dont les proportions d'exemples sont les
plus proches des cibles, en exigeant que chaque signe soit present partout.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def make_signer_split(
    participants: np.ndarray,
    labels: np.ndarray,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    n_trials: int = 20000,
    seed: int = 0,
) -> dict[str, list[int]]:
    ids, counts = np.unique(participants, return_counts=True)
    total = counts.sum()
    n_classes = len(np.unique(labels))
    signs_of = {int(p): set(np.unique(labels[participants == p]).tolist()) for p in ids}
    rng = np.random.default_rng(seed)

    best, best_score = None, np.inf
    for _ in range(n_trials):
        order = rng.permutation(len(ids))
        groups, acc, k = {"test": [], "val": []}, {"test": 0, "val": 0}, 0
        for name, frac in (("test", test_frac), ("val", val_frac)):
            while k < len(order) - 1 and acc[name] < frac * total:
                groups[name].append(int(ids[order[k]]))
                acc[name] += int(counts[order[k]])
                k += 1
        train = [int(ids[i]) for i in order[k:]]
        if not train or not groups["val"] or not groups["test"]:
            continue
        score = abs(acc["test"] / total - test_frac) + abs(acc["val"] / total - val_frac)
        if score >= best_score:
            continue
        if any(len(set().union(*(signs_of[p] for p in g))) < n_classes for g in (train, groups["val"], groups["test"])):
            continue
        best, best_score = {"train": sorted(train), "val": sorted(groups["val"]), "test": sorted(groups["test"])}, score
    if best is None:
        raise RuntimeError("aucune separation valide trouvee")
    return best


def split_indices(participants: np.ndarray, split: dict[str, list[int]]) -> dict[str, np.ndarray]:
    return {k: np.flatnonzero(np.isin(participants, v)) for k, v in split.items() if k in ("train", "val", "test")}


def describe(participants: np.ndarray, split: dict[str, list[int]]) -> str:
    idx = split_indices(participants, split)
    total = len(participants)
    return "\n".join(f"{k:5s}: {len(split[k]):2d} signeurs, {len(v):6d} exemples ({len(v) / total:.1%})" for k, v in idx.items())


def save_split(split: dict, path: str | Path, **meta) -> None:
    Path(path).write_text(json.dumps({**split, "meta": meta}, indent=2))


def load_split(path: str | Path) -> dict[str, list[int]]:
    return json.loads(Path(path).read_text())
