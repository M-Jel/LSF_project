"""Disposition des points ISLR, sous-ensemble retenu et pretraitement d'une sequence.

Disposition d'une frame ISLR (543 lignes) : face (468), left_hand (21), pose (33),
right_hand (21). Cet ordre est celui du starter de la competition ; il est controle
par ``lsf.data.islr.verify_layout`` dans le notebook d'exploration.

Les indices LIPS et FACE_OVAL sont ceux de la topologie FaceMesh de MediaPipe
(Apache 2.0, constantes FACEMESH_LIPS / FACEMESH_FACE_OVAL).
"""

from __future__ import annotations

import numpy as np

ROWS_PER_FRAME = 543
OFFSETS = {"face": 0, "left_hand": 468, "pose": 489, "right_hand": 522}

LIPS = [
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291,
    146, 91, 181, 84, 17, 314, 405, 321, 375,
    78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
    95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
]
FACE_OVAL = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378,
    400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21,
    54, 103, 67, 109,
]
# Haut du corps : nez, epaules, coudes, poignets, points de main de la pose.
POSE_UPPER = [0, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
POSE_MIRROR_PAIRS = [(11, 12), (13, 14), (15, 16), (17, 18), (19, 20), (21, 22)]


def _build_subset() -> tuple[np.ndarray, dict[str, slice]]:
    groups = [
        ("left_hand", [OFFSETS["left_hand"] + i for i in range(21)]),
        ("right_hand", [OFFSETS["right_hand"] + i for i in range(21)]),
        ("pose", [OFFSETS["pose"] + i for i in POSE_UPPER]),
        ("lips", [OFFSETS["face"] + i for i in LIPS]),
        ("face_oval", [OFFSETS["face"] + i for i in FACE_OVAL]),
    ]
    idx, slices, start = [], {}, 0
    for name, g in groups:
        idx.extend(g)
        slices[name] = slice(start, start + len(g))
        start += len(g)
    return np.asarray(idx, dtype=np.int64), slices


SUBSET_INDEX, GROUPS = _build_subset()
N_POINTS = len(SUBSET_INDEX)  # 131


def _mirror_permutation() -> np.ndarray:
    """Permutation du sous-ensemble pour un miroir gauche/droite.

    Mains et paires de la pose sont echangees. Les points du visage ne sont PAS
    permutes (seul x est inverse) : approximation assumee, faute de table de
    symetrie FaceMesh verifiee.
    """
    perm = np.arange(N_POINTS)
    lh, rh = GROUPS["left_hand"], GROUPS["right_hand"]
    perm[lh], perm[rh] = np.arange(rh.start, rh.stop), np.arange(lh.start, lh.stop)
    pose_pos = {p: GROUPS["pose"].start + k for k, p in enumerate(POSE_UPPER)}
    for a, b in POSE_MIRROR_PAIRS:
        perm[pose_pos[a]], perm[pose_pos[b]] = pose_pos[b], pose_pos[a]
    return perm


MIRROR_PERM = _mirror_permutation()


def preprocess_sequence(raw: np.ndarray, max_len: int = 64) -> tuple[np.ndarray, int]:
    """Transforme une sequence brute (T, 543, 3) en positions normalisees.

    Etapes : sous-ensemble de points (x, y) ; suppression des frames sans aucune
    main ; miroir pour que la main la plus presente soit toujours en position
    « right_hand » ; centrage sur le milieu des epaules et mise a l'echelle par la
    largeur d'epaules (sur toute la sequence, pour garder le deplacement global) ;
    sous-echantillonnage uniforme a ``max_len`` ; NaN -> 0 ; padding par zeros.

    Retourne ``(pos (max_len, N_POINTS, 2) float32, longueur utile)``.
    """
    if raw.ndim != 3 or raw.shape[1:] != (ROWS_PER_FRAME, 3):
        raise ValueError(f"forme attendue (T, 543, 3), recu {raw.shape}")
    x = raw[:, SUBSET_INDEX, :2].astype(np.float32)

    lh_ok = ~np.isnan(x[:, GROUPS["left_hand"], 0]).all(axis=1)
    rh_ok = ~np.isnan(x[:, GROUPS["right_hand"], 0]).all(axis=1)
    keep = lh_ok | rh_ok
    if keep.any():
        x, lh_ok, rh_ok = x[keep], lh_ok[keep], rh_ok[keep]

    if lh_ok.sum() > rh_ok.sum():
        x = x[:, MIRROR_PERM]
        x[..., 0] = 1.0 - x[..., 0]

    pose = GROUPS["pose"].start
    shoulders = x[:, [pose + 1, pose + 2]]  # indices 11 et 12 de la pose
    if not np.isnan(shoulders).all():
        center = np.nanmean(shoulders.reshape(-1, 2), axis=0)
        width = np.nanmean(np.linalg.norm(shoulders[:, 0] - shoulders[:, 1], axis=-1))
    else:
        center, width = np.nanmean(x.reshape(-1, 2), axis=0), np.nan
    if not np.isfinite(width) or width < 1e-3:
        width = np.nanstd(x) * 2 if np.isfinite(np.nanstd(x)) else 1.0
        width = max(float(width), 1e-3)
    if not np.isfinite(center).all():
        center = np.zeros(2, dtype=np.float32)
    x = (x - center) / width

    t = len(x)
    if t > max_len:
        x = x[np.linspace(0, t - 1, max_len).round().astype(int)]
        t = max_len
    x = np.nan_to_num(x, nan=0.0)
    out = np.zeros((max_len, N_POINTS, 2), dtype=np.float32)
    out[:t] = x
    return out, t
