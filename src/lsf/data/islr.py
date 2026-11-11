"""Chargement du jeu Google ISLR (Kaggle « asl-signs ») et construction du cache.

Usage (sur Kaggle) :
    python -m lsf.data.islr --root /kaggle/input/asl-signs --out /kaggle/working/islr_cache.npz
"""

from __future__ import annotations

import argparse
import json
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from lsf.features.landmarks import N_POINTS, OFFSETS, ROWS_PER_FRAME, preprocess_sequence


def load_sequence(path: str | Path) -> np.ndarray:
    """Lit un fichier parquet ISLR -> tableau (T, 543, 3) avec NaN pour les points absents."""
    import pandas as pd

    data = pd.read_parquet(path, columns=["x", "y", "z"])
    n = len(data) // ROWS_PER_FRAME
    if n * ROWS_PER_FRAME != len(data):
        raise ValueError(f"{path}: {len(data)} lignes, pas un multiple de {ROWS_PER_FRAME}")
    return data.to_numpy(dtype=np.float32).reshape(n, ROWS_PER_FRAME, 3)


def verify_layout(path: str | Path) -> dict[str, int]:
    """Verifie que l'ordre des types dans un fichier correspond a ``OFFSETS``."""
    import pandas as pd

    df = pd.read_parquet(path, columns=["frame", "type", "landmark_index"])
    first = df.iloc[:ROWS_PER_FRAME]
    if first["frame"].nunique() != 1:
        raise AssertionError("les 543 premieres lignes ne forment pas une seule frame")
    found = {t: int(first.index[first["type"] == t][0] - first.index[0]) for t in first["type"].unique()}
    if found != OFFSETS:
        raise AssertionError(f"disposition inattendue : {found} au lieu de {OFFSETS}")
    return found


def load_sign_map(root: Path, signs: list[str]) -> dict[str, int]:
    p = root / "sign_to_prediction_index_map.json"
    if p.exists():
        return json.loads(p.read_text())
    return {s: i for i, s in enumerate(sorted(set(signs)))}


def _work(args):
    path, max_len = args
    raw = load_sequence(path)
    pos, length = preprocess_sequence(raw, max_len=max_len)
    return pos.astype(np.float16), length, len(raw)


def build_cache(root: str | Path, out: str | Path, max_len: int = 64, workers: int = 4, limit: int | None = None) -> None:
    """Pretraite toutes les sequences et ecrit un .npz (positions en float16).

    Taille indicative : 94 481 x 64 x 131 x 2 x 2 octets, environ 3,2 Go.
    """
    import pandas as pd

    root = Path(root)
    df = pd.read_csv(root / "train.csv")
    if limit:
        df = df.iloc[:limit]
    sign_map = load_sign_map(root, df["sign"].tolist())
    jobs = [(root / p, max_len) for p in df["path"]]

    pos = np.zeros((len(df), max_len, N_POINTS, 2), dtype=np.float16)
    lengths = np.zeros(len(df), dtype=np.int16)
    raw_lengths = np.zeros(len(df), dtype=np.int32)
    t0 = time.time()
    with Pool(workers) as pool:
        for i, (p, l, rl) in enumerate(pool.imap(_work, jobs, chunksize=64)):
            pos[i], lengths[i], raw_lengths[i] = p, l, rl
            if (i + 1) % 5000 == 0:
                print(f"{i + 1}/{len(df)} sequences, {time.time() - t0:.0f} s", flush=True)

    np.savez(
        out,
        pos=pos,
        lengths=lengths,
        raw_lengths=raw_lengths,
        labels=df["sign"].map(sign_map).to_numpy(np.int64),
        participant=df["participant_id"].to_numpy(np.int64),
        sequence_id=df["sequence_id"].to_numpy(np.int64),
        signs=np.array(sorted(sign_map, key=sign_map.get)),
        max_len=np.array(max_len),
    )
    print(f"cache ecrit : {out} ({len(df)} sequences, {time.time() - t0:.0f} s)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-len", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    build_cache(a.root, a.out, a.max_len, a.workers, a.limit)


if __name__ == "__main__":
    main()
