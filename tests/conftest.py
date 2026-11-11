import numpy as np
import pytest

from lsf.features.landmarks import N_POINTS, ROWS_PER_FRAME, preprocess_sequence


def synthetic_raw(t=30, seed=0, left=False, right=True, face=True):
    rng = np.random.default_rng(seed)
    raw = rng.uniform(0.2, 0.8, size=(t, ROWS_PER_FRAME, 3)).astype(np.float32)
    if not left:
        raw[:, 468:489] = np.nan
    if not right:
        raw[:, 522:543] = np.nan
    if not face:
        raw[:, :468] = np.nan
    return raw


@pytest.fixture
def tiny_cache(tmp_path):
    """Cache synthetique : 6 signeurs, 4 signes, signal separable par la classe."""
    rng = np.random.default_rng(0)
    max_len, items = 16, []
    for participant in range(6):
        for label in range(4):
            for _ in range(10):
                raw = synthetic_raw(t=int(rng.integers(8, 24)), seed=int(rng.integers(1e9)))
                raw[:, 522:543, 0] = 0.2 + 0.15 * label + rng.normal(0, 0.01, (raw.shape[0], 21))
                pos, length = preprocess_sequence(raw, max_len)
                items.append((pos, length, label, 1000 + participant))
    pos, lengths, labels, part = map(np.array, zip(*items))
    path = tmp_path / "cache.npz"
    np.savez(path, pos=pos.astype(np.float16), lengths=lengths, labels=labels, participant=part,
             sequence_id=np.arange(len(labels)), signs=np.array(["a", "b", "c", "d"]), max_len=np.array(max_len))
    assert pos.shape[2] == N_POINTS
    return path
