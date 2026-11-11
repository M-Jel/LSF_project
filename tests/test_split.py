import numpy as np

from lsf.data.split import make_signer_split, split_indices


def test_split_is_by_signer_and_covers_all_signs():
    rng = np.random.default_rng(0)
    participants = np.repeat(np.arange(21), rng.integers(3000, 6000, 21))
    labels = rng.integers(0, 250, len(participants))
    split = make_signer_split(participants, labels, n_trials=2000, seed=0)
    sets = [set(split[k]) for k in ("train", "val", "test")]
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
    assert set.union(*sets) == set(range(21))
    idx = split_indices(participants, split)
    for k in ("val", "test"):
        assert abs(len(idx[k]) / len(participants) - 0.15) < 0.04
        assert len(np.unique(labels[idx[k]])) == 250


def test_split_is_deterministic():
    p = np.repeat(np.arange(10), 100)
    y = np.tile(np.arange(5), 200)
    assert make_signer_split(p, y, n_trials=500, seed=1) == make_signer_split(p, y, n_trials=500, seed=1)
