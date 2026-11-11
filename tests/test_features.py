import numpy as np
import torch

from lsf.features.landmarks import GROUPS, MIRROR_PERM, N_POINTS, preprocess_sequence
from lsf.features.torch_features import FrameFeatures, random_affine
from tests.conftest import synthetic_raw


def test_subset_size_and_mirror_is_involution():
    assert N_POINTS == 131
    assert (MIRROR_PERM[MIRROR_PERM] == np.arange(N_POINTS)).all()


def test_shapes_padding_and_no_nan():
    pos, t = preprocess_sequence(synthetic_raw(t=20), max_len=32)
    assert pos.shape == (32, N_POINTS, 2) and t == 20
    assert not np.isnan(pos).any()
    assert (pos[t:] == 0).all()


def test_long_sequence_is_subsampled():
    pos, t = preprocess_sequence(synthetic_raw(t=200), max_len=64)
    assert t == 64 and pos.shape[0] == 64


def test_frames_without_hands_are_dropped():
    raw = synthetic_raw(t=30)
    raw[:10, 522:543] = np.nan  # 10 frames sans aucune main
    _, t = preprocess_sequence(raw, max_len=64)
    assert t == 20


def test_left_handed_sequence_is_mirrored_to_right_slot():
    pos, _ = preprocess_sequence(synthetic_raw(left=True, right=False), max_len=32)
    assert (pos[:, GROUPS["left_hand"]] == 0).all()
    assert (pos[:20, GROUPS["right_hand"]] != 0).any()


def test_no_hands_at_all_does_not_crash():
    pos, t = preprocess_sequence(synthetic_raw(left=False, right=False, face=False), max_len=32)
    assert t == 30 and np.isfinite(pos).all()


def test_velocity_zero_on_padding_and_missing_points():
    pos = torch.randn(2, 10, N_POINTS, 2)
    pos[0, 6:] = 0
    pos[1, :, :5] = 0
    f = FrameFeatures(N_POINTS)(pos, torch.tensor([6, 10])).reshape(2, 10, N_POINTS, 4)
    assert (f[0, 6:] == 0).all()
    assert (f[1, :, :5] == 0).all()


def test_affine_keeps_missing_points_at_zero():
    pos = torch.randn(4, 8, N_POINTS, 2)
    pos[:, :, :3] = 0
    out = random_affine(pos)
    assert (out[:, :, :3] == 0).all() and not torch.equal(out, pos)
