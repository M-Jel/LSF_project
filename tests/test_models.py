import pytest
import torch

from lsf.features.landmarks import N_POINTS
from lsf.models import build_model


@pytest.mark.parametrize("name", ["baseline", "transformer"])
def test_output_shape(name):
    m = build_model({"model": name, "n_points": N_POINTS, "n_classes": 250, "max_len": 64}).eval()
    out = m(torch.randn(3, 64, N_POINTS, 2), torch.tensor([64, 10, 1]))
    assert out.shape == (3, 250) and torch.isfinite(out).all()


@pytest.mark.parametrize("name", ["baseline", "transformer"])
def test_padding_does_not_change_output(name):
    """Le masque est effectif : ajouter des frames de padding ne change rien."""
    torch.manual_seed(0)
    m = build_model({"model": name, "n_points": N_POINTS, "n_classes": 10, "max_len": 64}).eval()
    x = torch.randn(1, 20, N_POINTS, 2)
    padded = torch.zeros(1, 64, N_POINTS, 2)
    padded[:, :20] = x
    padded[:, 20:] = 5.0  # valeurs parasites : elles doivent etre ignorees
    with torch.no_grad():
        a = m(x, torch.tensor([20]))
        b = m(padded, torch.tensor([20]))
    assert torch.allclose(a, b, atol=1e-5)
