"""
test_chd_conditioning.py — local unit tests for the diagnosis-conditioning
primitives (torch-only; no dynamic_network_architectures needed).

Run:  python3 tools/test_chd_conditioning.py
"""

import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "nnUNet"))

from nnunetv2.training.nnUNetTrainer.dinov3.dinov3.models.chd_conditioning import (
    DiagnosisConditioner,
    FiLM3D,
)


def test_film_identity_at_init():
    torch.manual_seed(0)
    film = FiLM3D(cond_dim=256, num_channels=64)
    x = torch.randn(2, 64, 4, 5, 5)
    z = torch.randn(2, 256)
    y = film(x, z)
    assert torch.allclose(y, x, atol=1e-6), "FiLM must be exact identity at init"
    print("[ok] FiLM3D is identity at init (max|y-x|=%.2e)" % (y - x).abs().max())


def test_film_nonidentity_after_perturb():
    film = FiLM3D(cond_dim=8, num_channels=4)
    torch.nn.init.normal_(film.to_gamma.weight, std=0.1)
    x = torch.randn(1, 4, 2, 2, 2)
    z = torch.randn(1, 8)
    y = film(x, z)
    assert not torch.allclose(y, x), "FiLM should modulate once weights are nonzero"
    print("[ok] FiLM3D modulates after perturbation")


def test_conditioner_shapes_and_multilabel():
    torch.manual_seed(0)
    K, B, C = 5, 3, 256
    cond = DiagnosisConditioner(num_diagnoses=K, cond_dim=C)
    # multi-hot: case 0 has diagnoses {0,2}, case 1 has {1}, case 2 has none (UNK)
    d = torch.tensor([[1, 0, 1, 0, 0],
                      [0, 1, 0, 0, 0],
                      [0, 0, 0, 0, 0]], dtype=torch.float32)
    z = cond(d)
    assert z.shape == (B, C), f"expected ({B},{C}), got {tuple(z.shape)}"
    # zero (UNK) row should still produce a finite vector (bias only)
    assert torch.isfinite(z).all()
    # different multi-hot inputs -> generally different embeddings
    assert not torch.allclose(z[0], z[1])
    print("[ok] DiagnosisConditioner multi-hot -> [B, cond_dim], UNK row finite")


def test_conditioning_noop_when_film_zero():
    """End-to-end-ish: with zero-init FiLM, any z leaves features unchanged, so a
    'real diagnosis' and a 'zero diagnosis' give identical modulated features."""
    torch.manual_seed(1)
    cond = DiagnosisConditioner(num_diagnoses=5, cond_dim=256)
    film = FiLM3D(cond_dim=256, num_channels=32)  # identity at init
    x = torch.randn(2, 32, 3, 4, 4)
    z_real = cond(torch.tensor([[1, 0, 1, 0, 0], [0, 1, 0, 1, 0]], dtype=torch.float32))
    z_zero = cond(torch.zeros(2, 5))
    y_real = film(x, z_real)
    y_zero = film(x, z_zero)
    assert torch.allclose(y_real, y_zero, atol=1e-6)
    assert torch.allclose(y_real, x, atol=1e-6)
    print("[ok] conditioned == unconditioned == input at init (no-op guarantee)")


if __name__ == "__main__":
    test_film_identity_at_init()
    test_film_nonidentity_after_perturb()
    test_conditioner_shapes_and_multilabel()
    test_conditioning_noop_when_film_zero()
    print("\nALL CHD CONDITIONING TESTS PASSED")
