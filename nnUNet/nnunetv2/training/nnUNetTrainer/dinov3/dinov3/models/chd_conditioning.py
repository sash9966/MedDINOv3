"""
chd_conditioning.py — Diagnosis-prior conditioning primitives for the 3D
MedDINOv3 segmentation hybrid.

Two small modules:

  DiagnosisConditioner
    Multi-LABEL CHD diagnosis -> conditioning vector z.
    Input is a MULTI-HOT float vector [B, num_diagnoses] (a case may carry
    several co-occurring diagnoses, e.g. VSD + ASD). A Linear over the multi-hot
    vector is the correct encoder for multi-label input (nn.Embedding would only
    handle a single categorical id). An all-zero vector is the natural
    "unconditioned / UNK" input and is used by the zero-conditioning control.

  FiLM3D
    Feature-wise Linear Modulation for 5D tensors: F' = F * (1 + gamma) + beta,
    where (gamma, beta) are produced from z by two ZERO-initialised Linears.
    Zero-init makes the module an EXACT identity at initialisation, so a
    conditioned network starts numerically identical to the unconditioned
    baseline — essential for clean ablations.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DiagnosisConditioner(nn.Module):
    """Multi-hot diagnosis vector -> conditioning embedding z of size cond_dim."""

    def __init__(self, num_diagnoses: int, cond_dim: int = 256):
        super().__init__()
        self.num_diagnoses = num_diagnoses
        self.cond_dim = cond_dim
        self.proj = nn.Sequential(
            nn.Linear(num_diagnoses, cond_dim),
            nn.GELU(),
            nn.Linear(cond_dim, cond_dim),
        )

    def forward(self, diagnosis: torch.Tensor) -> torch.Tensor:
        # diagnosis: [B, num_diagnoses] float multi-hot (0.0/1.0).
        if diagnosis.dtype != self.proj[0].weight.dtype:
            diagnosis = diagnosis.to(self.proj[0].weight.dtype)
        return self.proj(diagnosis)  # [B, cond_dim]


class FiLM3D(nn.Module):
    """Identity-initialised FiLM for (B, C, D, H, W) tensors."""

    def __init__(self, cond_dim: int, num_channels: int):
        super().__init__()
        self.to_gamma = nn.Linear(cond_dim, num_channels)
        self.to_beta = nn.Linear(cond_dim, num_channels)
        # Zero-init => gamma=beta=0 => exact identity at start.
        nn.init.zeros_(self.to_gamma.weight)
        nn.init.zeros_(self.to_gamma.bias)
        nn.init.zeros_(self.to_beta.weight)
        nn.init.zeros_(self.to_beta.bias)

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        # x: [B, C, D, H, W];  z: [B, cond_dim]
        gamma = self.to_gamma(z).view(z.shape[0], -1, 1, 1, 1)
        beta = self.to_beta(z).view(z.shape[0], -1, 1, 1, 1)
        return x * (1.0 + gamma) + beta
