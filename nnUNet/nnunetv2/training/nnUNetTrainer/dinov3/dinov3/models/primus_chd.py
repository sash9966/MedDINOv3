"""
primus_chd.py — Diagnosis-conditioned variant of the 3D MedDINOv3 segmentation
hybrid (Primus_Multiscale3D) using identity-initialised FiLM.

Two additions over the baseline network, each independently toggleable so they
can be ablated one at a time (research requirement):

  CHD_FILM_BRIDGE=1  -> FiLM on each of the 4 ViT feature grids (768 ch) before
                        they are concatenated and fed to the decoder.
  CHD_FILM_DECODER=1 -> FiLM after every PatchDecode3D upsampling stage.

Because every FiLM is zero-initialised (exact identity at start), a conditioned
network begins numerically identical to the unconditioned baseline regardless of
which toggles are on — so any divergence during training is attributable to the
diagnosis prior, not to a different initialisation.

Diagnosis delivery
------------------
forward(x, diagnosis=None). If `diagnosis` is None the network falls back to
`self._pending_diagnosis` (set by the trainer/predictor right before the forward
call). This lets us reuse the parent train_step/validation_step/predictor forward
call sites — which call `network(data)` with no extra args — without copying
their long bodies. If neither is available, a zero vector is used (== the
unconditioned / UNK case), which with identity-init FiLM is an exact no-op.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn

from nnunetv2.training.nnUNetTrainer.dinov3.dinov3.models.primus import (
    Primus_Multiscale3D,
    PatchDecode3D,
)
from nnunetv2.training.nnUNetTrainer.dinov3.dinov3.models.chd_conditioning import (
    DiagnosisConditioner,
    FiLM3D,
)


class PatchDecode3D_FiLM(PatchDecode3D):
    """PatchDecode3D with optional per-stage FiLM.

    Identical channel schedule / weights to the parent, but the stages are kept
    in an addressable nn.ModuleList and a FiLM3D is attached after each upsampling
    stage (not after the final 1x1 conv). forward(x, cond=None): when cond is None
    or use_film is False, behaves exactly like the parent decoder.
    """

    def __init__(
        self,
        patch_size: int,
        d_patch: int,
        embed_dim: int,
        out_channels: int,
        cond_dim: int,
        use_film: bool,
        norm=None,
        activation=nn.GELU,
    ):
        # Build the parent so self.decode (nn.Sequential of stages) exists with the
        # exact same conv/norm/act weights, then re-expose the stages as a ModuleList.
        if norm is None:
            from dynamic_network_architectures.building_blocks.patch_encode_decode import LayerNormNd
            norm = LayerNormNd
        super().__init__(
            patch_size=patch_size,
            d_patch=d_patch,
            embed_dim=embed_dim,
            out_channels=out_channels,
            norm=norm,
            activation=activation,
        )
        # self.decode is nn.Sequential(stage_0, ..., stage_{n-1}, final_conv1x1).
        stage_list = list(self.decode.children())
        self.final_conv = stage_list[-1]          # bare Conv3d(k=1)
        self.up_stages = nn.ModuleList(stage_list[:-1])  # the n upsampling stages
        del self.decode  # avoid duplicate registration / accidental use

        self.use_film = use_film
        if use_film:
            # One FiLM per upsampling stage, sized to that stage's output channels.
            self.stage_film = nn.ModuleList(
                [FiLM3D(cond_dim, stage[0].out_channels) for stage in self.up_stages]
            )
        else:
            self.stage_film = None

    def forward(self, x, cond=None):
        for i, stage in enumerate(self.up_stages):
            x = stage(x)
            if self.use_film and cond is not None:
                x = self.stage_film[i](x, cond)
        return self.final_conv(x)


class Primus_Multiscale3D_CHD(Primus_Multiscale3D):
    """Diagnosis-conditioned Primus_Multiscale3D.

    Adds a multi-hot DiagnosisConditioner, optional bridge FiLM on each ViT grid,
    and a FiLM-aware decoder. All conditioning is identity at init.
    """

    def __init__(
        self,
        embed_dim: int,
        patch_embed_size: int,
        d_patch: int,
        num_classes: int,
        num_diagnoses: int,
        cond_dim: int = 256,
        film_bridge: bool = True,
        film_decoder: bool = False,
        decoder_norm=None,
        decoder_act=nn.GELU,
        dino_encoder=None,
        interaction_indices=None,
        use_input_adapter: bool = False,
    ):
        # Parent builds the standard up_projection (PatchDecode3D), dino_encoder,
        # input_adapter, etc. We then swap up_projection for the FiLM-aware decoder.
        super().__init__(
            embed_dim=embed_dim,
            patch_embed_size=patch_embed_size,
            d_patch=d_patch,
            num_classes=num_classes,
            decoder_norm=decoder_norm if decoder_norm is not None else self._default_norm(),
            decoder_act=decoder_act,
            dino_encoder=dino_encoder,
            interaction_indices=interaction_indices,
            use_input_adapter=use_input_adapter,
        )

        self.num_diagnoses = num_diagnoses
        self.cond_dim = cond_dim
        self.film_bridge = film_bridge
        self.film_decoder = film_decoder
        self._pending_diagnosis = None  # set by trainer/predictor before forward

        self.conditioner = DiagnosisConditioner(num_diagnoses, cond_dim=cond_dim)

        n_grids = len(self.interaction_indices)
        if film_bridge:
            self.bridge_film = nn.ModuleList(
                [FiLM3D(cond_dim, embed_dim) for _ in range(n_grids)]
            )
        else:
            self.bridge_film = None

        # Replace up_projection with the FiLM-aware decoder (same channel schedule).
        self.up_projection = PatchDecode3D_FiLM(
            patch_size=patch_embed_size,
            d_patch=d_patch,
            embed_dim=embed_dim * n_grids,
            out_channels=num_classes,
            cond_dim=cond_dim,
            use_film=film_decoder,
            norm=decoder_norm,
            activation=decoder_act,
        )
        from dynamic_network_architectures.initialization.weight_init import InitWeights_He
        self.up_projection.apply(InitWeights_He(1e-2))
        # Re-zero the decoder FiLM after He-init (He would have perturbed them).
        if film_decoder and self.up_projection.stage_film is not None:
            for film in self.up_projection.stage_film:
                self._zero_film(film)

    @staticmethod
    def _default_norm():
        from dynamic_network_architectures.building_blocks.patch_encode_decode import LayerNormNd
        return LayerNormNd

    @staticmethod
    def _zero_film(film: FiLM3D):
        nn.init.zeros_(film.to_gamma.weight); nn.init.zeros_(film.to_gamma.bias)
        nn.init.zeros_(film.to_beta.weight); nn.init.zeros_(film.to_beta.bias)

    def set_diagnosis(self, diagnosis):
        """Stash the multi-hot diagnosis [B, num_diagnoses] for the next forward."""
        self._pending_diagnosis = diagnosis

    def _resolve_cond(self, x, diagnosis):
        if diagnosis is None:
            diagnosis = self._pending_diagnosis
        if diagnosis is None:
            # Unconditioned / UNK: zero multi-hot -> with identity FiLM, exact no-op.
            diagnosis = torch.zeros(
                x.shape[0], self.num_diagnoses, device=x.device, dtype=x.dtype
            )
        diagnosis = diagnosis.to(x.device)
        return self.conditioner(diagnosis)  # [B, cond_dim]

    def forward(self, x, diagnosis=None, ret_mask=False):
        z = self._resolve_cond(x, diagnosis)

        enc_chans = self.dino_encoder.patch_embed.proj.weight.shape[1]
        if x.shape[1] != enc_chans:
            x = x.expand(-1, enc_chans, -1, -1, -1).contiguous()
        if self.input_adapter is not None:
            x = self.input_adapter(x)

        hier = self.dino_encoder.get_intermediate_layers(
            x, n=self.interaction_indices, reshape=True
        )  # tuple of (B, embed_dim, D', H', W')

        if self.film_bridge and self.bridge_film is not None:
            hier = [self.bridge_film[i](g, z) for i, g in enumerate(hier)]

        hier = torch.cat(tuple(hier), dim=1)  # (B, embed_dim*n_grids, D', H', W')
        return self.up_projection(hier, cond=z)  # (B, num_classes, D, H, W)


def film_flags_from_env():
    """Read CHD_FILM_BRIDGE / CHD_FILM_DECODER env toggles (default: bridge on)."""
    bridge = os.environ.get("CHD_FILM_BRIDGE", "1") == "1"
    decoder = os.environ.get("CHD_FILM_DECODER", "0") == "1"
    return bridge, decoder
