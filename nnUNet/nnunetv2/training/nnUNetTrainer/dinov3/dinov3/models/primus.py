import torch
from einops import rearrange
from dynamic_network_architectures.architectures.abstract_arch import (
    AbstractDynamicNetworkArchitectures,
)
from typing import Tuple
import torch
from torch import nn

from dynamic_network_architectures.architectures.abstract_arch import (
    AbstractDynamicNetworkArchitectures,
    test_submodules_loadable,
)
from dynamic_network_architectures.building_blocks.patch_encode_decode import LayerNormNd
from dynamic_network_architectures.initialization.weight_init import InitWeights_He
from einops import rearrange
import numpy as np

import math

class PatchDecode(nn.Module):
    """
    Loosely inspired by SAM decoder
    https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/mask_decoder.py#L53
    """

    def __init__(
        self,
        patch_size: int,
        embed_dim: int,
        out_channels: int,
        norm=LayerNormNd,
        activation=nn.GELU,
    ):
        """
        patch size must be 2^x, so 2, 4, 8, 16, 32, etc. Otherwise we die
        """
        super().__init__()
        assert patch_size > 0
        n = int(math.log2(patch_size))

        assert 2 ** n == patch_size and n >= 1

        ch = [embed_dim]
        for _ in range(n):
            ch.append(ch[-1]//2)
        ch.append(out_channels)

        stages = []
        for i in range(n):
            stages.append(
                nn.Sequential(
                    nn.ConvTranspose2d(ch[i], ch[i + 1], kernel_size=2, stride=2),
                    norm(ch[i + 1]),
                    activation(),
                )
            )
        stages.append(nn.Conv2d(ch[-2], ch[-1], kernel_size=1))
        self.decode = nn.Sequential(*stages)

    def forward(self, x):
        """
        Expects input of shape (B, embed_dim, px, py)! This will require you to reshape the output of your transformer!
        """
        return self.decode(x)

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.deep_supervision = False

class Primus(AbstractDynamicNetworkArchitectures):

    def __init__(
        self,
        embed_dim: int,
        patch_embed_size: int,
        num_classes: int,
        decoder_norm=LayerNormNd,
        decoder_act=nn.GELU,
        dino_encoder = None,
    ):
        """
        Architecture as proposed in the Primus paper (https://arxiv.org/pdf/2503.01835)
        `Primus: Enforcing Attention Usage for 3D Medical Image Segmentation`

        consists of simple patch_embedding, a EVA ViT encoder with a few adatptations and a simple patch decoder.
        """
        super().__init__()

        self.up_projection = PatchDecode(
            patch_embed_size, embed_dim, num_classes, norm=decoder_norm, activation=decoder_act
        )

        # we need to compute the ref_feat_shape for eva
        self.dino_encoder = dino_encoder
        self.decoder = Decoder()
        self.up_projection.apply(InitWeights_He(1e-2))

    def forward(self, x, ret_mask=False):
        assert x.shape[1] == 1
        x = x.repeat(1,3,1,1)
        x = self.dino_encoder.get_intermediate_layers(x,  n=1, reshape = True)[0]
        dec_out = self.up_projection(x)
        return dec_out

    def compute_conv_feature_map_size(self, input_size):
        raise NotImplementedError("yuck")


class Primus_Multiscale(AbstractDynamicNetworkArchitectures):


    def __init__(
        self,
        embed_dim: int,
        patch_embed_size: int,
        num_classes: int,
        decoder_norm=LayerNormNd,
        decoder_act=nn.GELU,
        dino_encoder = None,
        interaction_indices =[1,2,3,4]
    ):
        """
        We follow a similar design as ViT-adapter, using intermediate layers and concat along channel dimension.
        """
        super().__init__()

        self.up_projection = PatchDecode(
            patch_embed_size, embed_dim * len(interaction_indices), num_classes, norm=decoder_norm, activation=decoder_act
        )

        # we need to compute the ref_feat_shape for eva
        self.dino_encoder = dino_encoder
        self.decoder = Decoder()
        self.up_projection.apply(InitWeights_He(1e-2))
        self.interaction_indices=interaction_indices

    def forward(self, x, ret_mask=False):
        assert x.shape[1] == 1
        x = x.repeat(1,3,1,1)
        hier = self.dino_encoder.get_intermediate_layers(x,  n=self.interaction_indices, reshape = True)
        hier = torch.cat(hier, dim=1)
        dec_out = self.up_projection(hier)
        return dec_out

    def compute_conv_feature_map_size(self, input_size):
        raise NotImplementedError("yuck")


class PatchDecode3D(nn.Module):
    """
    3D counterpart of PatchDecode.

    Upsamples from (B, embed_dim, D', H', W') to (B, out_channels, D, H, W) where
    D = D' * d_patch and H = H' * patch_size, W = W' * patch_size.

    Both patch_size and d_patch must be powers of 2, with d_patch <= patch_size.

    Upsampling schedule across n = log2(patch_size) total stages:
      - First log2(d_patch) stages use stride (2, 2, 2) — upsample depth + spatial
      - Remaining stages use stride (1, 2, 2) — upsample spatial only
    """

    def __init__(
        self,
        patch_size: int,
        d_patch: int,
        embed_dim: int,
        out_channels: int,
        norm=LayerNormNd,
        activation=nn.GELU,
    ):
        super().__init__()
        assert patch_size > 0
        n = int(math.log2(patch_size))
        assert 2 ** n == patch_size and n >= 1, "patch_size must be a power of 2"
        n_depth = int(math.log2(d_patch)) if d_patch > 1 else 0
        assert 2 ** n_depth == d_patch, "d_patch must be a power of 2"
        assert n_depth <= n, "d_patch must be <= patch_size"

        ch = [embed_dim]
        for _ in range(n):
            ch.append(ch[-1] // 2)
        ch.append(out_channels)

        stages = []
        for i in range(n):
            stride = (2, 2, 2) if i < n_depth else (1, 2, 2)
            stages.append(
                nn.Sequential(
                    nn.ConvTranspose3d(ch[i], ch[i + 1], kernel_size=stride, stride=stride),
                    norm(ch[i + 1]),
                    activation(),
                )
            )
        stages.append(nn.Conv3d(ch[-2], ch[-1], kernel_size=1))
        self.decode = nn.Sequential(*stages)

    def forward(self, x):
        """Input: (B, embed_dim, D', H', W')"""
        return self.decode(x)


class Primus_Multiscale3D(AbstractDynamicNetworkArchitectures):
    """
    3D version of Primus_Multiscale. Accepts (B, 1, D, H, W) and returns
    (B, num_classes, D, H, W). Uses DinoVisionTransformer3D as encoder
    and PatchDecode3D as decoder.
    """

    def __init__(
        self,
        embed_dim: int,
        patch_embed_size: int,
        d_patch: int,
        num_classes: int,
        decoder_norm=LayerNormNd,
        decoder_act=nn.GELU,
        dino_encoder=None,
        interaction_indices=None,
    ):
        super().__init__()
        if interaction_indices is None:
            interaction_indices = [2, 5, 8, 11]

        self.up_projection = PatchDecode3D(
            patch_size=patch_embed_size,
            d_patch=d_patch,
            embed_dim=embed_dim * len(interaction_indices),
            out_channels=num_classes,
            norm=decoder_norm,
            activation=decoder_act,
        )
        self.dino_encoder = dino_encoder
        self.decoder = Decoder()
        self.up_projection.apply(InitWeights_He(1e-2))
        self.interaction_indices = interaction_indices

    def forward(self, x, ret_mask=False):
        # x: (B, 1, D, H, W)
        assert x.shape[1] == 1
        hier = self.dino_encoder.get_intermediate_layers(
            x, n=self.interaction_indices, reshape=True
        )  # list of (B, embed_dim, D', H', W')
        hier = torch.cat(hier, dim=1)  # (B, embed_dim*4, D', H', W')
        return self.up_projection(hier)  # (B, num_classes, D, H, W)

    def compute_conv_feature_map_size(self, input_size):
        raise NotImplementedError("yuck")
