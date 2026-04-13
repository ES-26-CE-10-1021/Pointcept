"""
3DETR Modular Components

Registered MODULES wrapping the encoder/decoder building blocks from
third_party/3detr so they can be instantiated from Pointcept configs.
This allows swapping encoders (e.g. PointNet++ → PointTransformerV3)
via config without touching model code.
"""

import math
import torch
import torch.nn as nn

from pointcept.models.builder import MODULES
from .dataset_config import _setup_3detr_path

_setup_3detr_path()

from third_party.pointnet2.pointnet2_modules import PointnetSAModuleVotes
from third_party.pointnet2.pointnet2_utils import furthest_point_sample
from models.transformer import (
    TransformerEncoder,
    TransformerEncoderLayer,
    MaskedTransformerEncoder,
    TransformerDecoder,
    TransformerDecoderLayer,
)


@MODULES.register_module()
class IdentityEncoder3DETR(nn.Module):
    """
    Passthrough encoder — returns inputs unchanged.

    Use when the pre_encoder (e.g. a PTv3-based component) already produces
    encoder-quality features and no additional transformer layers are needed
    before the 3DETR decoder.
    """

    def forward(self, features, xyz, padding_mask=None):
        """
        Args:
            features: (npoint, B, C) point features
            xyz:      (B, npoint, 3) coordinates
            padding_mask: ignored (passthrough encoder)

        Returns:
            xyz: (B, npoint, 3) unchanged
            features: (npoint, B, C) unchanged
            inds: None
        """
        return xyz, features, None


@MODULES.register_module()
class PointnetSAPreEncoder(nn.Module):
    """
    PointNet++ Set Abstraction pre-encoder used in 3DETR.

    Takes raw XYZ (+ optional features), sub-samples to `npoint` points
    using ball queries, and projects to `enc_dim` dimensions.

    Args:
        npoint (int): number of output points after FPS sampling
        radius (float): ball query radius
        nsample (int): max number of neighbors in ball query
        mlp_dims (list[int]): MLP channel dims where the first element is
            the *extra* feature channels beyond XYZ. PointnetSAModuleVotes
            adds 3 (XYZ) automatically when use_xyz=True (default). Use
            [0, 64, 128, 256] for XYZ-only or [3, 64, 128, 256] for RGB.
        normalize_xyz (bool): normalize XYZ coordinates in ball query
    """

    def __init__(
        self,
        npoint=2048,
        radius=0.2,
        nsample=64,
        mlp_dims=None,
        normalize_xyz=True,
    ):
        super().__init__()
        if mlp_dims is None:
            mlp_dims = [3, 64, 128, 256]
        self.sa_module = PointnetSAModuleVotes(
            radius=radius,
            nsample=nsample,
            npoint=npoint,
            mlp=mlp_dims,
            normalize_xyz=normalize_xyz,
        )

    def forward(self, xyz, features=None):
        """
        Args:
            xyz: (B, N, 3) point coordinates
            features: (B, C, N) point features or None

        Returns:
            xyz: (B, npoint, 3) sub-sampled coordinates
            features: (B, enc_dim, npoint) projected features
            inds: (B, npoint) FPS indices
        """
        return self.sa_module(xyz, features)


@MODULES.register_module()
class VanillaTransformerEncoder3DETR(nn.Module):
    """
    Standard (non-masking) Transformer encoder used in 3DETR.

    Input features are in (npoint, B, C) format (PyTorch nn.MultiHeadAttention
    convention). Returns same-size output — no downsampling.

    Args:
        encoder_dim (int): feature dimension
        nhead (int): number of attention heads
        nlayers (int): number of encoder layers
        ffn_dim (int): feedforward hidden dim
        dropout (float): dropout probability
        activation (str): activation function name
    """

    def __init__(
        self,
        encoder_dim=256,
        nhead=4,
        nlayers=3,
        ffn_dim=128,
        dropout=0.1,
        activation="relu",
    ):
        super().__init__()
        encoder_layer = TransformerEncoderLayer(
            d_model=encoder_dim,
            nhead=nhead,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation=activation,
        )
        self.encoder = TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=nlayers
        )

    def forward(self, features, xyz, padding_mask=None):
        """
        Args:
            features: (npoint, B, C) point features
            xyz: (B, npoint, 3) coordinates
            padding_mask: (B, npoint) bool or None. True = padded position.

        Returns:
            xyz: (B, npoint, 3)
            features: (npoint, B, C) encoded features
            inds: None (no downsampling)
        """
        enc_xyz, enc_features, enc_inds = self.encoder(
            features, xyz=xyz, src_key_padding_mask=padding_mask
        )
        return enc_xyz, enc_features, enc_inds


@MODULES.register_module()
class MaskedTransformerEncoder3DETR(nn.Module):
    """
    Masked Transformer encoder used in 3DETR.

    Applies local masking based on spatial radius at 3 stages.
    Also performs interim downsampling, halving the point count.

    Args:
        encoder_dim (int): feature dimension
        nhead (int): number of attention heads
        ffn_dim (int): feedforward hidden dim
        dropout (float): dropout probability
        activation (str): activation function name
        preenc_npoints (int): input point count (used to size interim downsample)
        interim_radius (float): ball query radius for interim downsampling
        interim_nsample (int): number of neighbors for interim downsampling
        masking_radius (list[float]): masking radii (squared) for 3 stages
    """

    def __init__(
        self,
        encoder_dim=256,
        nhead=4,
        ffn_dim=128,
        dropout=0.1,
        activation="relu",
        preenc_npoints=2048,
        interim_radius=0.4,
        interim_nsample=32,
        masking_radius=None,
    ):
        super().__init__()
        if masking_radius is None:
            masking_radius = [
                math.pow(x, 2) for x in [0.4, 0.8, 1.2]
            ]

        encoder_layer = TransformerEncoderLayer(
            d_model=encoder_dim,
            nhead=nhead,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation=activation,
        )
        interim_downsampling = PointnetSAModuleVotes(
            radius=interim_radius,
            nsample=interim_nsample,
            npoint=preenc_npoints // 2,
            mlp=[encoder_dim, 256, 256, encoder_dim],
            normalize_xyz=True,
        )
        self.encoder = MaskedTransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=3,
            interim_downsampling=interim_downsampling,
            masking_radius=masking_radius,
        )

    def forward(self, features, xyz, padding_mask=None):
        """
        Args:
            features: (npoint, B, C) point features
            xyz: (B, npoint, 3) coordinates
            padding_mask: (B, npoint) bool or None. True = padded position.

        Returns:
            xyz: (B, npoint//2, 3) downsampled coordinates
            features: (npoint//2, B, C) encoded features
            inds: (B, npoint//2) indices of kept points
        """
        enc_xyz, enc_features, enc_inds = self.encoder(
            features, xyz=xyz, src_key_padding_mask=padding_mask
        )
        return enc_xyz, enc_features, enc_inds


@MODULES.register_module()
class TransformerDecoder3DETR(nn.Module):
    """
    Transformer decoder used in 3DETR for box prediction.

    Attends query embeddings (one per predicted box) to encoder features
    via cross-attention. Stacks `nlayers` decoder layers and returns
    intermediate outputs.

    Args:
        decoder_dim (int): feature dimension
        nhead (int): number of attention heads
        nlayers (int): number of decoder layers
        ffn_dim (int): feedforward hidden dim
        dropout (float): dropout probability
    """

    def __init__(
        self,
        decoder_dim=256,
        nhead=4,
        nlayers=8,
        ffn_dim=256,
        dropout=0.1,
    ):
        super().__init__()
        decoder_layer = TransformerDecoderLayer(
            d_model=decoder_dim,
            nhead=nhead,
            dim_feedforward=ffn_dim,
            dropout=dropout,
        )
        self.decoder = TransformerDecoder(
            decoder_layer, num_layers=nlayers, return_intermediate=True
        )

    def forward(
        self, tgt, memory, query_pos=None, pos=None, memory_key_padding_mask=None
    ):
        """
        Args:
            tgt: (nqueries, B, C) target (initialised to zeros)
            memory: (npoints, B, C) encoder output
            query_pos: (nqueries, B, C) positional embeddings for queries
            pos: (npoints, B, C) positional embeddings for memory
            memory_key_padding_mask: (B, npoints) bool, True = ignore

        Returns:
            box_features: (nlayers, nqueries, B, C)
        """
        return self.decoder(
            tgt,
            memory,
            query_pos=query_pos,
            pos=pos,
            memory_key_padding_mask=memory_key_padding_mask,
        )
