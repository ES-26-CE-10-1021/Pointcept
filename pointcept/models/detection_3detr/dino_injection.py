"""
3DETR + DINO feature injection via a wrapping encoder module.

The injection lives entirely in the ``encoder`` config slot.
``DinoInjectionEncoder`` wraps the real transformer encoder and reads
``dino_feat`` from a side-channel set by ``Model3DETRDetectorWithDino``
immediately before ``run_encoder`` is called.  The side-channel is a
per-forward-pass attribute on the wrapper itself, set and cleared within
a single ``forward()`` call — safe for single-GPU and DDP (each process
has its own model replica).

Config shape
------------
model:
  type: Model3DETRDetectorWithDino
  dino_dim: 384
  pre_encoder:
    type: PTv3PreEncoderWithDino   # or PTv3m3PreEncoderWithDino
    ...
  encoder:
    type: DinoInjectionEncoder
    dino_dim: 384
    encoder_dim: 256
    zero_init: true
    inner_encoder:
      type: MaskedTransformerEncoder   # or TransformerEncoder
      ...
  decoder: ...
"""

import torch
import torch.nn as nn

from pointcept.models.builder import MODELS, MODULES
from pointcept.models.utils.structure import Point

from .model import (
    Model3DETRDetector,
    point2dense,
)
from .ptv3 import PTv3DinoMixin


# ---------------------------------------------------------------------------
# Wrapping encoder
# ---------------------------------------------------------------------------

@MODULES.register_module("DinoInjectionEncoder")
class DinoInjectionEncoder(nn.Module):
    """Wraps a transformer encoder and injects pooled DINO features.

    Sits in the ``encoder`` config slot of Model3DETRDetectorWithDino.
    The parent model sets ``self._dino_pooled`` (B, N', D) immediately
    before calling ``run_encoder``; this module reads and clears it.

    Args:
        inner_encoder (dict): config for the wrapped encoder MODULE
            (e.g. MaskedTransformerEncoder or TransformerEncoder).
        dino_dim (int): channel dimension D of pooled DINO features.
        encoder_dim (int): must match the encoder's feature dimension C.
        zero_init (bool): zero-initialise the dino projection weight so
            the model starts as a faithful copy of the base detector.
            Default True.
        dropout (float): dropout on the projected DINO features before
            addition. Default 0.0.
    """

    def __init__(
        self,
        inner_encoder: dict,
        dino_dim: int,
        encoder_dim: int,
        zero_init: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.inner_encoder = MODULES.build(inner_encoder)

        # Linear projection dino_dim → encoder_dim, no bias.
        # No bias: encoder features already carry normalisation offsets.
        self.dino_proj = nn.Linear(dino_dim, encoder_dim, bias=False)
        if zero_init:
            nn.init.zeros_(self.dino_proj.weight)

        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else None

        # Side-channel: set by Model3DETRDetectorWithDino before run_encoder.
        # Shape: (B, N', D) or None.
        self._dino_pooled: torch.Tensor | None = None

    def forward(self, features, xyz=None, padding_mask=None):
        """
        Args:
            features:     (N', B, C) — transformer convention from pre_encoder
            xyz:          (B, N', 3)
            padding_mask: (B, N') bool or None

        Returns:
            Same as inner_encoder.forward() — either a Point or
            (enc_xyz, enc_features, enc_inds).
        """
        dino_pooled = self._dino_pooled
        self._dino_pooled = None            # clear immediately — don't leak

        if dino_pooled is not None:
            # dino_pooled: (B, N', D) → project → (B, N', C)
            dino_proj = self.dino_proj(dino_pooled)
            if self.dropout is not None:
                dino_proj = self.dropout(dino_proj)
            # features: (N', B, C) += permuted projection (N', B, C)
            features = features + dino_proj.permute(1, 0, 2)

        return self.inner_encoder(features, xyz=xyz, padding_mask=padding_mask)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

@MODELS.register_module()
class Model3DETRDetectorWithDino(Model3DETRDetector):
    """3DETR detector with DINO feature injection through the encoder slot.

    The pre_encoder must be a PTv3DinoMixin subclass so that
    forward_with_dino() is available.  The encoder must be a
    DinoInjectionEncoder so that pooled DINO features are added
    before the transformer encoder runs.

    The only change versus Model3DETRDetector is in forward():
      1. Extract dino_feat from input_dict.
      2. Call pre_encoder.forward_with_dino() to get pooled dino_feat.
      3. Stash the pooled features on the DinoInjectionEncoder.
      4. Call run_encoder() as normal — it calls self.encoder(features, ...)
         which reads and applies the stashed features.

    Everything else — encoder_to_decoder_projection, query sampling,
    decoder, heads, loss — is inherited without modification.

    Args:
        **kwargs: forwarded verbatim to Model3DETRDetector.__init__().
            pre_encoder must be a PTv3DinoMixin subclass.
            encoder must be a DinoInjectionEncoder.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if not isinstance(self.pre_encoder, PTv3DinoMixin):
            raise TypeError(
                f"Model3DETRDetectorWithDino: pre_encoder must be a "
                f"PTv3DinoMixin subclass (PTv3PreEncoderWithDino or "
                f"PTv3m3PreEncoderWithDino), got "
                f"{type(self.pre_encoder).__name__}."
            )
        if not isinstance(self.encoder, DinoInjectionEncoder):
            raise TypeError(
                f"Model3DETRDetectorWithDino: encoder must be a "
                f"DinoInjectionEncoder, got {type(self.encoder).__name__}."
            )

    def forward(self, input_dict, encoder_only=False):
        """
        Args:
            input_dict (dict): same as Model3DETRDetector, plus optionally:
                'dino_feat': (B, D, N) float tensor — per-point DINO features
                at input resolution, aligned with input_dict['point_clouds'].
                If absent the model behaves identically to the base detector
                (dino_proj produces zeros due to zero_init, or is skipped).

        Returns:
            Same as Model3DETRDetector.forward.
        """
        point_clouds = input_dict["point_clouds"]
        dino_feat = input_dict.get("dino_feat", None)   # (B, D, N) or None

        # ---- Run pre_encoder with dino passthrough ----
        xyz = point_clouds[..., :3].contiguous()
        features = (
            point_clouds[..., 3:].transpose(1, 2).contiguous()
            if point_clouds.size(-1) > 3 else None
        )

        if dino_feat is not None:
            # pre_enc_result, dino_pooled = self.pre_encoder.forward_with_dino(
            #     xyz, features, dino_feat
            # )
            result = self.pre_encoder.forward_with_dino(
                xyz,
                features,
                dino_feat
            )

            if getattr(self.pre_encoder, "npoint", None) is not None:
                # FPS path
                enc_xyz, enc_features, enc_inds, dino_pooled = result

                # cache full tuple for run_encoder()
                pre_enc_result = (enc_xyz, enc_features, enc_inds)

                # convert (B, D, N) -> (B, N, D)
                dino_pooled = dino_pooled.permute(0, 2, 1).contiguous()

            else:
                # no-FPS path
                pre_enc_result, dino_pooled = result
            # dino_pooled layout depends on pre_encoder.npoint:
            #   npoint is None → (B, N_enc, D)   already (B, N', D)
            #   npoint is set  → (B, D, npoint)  needs transpose
            # if dino_pooled is not None and dino_pooled.shape[1] != dino_pooled.shape[2]:
            #     # Distinguish (B, N', D) from (B, D, npoint) by checking
            #     # whether the pre_encoder has npoint set.
            #     if getattr(self.pre_encoder, "npoint", None) is not None:
            #         dino_pooled = dino_pooled.permute(0, 2, 1)  # → (B, N', D)
            # Stash for DinoInjectionEncoder to consume in run_encoder.
            self.encoder._dino_pooled = dino_pooled
        else:
            # No dino_feat: pre_encoder runs its normal forward.
            # DinoInjectionEncoder._dino_pooled stays None → no injection.
            pass

        # ---- run_encoder calls self.pre_encoder(xyz, features) again ----
        # Problem: we already ran forward_with_dino above, and run_encoder
        # will call self.pre_encoder(xyz, features) a second time — double
        # encode.  We avoid this by temporarily replacing self.pre_encoder
        # with a thin wrapper that returns the already-computed result.
        if dino_feat is not None:
            _orig_pre_encoder = self.pre_encoder
            _cached_result = pre_enc_result

            class _CachedPreEncoder(nn.Module):
                def __init__(self, cached_result):
                    super().__init__()
                    self.cached_result = cached_result

                def forward(self, xyz_, features_):
                    return self.cached_result

                __call__ = forward
            self.pre_encoder = _CachedPreEncoder(_cached_result)
            try:
                enc_xyz, enc_features, enc_inds, padding_mask = self.run_encoder(
                    point_clouds
                )
            finally:
                self.pre_encoder = _orig_pre_encoder
        else:
            enc_xyz, enc_features, enc_inds, padding_mask = self.run_encoder(
                point_clouds
            )

        # ---- Everything below is identical to Model3DETRDetector.forward ----
        enc_features = self.encoder_to_decoder_projection(
            enc_features.permute(1, 2, 0)
        ).permute(2, 0, 1)

        if encoder_only:
            return enc_xyz, enc_features.transpose(0, 1)

        point_cloud_dims = [
            input_dict["point_cloud_dims_min"],
            input_dict["point_cloud_dims_max"],
        ]
        query_xyz, query_embed = self.get_query_embeddings(
            enc_xyz, point_cloud_dims, padding_mask=padding_mask
        )
        enc_pos = self.pos_embedding(enc_xyz, input_range=point_cloud_dims)

        enc_pos = enc_pos.permute(2, 0, 1)
        query_embed = query_embed.permute(2, 0, 1)
        tgt = torch.zeros_like(query_embed)
        box_features = self.decoder(
            tgt,
            enc_features,
            query_pos=query_embed,
            pos=enc_pos,
            memory_key_padding_mask=padding_mask,
        )[0]

        box_predictions = self.get_box_predictions(
            query_xyz, point_cloud_dims, box_features
        )

        if self.training and self.criterion is not None:
            loss, _ = self.criterion(box_predictions, input_dict)
            return dict(loss=loss)

        return box_predictions
