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
from .model import Model3DETRDetector
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
        self._dino_pooled = None  # clear immediately — don't leak

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

    def _run_pre_encoder(self, xyz, features):
        # If forward() stashed a precomputed pre-encoder result (via
        # forward_with_dino), reuse it instead of double-encoding.
        cached = getattr(self, "_cached_pre_enc", None)
        if cached is not None:
            return cached
        return super()._run_pre_encoder(xyz, features)

    def forward(self, input_dict, encoder_only=False):
        """
        Args:
            input_dict (dict): same as Model3DETRDetector, plus optionally:
                'dino_feat': (B, N, D) float tensor — per-point DINO features
                at input resolution, aligned with input_dict['point_clouds'].
                If absent the model behaves identically to the base detector
                (dino_proj produces zeros due to zero_init, or is skipped).

        Returns:
            Same as Model3DETRDetector.forward.
        """
        dino_feat = input_dict.get("dino_feat", None)

        if dino_feat is not None:
            point_clouds = input_dict["point_clouds"]
            xyz = point_clouds[..., :3].contiguous()
            features = (
                point_clouds[..., 3:].transpose(1, 2).contiguous()
                if point_clouds.size(-1) > 3
                else None
            )
            result = self.pre_encoder.forward_with_dino(xyz, features, dino_feat)

            if getattr(self.pre_encoder, "npoint", None) is not None:
                enc_xyz, enc_features, enc_inds, dino_pooled = result
                self._cached_pre_enc = (enc_xyz, enc_features, enc_inds)
                # FPS path emits (B, D, npoint); DinoInjectionEncoder wants (B, N', D).
                dino_pooled = dino_pooled.permute(0, 2, 1).contiguous()
            else:
                point, dino_pooled = result
                self._cached_pre_enc = point
            self.encoder._dino_pooled = dino_pooled

        try:
            return super().forward(input_dict, encoder_only=encoder_only)
        finally:
            self._cached_pre_enc = None
