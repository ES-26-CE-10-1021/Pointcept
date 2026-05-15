"""Seg-only PTv3 + dense (B, N, 3+C) input segmentor.

Architecture:

    input dense (B, N, 3+C) ─► dense2point ─► embedding ─► enc ─► dec ─► unpool ─► seg_head ─► seg_logits

Mirrors the segmentation branch of ``MultiTask3DETRSegmentor`` (multi_task.py)
without the 3DETR detection branch. Intended as the seg-only counterpart to the
detection-only ``Model3DETRDetector`` and the joint ``MultiTask3DETRSegmentor``,
so the three configs share an identical backbone + seg head and only differ in
which task losses are active.

Accepts the same dense ``input_dict["point_clouds"]`` format used by
``MultiTask3DETRSegmentor`` and ``Model3DETRDetector``, so it pairs directly
with ``AgcoBBoxV1`` (``load_segment=True``) and the detection-aware transforms
in ``pointcept/datasets/det_transform.py``.
"""

import torch
import torch.nn as nn

from pointcept.models.builder import MODELS, build_model
from pointcept.models.losses import build_criteria

from .model import dense2point


@MODELS.register_module()
class Dense3DETRSegmentor(nn.Module):
    """PTv3 backbone (full U-Net) + linear seg head over dense input.

    Args:
        backbone: dict — config for ``PT-v3m1`` or ``PTv3m3PreEncoder`` built
            via ``build_model``. Must have ``enc_mode=False`` so the decoder
            runs.
        backbone_grid_size: float — written into ``Point["grid_size"]`` so
            PTv3's serialization grid matches the upstream
            ``GridSampleDetection``.
        num_seg_classes: int — number of semseg output classes.
        backbone_out_channels: int — = ``backbone.dec_channels[0]``; channels
            of the decoder output that feed ``seg_head``.
        seg_criteria: list[dict] — seg loss configs (e.g. ``[CrossEntropyLoss,
            LovaszLoss]``). Built via ``build_criteria``.
        seg_ignore_index: int — labels equal to this value are ignored in the
            loss. (Forwarded to ``intersection_and_union_gpu`` by the
            evaluator/tester, not consumed here directly.)
    """

    def __init__(
        self,
        backbone,
        backbone_grid_size,
        num_seg_classes,
        backbone_out_channels,
        seg_criteria,
        seg_ignore_index=-1,
    ):
        super().__init__()

        # ── Backbone (full U-Net) ─────────────────────────────────────────
        self.backbone = build_model(backbone)
        assert not self.backbone.enc_mode, (
            "Dense3DETRSegmentor requires the backbone's decoder; set "
            "enc_mode=False in the backbone config."
        )
        self.backbone_grid_size = float(backbone_grid_size)

        # ── Seg head + criteria ───────────────────────────────────────────
        self.num_seg_classes = int(num_seg_classes)
        self.seg_head = (
            nn.Linear(backbone_out_channels, num_seg_classes)
            if num_seg_classes > 0
            else nn.Identity()
        )
        self.seg_criteria = build_criteria(seg_criteria)
        self.seg_ignore_index = int(seg_ignore_index)

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _unpool_to_root(point):
        """Walk PTv3's pooling-parent chain back to root resolution.

        Mirrors ``MultiTask3DETRSegmentor._unpool_to_root`` (multi_task.py:225)
        and ``DefaultSegmentorV2.forward`` (default.py:69-75).
        """
        while "pooling_parent" in point.keys():
            assert "pooling_inverse" in point.keys()
            parent = point.pop("pooling_parent")
            inverse = point.pop("pooling_inverse")
            parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
            point = parent
        return point.feat

    def _backbone_forward(self, xyz, features):
        """Full-U-Net backbone pass returning the decoder-output ``Point``.

        Wrapper-aware: if the backbone exposes ``forward_enc_dec_split``
        (``PTv3m3PreEncoder`` / Utonia), delegate so freeze semantics,
        channel padding, and graph reattachment are honoured. Otherwise call
        the raw ``PT-v3m1`` ``embedding`` / ``enc`` / ``dec`` directly.
        """
        if hasattr(self.backbone, "forward_enc_dec_split"):
            _, _, _, dec_point = self.backbone.forward_enc_dec_split(xyz, features)
            return dec_point

        point = dense2point(xyz, features)
        point["grid_size"] = self.backbone_grid_size

        point.serialization(
            order=self.backbone.order, shuffle_orders=self.backbone.shuffle_orders
        )
        point.sparsify()
        point = self.backbone.embedding(point)
        point = self.backbone.enc(point)
        point = self.backbone.dec(point)
        return point

    # ------------------------------------------------------------------ forward
    def forward(self, input_dict):
        # ── 1. Unpack the dense (B, N, 3+C) batch. ───────────────────────
        pc = input_dict["point_clouds"]
        xyz = pc[..., :3].contiguous()
        features = (
            pc[..., 3:].permute(0, 2, 1).contiguous() if pc.shape[-1] > 3 else None
        )

        # ── 2. Backbone forward (encoder + decoder). ─────────────────────
        dec_point = self._backbone_forward(xyz, features)

        # ── 3. Unpool to root resolution + linear seg head. ──────────────
        seg_feat = self._unpool_to_root(dec_point)
        seg_logits = self.seg_head(seg_feat)

        # ── 4. Loss / outputs. ───────────────────────────────────────────
        has_segment = "segment" in input_dict
        if has_segment:
            segment = input_dict["segment"].reshape(-1)
            loss = self.seg_criteria(seg_logits, segment)

        if self.training:
            assert has_segment, "Training forward requires `segment` in input_dict."
            return dict(
                loss=loss,
                loss_seg=loss.detach(),
                seg_logits=seg_logits,
            )

        # Eval: SemSegEvaluator (evaluator.py:140) reads both `loss` and
        # `seg_logits`. Return both when segment is available; otherwise
        # return seg_logits only (pure inference path).
        if has_segment:
            return dict(loss=loss, seg_logits=seg_logits)
        return dict(seg_logits=seg_logits)
