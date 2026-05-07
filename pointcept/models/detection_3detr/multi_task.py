"""Multi-task PTv3 + 3DETR model: shared encoder, branched seg / det heads.

Architecture:

    input dense (B, N, 3+C) ─► dense2point ─► embedding ─► enc ─┬─► point2dense ─► 3DETR enc/dec ─► boxes
                                                                │
                                                                └─► dec ─► unpool ─► seg_head ─► seg_logits

The semseg branch keeps PTv3's standard U-Net structure (encoder + decoder +
unpool to root resolution) so it matches a vanilla `DefaultSegmentorV2` setup.
The detection branch taps the encoder bottleneck (after `self.enc`, before
`self.dec` runs), mirroring `Model3DETRDetector`'s pipeline from that point on.

For standalone PTv3 segmentation use ``DefaultSegmentorV2`` + the segmentation
configs. For standalone 3DETR detection use ``Model3DETRDetector`` + a
PTv3-based pre-encoder. This class is purpose-built for the multi-task path.
"""

from functools import partial

import numpy as np
import torch
import torch.nn as nn

from pointcept.models.builder import MODELS, MODULES, build_model
from pointcept.models.losses import build_criteria, LOSSES
from pointcept.models.utils.structure import Point

from .dataset_config import _setup_3detr_path
from .model import BoxProcessor, dense2point, point2dense

_setup_3detr_path()

from third_party.pointnet2.pointnet2_utils import furthest_point_sample
from models.helpers import GenericMLP
from models.position_embedding import PositionEmbeddingCoordsSine


@MODELS.register_module()
class MultiTask3DETRSegmentor(nn.Module):
    """PTv3 backbone + semseg head + 3DETR detection branch.

    Args:
        backbone: dict — config for ``PT-v3m1`` built via ``build_model``.
            Must use ``enc_mode=False`` so the decoder runs.
        backbone_grid_size: float — written into ``Point["grid_size"]`` so
            PTv3's serialization grid matches the upstream ``GridSampleDetection``.
        num_seg_classes: int — semseg output classes.
        backbone_out_channels: int — = ``backbone.dec_channels[0]``; channels
            of the decoder output that feed ``seg_head``.
        seg_criteria: list[dict] — list of seg loss configs, e.g.
            ``[CrossEntropyLoss, LovaszLoss]``. Built via ``build_criteria``.
        seg_ignore_index: int — segments == this value are ignored in the loss.
        det_encoder, det_decoder: dict — 3DETR transformer encoder / decoder
            configs (``IdentityEncoder3DETR`` / ``VanillaTransformerEncoder3DETR``
            / ``TransformerDecoder3DETR``).
        det_dataset_config: dict — ``ScanNetDetectionConfig`` or
            ``AgcoBBoxConfig``; supplies ``num_semcls`` / ``num_angle_bin``.
        det_criterion: dict — ``SetCriterion3DETR`` config.
        encoder_dim: int — channel count of the encoder bottleneck (=
            ``backbone.enc_channels[-1]``). Passed to ``encoder_to_decoder_projection``.
        decoder_dim, num_queries, position_embedding, mlp_dropout, projection_norm:
            mirror ``Model3DETRDetector`` (``model.py:191``).
        seg_weight, det_weight: float — combined loss is
            ``seg_weight * loss_seg + det_weight * loss_det``.
    """

    def __init__(
        self,
        backbone,
        backbone_grid_size,
        num_seg_classes,
        backbone_out_channels,
        seg_criteria,
        det_encoder,
        det_decoder,
        det_dataset_config,
        det_criterion,
        encoder_dim,
        decoder_dim=256,
        num_queries=128,
        position_embedding="fourier",
        mlp_dropout=0.3,
        projection_norm="ln",
        seg_ignore_index=-1,
        seg_weight=1.0,
        det_weight=1.0,
    ):
        super().__init__()

        # ── Backbone (PT-v3m1, full U-Net) ────────────────────────────────
        self.backbone = build_model(backbone)
        assert not self.backbone.enc_mode, (
            "MultiTask3DETRSegmentor requires the backbone's decoder; "
            "set enc_mode=False in the backbone config."
        )
        self.backbone_grid_size = float(backbone_grid_size)

        # ── Semseg head ───────────────────────────────────────────────────
        self.num_seg_classes = int(num_seg_classes)
        self.seg_head = (
            nn.Linear(backbone_out_channels, num_seg_classes)
            if num_seg_classes > 0
            else nn.Identity()
        )
        self.seg_criteria = build_criteria(seg_criteria)
        self.seg_ignore_index = int(seg_ignore_index)

        # ── Detection branch ──────────────────────────────────────────────
        self.det_encoder = MODULES.build(det_encoder)
        self.det_decoder = MODULES.build(det_decoder)
        self.dataset_config = MODULES.build(det_dataset_config)
        self.det_criterion = LOSSES.build(det_criterion)

        # Mirror Model3DETRDetector.__init__ (model.py:225-260) for the
        # parts that operate after the encoder bottleneck.
        if hasattr(self.det_encoder, "encoder") and hasattr(
            self.det_encoder.encoder, "masking_radius"
        ):
            hidden_dims = [encoder_dim]
        else:
            hidden_dims = [encoder_dim, encoder_dim]
        self.encoder_to_decoder_projection = GenericMLP(
            input_dim=encoder_dim,
            hidden_dims=hidden_dims,
            output_dim=decoder_dim,
            norm_fn_name=projection_norm,
            activation="relu",
            use_conv=True,
            output_use_activation=True,
            output_use_norm=True,
            output_use_bias=False,
        )
        self.pos_embedding = PositionEmbeddingCoordsSine(
            d_pos=decoder_dim, pos_type=position_embedding, normalize=True
        )
        self.query_projection = GenericMLP(
            input_dim=decoder_dim,
            hidden_dims=[decoder_dim],
            output_dim=decoder_dim,
            use_conv=True,
            output_use_activation=True,
            hidden_use_bias=True,
        )
        self.num_queries = int(num_queries)
        self.box_processor = BoxProcessor(self.dataset_config)
        self._build_mlp_heads(decoder_dim, mlp_dropout)

        # ── Loss weights ──────────────────────────────────────────────────
        self.seg_weight = float(seg_weight)
        self.det_weight = float(det_weight)

    # ------------------------------------------------------------------ MLP heads
    def _build_mlp_heads(self, decoder_dim, mlp_dropout):
        """Identical to ``Model3DETRDetector._build_mlp_heads`` (model.py:265)."""
        mlp_func = partial(
            GenericMLP,
            norm_fn_name="bn1d",
            activation="relu",
            use_conv=True,
            hidden_dims=[decoder_dim, decoder_dim],
            dropout=mlp_dropout,
            input_dim=decoder_dim,
        )
        self.mlp_heads = nn.ModuleDict(
            [
                ("sem_cls_head", mlp_func(output_dim=self.dataset_config.num_semcls + 1)),
                ("center_head", mlp_func(output_dim=3)),
                ("size_head", mlp_func(output_dim=3)),
                ("angle_cls_head", mlp_func(output_dim=self.dataset_config.num_angle_bin)),
                ("angle_residual_head", mlp_func(output_dim=self.dataset_config.num_angle_bin)),
            ]
        )

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _unpool_to_root(point):
        """Walk PTv3's pooling-parent chain to bring features back to root resolution.

        Mirrors ``DefaultSegmentorV2.forward`` (default.py:69-75).
        """
        while "pooling_parent" in point.keys():
            assert "pooling_inverse" in point.keys()
            parent = point.pop("pooling_parent")
            inverse = point.pop("pooling_inverse")
            parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
            point = parent
        return point.feat

    def _backbone_forward(self, point):
        """Manual backbone pass capturing the encoder bottleneck.

        Mirrors ``PointTransformerV3.forward`` (point_transformer_v3m1_base.py:699-707).
        IMPORTANT: ``self.backbone.dec`` walks the pooling-parent chain
        in-place and mutates the bottleneck Point, so we must snapshot
        the bottleneck (xyz / features / padding mask) BEFORE running
        ``dec``. The seg-side then consumes the (now-mutated) ``dec_point``
        whose parent chain leads back to the input resolution as usual.

        Returns:
            (enc_xyz, enc_features, padding_mask, dec_point)
                - enc_xyz:      (B, N', 3)
                - enc_features: (N', B, C_enc) — transformer convention
                - padding_mask: (B, N') bool or None
                - dec_point:    Point at first-encoder-stage resolution
                                with the unpool parent chain still attached
        """
        point.serialization(
            order=self.backbone.order, shuffle_orders=self.backbone.shuffle_orders
        )
        point.sparsify()
        point = self.backbone.embedding(point)
        enc_point = self.backbone.enc(point)

        # Snapshot the bottleneck NOW. dec() will mutate enc_point by
        # popping its pooling_parent / pooling_inverse fields.
        enc_xyz, enc_features_dense, padding_mask = point2dense(enc_point)
        enc_features = enc_features_dense.permute(2, 0, 1).contiguous()  # (N', B, C)

        dec_point = self.backbone.dec(enc_point)
        return enc_xyz, enc_features, padding_mask, dec_point

    # ------------------------------------------------------------------ detection-branch helpers
    def _det_run_encoder(self, enc_xyz, enc_features, padding_mask):
        """Apply the 3DETR transformer encoder over the bottleneck features.

        Adapts the second half of ``Model3DETRDetector.run_encoder`` (model.py:334-352).

        Args:
            enc_xyz:      (B, N', 3)
            enc_features: (B, C, N') or (N', B, C). Caller passes
                (N', B, C) to match transformer convention.
            padding_mask: (B, N') bool or None.

        Returns:
            enc_xyz, enc_features, padding_mask — possibly downsampled if
            the encoder selects FPS indices internally.
        """
        result = self.det_encoder(
            enc_features, xyz=enc_xyz, padding_mask=padding_mask
        )
        if isinstance(result, Point):
            new_xyz, new_feat_dense, new_pad = point2dense(result)
            new_feat = new_feat_dense.permute(2, 0, 1)
            return new_xyz, new_feat, new_pad
        # Tuple convention: (xyz, features, inds)
        new_xyz, new_features, new_inds = result
        if new_inds is not None and padding_mask is not None:
            padding_mask = torch.gather(padding_mask, 1, new_inds.type(torch.int64))
        return new_xyz, new_features, padding_mask

    def _det_get_query_embeddings(self, enc_xyz, point_cloud_dims, padding_mask):
        """FPS query sampling + positional embeddings.

        Identical to ``Model3DETRDetector.get_query_embeddings`` (model.py:354).
        """
        if padding_mask is not None:
            fps_xyz = enc_xyz.clone()
            fps_xyz[padding_mask] = 1e6
        else:
            fps_xyz = enc_xyz

        query_inds = furthest_point_sample(fps_xyz, self.num_queries).long()

        if padding_mask is not None:
            on_padded = torch.gather(padding_mask, 1, query_inds)
            if on_padded.any():
                first_real = (~padding_mask).long().argmax(dim=1, keepdim=True)
                first_real = first_real.expand_as(query_inds)
                query_inds = torch.where(on_padded, first_real, query_inds)

        query_xyz = torch.stack(
            [torch.gather(enc_xyz[..., x], 1, query_inds) for x in range(3)], dim=-1
        )
        pos_embed = self.pos_embedding(query_xyz, input_range=point_cloud_dims)
        query_embed = self.query_projection(pos_embed)
        return query_xyz, query_embed

    def _det_get_box_predictions(self, query_xyz, point_cloud_dims, box_features):
        """Decode per-query box parameters from the transformer decoder output.

        Identical to ``Model3DETRDetector.get_box_predictions`` (model.py:391).
        """
        box_features = box_features.permute(0, 2, 3, 1)
        num_layers, batch, channel, num_queries = box_features.shape
        box_features = box_features.reshape(num_layers * batch, channel, num_queries)

        cls_logits = self.mlp_heads["sem_cls_head"](box_features).transpose(1, 2)
        center_offset = (
            self.mlp_heads["center_head"](box_features).sigmoid().transpose(1, 2) - 0.5
        )
        size_normalized = (
            self.mlp_heads["size_head"](box_features).sigmoid().transpose(1, 2)
        )
        angle_logits = self.mlp_heads["angle_cls_head"](box_features).transpose(1, 2)
        angle_residual_normalized = self.mlp_heads["angle_residual_head"](
            box_features
        ).transpose(1, 2)

        cls_logits = cls_logits.reshape(num_layers, batch, num_queries, -1)
        center_offset = center_offset.reshape(num_layers, batch, num_queries, -1)
        size_normalized = size_normalized.reshape(num_layers, batch, num_queries, -1)
        angle_logits = angle_logits.reshape(num_layers, batch, num_queries, -1)
        angle_residual_normalized = angle_residual_normalized.reshape(
            num_layers, batch, num_queries, -1
        )
        angle_residual = angle_residual_normalized * (
            np.pi / angle_residual_normalized.shape[-1]
        )

        outputs = []
        for l in range(num_layers):
            center_normalized, center_unnormalized = (
                self.box_processor.compute_predicted_center(
                    center_offset[l], query_xyz, point_cloud_dims
                )
            )
            angle_continuous = self.box_processor.compute_predicted_angle(
                angle_logits[l], angle_residual[l]
            )
            size_unnormalized = self.box_processor.compute_predicted_size(
                size_normalized[l], point_cloud_dims
            )
            box_corners = self.box_processor.box_parametrization_to_corners(
                center_unnormalized, size_unnormalized, angle_continuous
            )
            with torch.no_grad():
                semcls_prob, objectness_prob = (
                    self.box_processor.compute_objectness_and_cls_prob(cls_logits[l])
                )
            outputs.append(
                {
                    "sem_cls_logits": cls_logits[l],
                    "center_normalized": center_normalized.contiguous(),
                    "center_unnormalized": center_unnormalized,
                    "size_normalized": size_normalized[l],
                    "size_unnormalized": size_unnormalized,
                    "angle_logits": angle_logits[l],
                    "angle_residual": angle_residual[l],
                    "angle_residual_normalized": angle_residual_normalized[l],
                    "angle_continuous": angle_continuous,
                    "objectness_prob": objectness_prob,
                    "sem_cls_prob": semcls_prob,
                    "box_corners": box_corners,
                }
            )
        return {"outputs": outputs[-1], "aux_outputs": outputs[:-1]}

    # ------------------------------------------------------------------ forward
    def forward(self, input_dict):
        # ── 1. Build dense Point from (B, N, 3+C) tensor. ────────────────
        pc = input_dict["point_clouds"]
        xyz = pc[..., :3].contiguous()
        features = (
            pc[..., 3:].permute(0, 2, 1).contiguous() if pc.shape[-1] > 3 else None
        )
        point = dense2point(xyz, features)
        point["grid_size"] = self.backbone_grid_size

        # ── 2. Manual backbone forward, capturing bottleneck. ────────────
        enc_xyz, enc_features, det_padding_mask, dec_point = (
            self._backbone_forward(point)
        )

        # ── 3. Seg branch: unpool to root resolution + linear head. ──────
        seg_feat = self._unpool_to_root(dec_point)
        seg_logits = self.seg_head(seg_feat)  # (B*N, num_seg_classes)

        # ── 4. Detection branch: 3DETR enc → projection → queries → dec. ─
        enc_xyz, enc_features, det_padding_mask = self._det_run_encoder(
            enc_xyz, enc_features, det_padding_mask
        )
        # encoder_to_decoder_projection expects (B, C, N')
        enc_features = self.encoder_to_decoder_projection(
            enc_features.permute(1, 2, 0)
        ).permute(2, 0, 1)  # → (N', B, decoder_dim)

        point_cloud_dims = [
            input_dict["point_cloud_dims_min"],
            input_dict["point_cloud_dims_max"],
        ]
        query_xyz, query_embed = self._det_get_query_embeddings(
            enc_xyz, point_cloud_dims, det_padding_mask
        )
        enc_pos = self.pos_embedding(enc_xyz, input_range=point_cloud_dims)
        enc_pos = enc_pos.permute(2, 0, 1)
        query_embed = query_embed.permute(2, 0, 1)
        tgt = torch.zeros_like(query_embed)
        box_features = self.det_decoder(
            tgt,
            enc_features,
            query_pos=query_embed,
            pos=enc_pos,
            memory_key_padding_mask=det_padding_mask,
        )[0]
        box_predictions = self._det_get_box_predictions(
            query_xyz, point_cloud_dims, box_features
        )

        # ── 5. Losses (training) / outputs (eval). ───────────────────────
        if self.training:
            segment = input_dict["segment"].reshape(-1)
            loss_seg = self.seg_criteria(seg_logits, segment)
            loss_det, _ = self.det_criterion(box_predictions, input_dict)
            loss = self.seg_weight * loss_seg + self.det_weight * loss_det
            return dict(
                loss=loss,
                loss_seg=loss_seg.detach(),
                loss_det=loss_det.detach(),
            )

        return dict(
            seg_logits=seg_logits,
            outputs=box_predictions["outputs"],
            aux_outputs=box_predictions.get("aux_outputs", []),
        )
