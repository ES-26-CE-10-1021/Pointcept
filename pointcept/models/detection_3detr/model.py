"""
3DETR Model for Pointcept

Registers Model3DETRDetector with Pointcept's MODELS registry so it can be
instantiated from a config dict. Follows the DefaultSegmentor pattern:
forward() returns dict(loss=...) during training and dict(outputs=...) during eval.

The model is fully modular — pre_encoder, encoder, and decoder are each built
from separate config dicts (MODULES registry), so any component can be replaced
(e.g. swap in PointTransformerV3 as the encoder) by changing the config.

Setting pre_encoder=None skips the PointNet++ SA pre-encoding step entirely,
which is required when using a backbone like PTv3 that produces its own
features and coordinates.
"""

from functools import partial

import numpy as np
import torch
import torch.nn as nn

from pointcept.models.builder import MODELS, MODULES
from pointcept.models.losses.builder import LOSSES
from pointcept.models.utils.structure import Point
from pointcept.models.utils import offset2bincount
from .dataset_config import _setup_3detr_path

_setup_3detr_path()

from third_party.pointnet2.pointnet2_utils import furthest_point_sample
from models.helpers import GenericMLP
from models.position_embedding import PositionEmbeddingCoordsSine
from utils.pc_util import scale_points, shift_scale_points


def dense2point(xyz, features=None):
    """Convert dense tensors to a Pointcept Point object.

    Args:
        xyz:      (B, N, 3) point coordinates
        features: (B, C, N) point features, or None (uses xyz as feat)

    Returns:
        Point with coord (B*N, 3), feat (B*N, C), offset (B,)
        All scenes have equal length N, so offset = [N, 2N, ..., B*N].
    """
    B, N, _ = xyz.shape
    coord = xyz.reshape(B * N, 3)
    feat = (
        features.permute(0, 2, 1).reshape(B * N, features.shape[1])
        if features is not None
        else coord.clone()
    )
    offset = torch.arange(1, B + 1, device=xyz.device, dtype=torch.long) * N
    return Point(dict(coord=coord, feat=feat, offset=offset))


def point2dense(point):
    """Convert a Pointcept Point object to equal-length dense tensors.

    All scenes must have the same point count. The 3DETR encoder/decoder
    does not use a key-padding mask, so zero-padded rows would silently
    corrupt attention weights, FPS query selection, and predictions.

    Args:
        point: Point with coord (total, 3), feat (total, C), offset (B,)

    Returns:
        xyz_out:  (B, N, 3)
        feat_out: (B, C, N)

    Raises:
        ValueError: if scenes have different point counts.
    """
    counts = offset2bincount(point.offset)  # (B,)
    B = len(counts)
    max_n = counts.max().item()
    if (counts != max_n).any():
        raise ValueError(
            "point2dense received variable-length scenes (offset-derived counts "
            f"{counts.tolist()}); this 3DETR path assumes equal-length scenes "
            "because no key-padding mask is used. Please pre-pad/trim to a "
            "fixed length or extend the model to handle a padding mask."
        )
    enc_dim = point.feat.shape[-1]
    xyz_out = point.coord.new_zeros(B, max_n, 3)
    feat_out = point.feat.new_zeros(B, enc_dim, max_n)
    start = 0
    for b in range(B):
        n = counts[b].item()
        xyz_out[b, :n] = point.coord[start : start + n]
        feat_out[b, :, :n] = point.feat[start : start + n].T
        start += n
    return xyz_out, feat_out  # (B, max_n, 3), (B, C, max_n)


class BoxProcessor:
    """Converts MLP head outputs into bounding box parameters."""

    def __init__(self, dataset_config):
        self.dataset_config = dataset_config

    def compute_predicted_center(self, center_offset, query_xyz, point_cloud_dims):
        center_unnormalized = query_xyz + center_offset
        center_normalized = shift_scale_points(
            center_unnormalized, src_range=point_cloud_dims
        )
        return center_normalized, center_unnormalized

    def compute_predicted_size(self, size_normalized, point_cloud_dims):
        scene_scale = point_cloud_dims[1] - point_cloud_dims[0]
        scene_scale = torch.clamp(scene_scale, min=1e-1)
        size_unnormalized = scale_points(size_normalized, mult_factor=scene_scale)
        return size_unnormalized

    def compute_predicted_angle(self, angle_logits, angle_residual):
        if angle_logits.shape[-1] == 1:
            angle = angle_logits * 0 + angle_residual * 0
            angle = angle.squeeze(-1).clamp(min=0)
        else:
            angle_per_cls = 2 * np.pi / self.dataset_config.num_angle_bin
            pred_angle_class = angle_logits.argmax(dim=-1).detach()
            angle_center = angle_per_cls * pred_angle_class
            angle = angle_center + angle_residual.gather(
                2, pred_angle_class.unsqueeze(-1)
            ).squeeze(-1)
            mask = angle > np.pi
            angle[mask] = angle[mask] - 2 * np.pi
        return angle

    def compute_objectness_and_cls_prob(self, cls_logits):
        assert cls_logits.shape[-1] == self.dataset_config.num_semcls + 1
        cls_prob = torch.nn.functional.softmax(cls_logits, dim=-1)
        objectness_prob = 1 - cls_prob[..., -1]
        return cls_prob[..., :-1], objectness_prob

    def box_parametrization_to_corners(
        self, box_center_unnorm, box_size_unnorm, box_angle
    ):
        return self.dataset_config.box_parametrization_to_corners(
            box_center_unnorm, box_size_unnorm, box_angle
        )


@MODELS.register_module()
class Model3DETRDetector(nn.Module):
    """
    3DETR 3D object detector wrapped for the Pointcept framework.

    Architecture:
        1. pre_encoder (optional): PointNet++ SA → subsamples N points to N'
           and projects to encoder_dim. Set to None when using an external
           backbone (e.g. PTv3) that already produces (xyz, features).
        2. encoder: Transformer (vanilla or masked) over N' points.
        3. encoder_to_decoder_projection: linear projection to decoder_dim.
        4. Query sampling: FPS to select B query points from encoder output.
        5. decoder: cross-attention transformer over B queries × N'' encoder pts.
        6. mlp_heads: per-query MLP heads for class, center, size, angle.

    Args:
        pre_encoder (dict | None): config for pre-encoder MODULE. If None,
            encoder receives the raw point cloud directly (useful for PTv3).
        encoder (dict): config for encoder MODULE.
        decoder (dict): config for decoder MODULE.
        dataset_config (dict): config for ScanNetDetectionConfig MODULE.
        encoder_dim (int): encoder feature dimension.
        decoder_dim (int): decoder feature dimension.
        num_queries (int): number of box queries (= max detections).
        position_embedding (str): 'fourier' or 'sine'.
        mlp_dropout (float): dropout in prediction MLP heads.
        criterion (dict): config for SetCriterion3DETR loss.
    """

    def __init__(
        self,
        pre_encoder=None,
        encoder=None,
        decoder=None,
        dataset_config=None,
        encoder_dim=256,
        decoder_dim=256,
        num_queries=256,
        position_embedding="fourier",
        mlp_dropout=0.3,
        criterion=None,
        input_feature_dim=0,
    ):
        super().__init__()

        # Build sub-components from MODULES registry
        self.pre_encoder = (
            MODULES.build(pre_encoder) if pre_encoder is not None else None
        )

        # When pre_encoder is None, project raw coordinates (+ optional features)
        # to encoder_dim so the transformer receives the expected channel count.
        if self.pre_encoder is None:
            in_channels = 3 + input_feature_dim  # XYZ + any extra features
            self.input_projection = nn.Linear(in_channels, encoder_dim)
        else:
            self.input_projection = None

        self.encoder = MODULES.build(encoder)
        self.decoder = MODULES.build(decoder)
        self.dataset_config = MODULES.build(dataset_config)

        # Projection from encoder space to decoder space
        if hasattr(self.encoder, "encoder") and hasattr(
            self.encoder.encoder, "masking_radius"
        ):
            hidden_dims = [encoder_dim]
        else:
            hidden_dims = [encoder_dim, encoder_dim]
        self.encoder_to_decoder_projection = GenericMLP(
            input_dim=encoder_dim,
            hidden_dims=hidden_dims,
            output_dim=decoder_dim,
            norm_fn_name="bn1d",
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

        self.num_queries = num_queries
        self.box_processor = BoxProcessor(self.dataset_config)

        # Build MLP heads for box parameter prediction
        self._build_mlp_heads(decoder_dim, mlp_dropout)

        # Build criterion (built directly, not via Pointcept's Criteria wrapper)
        self.criterion = LOSSES.build(criterion) if criterion is not None else None

    def _build_mlp_heads(self, decoder_dim, mlp_dropout):
        mlp_func = partial(
            GenericMLP,
            norm_fn_name="bn1d",
            activation="relu",
            use_conv=True,
            hidden_dims=[decoder_dim, decoder_dim],
            dropout=mlp_dropout,
            input_dim=decoder_dim,
        )
        semcls_head = mlp_func(output_dim=self.dataset_config.num_semcls + 1)
        center_head = mlp_func(output_dim=3)
        size_head = mlp_func(output_dim=3)
        angle_cls_head = mlp_func(output_dim=self.dataset_config.num_angle_bin)
        angle_reg_head = mlp_func(output_dim=self.dataset_config.num_angle_bin)

        self.mlp_heads = nn.ModuleDict(
            [
                ("sem_cls_head", semcls_head),
                ("center_head", center_head),
                ("size_head", size_head),
                ("angle_cls_head", angle_cls_head),
                ("angle_residual_head", angle_reg_head),
            ]
        )

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None
        return xyz, features

    def run_encoder(self, point_clouds):
        xyz, features = self._break_up_pc(point_clouds)

        if self.pre_encoder is not None:
            result = self.pre_encoder(xyz, features)
            if isinstance(result, Point):
                # Point-returning pre-encoder (e.g. a PTv3-based component)
                pre_enc_xyz, pre_enc_features = point2dense(result)
                pre_enc_inds = None
            else:
                pre_enc_xyz, pre_enc_features, pre_enc_inds = result
            # nn.MultiHeadAttention expects (npoints, B, C)
            pre_enc_features = pre_enc_features.permute(2, 0, 1)
        else:
            # No pre-encoder: project raw XYZ (+ optional features) to encoder_dim.
            pre_enc_xyz = xyz  # (B, N, 3)
            pre_enc_inds = (
                torch.arange(xyz.shape[1], device=xyz.device)
                .unsqueeze(0)
                .expand(xyz.shape[0], -1)
            )
            if features is not None:
                # features: (B, C, N) → cat with xyz → (B, N, 3+C)
                inp = torch.cat([xyz, features.permute(0, 2, 1)], dim=-1)
            else:
                inp = xyz  # (B, N, 3)
            # Project to encoder_dim and convert to (N, B, C)
            pre_enc_features = self.input_projection(inp).permute(1, 0, 2)

        result = self.encoder(pre_enc_features, xyz=pre_enc_xyz)
        if isinstance(result, Point):
            # Point-returning encoder (e.g. a PTv3-based encoder component)
            enc_xyz, enc_features_dense = point2dense(result)
            enc_features = enc_features_dense.permute(2, 0, 1)  # → (N'', B, C)
            enc_inds = None
        else:
            enc_xyz, enc_features, enc_inds = result

        if enc_inds is None:
            enc_inds = pre_enc_inds
        elif pre_enc_inds is not None:
            enc_inds = torch.gather(pre_enc_inds, 1, enc_inds.type(torch.int64))

        return enc_xyz, enc_features, enc_inds

    def get_query_embeddings(self, encoder_xyz, point_cloud_dims):
        query_inds = furthest_point_sample(encoder_xyz, self.num_queries).long()
        query_xyz = torch.stack(
            [torch.gather(encoder_xyz[..., x], 1, query_inds) for x in range(3)],
            dim=-1,
        )
        pos_embed = self.pos_embedding(query_xyz, input_range=point_cloud_dims)
        query_embed = self.query_projection(pos_embed)
        return query_xyz, query_embed

    def get_box_predictions(self, query_xyz, point_cloud_dims, box_features):
        """
        Args:
            query_xyz: (B, nqueries, 3)
            point_cloud_dims: [min (B,3), max (B,3)]
            box_features: (nlayers, nqueries, B, C)

        Returns:
            dict with 'outputs' (last layer) and 'aux_outputs' (intermediate layers)
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

        return {
            "outputs": outputs[-1],
            "aux_outputs": outputs[:-1],
        }

    def forward(self, input_dict, encoder_only=False):
        """
        Args:
            input_dict (dict): must contain:
                - 'point_clouds': (B, N, 3+C) float tensor
                - 'point_cloud_dims_min': (B, 3) float tensor
                - 'point_cloud_dims_max': (B, 3) float tensor
                - GT keys for loss computation during training

        Returns:
            During training: dict(loss=scalar, loss_dict=dict)
            During eval: dict(outputs=box_pred_dict, aux_outputs=list)
        """
        point_clouds = input_dict["point_clouds"]

        enc_xyz, enc_features, enc_inds = self.run_encoder(point_clouds)
        enc_features = self.encoder_to_decoder_projection(
            enc_features.permute(1, 2, 0)
        ).permute(2, 0, 1)

        if encoder_only:
            return enc_xyz, enc_features.transpose(0, 1)

        point_cloud_dims = [
            input_dict["point_cloud_dims_min"],
            input_dict["point_cloud_dims_max"],
        ]
        query_xyz, query_embed = self.get_query_embeddings(enc_xyz, point_cloud_dims)
        enc_pos = self.pos_embedding(enc_xyz, input_range=point_cloud_dims)

        enc_pos = enc_pos.permute(2, 0, 1)
        query_embed = query_embed.permute(2, 0, 1)
        tgt = torch.zeros_like(query_embed)
        box_features = self.decoder(
            tgt, enc_features, query_pos=query_embed, pos=enc_pos
        )[0]

        box_predictions = self.get_box_predictions(
            query_xyz, point_cloud_dims, box_features
        )

        if self.training and self.criterion is not None:
            loss, _ = self.criterion(box_predictions, input_dict)
            return dict(loss=loss)

        return box_predictions
