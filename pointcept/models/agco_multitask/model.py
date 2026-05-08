"""
Point Transformer - V3 Mode1

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.

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
from addict import Dict
import math
from pointcept.models.detection_3detr.model import dense2point
import torch
import torch.nn as nn
import spconv.pytorch as spconv
import torch_scatter
from timm.layers import DropPath

try:
    import flash_attn
except ImportError:
    flash_attn = None

from pointcept.models.point_prompt_training import PDNorm
from pointcept.models.builder import MODELS
from pointcept.models.utils.misc import offset2bincount
from pointcept.models.utils.structure import Point
from pointcept.models.modules import PointModule, PointSequential

from functools import partial

import numpy as np
import torch
import torch.nn as nn

from pointcept.models.builder import MODELS, MODULES
from pointcept.models.losses.builder import LOSSES
from pointcept.models.utils.structure import Point
from pointcept.models.utils import offset2bincount
# from .dataset_config import _setup_3detr_path

# _setup_3detr_path()

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
    """Convert a Pointcept Point object to dense tensors with optional padding.

    When all scenes have the same point count, uses reshape (zero-copy,
    fully differentiable). When scenes have variable lengths (e.g. after
    PTv3 voxelization), allocates dense tensors of length max_n and scatters
    each point into its per-scene position via indexed assignment
    (preserves autograd), then returns a padding mask for downstream
    attention layers.

    Args:
        point: Point with coord (total, 3), feat (total, C), offset (B,)

    Returns:
        xyz_out:      (B, N, 3)
        feat_out:     (B, C, N)
        padding_mask: (B, N) bool tensor where True = padded position,
                      or None if all scenes have equal length.
    """
    counts = offset2bincount(point.offset)  # (B,)
    B = len(counts)
    max_n = counts.max().item()
    enc_dim = point.feat.shape[-1]

    if (counts == max_n).all():
        # Equal-length fast path: reshape preserves autograd (no copy).
        xyz_out = point.coord.reshape(B, max_n, 3)
        feat_out = point.feat.reshape(B, max_n, enc_dim).permute(0, 2, 1).contiguous()
        return xyz_out, feat_out, None

    # Variable-length: scatter into pre-allocated dense tensors (no GPU→CPU sync).
    device = point.coord.device
    # Per-point position within its scene.
    offsets_shifted = torch.cat([counts.new_zeros(1), point.offset[:-1]])
    pos_in_scene = torch.arange(point.coord.shape[0], device=device) - offsets_shifted[point.batch]

    # Build (B, max_n, *) dense tensors via index_put (differentiable).
    xyz_out = point.coord.new_zeros(B, max_n, 3)
    feat_out = point.feat.new_zeros(B, max_n, enc_dim)
    xyz_out[point.batch, pos_in_scene] = point.coord
    feat_out[point.batch, pos_in_scene] = point.feat

    # True = padded (ignored in attention)
    padding_mask = (
        torch.arange(max_n, device=device).unsqueeze(0)
        >= counts.unsqueeze(1)
    )

    feat_out = feat_out.permute(0, 2, 1).contiguous()
    return xyz_out, feat_out, padding_mask


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
class RPE(torch.nn.Module):
    def __init__(self, patch_size, num_heads):
        super().__init__()
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.pos_bnd = int((4 * patch_size) ** (1 / 3) * 2)
        self.rpe_num = 2 * self.pos_bnd + 1
        self.rpe_table = torch.nn.Parameter(torch.zeros(3 * self.rpe_num, num_heads))
        torch.nn.init.trunc_normal_(self.rpe_table, std=0.02)

    def forward(self, coord):
        idx = (
            coord.clamp(-self.pos_bnd, self.pos_bnd)  # clamp into bnd
            + self.pos_bnd  # relative position to positive index
            + torch.arange(3, device=coord.device) * self.rpe_num  # x, y, z stride
        )
        out = self.rpe_table.index_select(0, idx.reshape(-1))
        out = out.view(idx.shape + (-1,)).sum(3)
        out = out.permute(0, 3, 1, 2)  # (N, K, K, H) -> (N, H, K, K)
        return out


class SerializedAttention(PointModule):
    def __init__(
        self,
        channels,
        num_heads,
        patch_size,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        order_index=0,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=True,
        upcast_softmax=True,
    ):
        super().__init__()
        assert channels % num_heads == 0
        self.channels = channels
        self.num_heads = num_heads
        self.scale = qk_scale or (channels // num_heads) ** -0.5
        self.order_index = order_index
        self.upcast_attention = upcast_attention
        self.upcast_softmax = upcast_softmax
        self.enable_rpe = enable_rpe
        self.enable_flash = enable_flash
        if enable_flash:
            assert (
                enable_rpe is False
            ), "Set enable_rpe to False when enable Flash Attention"
            assert (
                upcast_attention is False
            ), "Set upcast_attention to False when enable Flash Attention"
            assert (
                upcast_softmax is False
            ), "Set upcast_softmax to False when enable Flash Attention"
            assert flash_attn is not None, "Make sure flash_attn is installed."
            self.patch_size = patch_size
            self.attn_drop = attn_drop
        else:
            # when disable flash attention, we still don't want to use mask
            # consequently, patch size will auto set to the
            # min number of patch_size_max and number of points
            self.patch_size_max = patch_size
            self.patch_size = 0
            self.attn_drop = torch.nn.Dropout(attn_drop)

        self.qkv = torch.nn.Linear(channels, channels * 3, bias=qkv_bias)
        self.proj = torch.nn.Linear(channels, channels)
        self.proj_drop = torch.nn.Dropout(proj_drop)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.rpe = RPE(patch_size, num_heads) if self.enable_rpe else None

    @torch.no_grad()
    def get_rel_pos(self, point, order):
        K = self.patch_size
        rel_pos_key = f"rel_pos_{self.order_index}"
        if rel_pos_key not in point.keys():
            grid_coord = point.grid_coord[order]
            grid_coord = grid_coord.reshape(-1, K, 3)
            point[rel_pos_key] = grid_coord.unsqueeze(2) - grid_coord.unsqueeze(1)
        return point[rel_pos_key]

    @torch.no_grad()
    def get_padding_and_inverse(self, point):
        pad_key = "pad"
        unpad_key = "unpad"
        cu_seqlens_key = "cu_seqlens_key"
        if (
            pad_key not in point.keys()
            or unpad_key not in point.keys()
            or cu_seqlens_key not in point.keys()
        ):
            offset = point.offset
            bincount = offset2bincount(offset)
            bincount_pad = (
                torch.div(
                    bincount + self.patch_size - 1,
                    self.patch_size,
                    rounding_mode="trunc",
                )
                * self.patch_size
            )
            # only pad point when num of points larger than patch_size
            mask_pad = bincount > self.patch_size
            bincount_pad = ~mask_pad * bincount + mask_pad * bincount_pad
            _offset = nn.functional.pad(offset, (1, 0))
            _offset_pad = nn.functional.pad(torch.cumsum(bincount_pad, dim=0), (1, 0))
            pad = torch.arange(_offset_pad[-1], device=offset.device)
            unpad = torch.arange(_offset[-1], device=offset.device)
            cu_seqlens = []
            for i in range(len(offset)):
                unpad[_offset[i] : _offset[i + 1]] += _offset_pad[i] - _offset[i]
                if bincount[i] != bincount_pad[i]:
                    pad[
                        _offset_pad[i + 1]
                        - self.patch_size
                        + (bincount[i] % self.patch_size) : _offset_pad[i + 1]
                    ] = pad[
                        _offset_pad[i + 1]
                        - 2 * self.patch_size
                        + (bincount[i] % self.patch_size) : _offset_pad[i + 1]
                        - self.patch_size
                    ]
                pad[_offset_pad[i] : _offset_pad[i + 1]] -= _offset_pad[i] - _offset[i]
                cu_seqlens.append(
                    torch.arange(
                        _offset_pad[i],
                        _offset_pad[i + 1],
                        step=self.patch_size,
                        dtype=torch.int32,
                        device=offset.device,
                    )
                )
            point[pad_key] = pad
            point[unpad_key] = unpad
            point[cu_seqlens_key] = nn.functional.pad(
                torch.concat(cu_seqlens), (0, 1), value=_offset_pad[-1]
            )
        return point[pad_key], point[unpad_key], point[cu_seqlens_key]

    def forward(self, point):
        if not self.enable_flash:
            self.patch_size = min(
                offset2bincount(point.offset).min().tolist(), self.patch_size_max
            )

        H = self.num_heads
        K = self.patch_size
        C = self.channels

        pad, unpad, cu_seqlens = self.get_padding_and_inverse(point)

        order = point.serialized_order[self.order_index][pad]
        inverse = unpad[point.serialized_inverse[self.order_index]]

        # padding and reshape feat and batch for serialized point patch
        qkv = self.qkv(point.feat)[order]

        if not self.enable_flash:
            # encode and reshape qkv: (N', K, 3, H, C') => (3, N', H, K, C')
            q, k, v = (
                qkv.reshape(-1, K, 3, H, C // H).permute(2, 0, 3, 1, 4).unbind(dim=0)
            )
            # attn
            if self.upcast_attention:
                q = q.float()
                k = k.float()
            attn = (q * self.scale) @ k.transpose(-2, -1)  # (N', H, K, K)
            if self.enable_rpe:
                attn = attn + self.rpe(self.get_rel_pos(point, order))
            if self.upcast_softmax:
                attn = attn.float()
            attn = self.softmax(attn)
            attn = self.attn_drop(attn).to(qkv.dtype)
            feat = (attn @ v).transpose(1, 2).reshape(-1, C)
        else:
            feat = flash_attn.flash_attn_varlen_qkvpacked_func(
                qkv.to(torch.bfloat16).reshape(-1, 3, H, C // H),
                cu_seqlens,
                max_seqlen=self.patch_size,
                dropout_p=self.attn_drop if self.training else 0,
                softmax_scale=self.scale,
            ).reshape(-1, C)
            feat = feat.to(qkv.dtype)
        feat = feat[inverse]

        # ffn
        feat = self.proj(feat)
        feat = self.proj_drop(feat)
        point.feat = feat
        return point


class MLP(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels=None,
        out_channels=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_channels = out_channels or in_channels
        hidden_channels = hidden_channels or in_channels
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_channels, out_channels)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(PointModule):
    def __init__(
        self,
        channels,
        num_heads,
        patch_size=48,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
        pre_norm=True,
        order_index=0,
        cpe_indice_key=None,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=True,
        upcast_softmax=True,
    ):
        super().__init__()
        self.channels = channels
        self.pre_norm = pre_norm

        self.cpe = PointSequential(
            spconv.SubMConv3d(
                channels,
                channels,
                kernel_size=3,
                bias=True,
                indice_key=cpe_indice_key,
            ),
            nn.Linear(channels, channels),
            norm_layer(channels),
        )

        self.norm1 = PointSequential(norm_layer(channels))
        self.attn = SerializedAttention(
            channels=channels,
            patch_size=patch_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            order_index=order_index,
            enable_rpe=enable_rpe,
            enable_flash=enable_flash,
            upcast_attention=upcast_attention,
            upcast_softmax=upcast_softmax,
        )
        self.norm2 = PointSequential(norm_layer(channels))
        self.mlp = PointSequential(
            MLP(
                in_channels=channels,
                hidden_channels=int(channels * mlp_ratio),
                out_channels=channels,
                act_layer=act_layer,
                drop=proj_drop,
            )
        )
        self.drop_path = PointSequential(
            DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        )

    def forward(self, point: Point):
        shortcut = point.feat
        point = self.cpe(point)
        point.feat = shortcut + point.feat
        shortcut = point.feat
        if self.pre_norm:
            point = self.norm1(point)
        point = self.drop_path(self.attn(point))
        point.feat = shortcut + point.feat
        if not self.pre_norm:
            point = self.norm1(point)

        shortcut = point.feat
        if self.pre_norm:
            point = self.norm2(point)
        point = self.drop_path(self.mlp(point))
        point.feat = shortcut + point.feat
        if not self.pre_norm:
            point = self.norm2(point)
        point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(point.feat)
        return point


class SerializedPooling(PointModule):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride=2,
        norm_layer=None,
        act_layer=None,
        reduce="max",
        shuffle_orders=True,
        traceable=True,  # record parent and cluster
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        assert stride == 2 ** (math.ceil(stride) - 1).bit_length()  # 2, 4, 8
        # TODO: add support to grid pool (any stride)
        self.stride = stride
        assert reduce in ["sum", "mean", "min", "max"]
        self.reduce = reduce
        self.shuffle_orders = shuffle_orders
        self.traceable = traceable

        self.proj = nn.Linear(in_channels, out_channels)
        if norm_layer is not None:
            self.norm = PointSequential(norm_layer(out_channels))
        if act_layer is not None:
            self.act = PointSequential(act_layer())

    def forward(self, point: Point):
        pooling_depth = (math.ceil(self.stride) - 1).bit_length()
        if pooling_depth > point.serialized_depth:
            pooling_depth = 0
        assert {
            "serialized_code",
            "serialized_order",
            "serialized_inverse",
            "serialized_depth",
        }.issubset(
            point.keys()
        ), "Run point.serialization() point cloud before SerializedPooling"

        code = point.serialized_code >> pooling_depth * 3
        code_, cluster, counts = torch.unique(
            code[0],
            sorted=True,
            return_inverse=True,
            return_counts=True,
        )
        # indices of point sorted by cluster, for torch_scatter.segment_csr
        _, indices = torch.sort(cluster)
        # index pointer for sorted point, for torch_scatter.segment_csr
        idx_ptr = torch.cat([counts.new_zeros(1), torch.cumsum(counts, dim=0)])
        # head_indices of each cluster, for reduce attr e.g. code, batch
        head_indices = indices[idx_ptr[:-1]]
        # generate down code, order, inverse
        code = code[:, head_indices]
        order = torch.argsort(code)
        inverse = torch.zeros_like(order).scatter_(
            dim=1,
            index=order,
            src=torch.arange(0, code.shape[1], device=order.device).repeat(
                code.shape[0], 1
            ),
        )

        if self.shuffle_orders:
            perm = torch.randperm(code.shape[0])
            code = code[perm]
            order = order[perm]
            inverse = inverse[perm]

        # collect information
        point_dict = Dict(
            feat=torch_scatter.segment_csr(
                self.proj(point.feat)[indices], idx_ptr, reduce=self.reduce
            ),
            coord=torch_scatter.segment_csr(
                point.coord[indices], idx_ptr, reduce="mean"
            ),
            grid_coord=point.grid_coord[head_indices] >> pooling_depth,
            serialized_code=code,
            serialized_order=order,
            serialized_inverse=inverse,
            serialized_depth=point.serialized_depth - pooling_depth,
            batch=point.batch[head_indices],
        )

        if "condition" in point.keys():
            point_dict["condition"] = point.condition
        if "context" in point.keys():
            point_dict["context"] = point.context

        if self.traceable:
            point_dict["pooling_inverse"] = cluster
            point_dict["pooling_parent"] = point
        point = Point(point_dict)
        if self.norm is not None:
            point = self.norm(point)
        if self.act is not None:
            point = self.act(point)
        point.sparsify()
        return point


class SerializedUnpooling(PointModule):
    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        norm_layer=None,
        act_layer=None,
        traceable=False,  # record parent and cluster
    ):
        super().__init__()
        self.proj = PointSequential(nn.Linear(in_channels, out_channels))
        self.proj_skip = PointSequential(nn.Linear(skip_channels, out_channels))

        if norm_layer is not None:
            self.proj.add(norm_layer(out_channels))
            self.proj_skip.add(norm_layer(out_channels))

        if act_layer is not None:
            self.proj.add(act_layer())
            self.proj_skip.add(act_layer())

        self.traceable = traceable

    def forward(self, point):
        assert "pooling_parent" in point.keys()
        assert "pooling_inverse" in point.keys()
        parent = point.pop("pooling_parent")
        inverse = point.pop("pooling_inverse")
        point = self.proj(point)
        parent = self.proj_skip(parent)
        parent.feat = parent.feat + point.feat[inverse]

        if self.traceable:
            parent["unpooling_parent"] = point
        return parent


class Embedding(PointModule):
    def __init__(
        self,
        in_channels,
        embed_channels,
        norm_layer=None,
        act_layer=None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.embed_channels = embed_channels

        # TODO: check remove spconv
        self.stem = PointSequential(
            conv=spconv.SubMConv3d(
                in_channels,
                embed_channels,
                kernel_size=5,
                padding=1,
                bias=False,
                indice_key="stem",
            )
        )
        if norm_layer is not None:
            self.stem.add(norm_layer(embed_channels), name="norm")
        if act_layer is not None:
            self.stem.add(act_layer(), name="act")

    def forward(self, point: Point):
        point = self.stem(point)
        return point


@MODELS.register_module("MultiTaskDETR")
class MultiTaskDETR(PointModule):
    def __init__(
        self,
        num_seg_classes,
        in_channels=6,
        order=("z", "z-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(32, 64, 128, 256, 512),
        enc_num_head=(2, 4, 8, 16, 32),
        enc_patch_size=(48, 48, 48, 48, 48),
        dec_depths=(2, 2, 2, 2),
        dec_channels=(64, 64, 128, 256),
        dec_num_head=(4, 4, 8, 16),
        dec_patch_size=(48, 48, 48, 48),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        pre_norm=True,
        shuffle_orders=True,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=False,
        upcast_softmax=False,
        enc_mode=False,
        pdnorm_bn=False,
        pdnorm_ln=False,
        pdnorm_decouple=True,
        pdnorm_adaptive=False,
        pdnorm_affine=True,
        pdnorm_conditions=("ScanNet", "S3DIS", "Structured3D"),
        det_encoder=None,
        det_decoder=None,
        dataset_config=None,
        encoder_dim=256,
        decoder_dim=256,
        num_queries=256,
        position_embedding="fourier",
        mlp_dropout=0.3,
        det_criterion=None,
        seg_criterion=None,
        input_feature_dim=0,
        projection_norm="bn1d",
        det_npoints=2048,
        alpha_weight=0.5,
        grid_size=0.05,
    ):
        super().__init__()
        self.num_stages = len(enc_depths)
        self.order = [order] if isinstance(order, str) else order
        self.enc_mode = enc_mode
        self.shuffle_orders = shuffle_orders
        self.alpha_weight = alpha_weight
        self.grid_size = grid_size
        self.det_npoints = det_npoints
        
        assert self.num_stages == len(stride) + 1
        assert self.num_stages == len(enc_depths)
        assert self.num_stages == len(enc_channels)
        assert self.num_stages == len(enc_num_head)
        assert self.num_stages == len(enc_patch_size)
        assert self.enc_mode or self.num_stages == len(dec_depths) + 1
        assert self.enc_mode or self.num_stages == len(dec_channels) + 1
        assert self.enc_mode or self.num_stages == len(dec_num_head) + 1
        assert self.enc_mode or self.num_stages == len(dec_patch_size) + 1
       
        # norm layers
        if pdnorm_bn:
            bn_layer = partial(
                PDNorm,
                norm_layer=partial(
                    nn.BatchNorm1d, eps=1e-3, momentum=0.01, affine=pdnorm_affine
                ),
                conditions=pdnorm_conditions,
                decouple=pdnorm_decouple,
                adaptive=pdnorm_adaptive,
            )
        else:
            bn_layer = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        if pdnorm_ln:
            ln_layer = partial(
                PDNorm,
                norm_layer=partial(nn.LayerNorm, elementwise_affine=pdnorm_affine),
                conditions=pdnorm_conditions,
                decouple=pdnorm_decouple,
                adaptive=pdnorm_adaptive,
            )
        else:
            ln_layer = nn.LayerNorm
        # activation layers
        act_layer = nn.GELU

        self.embedding = Embedding(
            in_channels=in_channels,
            embed_channels=enc_channels[0],
            norm_layer=bn_layer,
            act_layer=act_layer,
        )

        # encoder
        enc_drop_path = [
            x.item() for x in torch.linspace(0, drop_path, sum(enc_depths))
        ]
        self.enc = PointSequential()
        for s in range(self.num_stages):
            enc_drop_path_ = enc_drop_path[
                sum(enc_depths[:s]) : sum(enc_depths[: s + 1])
            ]
            enc = PointSequential()
            if s > 0:
                enc.add(
                    SerializedPooling(
                        in_channels=enc_channels[s - 1],
                        out_channels=enc_channels[s],
                        stride=stride[s - 1],
                        norm_layer=bn_layer,
                        act_layer=act_layer,
                    ),
                    name="down",
                )
            for i in range(enc_depths[s]):
                enc.add(
                    Block(
                        channels=enc_channels[s],
                        num_heads=enc_num_head[s],
                        patch_size=enc_patch_size[s],
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        attn_drop=attn_drop,
                        proj_drop=proj_drop,
                        drop_path=enc_drop_path_[i],
                        norm_layer=ln_layer,
                        act_layer=act_layer,
                        pre_norm=pre_norm,
                        order_index=i % len(self.order),
                        cpe_indice_key=f"stage{s}",
                        enable_rpe=enable_rpe,
                        enable_flash=enable_flash,
                        upcast_attention=upcast_attention,
                        upcast_softmax=upcast_softmax,
                    ),
                    name=f"block{i}",
                )
            if len(enc) != 0:
                self.enc.add(module=enc, name=f"enc{s}")

        # decoder
        if not self.enc_mode:
            dec_drop_path = [
                x.item() for x in torch.linspace(0, drop_path, sum(dec_depths))
            ]
            self.dec = PointSequential()
            dec_channels = list(dec_channels) + [enc_channels[-1]]
            for s in reversed(range(self.num_stages - 1)):
                dec_drop_path_ = dec_drop_path[
                    sum(dec_depths[:s]) : sum(dec_depths[: s + 1])
                ]
                dec_drop_path_.reverse()
                dec = PointSequential()
                dec.add(
                    SerializedUnpooling(
                        in_channels=dec_channels[s + 1],
                        skip_channels=enc_channels[s],
                        out_channels=dec_channels[s],
                        norm_layer=bn_layer,
                        act_layer=act_layer,
                    ),
                    name="up",
                )
                for i in range(dec_depths[s]):
                    dec.add(
                        Block(
                            channels=dec_channels[s],
                            num_heads=dec_num_head[s],
                            patch_size=dec_patch_size[s],
                            mlp_ratio=mlp_ratio,
                            qkv_bias=qkv_bias,
                            qk_scale=qk_scale,
                            attn_drop=attn_drop,
                            proj_drop=proj_drop,
                            drop_path=dec_drop_path_[i],
                            norm_layer=ln_layer,
                            act_layer=act_layer,
                            pre_norm=pre_norm,
                            order_index=i % len(self.order),
                            cpe_indice_key=f"stage{s}",
                            enable_rpe=enable_rpe,
                            enable_flash=enable_flash,
                            upcast_attention=upcast_attention,
                            upcast_softmax=upcast_softmax,
                        ),
                        name=f"block{i}",
                    )
                self.dec.add(module=dec, name=f"dec{s}")
        self.seg_head = nn.Linear(dec_channels[0], num_seg_classes)
        
        ########## detection decoder ###########
        self.input_projection = None

        self.det_npoints = det_npoints
        self.det_encoder = MODULES.build(det_encoder)
        self.det_decoder = MODULES.build(det_decoder)
        self.dataset_config = MODULES.build(dataset_config)

        # Projection from encoder space to decoder space
        # if hasattr(self.encoder, "encoder") and hasattr(
        #     self.encoder.encoder, "masking_radius"
        # ):
        #     hidden_dims = [encoder_dim]
        # else:
        
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

        self.num_queries = num_queries
        self.box_processor = BoxProcessor(self.dataset_config)

        # Build MLP heads for box parameter prediction
        self._build_mlp_heads(decoder_dim, mlp_dropout)

        # Build criterion (built directly, not via Pointcept's Criteria wrapper)
        self.det_criterion = LOSSES.build(det_criterion) if det_criterion is not None else None
        self.seg_criterion = LOSSES.build(seg_criterion) if seg_criterion is not None else None

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None
        return xyz, features

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


    def get_query_embeddings(self, encoder_xyz, point_cloud_dims, padding_mask=None):
        """Sample query points via FPS and compute positional embeddings.

        Args:
            encoder_xyz:    (B, N, 3) encoder output coordinates
            point_cloud_dims: [min (B,3), max (B,3)]
            padding_mask:   (B, N) bool or None. True = padded position.
                            Padded positions are moved far away so FPS
                            naturally ignores them.
        """
        if padding_mask is not None:
            # Push padded coords far from real points so FPS won't select them.
            fps_xyz = encoder_xyz.clone()
            fps_xyz[padding_mask] = 1e6
        else:
            fps_xyz = encoder_xyz

        query_inds = furthest_point_sample(fps_xyz, self.num_queries).long()

        if padding_mask is not None:
            # If a scene has fewer real points than num_queries, FPS may select
            # padded positions. Replace those with the first real index per scene.
            on_padded = torch.gather(padding_mask, 1, query_inds)
            if on_padded.any():
                # First real (non-padded) index per scene — always exists.
                first_real = (~padding_mask).long().argmax(dim=1, keepdim=True)
                first_real = first_real.expand_as(query_inds)
                query_inds = torch.where(on_padded, first_real, query_inds)

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



    def forward(self, data_dict):


        point_clouds = data_dict["point_clouds"]

        xyz, features = self._break_up_pc(point_clouds)
        
        point = dense2point(xyz, features)
        point["grid_size"] = self.grid_size 

        point.serialization(order=self.order, shuffle_orders=self.shuffle_orders)
        point.sparsify()

        point = self.embedding(point)
        point = self.enc(point)

        
        ### run detection stuff ###
        dense_xyz, dense_features, padding_mask = point2dense(point)

        # Push padded positions far away so FPS ignores them.
        if padding_mask is not None:
            fps_xyz = dense_xyz.clone()
            fps_xyz[padding_mask] = 1e6
        else:
            fps_xyz = dense_xyz

        fps_inds = furthest_point_sample(fps_xyz, self.det_npoints).long()

        if padding_mask is not None:
            # If npoint exceeds real points in a scene, FPS may select padded
            # positions. Replace those with the first real index per scene.
            on_padded = torch.gather(padding_mask, 1, fps_inds)
            if on_padded.any():
                first_real = (~padding_mask).long().argmax(dim=1, keepdim=True)
                fps_inds = torch.where(on_padded, first_real.expand_as(fps_inds), fps_inds)

        # Gather xyz: (B, npoint, 3)
        det_xyz = torch.gather(
            dense_xyz, 1, fps_inds.unsqueeze(-1).expand(-1, -1, 3)
        )
        # Gather features: (B, C, npoint) from (B, C, max_n)
        det_features = torch.gather(
            dense_features, 2, fps_inds.unsqueeze(1).expand(-1, dense_features.shape[1], -1)
        )
    
        det_padding_mask = None

        if padding_mask is not None:
            det_padding_mask = torch.gather(
                padding_mask,
                1,
                fps_inds
            )
        det_features = det_features.permute(2, 0, 1).contiguous()
        result = self.det_encoder(det_features, xyz=det_xyz, padding_mask=det_padding_mask) 
        
        if isinstance(result, Point):
            # Point-returning encoder (e.g. a PTv3-based encoder component)
            enc_xyz, enc_features_dense, padding_mask = point2dense(result)
            enc_features = enc_features_dense.permute(2, 0, 1)   # → (N'', B, C)
            enc_inds = None
        else:
            enc_xyz, enc_features, enc_inds = result

        # if enc_inds is None:
        #     enc_inds = pre_enc_inds
        # else:
        #     # Encoder downsampled: gather padding_mask to match enc_xyz resolution.
        #     if padding_mask is not None:
        #         padding_mask = torch.gather(padding_mask, 1, enc_inds.type(torch.int64))
        #     if pre_enc_inds is not None:
        #         enc_inds = torch.gather(pre_enc_inds, 1, enc_inds.type(torch.int64))
        
        enc_features = self.encoder_to_decoder_projection(enc_features.permute(1, 2, 0))
        enc_features = enc_features.permute(2, 0, 1)
        
        
        point_cloud_dims = [
            data_dict["point_cloud_dims_min"],
            data_dict["point_cloud_dims_max"],
        ]
        query_xyz, query_embed = self.get_query_embeddings(
            enc_xyz, point_cloud_dims, padding_mask=det_padding_mask
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

        box_predictions = self.get_box_predictions(
            query_xyz, point_cloud_dims, box_features
        )

        point = self.dec(point)
        seg_logits = self.seg_head(point.feat)

        if self.training and self.det_criterion and self.seg_criterion:
            det_loss, _ = self.det_criterion(box_predictions, data_dict)

            seg_target = data_dict["segment"].reshape(-1).long()
            seg_loss = self.seg_criterion(seg_logits, seg_target)
            loss = self.alpha_weight * seg_loss + (1-self.alpha_weight) * det_loss
            return dict(loss=loss)

        if self.training and self.det_criterion is not None:
            det_loss, _ = self.det_criterion(box_predictions, data_dict)
            return dict(loss=det_loss) 
        
        if self.training and self.seg_criterion is not None:
            seg_target = data_dict["segment"].reshape(-1).long()
            seg_loss = self.seg_criterion(seg_logits, seg_target)
            return dict(loss=seg_loss)

        return dict(
            seg_logits=seg_logits,
            pred_boxes=box_predictions["outputs"],
            aux_outputs=box_predictions["aux_outputs"],
        )

        return dict(
            seg_logits=seg_logits,
            pred_boxes=box_predictions,
        )
        
