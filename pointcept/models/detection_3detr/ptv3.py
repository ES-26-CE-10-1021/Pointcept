"""
PTv3 adapter for the 3DETR pre_encoder slot.

Bridges the (xyz, features) dense-tensor interface expected by
Model3DETRDetector.run_encoder to the Point-based interface
expected by PointTransformerV3.

After encoding, PTv3's voxelization and pooling strides produce
variable-length outputs per scene. Two modes are supported:

1. npoint=None (default): returns the Point directly. Downstream,
   point2dense() pads to max scene length and returns a padding mask
   threaded through the decoder's cross-attention.

2. npoint=K: applies FPS on the voxel coordinates to select K
   spatially well-distributed points per scene, returning a fixed-
   length (xyz, features, inds) tuple matching PointnetSAPreEncoder's
   interface. No padding mask is needed downstream.
"""

import torch
from pointcept.models.builder import MODULES
from pointcept.models.point_transformer_v3.point_transformer_v3m1_base import (
    PointTransformerV3,
)
from pointcept.models.detection_3detr.model import dense2point, point2dense
from third_party.pointnet2.pointnet2_utils import furthest_point_sample


@MODULES.register_module("PTv3PreEncoder")
class PTv3PreEncoder(PointTransformerV3):
    """PointTransformerV3 wrapped as a 3DETR pre_encoder.

    Accepts (xyz, features) in dense format, converts to Point,
    runs PTv3 encoder, and optionally applies FPS downsampling.

    Args:
        grid_size (float): Voxel size for PTv3 serialization.
        npoint (int or None): If set, apply FPS after encoding to
            select this many points per scene. Returns a tuple
            (xyz, features, inds) matching PointnetSAPreEncoder.
            If None, returns the Point directly (variable-length).
        **kwargs: Forwarded to PointTransformerV3.__init__().
    """

    def __init__(self, grid_size=0.02, npoint=None, **kwargs):
        kwargs["enc_mode"] = True
        super().__init__(**kwargs)
        self.grid_size = grid_size
        self.npoint = npoint

    def forward(self, xyz, features=None):
        """
        Args:
            xyz:      (B, N, 3) point coordinates
            features: (B, C, N) point features, or None

        Returns:
            If npoint is None:
                Point with encoded features (variable-length scenes).
            If npoint is set:
                xyz:      (B, npoint, 3) FPS-selected coordinates
                features: (B, C, npoint) gathered features
                inds:     (B, npoint) FPS indices
        """
        point = dense2point(xyz, features)
        point["grid_size"] = self.grid_size

        point.serialization(order=self.order, shuffle_orders=self.shuffle_orders)
        point.sparsify()

        point = self.embedding(point)
        point = self.enc(point)

        if self.npoint is None:
            return point

        # Convert variable-length Point to dense padded tensors,
        # then FPS downsample to fixed npoint per scene.
        dense_xyz, dense_features, padding_mask = point2dense(point)

        # Push padded positions far away so FPS ignores them.
        if padding_mask is not None:
            fps_xyz = dense_xyz.clone()
            fps_xyz[padding_mask] = 1e6
        else:
            fps_xyz = dense_xyz

        fps_inds = furthest_point_sample(fps_xyz, self.npoint).long()

        # Gather xyz: (B, npoint, 3)
        out_xyz = torch.gather(
            dense_xyz, 1, fps_inds.unsqueeze(-1).expand(-1, -1, 3)
        )
        # Gather features: (B, C, npoint) from (B, C, max_n)
        out_features = torch.gather(
            dense_features, 2, fps_inds.unsqueeze(1).expand(-1, dense_features.shape[1], -1)
        )

        return out_xyz, out_features, fps_inds
