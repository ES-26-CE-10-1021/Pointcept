"""
PTv3 adapter for the 3DETR pre_encoder slot.

Bridges the (xyz, features) dense-tensor interface expected by
Model3DETRDetector.run_encoder to the Point-based interface
expected by PointTransformerV3.

After encoding, PTv3's voxelization and pooling strides produce
variable-length outputs per scene. The returned Point is converted
to padded dense tensors by point2dense(), which also returns a
padding mask threaded through the 3DETR decoder's cross-attention.
"""

from pointcept.models.builder import MODULES
from pointcept.models.point_transformer_v3.point_transformer_v3m1_base import (
    PointTransformerV3,
)
from pointcept.models.detection_3detr.model import dense2point


@MODULES.register_module("PTv3PreEncoder")
class PTv3PreEncoder(PointTransformerV3):
    """PointTransformerV3 wrapped as a 3DETR pre_encoder.

    Accepts (xyz, features) in dense format, converts to Point,
    runs PTv3 encoder, and returns the Point directly. Variable-length
    scenes are handled downstream by point2dense() which pads to
    max_n and returns a padding mask for the decoder's attention.

    Args:
        grid_size (float): Voxel size for PTv3 serialization.
        **kwargs: Forwarded to PointTransformerV3.__init__().
    """

    def __init__(self, grid_size=0.02, **kwargs):
        kwargs["enc_mode"] = True
        super().__init__(**kwargs)
        self.grid_size = grid_size

    def forward(self, xyz, features=None):
        """
        Args:
            xyz:      (B, N, 3) point coordinates
            features: (B, C, N) point features, or None

        Returns:
            Point with encoded features (variable-length scenes).
        """
        point = dense2point(xyz, features)
        point["grid_size"] = self.grid_size

        point.serialization(order=self.order, shuffle_orders=self.shuffle_orders)
        point.sparsify()

        point = self.embedding(point)
        point = self.enc(point)
        return point


@MODULES.register_module("PTv3UNetPreEncoder")
class PTv3UNetPreEncoder(PointTransformerV3):
    """PTv3 with full U-Net (encoder + decoder) as a 3DETR pre-encoder.

    Unlike PTv3PreEncoder (encoder-only, coarse output), this runs the
    full U-Net so the 3DETR decoder cross-attends to high-resolution
    features with multi-scale context from skip connections.

    Output resolution matches the initial voxel grid (grid_size), giving
    spatially precise coordinates for positional embeddings and FPS
    query generation.

    Args:
        grid_size (float): Voxel size for PTv3 serialization.
        **kwargs: Forwarded to PointTransformerV3.__init__().
    """

    def __init__(self, grid_size=0.02, **kwargs):
        kwargs["enc_mode"] = False  # full U-Net
        super().__init__(**kwargs)
        self.grid_size = grid_size

    def forward(self, xyz, features=None):
        """
        Args:
            xyz:      (B, N, 3) point coordinates
            features: (B, C, N) point features, or None

        Returns:
            Point with decoded features at initial voxel resolution.
        """
        point = dense2point(xyz, features)
        point["grid_size"] = self.grid_size

        point.serialization(order=self.order, shuffle_orders=self.shuffle_orders)
        point.sparsify()

        point = self.embedding(point)
        point = self.enc(point)
        point = self.dec(point)
        return point
