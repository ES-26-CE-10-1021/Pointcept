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

The module also provides ``PTv3m3PreEncoder``, which wraps the
``PT-v3m3`` (Utonia) variant as a frozen, pretrained VFM feature
extractor. Checkpoint weights are pulled from HuggingFace on first
use and never updated during training.
"""

import logging

import torch
from pointcept.models.builder import MODELS, MODULES
from pointcept.models.point_transformer_v3.point_transformer_v3m1_base import (
    PointTransformerV3,
)
from pointcept.models.point_transformer_v3.point_transformer_v3m3_utonia import (
    PointTransformerV3 as PointTransformerV3m3,
)
from pointcept.models.detection_3detr.model import dense2point, point2dense
from third_party.pointnet2.pointnet2_utils import furthest_point_sample

_logger = logging.getLogger(__name__)


import torch_scatter
from pointcept.models.point_transformer_v3.point_transformer_v3m1_base import (
    SerializedPooling,
)

"""
PTv3 adapter for the 3DETR pre_encoder slot.
...
"""

import logging

import torch
import torch_scatter
from pointcept.models.builder import MODELS, MODULES
from pointcept.models.point_transformer_v3.point_transformer_v3m1_base import (
    PointTransformerV3,
    SerializedPooling,
)
from pointcept.models.point_transformer_v3.point_transformer_v3m3_utonia import (
    PointTransformerV3 as PointTransformerV3m3,
)
from pointcept.models.detection_3detr.model import dense2point, point2dense
from pointcept.models.utils.misc import offset2bincount
from third_party.pointnet2.pointnet2_utils import furthest_point_sample

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DINO helpers
# ---------------------------------------------------------------------------
def _flatten_dino(xyz, dino_feat):
    """
    Args:
        xyz:       (B, N, 3)
        dino_feat: (B, D, N)

    Returns:
        (B*N, D)
    """
    # print("incoming dino", dino_feat.shape)
    B, N, D = dino_feat.shape

    return (
        dino_feat
        .reshape(B * N, D)
        .contiguous()
    )

def _pack_voxel_keys(keys, sparse_shape):
    """Encode (batch, gx, gy, gz) rows as unique int64 scalars.

    Uses the spatial strides implied by sparse_shape so that every valid
    (batch, gx, gy, gz) combination maps to a distinct integer with no
    collision.  Values stay well within int64 range for any realistic
    sparse_shape (grid coords are bounded by sparse_shape which is at
    most a few thousand in each dimension).

    Args:
        keys:         (N, 4) int32/int64 — columns: [batch, gx, gy, gz]
        sparse_shape: sequence of 3 ints [X, Y, Z] from point.sparse_shape

    Returns:
        (N,) int64 packed keys
    """
    sx, sy, sz = int(sparse_shape[0]), int(sparse_shape[1]), int(sparse_shape[2])
    b = keys[:, 0].long()
    x = keys[:, 1].long()
    y = keys[:, 2].long()
    z = keys[:, 3].long()
    return b * (sx * sy * sz) + x * (sy * sz) + y * sz + z


def _raw_to_voxel_cluster(point):
    """Map each raw point to its voxel row index in sparse_conv_feat.

    Must be called after point.sparsify() so that sparse_conv_feat.indices
    (the ground-truth voxel key table in spconv's internal order) exists.

    Uses coordinate-based hashing against sparse_conv_feat.indices rather
    than assuming any particular sort order from spconv or torch.unique.

    Args:
        point: Point after sparsify(), carrying .batch, .grid_coord,
               .sparse_shape, .sparse_conv_feat.

    Returns:
        cluster: (N_raw,) long — cluster[i] = row in sparse_conv_feat
                 that raw point i belongs to.
    """
    sparse_shape = point.sparse_shape           # [X, Y, Z]
    vox_keys = point.sparse_conv_feat.indices   # (N_vox, 4) int32: [b,x,y,z]
    raw_keys = torch.cat(
        [point.batch.unsqueeze(1).int(), point.grid_coord.int()], dim=1
    )                                           # (N_raw, 4)

    vox_packed = _pack_voxel_keys(vox_keys, sparse_shape)   # (N_vox,)
    raw_packed = _pack_voxel_keys(raw_keys, sparse_shape)   # (N_raw,)

    # Build a lookup table: packed_key → voxel row index.
    # vox_packed is unique by construction (spconv guarantees one entry
    # per unique voxel).  We size the table by the max packed key value.
    n_vox = vox_packed.shape[0]
    max_key = int(vox_packed.max().item()) + 1
    lookup = vox_packed.new_zeros(max_key)      # int64, zero-initialised
    lookup[vox_packed] = torch.arange(
        n_vox, device=vox_packed.device, dtype=vox_packed.dtype
    )

    return lookup[raw_packed]                   # (N_raw,) in [0, N_vox)


def _scatter_to_dense(dino_flat, point, n_max):
    """Scatter flat (N_total, D) dino features into (B, n_max, D) dense layout.

    Fully vectorised.  Padded positions are zero-filled.

    Args:
        dino_flat: (N_total, D)
        point:     encoded Point with .offset (length-B cumulative counts)
        n_max:     int — padded sequence length (= max scene length in batch)

    Returns:
        (B, n_max, D)
    """
    bincount = offset2bincount(point.offset)    # (B,)
    B = int(bincount.shape[0])
    N_total = int(dino_flat.shape[0])
    D = int(dino_flat.shape[1])
    device = dino_flat.device

    # scene_idx[i] = which scene raw/enc point i belongs to
    scene_idx = torch.repeat_interleave(
        torch.arange(B, device=device), bincount
    )                                           # (N_total,)

    # local_idx[i] = position of point i within its scene (0-based)
    starts = torch.cat(
        [bincount.new_zeros(1), bincount.cumsum(0)[:-1]]
    )                                           # (B,) first global idx per scene
    local_idx = (
        torch.arange(N_total, device=device)
        - torch.repeat_interleave(starts, bincount)
    )                                           # (N_total,) in [0, n_i)

    flat_idx = scene_idx * n_max + local_idx   # (N_total,) into (B*n_max,)

    out = dino_flat.new_zeros(B * n_max, D)
    out.scatter_(0, flat_idx.unsqueeze(1).expand(-1, D), dino_flat)
    return out.view(B, n_max, D)


# ---------------------------------------------------------------------------
# Mixin
# ---------------------------------------------------------------------------

class PTv3DinoMixin:
    """Threads dino_feat through voxelization and encoder pooling.

    Subclasses must implement:
      _build_point(xyz, features) -> Point
          Build and serialise the Point up to (but not including) sparsify().
          grid_coord must exist on return (serialization() computes it).
      _forward_from_point(point) -> Point
          Run sparsify() + embedding + enc (+ optional dec). FPS tail omitted.

    Call _register_pooling_hooks() once at the end of __init__.

    Voxelization alignment
    ----------------------
    After point.sparsify() runs, sparse_conv_feat.indices contains the
    (batch, gx, gy, gz) of every voxel in spconv's actual internal order.
    _raw_to_voxel_cluster() matches each raw point's grid_coord to its
    voxel row by hashing, avoiding any assumption about spconv's sort order.

    SerializedPooling stages
    ------------------------
    A forward hook on each SerializedPooling module mirrors the same
    scatter_mean onto point.dino_feat using the pooling_inverse cluster
    map that SerializedPooling writes onto the output Point.  This keeps
    dino_feat aligned with point.feat through every downsampling stage.
    """

    def _register_pooling_hooks(self):
        """Attach a forward hook to every SerializedPooling in self.enc."""
        self._pooling_hooks = []

        def _make_hook():
            def hook(module, inputs, output):
                parent = output.get("pooling_parent")
                if parent is None or "dino_feat" not in parent:
                    return
                inv = output["pooling_inverse"]       # (N_parent,)
                n_clusters = output["feat"].shape[0]
                output["dino_feat"] = torch_scatter.scatter(
                    parent["dino_feat"], 
                    inv[:, None].expand(-1, parent["dino_feat"].shape[1]),
                    dim=0, 
                    dim_size=n_clusters, 
                    reduce="mean",
                )                                     # (N_clusters, D)
            return hook

        for _, module in self.enc.named_modules():
            if isinstance(module, SerializedPooling):
                self._pooling_hooks.append(
                    module.register_forward_hook(_make_hook())
                )

    @staticmethod
    def _gather_dino_fps(dino_out, point, n_max, fps_inds):
        """Gather dino at FPS indices → (B, D, npoint)."""
        dino_dense = _scatter_to_dense(dino_out, point, n_max)  # (B, n_max, D)
        gathered = torch.gather(
            dino_dense, 1,
            fps_inds.unsqueeze(-1).expand(-1, -1, dino_dense.shape[-1]),
        )                                                        # (B, npoint, D)
        return gathered.permute(0, 2, 1).contiguous()           # (B, D, npoint)

    def forward_with_dino(self, xyz, features=None, dino_feat=None):
        """Forward pass threading dino_feat through the full encoder pipeline.

        Args:
            xyz:       (B, N, 3)
            features:  (B, C, N) or None
            dino_feat: (B, D, N) or None

        Returns:
            npoint is None:
                (point, dino_dense)
                  point:      encoded Point at encoder/decoder resolution
                  dino_dense: (B, N_enc, D) aligned with point2dense, or None
            npoint is set:
                (out_xyz, out_features, fps_inds, dino_fps)
                  out_xyz:      (B, npoint, 3)
                  out_features: (B, C, npoint)
                  fps_inds:     (B, npoint)
                  dino_fps:     (B, D, npoint) or None
        """
        point = self._build_point(xyz, features)

        if dino_feat is not None:
            point["dino_feat"] = _flatten_dino(xyz, dino_feat)
            # point["dino_feat"] = point["dino_feat"].transpose(0, 1).contiguous()

        # _orig_sparsify = point.sparsify
        #
        # def _patched_sparsify():
        #     _orig_sparsify()
        #     if "dino_feat" not in point:
        #         return
        #     # _raw_to_voxel_cluster reads sparse_conv_feat.indices which
        #     # now exists (set by _orig_sparsify above).  This is the only
        #     # safe moment to derive the cluster map: raw grid_coord is still
        #     # present, and spconv's internal voxel ordering is now fixed.
        #     cluster = _raw_to_voxel_cluster(point)  # (N_raw,)
        #     n_vox = point.sparse_conv_feat.features.shape[0]
        #     point["dino_feat"] = torch_scatter.scatter(
        #         point["dino_feat"], cluster,
        #         dim=0, dim_size=n_vox, reduce="mean",
        #     )                                       # (N_vox, D)
        #
        # point.sparsify = _patched_sparsify
        #
        # point = self._forward_from_point(point)

        point.sparsify()

        if "dino_feat" in point:
            cluster = _raw_to_voxel_cluster(point)

            n_vox = point.sparse_conv_feat.features.shape[0]
            # print("before scatter", point["dino_feat"].shape)
            # print("cluster", cluster.shape)
            point["dino_feat"] = torch_scatter.scatter(
                point["dino_feat"],
                # cluster.unsqueeze(0).expand(point["dino_feat"].shape[0], -1),
                cluster,
                dim=0,
                dim_size=n_vox,
                reduce="mean",
            )
        

        # print("point.feat", point.feat.shape)
        # print("dino_feat", point["dino_feat"].shape if "dino_feat" in point else None)
        # print("embedding expects", self.embedding.in_channels)    
        point = self.embedding(point)
        point = self.enc(point)

        # ---- Extract pooled dino_feat ----
        dino_out = point.get("dino_feat", None)     # (N_enc_total, D) or None

        if self.npoint is None:
            dense_xyz, _, _ = point2dense(point)
            dino_dense = (
                _scatter_to_dense(dino_out, point, dense_xyz.shape[1])
                if dino_out is not None else None
            )
            return point, dino_dense

        # FPS path — shares fps_inds across xyz, features, and dino.
        dense_xyz, dense_features, padding_mask = point2dense(point)

        fps_xyz = dense_xyz.clone()
        if padding_mask is not None:
            fps_xyz[padding_mask] = 1e6

        fps_inds = furthest_point_sample(fps_xyz, self.npoint).long()

        if padding_mask is not None:
            on_padded = torch.gather(padding_mask, 1, fps_inds)
            if on_padded.any():
                first_real = (~padding_mask).long().argmax(dim=1, keepdim=True)
                fps_inds = torch.where(
                    on_padded, first_real.expand_as(fps_inds), fps_inds
                )

        out_xyz = torch.gather(
            dense_xyz, 1,
            fps_inds.unsqueeze(-1).expand(-1, -1, 3),
        )
        out_features = torch.gather(
            dense_features, 2,
            fps_inds.unsqueeze(1).expand(-1, dense_features.shape[1], -1),
        )
        dino_fps = (
            self._gather_dino_fps(dino_out, point, dense_xyz.shape[1], fps_inds)
            if dino_out is not None else None
        )

        return out_xyz, out_features, fps_inds, dino_fps

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
        # PTv3 was designed with feat = [xyz, rgb, ...] (matching its semseg
        # pretraining). When dense features are provided, concat xyz so the
        # embedding sees the full channel layout (in_channels = 3 + features.C).
        # When features=None, dense2point falls back to feat = xyz.
        if features is not None:
            features = torch.cat(
                [xyz.transpose(1, 2).contiguous(), features], dim=1
            )
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

        if padding_mask is not None:
            # If npoint exceeds real points in a scene, FPS may select padded
            # positions. Replace those with the first real index per scene.
            on_padded = torch.gather(padding_mask, 1, fps_inds)
            if on_padded.any():
                first_real = (~padding_mask).long().argmax(dim=1, keepdim=True)
                fps_inds = torch.where(on_padded, first_real.expand_as(fps_inds), fps_inds)

        # Gather xyz: (B, npoint, 3)
        out_xyz = torch.gather(
            dense_xyz, 1, fps_inds.unsqueeze(-1).expand(-1, -1, 3)
        )
        # Gather features: (B, C, npoint) from (B, C, max_n)
        out_features = torch.gather(
            dense_features, 2, fps_inds.unsqueeze(1).expand(-1, dense_features.shape[1], -1)
        )

        return out_xyz, out_features, fps_inds


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
        # See PTv3PreEncoder.forward: prepend xyz to feat to match PTv3's
        # [xyz, rgb, ...] embedding convention when features are provided.
        if features is not None:
            features = torch.cat(
                [xyz.transpose(1, 2).contiguous(), features], dim=1
            )
        point = dense2point(xyz, features)
        point["grid_size"] = self.grid_size

        point.serialization(order=self.order, shuffle_orders=self.shuffle_orders)
        point.sparsify()

        point = self.embedding(point)
        point = self.enc(point)
        point = self.dec(point)
        return point


def _ddp_safe_utonia_load(pretrained, download_root):
    """Download the Utonia checkpoint in a DDP-safe manner.

    Rank 0 triggers the (potentially slow) HuggingFace download while all
    other ranks wait at a barrier. Once rank 0 has populated the local
    cache, every rank re-invokes ``load(ckpt_only=True)`` to read from the
    cached copy — this avoids file-lock contention under
    ``hf_hub_download``'s internal locking, which is known to hang under
    sustained multi-process races.
    """
    # Lazy import: importing third_party.utonia.utonia.model pulls in
    # spconv / torch_scatter / timm / huggingface_hub / flash_attn at
    # module import time. Deferring until actually loading a checkpoint
    # keeps this file importable in environments without third_party set
    # up (e.g. unit tests that use ``pretrained=None``).
    from third_party.utonia.utonia.model import load as utonia_load

    import torch.distributed as dist

    ddp = dist.is_available() and dist.is_initialized()
    if ddp:
        if dist.get_rank() == 0:
            utonia_load(
                name=pretrained,
                download_root=download_root,
                ckpt_only=True,
            )
        dist.barrier()
    return utonia_load(
        name=pretrained,
        download_root=download_root,
        ckpt_only=True,
    )


_FREEZE_MODES = ("enc", "enc_finetune", "none")


@MODELS.register_module("PTv3m3PreEncoder")
@MODULES.register_module("PTv3m3PreEncoder")
class PTv3m3PreEncoder(PointTransformerV3m3):
    """PT-v3m3 (Utonia) wrapped as a pretrained 3DETR pre_encoder.

    Loads a pretrained Utonia checkpoint from HuggingFace and runs it as
    a feature extractor. Supports both encoder-only (``enc_mode=True``)
    and full U-Net (``enc_mode=False``) operation, and optional FPS
    downsampling of the output to a fixed token count.

    Detection datasets provide XYZ (and optionally RGB), whereas Utonia
    was pretrained on a 9-dim ``[xyz, rgb, normal]`` input. Missing
    modalities are zero-padded on-device; Causal Modality Blinding makes
    Utonia tolerate absent modalities as long as their channels are
    explicit zeros.

    **Freeze semantics** are controlled by ``freeze_backbone``:

    - ``"enc"``: embedding + encoder are frozen (``requires_grad=False``,
      held in ``eval()`` mode, forward under ``torch.no_grad()``). The
      decoder, when present (``enc_mode=False``), remains trainable — the
      encoder output leaf is detached + ``requires_grad_(True)`` so the
      decoder can build a fresh graph. Default.
    - ``"enc_finetune"``: embedding + all-but-last encoder stages frozen;
      the **last encoder stage** remains trainable. The frozen prefix runs
      under ``no_grad`` for VRAM, then the leaf is bridged and the last
      stage (and decoder, if present) build a real graph.
    - ``"none"``: nothing frozen; full gradient flow through the backbone.

    Note: the distributed Utonia checkpoint is encoder-only (``dec_*``
    fields are ``None``). With ``enc_mode=False`` you are building and
    training a **fresh, randomly-initialized** decoder on top of the
    frozen encoder — supply ``dec_*`` kwargs explicitly in your config.

    Args:
        pretrained (str | None): HuggingFace model name (e.g. ``"utonia"``)
            or a local ``.pth`` path. If ``None`` / falsy, the backbone is
            built from the user's config without loading any checkpoint
            (useful for unit tests that don't require HF access).
        download_root (str | None): forwarded to the Utonia ``load()``
            helper; defaults to ``~/.cache/utonia/ckpt``.
        config_overrides (dict | None): merged over the checkpoint's
            recorded config before instantiating the backbone. Cannot
            change ``in_channels`` (which would desync the embedding
            weight shape from the checkpoint).
        freeze_backbone (str): one of ``"enc"``, ``"enc_finetune"``,
            ``"none"``. See above.
        grid_size (float): voxel size used to serialize the input point
            cloud before running the backbone.
        npoint (int | None): if set, apply FPS on the backbone output to
            select this many points per scene, returning a fixed-length
            ``(xyz, features, inds)`` tuple matching
            ``PointnetSAPreEncoder``'s interface. If ``None``, returns the
            variable-length ``Point`` directly.
        enc_mode (bool): if True (default), build and run only the encoder
            pyramid; the deepest-stage output is returned. If False, build
            and run the full U-Net; the shallowest decoder stage output is
            returned. Overrides the checkpoint config's ``enc_mode``.
        **overrides: any remaining kwargs are merged into the checkpoint
            config (same precedence as ``config_overrides``).
    """

    def __init__(
        self,
        pretrained="utonia",
        download_root=None,
        config_overrides=None,
        freeze_backbone="enc",
        grid_size=0.02,
        npoint=None,
        enc_mode=True,
        **overrides,
    ):
        if freeze_backbone not in _FREEZE_MODES:
            raise ValueError(
                f"PTv3m3PreEncoder: freeze_backbone={freeze_backbone!r} "
                f"is not one of {_FREEZE_MODES}."
            )

        # Merge priority (highest wins): **overrides > config_overrides >
        # checkpoint config > backbone defaults.
        ckpt_state_dict = None
        ckpt_in_channels = None
        ckpt_enc_mode = None
        if pretrained:
            ckpt = _ddp_safe_utonia_load(pretrained, download_root)
            base_config = dict(ckpt["config"])
            ckpt_state_dict = ckpt["state_dict"]
            ckpt_in_channels = base_config.get("in_channels")
            ckpt_enc_mode = base_config.get("enc_mode")
        else:
            base_config = {}

        if config_overrides:
            base_config.update(config_overrides)
        if overrides:
            base_config.update(overrides)

        # Guard: the embedding's in_channels must track the checkpoint
        # exactly — otherwise the first Linear will fail at load_state_dict.
        if ckpt_in_channels is not None:
            merged_in = base_config.get("in_channels", ckpt_in_channels)
            if merged_in != ckpt_in_channels:
                raise ValueError(
                    f"PTv3m3PreEncoder: config_overrides changed in_channels "
                    f"from {ckpt_in_channels} (checkpoint) to {merged_in}. "
                    f"This would desync the embedding layer from the "
                    f"checkpoint weights — refusing to proceed."
                )

        # enc_mode is the one architecture knob we pin from the wrapper
        # signature so downstream configs can flip it without reaching
        # into config_overrides. freeze_encoder is wired from the high-
        # level freeze_backbone enum; "enc_finetune" still freezes via
        # the base class, then we unfreeze the last stage below.
        base_config["enc_mode"] = bool(enc_mode)
        base_config["freeze_encoder"] = freeze_backbone in ("enc", "enc_finetune")

        super().__init__(**base_config)

        self.grid_size = grid_size
        self.npoint = npoint
        self.freeze_backbone = freeze_backbone
        self._pretrained = pretrained
        # Stash the ckpt's original enc_mode so _load_pretrained_state can
        # recognize "ckpt was enc-only, user built full U-Net" and treat
        # the missing dec.* keys as expected (fresh decoder).
        self._ckpt_enc_mode = ckpt_enc_mode

        if ckpt_state_dict is not None:
            self._load_pretrained_state(ckpt_state_dict)

        # "enc_finetune": undo the base class's freeze on the last encoder
        # stage so it trains end-to-end with the downstream heads.
        if self.freeze_backbone == "enc_finetune":
            for p in self.enc[-1].parameters():
                p.requires_grad = True

        # Put the frozen blocks into eval mode immediately so a caller
        # that never calls .train() still sees deterministic features.
        self._apply_freeze_eval()

    def _load_pretrained_state(self, state_dict):
        """Load a Utonia state_dict into this pointcept-side PT-v3m3.

        The upstream and pointcept-side ``PointTransformerV3`` classes
        are structurally equivalent but defined in separate files, so
        they can drift. We try ``strict=True`` first for a clean signal,
        and fall back to ``strict=False`` — with a warning that lists
        the dropped keys — if the checkpoint carries extras (e.g.
        pretraining heads). Missing keys remain a hard error: those
        indicate real structural divergence.
        """
        try:
            self.load_state_dict(state_dict, strict=True)
            _logger.info(
                "PTv3m3PreEncoder: loaded Utonia checkpoint '%s' with zero "
                "missing / unexpected keys.",
                self._pretrained,
            )
            return
        except RuntimeError:
            incompatible = self.load_state_dict(state_dict, strict=False)

        missing = list(incompatible.missing_keys)
        unexpected = list(incompatible.unexpected_keys)

        # Fresh-decoder configs (ckpt saved with enc_mode=True, user built
        # with enc_mode=False) will report every dec.* parameter as missing.
        # That's intentional — the decoder is trained from scratch — so drop
        # them from the missing list and log at INFO instead of raising.
        dec_missing = []
        if (
            self._ckpt_enc_mode is True
            and not self.enc_mode
            and hasattr(self, "dec")
        ):
            dec_missing = [k for k in missing if k.startswith("dec.")]
            missing = [k for k in missing if not k.startswith("dec.")]
            if dec_missing:
                _logger.info(
                    "PTv3m3PreEncoder: %d decoder key(s) missing from the "
                    "checkpoint — decoder will train from random "
                    "initialization: %s%s",
                    len(dec_missing),
                    dec_missing[:4],
                    " ..." if len(dec_missing) > 4 else "",
                )

        if missing:
            raise RuntimeError(
                f"PTv3m3PreEncoder: {len(missing)} key(s) missing from the "
                f"Utonia checkpoint but required by PT-v3m3: "
                f"{missing[:8]}{' ...' if len(missing) > 8 else ''}"
            )
        if unexpected:
            _logger.warning(
                "PTv3m3PreEncoder: dropping %d unexpected key(s) from the "
                "Utonia checkpoint (likely pretraining-head buffers): %s%s",
                len(unexpected),
                unexpected[:8],
                " ..." if len(unexpected) > 8 else "",
            )
        else:
            _logger.info(
                "PTv3m3PreEncoder: loaded Utonia checkpoint '%s' with zero "
                "missing / unexpected keys.",
                self._pretrained,
            )

    def _apply_freeze_eval(self):
        """Force frozen submodules into eval() per the freeze enum.

        The default ``nn.Module.train()`` recursively flips every
        submodule, which would re-enable dropout / drop_path inside the
        frozen backbone on every step — nondeterministic features for
        identical inputs. Under ``"enc_finetune"`` the last encoder stage
        must follow the outer module's train/eval state.
        """
        if self.freeze_backbone in ("enc", "enc_finetune"):
            self.embedding.eval()
            self.enc.eval()
        if self.freeze_backbone == "enc_finetune":
            self.enc[-1].train(self.training)

    def train(self, mode=True):
        """Override ``train()`` so the frozen blocks stay in eval mode."""
        super().train(mode)
        self._apply_freeze_eval()
        return self

    def _build_padded_feat(self, xyz, features):
        """Assemble the (B, target_c, N) feature tensor on-device.

        Channels laid out as:
            0..2                : xyz (always present)
            3..(3+Cf-1)         : user-provided features (RGB, etc.)
            (3+Cf)..(target_c-1): zero-padded — normals and any further
                                  modalities that the detection dataset
                                  never carries.

        ``target_c`` is read at runtime from the checkpoint-driven
        ``self.embedding.in_channels`` so it tracks whatever hyperparameter
        the checkpoint was trained with, without hardcoding 9.

        The zero pad is allocated with ``xyz.new_zeros`` so device and
        dtype are inherited directly — building it on CPU and moving it
        would force a CPU→GPU sync on every forward step.
        """
        B, N, _ = xyz.shape
        target_c = self.embedding.in_channels
        # (B, 3, N) xyz channels
        parts = [xyz.transpose(1, 2).contiguous()]
        used = 3
        if features is not None:
            feat_c = features.shape[1]
            if used + feat_c > target_c:
                raise ValueError(
                    f"PTv3m3PreEncoder: dense features contribute {feat_c} "
                    f"channel(s) on top of xyz ({used + feat_c} total) but "
                    f"the backbone's embedding only accepts {target_c}. "
                    f"Truncation is unsafe — adjust the dataset or pick a "
                    f"checkpoint with more input channels."
                )
            parts.append(features)
            used += feat_c
        if used < target_c:
            pad_c = target_c - used
            parts.append(xyz.new_zeros(B, pad_c, N))
        return torch.cat(parts, dim=1)

    def _bridge_leaf(self, point):
        """Detach point.feat and (in training) flip requires_grad on.

        Used after a no_grad section so the downstream graph starts
        fresh at this leaf — gradients stop here and never touch the
        frozen parameters.
        """
        point.feat = point.feat.detach()
        if self.training:
            point.feat.requires_grad_(True)

    @staticmethod
    def _clear_subm_pair_cache(point):
        """Walk the pooling-parent chain and drop cached SubMConv indice pairs.

        Each ``SparseConvTensor`` caches indice-pair tables in
        ``indice_dict[indice_key]`` for SubMConv reuse. When the encoder's
        first SubMConv at a stage ran under ``torch.no_grad()`` (or with
        all-frozen inputs/weights so autograd's Function wasn't engaged),
        the cached pair carries no backward indices. The decoder later
        reuses the same key (``stage{s}``) at the same resolution; backward
        then hits ``!indices.empty()`` in ``implicit_gemm_backward``.

        Clearing the cache before the dec runs forces fresh pair generation
        through spconv's autograd.Function (since the dec has trainable
        params), which populates both forward and backward indices.
        """
        cur = point
        while cur is not None:
            sct = getattr(cur, "sparse_conv_feat", None)
            if sct is not None and getattr(sct, "indice_dict", None):
                sct.indice_dict.clear()
            cur = cur.get("pooling_parent") if "pooling_parent" in cur.keys() else None

    def forward_enc_dec_split(self, xyz, features=None):
        """Multi-task entry point: split encoder bottleneck from decoder output.

        Mirrors :meth:`forward` up to the ``self.dec`` call but, instead of
        running the full encoder→decoder→FPS pipeline in one shot, snapshots
        the encoder bottleneck (xyz / features / padding mask) **before**
        ``self.dec`` mutates the bottleneck Point's pooling-parent chain.

        Required by ``MultiTask3DETRSegmentor`` so the detection branch can
        consume the bottleneck (after an external FPS) while the seg branch
        consumes the decoder output (unpooled back to root resolution).

        Args:
            xyz:      (B, N, 3) point coordinates.
            features: (B, C, N) point features, or ``None``.

        Returns:
            Tuple ``(enc_xyz, enc_features, padding_mask, dec_point)``:
              - ``enc_xyz``:      (B, N', 3) bottleneck coordinates (padded).
              - ``enc_features``: (N', B, C_enc) transformer convention.
              - ``padding_mask``: (B, N') bool, or ``None``.
              - ``dec_point``:    Point at first-encoder-stage resolution
                                  with the unpool parent chain still attached.
        """
        assert self.npoint is None, (
            "forward_enc_dec_split requires npoint=None — the multi-task "
            "model applies its own FPS to the bottleneck."
        )
        assert not self.enc_mode, (
            "forward_enc_dec_split requires enc_mode=False — the seg branch "
            "consumes the decoder output."
        )

        feat_padded = self._build_padded_feat(xyz, features)
        point = dense2point(xyz, feat_padded)
        point["grid_size"] = self.grid_size

        if self.freeze_backbone == "enc_finetune":
            with torch.no_grad():
                point.serialization(
                    order=self.order, shuffle_orders=self.shuffle_orders
                )
                point.sparsify()
                point = self.embedding(point)
                for _, stage in list(self.enc.named_children())[:-1]:
                    point = stage(point)
            self._bridge_leaf(point)
            point = self.enc[-1](point)
        elif self.freeze_backbone == "enc":
            with torch.no_grad():
                point.serialization(
                    order=self.order, shuffle_orders=self.shuffle_orders
                )
                point.sparsify()
                point = self.embedding(point)
                point = self.enc(point)
            # Decoder follows (enc_mode=False) so bridge here to start a
            # fresh graph for the trainable dec.
            self._bridge_leaf(point)
        else:  # "none"
            point.serialization(
                order=self.order, shuffle_orders=self.shuffle_orders
            )
            point.sparsify()
            point = self.embedding(point)
            point = self.enc(point)

        # Snapshot the bottleneck NOW. self.dec() will mutate point by
        # popping its pooling_parent / pooling_inverse fields.
        enc_xyz, enc_features_dense, padding_mask = point2dense(point)
        enc_features = enc_features_dense.permute(2, 0, 1).contiguous()  # (N', B, C)

        # Drop the encoder's cached SubMConv indice pairs from every
        # pooling-parent SparseConvTensor so the dec regenerates pairs
        # through spconv's autograd path (which populates the backward
        # indices implicit_gemm_backward needs). See _clear_subm_pair_cache.
        if self.freeze_backbone in ("enc", "enc_finetune"):
            self._clear_subm_pair_cache(point)

        dec_point = self.dec(point)
        return enc_xyz, enc_features, padding_mask, dec_point

    def forward(self, xyz, features=None):
        """
        Args:
            xyz:      (B, N, 3) point coordinates
            features: (B, C, N) point features, or None

        Returns:
            If ``npoint`` is None:
                Point with backbone features (variable-length per scene).
            If ``npoint`` is set:
                xyz:      (B, npoint, 3) FPS-selected coordinates
                features: (B, C, npoint) gathered features
                inds:     (B, npoint) FPS indices
        """
        feat_padded = self._build_padded_feat(xyz, features)
        point = dense2point(xyz, feat_padded)
        point["grid_size"] = self.grid_size

        if self.freeze_backbone == "enc_finetune":
            # Frozen prefix (embedding + all-but-last enc stages) under
            # no_grad for VRAM; bridge the leaf; trainable last stage
            # builds a fresh graph.
            with torch.no_grad():
                point.serialization(
                    order=self.order, shuffle_orders=self.shuffle_orders
                )
                point.sparsify()
                point = self.embedding(point)
                for _, stage in list(self.enc.named_children())[:-1]:
                    point = stage(point)
            self._bridge_leaf(point)
            point = self.enc[-1](point)
        elif self.freeze_backbone == "enc":
            with torch.no_grad():
                point.serialization(
                    order=self.order, shuffle_orders=self.shuffle_orders
                )
                point.sparsify()
                point = self.embedding(point)
                point = self.enc(point)
            # If a trainable decoder follows, bridge here so it builds a
            # graph. Otherwise (enc_mode=True) bridge *after* this block
            # so the final output leaf carries requires_grad for the
            # downstream 3DETR layers.
            if not self.enc_mode:
                self._bridge_leaf(point)
        else:  # "none"
            point.serialization(
                order=self.order, shuffle_orders=self.shuffle_orders
            )
            point.sparsify()
            point = self.embedding(point)
            point = self.enc(point)

        # Decoder (when present) is always trainable — fresh weights on
        # top of whatever the encoder produced. Drop the encoder's cached
        # SubMConv indice pairs first so the dec regenerates pairs through
        # spconv's autograd path (populates the backward indices that
        # implicit_gemm_backward needs). See _clear_subm_pair_cache.
        if not self.enc_mode:
            if self.freeze_backbone in ("enc", "enc_finetune"):
                self._clear_subm_pair_cache(point)
            point = self.dec(point)

        # Encoder-only + "enc": nothing has built a graph on point.feat
        # yet. Bridge so downstream 3DETR layers can attach.
        if self.freeze_backbone == "enc" and self.enc_mode:
            self._bridge_leaf(point)

        if self.npoint is None:
            return point

        # FPS downsample — mirror PTv3PreEncoder.forward's pattern exactly.
        dense_xyz, dense_features, padding_mask = point2dense(point)

        if padding_mask is not None:
            fps_xyz = dense_xyz.clone()
            fps_xyz[padding_mask] = 1e6
        else:
            fps_xyz = dense_xyz

        fps_inds = furthest_point_sample(fps_xyz, self.npoint).long()

        if padding_mask is not None:
            on_padded = torch.gather(padding_mask, 1, fps_inds)
            if on_padded.any():
                first_real = (~padding_mask).long().argmax(dim=1, keepdim=True)
                fps_inds = torch.where(
                    on_padded, first_real.expand_as(fps_inds), fps_inds
                )

        out_xyz = torch.gather(
            dense_xyz, 1, fps_inds.unsqueeze(-1).expand(-1, -1, 3)
        )
        out_features = torch.gather(
            dense_features,
            2,
            fps_inds.unsqueeze(1).expand(-1, dense_features.shape[1], -1),
        )

        return out_xyz, out_features, fps_inds
@MODELS.register_module("PTv3PreEncoderWithDino")
@MODULES.register_module("PTv3PreEncoderWithDino")
class PTv3PreEncoderWithDino(PTv3DinoMixin, PTv3PreEncoder):
    """PTv3PreEncoder that threads pooled DINO features through the encoder."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._register_pooling_hooks()

    def _build_point(self, xyz, features):
        if features is not None:
            features = torch.cat(
                [xyz.transpose(1, 2).contiguous(), features], dim=1
            )
        point = dense2point(xyz, features)
        point["grid_size"] = self.grid_size
        point.serialization(order=self.order, shuffle_orders=self.shuffle_orders)
        return point

    def _forward_from_point(self, point):
        point.sparsify()
        point = self.embedding(point)
        point = self.enc(point)
        return point


@MODULES.register_module("PTv3m3PreEncoderWithDino")
class PTv3m3PreEncoderWithDino(PTv3DinoMixin, PTv3m3PreEncoder):
    """PTv3m3PreEncoder (Utonia) that threads pooled DINO features.

    All freeze_backbone semantics preserved from PTv3m3PreEncoder.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._register_pooling_hooks()

    def _build_point(self, xyz, features):
        feat_padded = self._build_padded_feat(xyz, features)
        point = dense2point(xyz, feat_padded)
        point["grid_size"] = self.grid_size
        point.serialization(order=self.order, shuffle_orders=self.shuffle_orders)
        return point

    def _forward_from_point(self, point):
        if self.freeze_backbone == "enc_finetune":
            with torch.no_grad():
                point.sparsify()
                point = self.embedding(point)
                for _, stage in list(self.enc.named_children())[:-1]:
                    point = stage(point)
            self._bridge_leaf(point)
            point = self.enc[-1](point)

        elif self.freeze_backbone == "enc":
            with torch.no_grad():
                point.sparsify()
                point = self.embedding(point)
                point = self.enc(point)
            if not self.enc_mode:
                self._bridge_leaf(point)

        else:  # "none"
            point.sparsify()
            point = self.embedding(point)
            point = self.enc(point)

        if not self.enc_mode:
            if self.freeze_backbone in ("enc", "enc_finetune"):
                self._clear_subm_pair_cache(point)
            point = self.dec(point)

        if self.freeze_backbone == "enc" and self.enc_mode:
            self._bridge_leaf(point)

        return point
