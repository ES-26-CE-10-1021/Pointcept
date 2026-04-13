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

import contextlib
import logging

import torch
from pointcept.models.builder import MODULES
from pointcept.models.point_transformer_v3.point_transformer_v3m1_base import (
    PointTransformerV3,
)
from pointcept.models.point_transformer_v3.point_transformer_v3m3_utonia import (
    PointTransformerV3 as PointTransformerV3m3,
)
from pointcept.models.detection_3detr.model import dense2point, point2dense
from third_party.pointnet2.pointnet2_utils import furthest_point_sample

_logger = logging.getLogger(__name__)


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


@MODULES.register_module("PTv3m3PreEncoder")
class PTv3m3PreEncoder(PointTransformerV3m3):
    """PT-v3m3 (Utonia) wrapped as a frozen 3DETR pre_encoder.

    Loads a pretrained Utonia checkpoint from HuggingFace and runs its
    encoder as a fixed feature extractor. The backbone parameters are
    never updated; only the downstream 3DETR encoder / decoder / heads
    receive gradients. Three independent mechanisms ensure that the
    freeze is both correct and memory-efficient:

    1. ``freeze`` sets ``requires_grad=False`` on the embedding and
       encoder parameters (via the base class's ``freeze_encoder`` flag).
    2. ``freeze_eval`` forces the backbone into ``eval()`` mode on every
       ``train()`` call, so dropout / drop_path / stochastic depth are
       deterministic across iterations.
    3. ``freeze_no_grad`` wraps the backbone forward in
       ``torch.no_grad()``. This is the VRAM fix — without it, autograd
       still builds a graph over every op inside the frozen module,
       storing activations for a backward pass that never uses them.
       After the context exits we detach and (in training mode) re-enable
       grad on the output so the downstream projection can build a fresh
       graph from that leaf.

    Detection datasets provide XYZ (and optionally RGB), whereas Utonia
    was pretrained on a 9-dim ``[xyz, rgb, normal]`` input. Missing
    modalities are zero-padded on-device; Causal Modality Blinding makes
    Utonia tolerate absent modalities as long as their channels are
    explicit zeros.

    Args:
        pretrained (str | None): HuggingFace model name (e.g. ``"utonia"``)
            or a local ``.pth`` path. If ``None`` / falsy, the backbone is
            built from the user's config without loading any checkpoint.
            Useful for unit tests that don't require HF access.
        download_root (str | None): forwarded to the Utonia ``load()``
            helper; defaults to ``~/.cache/utonia/ckpt``.
        config_overrides (dict | None): merged over the checkpoint's
            recorded config before instantiating the backbone. Cannot
            change ``in_channels`` (which would desync the embedding
            weight shape from the checkpoint).
        freeze (bool): if True, set ``requires_grad=False`` on embedding
            + encoder parameters (via ``freeze_encoder``).
        freeze_eval (bool): if True, force the backbone into eval mode on
            every ``train()`` call. Ignored when ``freeze=False``.
        freeze_no_grad (bool): if True, wrap the backbone forward in
            ``torch.no_grad()`` and reattach grad on the output. Ignored
            when ``freeze=False``. Set to False only to ablate Utonia
            fine-tuning.
        grid_size (float): voxel size used to serialize the input point
            cloud before running the backbone.
        **overrides: any remaining kwargs are merged into the checkpoint
            config (same precedence as ``config_overrides``).
    """

    def __init__(
        self,
        pretrained="utonia",
        download_root=None,
        config_overrides=None,
        freeze=True,
        freeze_eval=True,
        freeze_no_grad=True,
        grid_size=0.02,
        **overrides,
    ):
        # Merge priority (highest wins): **overrides > config_overrides >
        # checkpoint config > backbone defaults. Then force enc_mode /
        # freeze_encoder regardless of what anyone tried to set.
        ckpt_state_dict = None
        ckpt_in_channels = None
        if pretrained:
            ckpt = _ddp_safe_utonia_load(pretrained, download_root)
            base_config = dict(ckpt["config"])
            ckpt_state_dict = ckpt["state_dict"]
            ckpt_in_channels = base_config.get("in_channels")
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

        base_config["enc_mode"] = True
        base_config["freeze_encoder"] = bool(freeze)

        super().__init__(**base_config)

        self.grid_size = grid_size
        self.freeze = bool(freeze)
        self.freeze_eval = bool(freeze_eval)
        self.freeze_no_grad = bool(freeze_no_grad)
        self._pretrained = pretrained

        if ckpt_state_dict is not None:
            self._load_pretrained_state(ckpt_state_dict)

        # Put the frozen blocks into eval mode immediately so a caller
        # that never calls .train() still sees deterministic features.
        if self.freeze and self.freeze_eval:
            self.embedding.eval()
            self.enc.eval()

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
        incompatible = self.load_state_dict(state_dict, strict=False)
        missing = list(incompatible.missing_keys)
        unexpected = list(incompatible.unexpected_keys)
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

    def train(self, mode=True):
        """Override ``train()`` so the frozen blocks stay in eval mode.

        The default ``nn.Module.train()`` recursively flips every
        submodule, which would re-enable dropout / drop_path inside the
        frozen backbone on every step — nondeterministic features for
        identical inputs.
        """
        super().train(mode)
        if self.freeze and self.freeze_eval:
            self.embedding.eval()
            self.enc.eval()
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

    def forward(self, xyz, features=None):
        """
        Args:
            xyz:      (B, N, 3) point coordinates
            features: (B, C, N) point features, or None

        Returns:
            Point with encoded features (variable-length per scene).
        """
        feat_padded = self._build_padded_feat(xyz, features)
        point = dense2point(xyz, feat_padded)
        point["grid_size"] = self.grid_size

        use_no_grad = self.freeze and self.freeze_no_grad
        freeze_ctx = (
            torch.no_grad() if use_no_grad else contextlib.nullcontext()
        )
        with freeze_ctx:
            point.serialization(
                order=self.order, shuffle_orders=self.shuffle_orders
            )
            point.sparsify()
            point = self.embedding(point)
            point = self.enc(point)

        if use_no_grad:
            # The backbone built no graph, so point.feat has
            # grad_fn=None. Detach to be explicit (safe no-op) and then
            # flip requires_grad on so the downstream encoder / decoder /
            # heads can attach a fresh graph at this leaf. Gradients stop
            # here and never touch the Utonia parameters.
            point.feat = point.feat.detach()
            if self.training:
                point.feat.requires_grad_(True)
        return point
