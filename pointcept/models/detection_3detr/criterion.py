"""
3DETR Detection Criterion

Adapted from third_party/3detr/criterion.py.
Provides a Pointcept LOSSES-registered criterion for 3DETR training.

Loss components:
  - Semantic class prediction (cross-entropy)
  - Angle class + residual regression
  - Center regression (L1)
  - Size regression (L1)
  - Generalized 3D IoU loss
  - Cardinality error (logged only, no gradient)

Uses Hungarian matching (Matcher) to assign predictions to GT boxes.
"""

import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

logger = logging.getLogger(__name__)

from pointcept.models.losses.builder import LOSSES
from .dataset_config import _setup_3detr_path

_setup_3detr_path()

from utils.box_util import generalized_box3d_iou
from utils.dist import all_reduce_average
from utils.misc import huber_loss


class Matcher(nn.Module):
    """
    Bipartite matcher using Hungarian algorithm.
    Computes assignment between predictions and GT boxes.
    """

    def __init__(self, cost_class=1.0, cost_objectness=0.1, cost_giou=1.0, cost_center=5.0):
        super().__init__()
        self.cost_class = cost_class
        self.cost_objectness = cost_objectness
        self.cost_giou = cost_giou
        self.cost_center = cost_center

    @torch.no_grad()
    def forward(self, outputs, targets):
        batchsize = outputs["sem_cls_prob"].shape[0]
        nqueries = outputs["sem_cls_prob"].shape[1]
        ngt = targets["gt_box_sem_cls_label"].shape[1]
        nactual_gt = targets["nactual_gt"]

        pred_cls_prob = outputs["sem_cls_prob"]
        gt_box_sem_cls_labels = (
            targets["gt_box_sem_cls_label"]
            .unsqueeze(1)
            .expand(batchsize, nqueries, ngt)
        )
        class_mat = -torch.gather(pred_cls_prob, 2, gt_box_sem_cls_labels)
        objectness_mat = -outputs["objectness_prob"].unsqueeze(-1)
        center_mat = outputs["center_dist"].detach()
        giou_mat = -outputs["gious"].detach()

        final_cost = (
            self.cost_class * class_mat
            + self.cost_objectness * objectness_mat
            + self.cost_center * center_mat
            + self.cost_giou * giou_mat
        )

        final_cost = final_cost.detach().cpu().numpy()
        batch_size, nprop = final_cost.shape[0], final_cost.shape[1]
        per_prop_gt_inds = torch.zeros(
            [batch_size, nprop], dtype=torch.int64, device=pred_cls_prob.device
        )
        proposal_matched_mask = torch.zeros(
            [batch_size, nprop], dtype=torch.float32, device=pred_cls_prob.device
        )
        assignments = []
        for b in range(batchsize):
            assign = []
            if nactual_gt[b] > 0:
                assign = linear_sum_assignment(final_cost[b, :, : nactual_gt[b]])
                assign = [
                    torch.from_numpy(x).long().to(device=pred_cls_prob.device)
                    for x in assign
                ]
                per_prop_gt_inds[b, assign[0]] = assign[1]
                proposal_matched_mask[b, assign[0]] = 1
            assignments.append(assign)

        return {
            "assignments": assignments,
            "per_prop_gt_inds": per_prop_gt_inds,
            "proposal_matched_mask": proposal_matched_mask,
        }


@LOSSES.register_module()
class SetCriterion3DETR(nn.Module):
    """
    3DETR set-based detection criterion.

    Performs Hungarian matching then computes a weighted sum of
    classification, angle, center, size, and GIoU losses.

    Args:
        matcher_cfg (dict): kwargs for Matcher (cost_class, cost_objectness,
            cost_giou, cost_center)
        loss_weight_dict (dict): per-loss weight keys, e.g.
            ``loss_sem_cls_weight``, ``loss_no_object_weight``,
            ``loss_angle_cls_weight``, ``loss_angle_reg_weight``,
            ``loss_center_weight``, ``loss_size_weight``,
            ``loss_giou_weight``
        num_semcls (int): number of semantic classes (18 for ScanNet)
        num_angle_bin (int): number of angle bins (1 for ScanNet = axis-aligned)
    """

    def __init__(
        self,
        matcher_cfg=None,
        loss_weight_dict=None,
        num_semcls=18,
        num_angle_bin=1,
        giou_on_aux_outputs=True,
        giou_mode="optimized",
    ):
        super().__init__()
        if matcher_cfg is None:
            matcher_cfg = {}
        self.matcher = Matcher(**matcher_cfg)

        if loss_weight_dict is None:
            loss_weight_dict = {
                "loss_giou_weight": 1.0,
                "loss_sem_cls_weight": 1.0,
                "loss_no_object_weight": 0.25,
                "loss_angle_cls_weight": 0.1,
                "loss_angle_reg_weight": 0.5,
                "loss_center_weight": 5.0,
                "loss_size_weight": 1.0,
            }
        self.loss_weight_dict = loss_weight_dict

        self.num_semcls = num_semcls
        self.num_angle_bin = num_angle_bin
        self.giou_on_aux_outputs = bool(giou_on_aux_outputs)
        self.giou_mode = str(giou_mode)
        if self.giou_mode not in {"legacy", "optimized"}:
            raise ValueError(
                f"Unsupported giou_mode='{self.giou_mode}'. "
                "Expected 'legacy' or 'optimized'."
            )

        # Counts how many times the GIoU clamp has fired; used for rate-limited
        # logging so anomalies are visible without spamming the training log.
        self._giou_anomaly_count = 0

        semcls_percls_weights = torch.ones(num_semcls + 1)
        semcls_percls_weights[-1] = loss_weight_dict.get("loss_no_object_weight", 0.25)
        self.register_buffer("semcls_percls_weights", semcls_percls_weights)

        # Remove no_object_weight from the main weight dict (it's absorbed above)
        self._loss_weight_dict = {
            k: v for k, v in loss_weight_dict.items()
            if k != "loss_no_object_weight"
        }

        self.loss_functions = {
            "loss_sem_cls": self._loss_sem_cls,
            "loss_angle": self._loss_angle,
            "loss_center": self._loss_center,
            "loss_size": self._loss_size,
            "loss_giou": self._loss_giou,
            "loss_cardinality": self._loss_cardinality,
        }

    @torch.no_grad()
    def _loss_cardinality(self, outputs, targets, assignments):
        pred_logits = outputs["sem_cls_logits"]
        pred_objects = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(pred_objects.float(), targets["nactual_gt"].float())
        return {"loss_cardinality": card_err}

    def _loss_sem_cls(self, outputs, targets, assignments):
        pred_logits = outputs["sem_cls_logits"]
        gt_box_label = torch.gather(
            targets["gt_box_sem_cls_label"], 1, assignments["per_prop_gt_inds"]
        )
        gt_box_label[assignments["proposal_matched_mask"].int() == 0] = (
            pred_logits.shape[-1] - 1
        )
        loss = F.cross_entropy(
            pred_logits.transpose(2, 1),
            gt_box_label,
            self.semcls_percls_weights,
            reduction="mean",
        )
        return {"loss_sem_cls": loss}

    def _loss_angle(self, outputs, targets, assignments):
        angle_logits = outputs["angle_logits"]
        angle_residual = outputs["angle_residual_normalized"]

        if targets["num_boxes_replica"] > 0:
            gt_angle_label = targets["gt_angle_class_label"]
            gt_angle_residual = targets["gt_angle_residual_label"]
            gt_angle_residual_normalized = gt_angle_residual / (
                np.pi / self.num_angle_bin
            )
            gt_angle_label = torch.gather(
                gt_angle_label, 1, assignments["per_prop_gt_inds"]
            )
            angle_cls_loss = F.cross_entropy(
                angle_logits.transpose(2, 1), gt_angle_label, reduction="none"
            )
            angle_cls_loss = (
                angle_cls_loss * assignments["proposal_matched_mask"]
            ).sum()

            gt_angle_residual_normalized = torch.gather(
                gt_angle_residual_normalized, 1, assignments["per_prop_gt_inds"]
            )
            gt_angle_label_one_hot = torch.zeros_like(angle_residual, dtype=torch.float32)
            gt_angle_label_one_hot.scatter_(2, gt_angle_label.unsqueeze(-1), 1)
            angle_residual_for_gt_class = torch.sum(
                angle_residual * gt_angle_label_one_hot, -1
            )
            angle_reg_loss = huber_loss(
                angle_residual_for_gt_class - gt_angle_residual_normalized, delta=1.0
            )
            angle_reg_loss = (
                angle_reg_loss * assignments["proposal_matched_mask"]
            ).sum()

            angle_cls_loss /= targets["num_boxes"]
            angle_reg_loss /= targets["num_boxes"]
        else:
            angle_cls_loss = torch.zeros(1, device=angle_logits.device).squeeze()
            angle_reg_loss = torch.zeros(1, device=angle_logits.device).squeeze()
        return {"loss_angle_cls": angle_cls_loss, "loss_angle_reg": angle_reg_loss}

    def _loss_center(self, outputs, targets, assignments):
        center_dist = outputs["center_dist"]
        if targets["num_boxes_replica"] > 0:
            center_loss = torch.gather(
                center_dist, 2, assignments["per_prop_gt_inds"].unsqueeze(-1)
            ).squeeze(-1)
            center_loss = center_loss * assignments["proposal_matched_mask"]
            center_loss = center_loss.sum()
            if targets["num_boxes"] > 0:
                center_loss /= targets["num_boxes"]
        else:
            center_loss = torch.zeros(1, device=center_dist.device).squeeze()
        return {"loss_center": center_loss}

    def _loss_giou(self, outputs, targets, assignments):
        if self.giou_mode == "legacy":
            gious_dist = 1 - outputs["gious"]
            giou_loss = torch.gather(
                gious_dist, 2, assignments["per_prop_gt_inds"].unsqueeze(-1)
            ).squeeze(-1)
            giou_loss = giou_loss * assignments["proposal_matched_mask"]
        else:
            # `gious_matched` is (B, K1), already evaluated at the matcher-assigned
            # GT for every proposal (with grad). For rotated boxes that compute is
            # ~K2× cheaper than the full (B, K1, K2) grid the matcher uses.
            gious_matched = outputs["gious_matched"]
            giou_loss = (1 - gious_matched) * assignments["proposal_matched_mask"]
        giou_loss = giou_loss.sum()
        if targets["num_boxes"] > 0:
            giou_loss /= targets["num_boxes"]
        return {"loss_giou": giou_loss}

    def _loss_size(self, outputs, targets, assignments):
        gt_box_sizes = targets["gt_box_sizes_normalized"]
        pred_box_sizes = outputs["size_normalized"]

        if targets["num_boxes_replica"] > 0:
            gt_box_sizes = torch.stack(
                [
                    torch.gather(
                        gt_box_sizes[:, :, x], 1, assignments["per_prop_gt_inds"]
                    )
                    for x in range(gt_box_sizes.shape[-1])
                ],
                dim=-1,
            )
            size_loss = F.l1_loss(pred_box_sizes, gt_box_sizes, reduction="none").sum(
                dim=-1
            )
            size_loss *= assignments["proposal_matched_mask"]
            size_loss = size_loss.sum()
            size_loss /= targets["num_boxes"]
        else:
            size_loss = torch.zeros(1, device=pred_box_sizes.device).squeeze()
        return {"loss_size": size_loss}

    def _use_rotated_boxes(self) -> bool:
        """True when the dataset uses oriented boxes (num_angle_bin > 1).

        Deciding from config rather than from batch GT angles avoids a silent
        failure mode: if a batch happens to have all GT yaw = 0 (early training,
        or a fully axis-aligned scene), torch.any(gt_angles > 0) returns False
        and the axis-aligned AABB path would be used even though predictions may
        have non-zero yaw, producing wrong GIoU.
        """
        return self.num_angle_bin > 1

    def _sanitize_gious(self, gious, outputs, targets, tag):
        """Detect NaN/Inf/out-of-range GIoU, log, then nan_to_num + clamp.

        GIoU ∈ [-1, 1] by construction. Values outside that range or
        non-finite values signal a geometry failure in the polygon
        intersection code. Root-cause fixes live in box_util.py; this is
        the final safety net.
        """
        _GIOU_TOL = 1e-4
        finite_mask = gious.isfinite()
        num_nan = int(gious.isnan().sum())
        num_posinf = int(gious.isposinf().sum())
        num_neginf = int(gious.isneginf().sum())
        if finite_mask.any():
            finite_gious = gious[finite_mask]
            gious_min = finite_gious.min().item()
            gious_max = finite_gious.max().item()
        else:
            gious_min = float("nan")
            gious_max = float("nan")

        has_nonfinite = (num_nan + num_posinf + num_neginf) > 0
        has_out_of_range = gious_max > 1.0 + _GIOU_TOL or gious_min < -1.0 - _GIOU_TOL
        if has_nonfinite or has_out_of_range:
            self._giou_anomaly_count += 1
            n = self._giou_anomaly_count
            if n == 1 or n % 100 == 0:
                logger.warning(
                    "GIoU anomaly #%d (%s): finite min=%.4g max=%.4g "
                    "nan=%d +inf=%d -inf=%d. "
                    "pred_angles_deg=[%.1f, %.1f]  "
                    "gt_angles_deg=[%.1f, %.1f]. "
                    "Sanitising with nan_to_num then clamp(-1, 1).",
                    n,
                    tag,
                    gious_min,
                    gious_max,
                    num_nan,
                    num_posinf,
                    num_neginf,
                    float(outputs["angle_continuous"].min()) * 180.0 / 3.14159,
                    float(outputs["angle_continuous"].max()) * 180.0 / 3.14159,
                    float(targets["gt_box_angles"].min()) * 180.0 / 3.14159,
                    float(targets["gt_box_angles"].max()) * 180.0 / 3.14159,
                )

        # nan_to_num first: clamp(-1, 1) passes NaNs through unchanged.
        gious = torch.nan_to_num(gious, nan=-1.0, posinf=1.0, neginf=-1.0)
        return gious.clamp(-1.0, 1.0)

    def single_output_forward(self, outputs, targets, compute_giou_loss=True):
        if self.giou_mode == "legacy":
            gious = generalized_box3d_iou(
                outputs["box_corners"],
                targets["gt_box_corners"],
                targets["nactual_gt"],
                rotated_boxes=torch.any(targets["gt_box_angles"] > 0).item(),
                needs_grad=(self._loss_weight_dict.get("loss_giou_weight", 0) > 0),
            )
            outputs["gious"] = gious.clamp(-1.0, 1.0)

            center_dist = torch.cdist(
                outputs["center_normalized"],
                targets["gt_box_centers_normalized"],
                p=1,
            )
            outputs["center_dist"] = center_dist
            assignments = self.matcher(outputs, targets)

            losses = {}
            for k in self.loss_functions:
                loss_wt_key = k + "_weight"
                if (
                    loss_wt_key in self._loss_weight_dict
                    and self._loss_weight_dict[loss_wt_key] > 0
                ) or loss_wt_key not in self._loss_weight_dict:
                    curr_loss = self.loss_functions[k](outputs, targets, assignments)
                    losses.update(curr_loss)

            final_loss = 0
            for k in self._loss_weight_dict:
                loss_key = k.replace("_weight", "")
                if self._loss_weight_dict[k] > 0 and loss_key in losses:
                    losses[loss_key] = losses[loss_key] * self._loss_weight_dict[k]
                    final_loss += losses[loss_key]
            return final_loss, losses

        # ── Matcher-side GIoU: full (B, K1, K2) grid, NO grad ───────────────
        # With needs_grad=False this uses the cythonized box_intersection
        # (single batched C call), which is ~K2× faster than the JIT path
        # for rotated boxes. The matcher only consumes .detach()-ed gious
        # via the cost matrix, so dropping grad here is exact, not an
        # approximation.
        with torch.no_grad():
            gious_full = generalized_box3d_iou(
                outputs["box_corners"],
                targets["gt_box_corners"],
                targets["nactual_gt"],
                rotated_boxes=self._use_rotated_boxes(),
                needs_grad=False,
            )
        gious_full = self._sanitize_gious(gious_full, outputs, targets, tag="matcher")
        outputs["gious"] = gious_full

        center_dist = torch.cdist(
            outputs["center_normalized"], targets["gt_box_centers_normalized"], p=1
        )
        outputs["center_dist"] = center_dist
        assignments = self.matcher(outputs, targets)

        # ── Loss-side GIoU: only matched (prop, gt) pairs, WITH grad ────────
        # The full (B, K1, K2) JIT computation runs the Sutherland-Hodgman
        # polygon clip in a Python loop over every pair; for rotated boxes
        # that's the dominant cost. We only need GIoU at the matched
        # (proposal, gt) pair for each proposal — gather the matched GT
        # corners and shape the call as (B*K1, 1, 8, 3) so the JIT path does
        # B*K1 polygon clips instead of B*K1*K2.
        giou_wt = self._loss_weight_dict.get("loss_giou_weight", 0)
        if giou_wt > 0 and targets["num_boxes_replica"] > 0:
            if not compute_giou_loss:
                gious_matched = torch.zeros(
                    outputs["box_corners"].shape[:2],
                    device=outputs["box_corners"].device,
                    dtype=outputs["box_corners"].dtype,
                )
                outputs["gious_matched"] = gious_matched
            else:
                per_prop_gt_inds = assignments["per_prop_gt_inds"]  # (B, K1)
                B, K1 = per_prop_gt_inds.shape
                gt_corners = targets["gt_box_corners"]  # (B, K2, 8, 3)
                gt_matched = torch.gather(
                    gt_corners,
                    1,
                    per_prop_gt_inds[:, :, None, None].expand(B, K1, 8, 3),
                )  # (B, K1, 8, 3)
                pred_corners = outputs["box_corners"]  # (B, K1, 8, 3)

                pred_flat = pred_corners.reshape(B * K1, 1, 8, 3)
                gt_flat = gt_matched.reshape(B * K1, 1, 8, 3)
                nums_k2_ones = torch.ones(
                    B * K1, dtype=torch.long, device=pred_flat.device
                )
                gious_matched = generalized_box3d_iou(
                    pred_flat,
                    gt_flat,
                    nums_k2_ones,
                    rotated_boxes=self._use_rotated_boxes(),
                    needs_grad=True,
                ).reshape(B, K1)
                gious_matched = self._sanitize_gious(
                    gious_matched, outputs, targets, tag="loss"
                )
                outputs["gious_matched"] = gious_matched
        else:
            gious_matched = torch.zeros(
                outputs["box_corners"].shape[:2],
                device=outputs["box_corners"].device,
                dtype=outputs["box_corners"].dtype,
            )
            outputs["gious_matched"] = gious_matched

        losses = {}
        for k in self.loss_functions:
            loss_wt_key = k + "_weight"
            if (
                loss_wt_key in self._loss_weight_dict
                and self._loss_weight_dict[loss_wt_key] > 0
            ) or loss_wt_key not in self._loss_weight_dict:
                curr_loss = self.loss_functions[k](outputs, targets, assignments)
                losses.update(curr_loss)

        final_loss = 0
        for k in self._loss_weight_dict:
            loss_key = k.replace("_weight", "")
            if self._loss_weight_dict[k] > 0 and loss_key in losses:
                losses[loss_key] = losses[loss_key] * self._loss_weight_dict[k]
                final_loss += losses[loss_key]
        return final_loss, losses

    def forward(self, outputs, targets):
        """
        Args:
            outputs (dict): model output with keys 'outputs' and 'aux_outputs'
            targets (dict): batch of GT labels (already on CUDA)

        Returns:
            (total_loss, loss_dict)
        """
        nactual_gt = targets["gt_box_present"].sum(axis=1).long()
        num_boxes = torch.clamp(all_reduce_average(nactual_gt.sum()), min=1).item()
        targets["nactual_gt"] = nactual_gt
        targets["num_boxes"] = num_boxes
        targets["num_boxes_replica"] = nactual_gt.sum().item()

        loss, loss_dict = self.single_output_forward(outputs["outputs"], targets)

        if "aux_outputs" in outputs:
            for k, aux_out in enumerate(outputs["aux_outputs"]):
                interm_loss, interm_loss_dict = self.single_output_forward(
                    aux_out,
                    targets,
                    compute_giou_loss=self.giou_on_aux_outputs,
                )
                loss += interm_loss
                for key in interm_loss_dict:
                    loss_dict[f"{key}_{k}"] = interm_loss_dict[key]

        return loss, loss_dict
