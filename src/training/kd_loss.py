"""
src/training/kd_loss.py
-----------------------
Composite Knowledge Distillation loss for YOLO-based weed detection.

Total loss = α·L_det + β·L_feat + γ·L_kl

  L_det  — Standard YOLOv8 detection loss (box + cls + dfl)
  L_feat — MSE between adapted student features and teacher features
           at backbone layers 4, 6, 9
  L_kl   — KL divergence between temperature-scaled student and teacher
           class logits (soft-label distillation)

Constraint: α + β + γ = 1.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from types import SimpleNamespace
from typing import Dict, Optional, Tuple


class KDLoss(nn.Module):
    """
    Composite KD loss: detection + feature distillation + KL soft labels.

    Args:
        alpha: Weight for detection loss.
        beta: Weight for feature distillation loss.
        gamma: Weight for KL soft-label loss.
        temperature: Softmax temperature for KL distillation.
        nc: Number of classes.
    """

    def __init__(
        self,
        alpha: float = 0.5,
        beta: float = 0.3,
        gamma: float = 0.2,
        temperature: float = 3.0,
        nc: int = 19
    ):
        super().__init__()
        assert abs(alpha + beta + gamma - 1.0) < 1e-6, \
            f"alpha + beta + gamma must equal 1.0 (got {alpha + beta + gamma:.4f})"

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.T = temperature
        self.nc = nc
        self._det_criterion = None

    def set_criterion(self, student_model: nn.Module) -> None:
        """
        Initialize the YOLOv8 detection loss using the student model's architecture.
        Must be called once before training begins.

        Args:
            student_model: The student's underlying DetectionModel (student.model).
        """
        from ultralytics.utils.loss import v8DetectionLoss

        student_model.args = SimpleNamespace(
            box=7.5, cls=0.5, dfl=1.5,
            pose=12.0, kobj=1.0,
            label_smoothing=0.0, nc=self.nc,
            overlap_mask=True, mask_ratio=4
        )
        self._det_criterion = v8DetectionLoss(student_model)

    def detection_loss(
        self,
        s_out: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute standard YOLOv8 detection loss.

        Args:
            s_out: Student model output.
            targets: Batch targets tensor [N, 6] (batch_idx, cls, cx, cy, w, h).

        Returns:
            Scalar detection loss.
        """
        if self._det_criterion is None:
            raise RuntimeError("Call set_criterion(student.model) before training.")

        if targets.numel() == 0:
            device = next(iter(self._det_criterion.model.parameters())).device
            return torch.tensor(0.0, device=device)

        batch = {
            "batch_idx": targets[:, 0].long(),
            "cls":       targets[:, 1].long(),
            "bboxes":    targets[:, 2:6],
        }
        loss, _ = self._det_criterion(s_out, batch)
        return loss.mean() if loss.ndim > 0 else loss

    def feature_loss(
        self,
        s_feats: Dict[str, torch.Tensor],
        t_feats: Dict[str, torch.Tensor],
        adapters: nn.ModuleDict
    ) -> torch.Tensor:
        """
        MSE loss between adapted student features and frozen teacher features.

        Args:
            s_feats: Student intermediate features {'l4', 'l6', 'l9'}.
            t_feats: Teacher intermediate features {'l4', 'l6', 'l9'}.
            adapters: Channel-alignment adapters (nn.ModuleDict).

        Returns:
            Mean MSE loss across available layers.
        """
        total, count = 0.0, 0
        for key in ["l4", "l6", "l9"]:
            if key in s_feats and key in t_feats and key in adapters:
                adapted = adapters[key](s_feats[key])
                total = total + F.mse_loss(adapted, t_feats[key])
                count += 1

        if count == 0:
            device = next(adapters.parameters()).device
            return torch.tensor(0.0, device=device)
        return total / count

    def kl_loss(
        self,
        s_out: torch.Tensor,
        t_out: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        KL divergence between temperature-scaled student and teacher class logits.
        Operates on predicted class scores from the detection head output.

        Args:
            s_out: Student output tensor (or list of tensors).
            t_out: Teacher output tensor (or list of tensors).
            targets: Batch targets (used only for device reference).

        Returns:
            Scalar KL divergence loss.
        """
        device = targets.device

        def _extract_cls_logits(out):
            if isinstance(out, (list, tuple)):
                for o in out:
                    if isinstance(o, torch.Tensor) and o.ndim >= 2:
                        return o.reshape(-1, o.shape[-1])
            if isinstance(out, torch.Tensor):
                return out.reshape(-1, out.shape[-1])
            return None

        s_logits = _extract_cls_logits(s_out)
        t_logits = _extract_cls_logits(t_out)

        if s_logits is None or t_logits is None:
            return torch.tensor(0.0, device=device)

        # Align shapes
        min_rows = min(s_logits.shape[0], t_logits.shape[0])
        if min_rows == 0:
            return torch.tensor(0.0, device=device)

        s_logits = s_logits[:min_rows]
        t_logits = t_logits[:min_rows]

        s_log_probs = F.log_softmax(s_logits / self.T, dim=-1)
        t_probs     = F.softmax(t_logits / self.T, dim=-1)

        kl = F.kl_div(s_log_probs, t_probs, reduction='batchmean')
        return (self.T ** 2) * kl

    def forward(
        self,
        s_out: torch.Tensor,
        t_out: torch.Tensor,
        s_feats: Dict[str, torch.Tensor],
        t_feats: Dict[str, torch.Tensor],
        adapters: nn.ModuleDict,
        targets: torch.Tensor
    ) -> Tuple[torch.Tensor, float, float, float]:
        """
        Compute total KD loss.

        Returns:
            (total_loss, det_loss_val, feat_loss_val, kl_loss_val)
        """
        l_det  = self.detection_loss(s_out, targets)
        l_feat = self.feature_loss(s_feats, t_feats, adapters)
        l_kl   = self.kl_loss(s_out, t_out, targets)

        total = self.alpha * l_det + self.beta * l_feat + self.gamma * l_kl

        return (
            total,
            float(l_det),
            float(l_feat),
            float(l_kl)
        )
