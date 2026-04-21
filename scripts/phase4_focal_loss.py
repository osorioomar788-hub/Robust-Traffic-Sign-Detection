"""
Phase 4 — Focal Loss integration for Ultralytics YOLOv8.

Implements Focal Loss for dense object detection (Lin et al., 2017,
arXiv:1708.02002) as a drop-in replacement for the BCE classification
loss used by ``ultralytics.utils.loss.v8DetectionLoss``.

Motivation (audit F-1): TT100K is extremely imbalanced — the 143 dense
classes span from ~10 images on the tail to >1000 on the head. Stock
``BCEWithLogitsLoss`` over-weights easy, frequent classes. Focal loss
down-weights well-classified examples via ``(1 - p_t)**gamma`` and
rebalances positives/negatives via ``alpha``.

Parameters (paper defaults, audit A-1/A-3):
    gamma = 2.0     focusing parameter
    alpha = 0.25    positive class weight

Public API:
    FocalBCE              : nn.Module, BCE-with-logits + focal term
    register_custom_loss(): monkey-patches v8DetectionLoss to use FocalBCE.
                            Idempotent; safe to call before YOLO().

Usage (from train.py):
    from scripts.phase4_focal_loss import register_custom_loss
    register_custom_loss(gamma=2.0, alpha=0.25)
    model = YOLO(...); model.train(...)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalBCE(nn.Module):
    """Binary-cross-entropy-with-logits + focal modulation (Lin et al. 2017).

    Signature matches ``nn.BCEWithLogitsLoss(reduction="none")`` so it can
    be dropped into ``v8DetectionLoss.bce`` without any call-site changes:
    forward(pred, target) -> tensor of same shape as pred/target.

    FL(p_t) = -alpha_t * (1 - p_t)**gamma * log(p_t)

    Implemented via stable BCE-with-logits to avoid log(sigmoid) overflow.
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        super().__init__()
        self.gamma = float(gamma)
        self.alpha = float(alpha)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
        p = torch.sigmoid(pred)
        p_t = p * target + (1.0 - p) * (1.0 - target)
        alpha_t = self.alpha * target + (1.0 - self.alpha) * (1.0 - target)
        return alpha_t * (1.0 - p_t).pow(self.gamma) * bce


def register_custom_loss(gamma: float = 2.0, alpha: float = 0.25) -> None:
    """Monkey-patch ``v8DetectionLoss`` so every instance uses FocalBCE.

    Must be called BEFORE ``YOLO(...).train(...)``. Idempotent.

    Wraps ``v8DetectionLoss.__init__`` so that, after the original
    initialiser runs, ``self.bce`` is replaced with a FocalBCE module
    carrying the requested (gamma, alpha). The rest of the loss pipeline
    (objectness, DFL, box IoU) is untouched — only the class head
    classification term is modified. This matches the scope of audit F-1.
    """
    from ultralytics.utils import loss as ult_loss

    cls = ult_loss.v8DetectionLoss
    original_init = cls.__init__

    if getattr(original_init, "_focal_patched", False):
        return

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.bce = FocalBCE(gamma=gamma, alpha=alpha)

    patched_init._focal_patched = True
    patched_init._focal_gamma = gamma
    patched_init._focal_alpha = alpha
    cls.__init__ = patched_init
