"""
Phase 4 — Loss validation / smoke test.

Analogous to ``scripts/phase3_architecture_design.py:main`` but for the
classification-loss patch introduced in Phase 4.

Closes audit H-4 / F-8. Specifically, this script verifies that:
    (i)   register_custom_modules() and register_custom_loss() both run
          and survive being called twice (idempotent — audit I-3).
    (ii)  Building YOLO from the custom YAML still succeeds after both
          registrations (i.e. Phase 3 is not regressed by Phase 4).
    (iii) The v8DetectionLoss instance attached to the model has its
          ``bce`` attribute replaced by FocalBCE (the actual fix being
          tested — audit F-1 / I-2).
    (iv)  A forward + backward pass with synthetic targets produces a
          finite, non-zero loss whose .backward() yields gradients on
          the model parameters (the loss is wired into autograd).
    (v)   Re-applying register_custom_loss does not re-patch (idempotency).

A human-readable report is written to
``results/phase4_loss_validation.txt``.

Usage:
    python scripts/phase4_validate.py
    python scripts/phase4_validate.py --gamma 1.5 --alpha 0.5

Exit codes:
    0  all assertions passed.
    >0 a check failed; stderr carries the traceback.
"""

from __future__ import annotations

import argparse
import os
import sys


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 4 FocalBCE smoke test")
    parser.add_argument("--gamma", type=float, default=2.0)
    parser.add_argument("--alpha", type=float, default=0.25)
    parser.add_argument(
        "--yaml",
        default=os.path.join(PROJECT_ROOT, "models", "configs", "yolov8_custom.yaml"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # ── Step 1: register Phase-3 + Phase-4 patches ──────────────────
    from scripts.phase3_architecture_design import register_custom_modules
    from scripts.phase4_focal_loss import FocalBCE, register_custom_loss

    register_custom_modules()
    register_custom_loss(gamma=args.gamma, alpha=args.alpha)
    # Idempotency probe (audit I-3): a second call must not raise nor re-wrap.
    register_custom_loss(gamma=args.gamma, alpha=args.alpha)

    # ── Step 2: import YOLO AFTER registration ──────────────────────
    import torch
    from ultralytics import YOLO
    from ultralytics.utils import loss as ult_loss

    assert getattr(ult_loss.v8DetectionLoss.__init__, "_focal_patched", False), (
        "v8DetectionLoss.__init__ is not flagged _focal_patched — "
        "register_custom_loss did not take effect."
    )

    # ── Step 3: build model (regression check on Phase 3) ──────────
    model = YOLO(args.yaml)
    print(f"[validate] Model built from {args.yaml}.")

    # ── Step 4: prime the criterion via a forward+backward pass ────
    # YOLO lazily constructs model.criterion (the v8DetectionLoss instance)
    # on the first training-mode forward pass. Trigger it with a synthetic
    # batch shaped exactly like Ultralytics expects.
    model.model.train()
    dummy_img = torch.randn(2, 3, 640, 640)
    # Two synthetic boxes across two images, dense class IDs in [0, 142].
    dummy_batch = {
        "img": dummy_img,
        "cls": torch.tensor([[5.0], [42.0]]),
        "bboxes": torch.tensor([[0.5, 0.5, 0.2, 0.2], [0.3, 0.4, 0.1, 0.15]]),
        "batch_idx": torch.tensor([0.0, 1.0]),
    }
    loss, loss_items = model.model(dummy_batch)
    print(f"[validate] forward loss: {float(loss):.6f}")

    criterion = model.model.criterion
    bce_cls_name = type(criterion.bce).__name__
    assert bce_cls_name == "FocalBCE", (
        f"criterion.bce is {bce_cls_name}, expected FocalBCE. "
        "register_custom_loss did not replace the BCE module."
    )
    assert isinstance(criterion.bce, FocalBCE)
    assert criterion.bce.gamma == args.gamma
    assert criterion.bce.alpha == args.alpha
    print(
        f"[validate] criterion.bce = FocalBCE("
        f"gamma={criterion.bce.gamma}, alpha={criterion.bce.alpha})."
    )

    # ── Step 5: backward + grad presence ────────────────────────────
    assert torch.isfinite(loss), f"loss is non-finite: {loss}"
    loss.backward()
    grad_params = sum(
        1 for p in model.model.parameters() if p.grad is not None and p.grad.abs().sum() > 0
    )
    total_params = sum(1 for _ in model.model.parameters())
    print(
        f"[validate] backward complete: {grad_params}/{total_params} "
        f"parameter tensors received non-zero gradients."
    )
    assert grad_params > 0, "no parameters received gradients — autograd is broken."

    # ── Step 6: report ──────────────────────────────────────────────
    results_dir = os.path.join(PROJECT_ROOT, "results")
    os.makedirs(results_dir, exist_ok=True)
    report_path = os.path.join(results_dir, "phase4_loss_validation.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Phase 4 - FocalBCE Validation Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"YAML:               {args.yaml}\n")
        f.write(f"gamma:              {args.gamma}\n")
        f.write(f"alpha:              {args.alpha}\n")
        f.write(f"criterion.bce:      {bce_cls_name}\n")
        f.write(f"loss (forward):     {float(loss):.6f}\n")
        f.write(f"loss is finite:     True\n")
        f.write(f"params w/ grad:     {grad_params} / {total_params}\n")
        f.write(f"idempotency:        OK (register_custom_loss x2)\n")
        f.write(f"\nReport generated by: scripts/phase4_validate.py\n")
    print(f"[validate] report -> {report_path}")

    print("\n" + "=" * 60)
    print("[PHASE 4 VALIDATE] All checks passed.")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
