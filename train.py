"""
Phase 4 — MICAI training driver.

Wires together Phase 3 (custom architecture: SubPixelConv + CoordAtt) and
Phase 4 (FocalBCE classification loss) and launches Ultralytics training.

Audit closures (docs/audits/2026-04-17-phase4-audit.md):
    F-6  argparse-driven entry point (replaces hardcoded driver).
    B-3  loads custom YAML — not stock yolov8n.pt — so the MICAI
         architecture actually receives gradients.
    M-1  Phase-3 / Phase-4 registration ImportErrors are NOT swallowed.
    M-2  device / workers / epochs / batch / imgsz exposed via CLI with
         portable defaults.
    M-3  run name embeds (gamma, alpha, epochs, seed) — no more 'test2_focal'.

Critical ordering (do not reshuffle):
    1. sys.path bootstrap.
    2. register_custom_modules()    — patches parse_model BEFORE YOLO sees the YAML.
    3. register_custom_loss(...)    — patches v8DetectionLoss BEFORE it is constructed.
    4. from ultralytics import YOLO — only AFTER both registrations.
    5. YOLO(yaml).train(...).

Reversing 2/3/4 silently regresses to stock Ultralytics behaviour.

Usage:
    python train.py --epochs 50 --batch 16
    python train.py --device cpu --workers 2 --epochs 1   # smoke test
    python train.py --gamma 1.5 --alpha 0.5               # ablation
"""

from __future__ import annotations

import argparse
import os
import sys


def parse_args() -> argparse.Namespace:
    """CLI for the Phase 4 training driver (audit M-2)."""
    parser = argparse.ArgumentParser(
        description="MICAI Phase 4 — YOLOv8 + custom modules + FocalBCE",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data",
        default="data/processed/dataset.yaml",
        help="Path to dataset YAML (must declare nc=143 for the custom head).",
    )
    parser.add_argument(
        "--model-yaml",
        default="models/configs/yolov8_custom.yaml",
        help="Architecture YAML with SubPixelConv + CoordAtt.",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument(
        "--device",
        default=None,
        help="'0' for first GPU, 'cpu', or comma-separated indices. "
        "Default: auto-detect CUDA, fall back to CPU.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Dataloader workers. Default: min(8, os.cpu_count() // 2).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gamma", type=float, default=2.0, help="Focal loss gamma.")
    parser.add_argument("--alpha", type=float, default=0.25, help="Focal loss alpha.")
    parser.add_argument("--loss", type=str, default="focal", choices=["bce", "focal"], help="Elegir entre BCE normal o Focal Loss.")
    parser.add_argument("--project", default="runs/detect")
    parser.add_argument(
        "--name",
        default=None,
        help="Run name. Default: focal_g{gamma}_a{alpha}_ep{epochs}_seed{seed}.",
    )
    return parser.parse_args()


def resolve_device(requested: str | None) -> str | int:
    """Pick a sensible device default when --device is not given (audit M-2)."""
    if requested is not None:
        return requested
    import torch

    return 0 if torch.cuda.is_available() else "cpu"


def resolve_workers(requested: int | None) -> int:
    """Pick a portable dataloader-worker default (audit M-2)."""
    if requested is not None:
        return requested
    cpu_count = os.cpu_count() or 2
    return min(8, max(1, cpu_count // 2))


def main() -> None:
    args = parse_args()

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    # ── Step 1: register Phase-3 custom modules ─────────────────────
    # No silent except (audit M-1): if Phase 3 cannot be imported, the
    # custom YAML will fail to parse, so we MUST surface the error.
    from scripts.phase3_architecture_design import register_custom_modules

    register_custom_modules()
    print("[train] Phase-3 modules registered (SubPixelConv, CoordAtt).")

    # ── Step 2: register Phase-4 focal loss ─────────────────────────
    if args.loss == "focal":
        try:
            from scripts.phase4_focal_loss import register_custom_loss
            register_custom_loss(gamma=args.gamma, alpha=args.alpha)
            print(f"[train] Phase-4 FocalBCE registrada (gamma={args.gamma}, alpha={args.alpha}).")
        except ImportError as e:
            raise RuntimeError("Falla Crítica: No se encontró scripts/phase4_focal_loss.py.") from e
    else:
        print("[train] Modo Baseline: Usando BCE estándar. Focal Loss desactivada.")

    # ── Step 3: import YOLO AFTER both registrations ───────────────
    from ultralytics import YOLO

    # ── Step 4: build model from custom YAML (audit B-3) ───────────
    model = YOLO(args.model_yaml)
    print(f"[train] Model built from {args.model_yaml}.")

    device = resolve_device(args.device)
    workers = resolve_workers(args.workers)
    name = args.name or f"focal_g{args.gamma}_a{args.alpha}_ep{args.epochs}_seed{args.seed}"

    print(
        f"[train] Launching: device={device} workers={workers} "
        f"epochs={args.epochs} batch={args.batch} imgsz={args.imgsz} name={name}"
    )

    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        workers=workers,
        seed=args.seed,
        project=args.project,
        name=name,
    )


if __name__ == "__main__":
    main()
