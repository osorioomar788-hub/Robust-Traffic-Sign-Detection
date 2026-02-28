"""
Fase 2 - Paso 2: Balanceo de Clases (Long-Tail Problem)
Proyecto: Robust Traffic Sign Detection (MICAI 25)
Responsable: Yael

Compatibilidad: albumentations==1.3.1 (sin parámetro seed en Compose)

Uso:
    python scripts/phase2_class_balancing.py
    python scripts/phase2_class_balancing.py --min_samples 500
"""

import os
import cv2
import json
import random
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import albumentations as A
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# ─────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────
DEFAULT_INPUT_DIR  = "data/augmented/train"
DEFAULT_OUTPUT_DIR = "data/augmented/train"
MIN_SAMPLES        = 500
SEED               = 42

random.seed(SEED)
np.random.seed(SEED)

# ─────────────────────────────────────────────
# PIPELINE PARA BALANCEO (compatible 1.3.1)
# ─────────────────────────────────────────────
bp = A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.3)

BALANCE_PIPELINE = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.8),
    A.GaussNoise(var_limit=(5.0, 30.0), p=0.5),
    A.GaussianBlur(blur_limit=(3, 5), p=0.3),
    A.ShiftScaleRotate(shift_limit=0.03, scale_limit=0.08, rotate_limit=3,
                       border_mode=cv2.BORDER_REFLECT_101, p=0.6),
    A.HueSaturationValue(hue_shift_limit=8, sat_shift_limit=20, val_shift_limit=15, p=0.5),
], bbox_params=bp)


# ─────────────────────────────────────────────
# UTILIDADES YOLO
# ─────────────────────────────────────────────

def read_yolo_label(label_path: Path):
    class_ids, bboxes = [], []
    if not label_path.exists():
        return class_ids, bboxes
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                class_ids.append(int(parts[0]))
                bboxes.append([float(x) for x in parts[1:]])
    return class_ids, bboxes


def write_yolo_label(label_path: Path, class_ids, bboxes):
    with open(label_path, "w") as f:
        for cid, bb in zip(class_ids, bboxes):
            bb = [max(0.0, min(1.0, v)) for v in bb]
            f.write(f"{cid} {bb[0]:.6f} {bb[1]:.6f} {bb[2]:.6f} {bb[3]:.6f}\n")


# ─────────────────────────────────────────────
# ANÁLISIS DE DISTRIBUCIÓN
# ─────────────────────────────────────────────

def analyze_distribution(input_dir: Path):
    lbl_dir = input_dir / "labels"
    img_dir = input_dir / "images"

    class_count   = defaultdict(int)
    class_to_imgs = defaultdict(list)

    label_files = list(lbl_dir.glob("*.txt"))
    print(f"\n📊 Analizando {len(label_files):,} archivos de etiquetas...")

    for lbl_path in tqdm(label_files, desc="  Analizando clases"):
        class_ids, _ = read_yolo_label(lbl_path)

        img_path = img_dir / (lbl_path.stem + ".jpg")
        if not img_path.exists():
            img_path = img_dir / (lbl_path.stem + ".png")

        for cid in class_ids:
            class_count[cid] += 1
            if img_path.exists() and img_path not in class_to_imgs[cid]:
                class_to_imgs[cid].append(img_path)

    return dict(class_count), dict(class_to_imgs)


# ─────────────────────────────────────────────
# BALANCEO PRINCIPAL
# ─────────────────────────────────────────────

def balance_classes(input_dir: str, output_dir: str, min_samples: int):
    input_dir  = Path(input_dir)
    output_dir = Path(output_dir)
    img_dir    = output_dir / "images"
    lbl_dir    = output_dir / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    class_count, class_to_imgs = analyze_distribution(input_dir)
    minority = {cid: min_samples - cnt
                for cid, cnt in class_count.items() if cnt < min_samples}

    if not minority:
        print(f"\n✅ Todas las clases tienen ≥ {min_samples} instancias. No se requiere balanceo.")
        _plot_distribution(class_count, {}, output_dir, min_samples)
        return

    print(f"\n⚠️  Clases minoritarias: {len(minority)}")
    print(f"   {'Clase':<10} {'Actual':<10} {'Faltantes'}")
    print(f"   {'-'*35}")
    for cid, needed in sorted(minority.items()):
        print(f"   {cid:<10} {class_count.get(cid,0):<10} {needed}")

    generated_log = []

    for cid, needed in tqdm(minority.items(), desc="Balanceando clases"):
        source_images = class_to_imgs.get(cid, [])
        if not source_images:
            continue

        generated = 0
        attempts  = 0

        while generated < needed and attempts < needed * 5:
            attempts += 1
            src_img_path = random.choice(source_images)
            src_lbl_path = (input_dir / "labels" / src_img_path.stem).with_suffix(".txt")

            image = cv2.imread(str(src_img_path))
            if image is None:
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            class_ids, bboxes = read_yolo_label(src_lbl_path)
            if not bboxes:
                continue

            try:
                result     = BALANCE_PIPELINE(image=image, bboxes=bboxes, class_labels=class_ids)
                aug_image  = result["image"]
                aug_bboxes = [[max(0.0, min(1.0, v)) for v in bb] for bb in result["bboxes"]]
                aug_labels = list(result["class_labels"])
            except Exception:
                continue

            if cid not in aug_labels:
                continue

            out_name = f"{src_img_path.stem}_bal_cls{cid}_{generated}{src_img_path.suffix}"
            cv2.imwrite(str(img_dir / out_name), cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR))
            write_yolo_label(lbl_dir / (Path(out_name).stem + ".txt"), aug_labels, aug_bboxes)

            generated_log.append({"source": src_img_path.name, "generated": out_name, "target_class": cid})
            generated += 1

        print(f"   Clase {cid}: generadas {generated}/{needed}")

    # Estadísticas finales
    class_count_after, _ = analyze_distribution(output_dir)

    log = {
        "min_samples_threshold": min_samples,
        "minority_classes": {str(k): v for k, v in minority.items()},
        "generated": generated_log,
        "class_distribution_before": {str(k): v for k, v in class_count.items()},
        "class_distribution_after":  {str(k): v for k, v in class_count_after.items()},
    }

    with open(output_dir / "balancing_log.json", "w") as f:
        json.dump(log, f, indent=2)

    print(f"\n✅ Balanceo completado. Log: {output_dir / 'balancing_log.json'}")
    _plot_distribution(class_count, class_count_after, output_dir, min_samples)


def _plot_distribution(before: dict, after: dict, output_dir: Path, threshold: int):
    all_classes = sorted(set(list(before.keys()) + list(after.keys())))
    before_vals = [before.get(c, 0) for c in all_classes]

    fig, axes = plt.subplots(1, 2 if after else 1, figsize=(16, 5))

    def _bar(ax, values, title, color):
        x = np.arange(len(all_classes))
        ax.bar(x, values, color=color, alpha=0.85)
        ax.axhline(y=threshold, color='red', linestyle='--', linewidth=1.5, label=f'Umbral ({threshold})')
        ax.set_title(title)
        ax.set_xlabel("Clase ID")
        ax.set_ylabel("Instancias")
        ax.set_xticks(x)
        ax.set_xticklabels([str(c) for c in all_classes], rotation=90, fontsize=6)
        ax.legend()
        ax.grid(axis='y', alpha=0.4)

    if after:
        after_vals = [after.get(c, 0) for c in all_classes]
        _bar(axes[0], before_vals, "ANTES del Balanceo", "#4C72B0")
        _bar(axes[1], after_vals,  "DESPUÉS del Balanceo", "#55A868")
    else:
        _bar(axes, before_vals, "Distribución de Clases", "#4C72B0")

    plt.suptitle("Análisis Long-Tail — Fase 2 Robust-TT100K", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plot_path = output_dir / "class_balancing.png"
    plt.savefig(str(plot_path), dpi=150)
    plt.close()
    print(f"   Gráfica: {plot_path}")


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir",   default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output_dir",  default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--min_samples", type=int, default=MIN_SAMPLES)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    balance_classes(args.input_dir, args.output_dir, args.min_samples)
