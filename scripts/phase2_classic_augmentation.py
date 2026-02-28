"""
Fase 2 - Paso 1: Aumentación Clásica con Albumentations
Proyecto: Robust Traffic Sign Detection (MICAI 25)
Responsable: Yael

Compatibilidad: albumentations==1.3.1 (sin parámetro seed en Compose)

Uso:
    python scripts/phase2_classic_augmentation.py
    python scripts/phase2_classic_augmentation.py --multiplier 3
"""

import os
import cv2
import json
import random
import argparse
import numpy as np
import albumentations as A
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# ─────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────
DEFAULT_INPUT_DIR  = "data/split/train"
DEFAULT_OUTPUT_DIR = "data/augmented/train"
DEFAULT_MULTIPLIER = 3
SEED               = 42
IMG_EXTENSIONS     = {".jpg", ".jpeg", ".png", ".bmp"}


# ─────────────────────────────────────────────
# PIPELINES DE AUMENTACIÓN
# ─────────────────────────────────────────────

def build_pipelines(seed: int) -> dict:
    """Construye los pipelines. Compatible con albumentations 1.3.1."""
    random.seed(seed)
    np.random.seed(seed)

    bp = A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.3)

    general = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.8),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=30, val_shift_limit=20, p=0.5),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.4),
        A.GaussianBlur(blur_limit=(3, 5), p=0.3),
        A.CLAHE(clip_limit=4.0, p=0.3),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=5,
                           border_mode=cv2.BORDER_REFLECT_101, p=0.5),
    ], bbox_params=bp)

    rain = A.Compose([
        A.MotionBlur(blur_limit=(7, 15), p=0.9),
        A.RandomBrightnessContrast(brightness_limit=(-0.4, -0.1), contrast_limit=(-0.2, 0.1), p=1.0),
        A.GaussNoise(var_limit=(20.0, 80.0), p=0.7),
        A.RandomRain(slant_lower=-10, slant_upper=10, drop_length=20,
                     drop_width=1, drop_color=(180, 180, 180),
                     rain_type="drizzle", p=0.8),
        A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=-20, val_shift_limit=-30, p=0.6),
    ], bbox_params=bp)

    fog = A.Compose([
        A.RandomFog(fog_coef_lower=0.3, fog_coef_upper=0.7, alpha_coef=0.1, p=0.9),
        A.RandomBrightnessContrast(brightness_limit=(0.0, 0.3), contrast_limit=(-0.4, -0.1), p=0.8),
        A.GaussianBlur(blur_limit=(5, 11), p=0.7),
        A.HueSaturationValue(hue_shift_limit=0, sat_shift_limit=-40, val_shift_limit=20, p=0.6),
    ], bbox_params=bp)

    night = A.Compose([
        A.RandomBrightnessContrast(brightness_limit=(-0.6, -0.3), contrast_limit=(-0.2, 0.2), p=1.0),
        A.GaussNoise(var_limit=(30.0, 100.0), p=0.8),
        A.RGBShift(r_shift_limit=-10, g_shift_limit=-10, b_shift_limit=20, p=0.7),
        A.GaussianBlur(blur_limit=(3, 7), p=0.4),
        A.CLAHE(clip_limit=2.0, p=0.5),
    ], bbox_params=bp)

    return {"general": general, "rain": rain, "fog": fog, "night": night}


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


def write_yolo_label(label_path: Path, class_ids: list, bboxes: list):
    with open(label_path, "w") as f:
        for cid, bb in zip(class_ids, bboxes):
            bb = [max(0.0, min(1.0, v)) for v in bb]
            f.write(f"{cid} {bb[0]:.6f} {bb[1]:.6f} {bb[2]:.6f} {bb[3]:.6f}\n")


# ─────────────────────────────────────────────
# AUMENTACIÓN PRINCIPAL
# ─────────────────────────────────────────────

def augment_dataset(input_dir: str, output_dir: str, multiplier: int, seed: int):
    input_dir  = Path(input_dir)
    output_dir = Path(output_dir)

    img_input  = input_dir / "images"
    lbl_input  = input_dir / "labels"
    img_output = output_dir / "images"
    lbl_output = output_dir / "labels"

    img_output.mkdir(parents=True, exist_ok=True)
    lbl_output.mkdir(parents=True, exist_ok=True)

    pipelines      = build_pipelines(seed)
    pipeline_names = list(pipelines.keys())

    image_paths = [p for p in img_input.iterdir() if p.suffix.lower() in IMG_EXTENSIONS]

    if not image_paths:
        print(f"⚠️  No se encontraron imágenes en: {img_input}")
        return

    print(f"\n✅ Imágenes encontradas : {len(image_paths):,}")
    print(f"📦 Versiones por imagen : x{multiplier} ({len(pipeline_names)} pipelines)")
    print(f"📊 Total esperado       : ~{len(image_paths) * (multiplier + 1):,} imágenes\n")

    log = {"total_original": len(image_paths), "multiplier": multiplier, "augmentations": []}
    class_before, class_after = {}, {}

    for img_path in tqdm(image_paths, desc="Aumentando"):
        lbl_path  = lbl_input / (img_path.stem + ".txt")
        class_ids, bboxes = read_yolo_label(lbl_path)

        for cid in class_ids:
            class_before[cid] = class_before.get(cid, 0) + 1

        image = cv2.imread(str(img_path))
        if image is None:
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # ── Copiar original ──
        cv2.imwrite(str(img_output / img_path.name), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        write_yolo_label(lbl_output / (img_path.stem + ".txt"), class_ids, bboxes)
        for cid in class_ids:
            class_after[cid] = class_after.get(cid, 0) + 1

        # ── Versiones aumentadas ──
        for i in range(multiplier):
            pipeline_name = pipeline_names[i % len(pipeline_names)]
            pipeline      = pipelines[pipeline_name]

            try:
                result     = pipeline(image=image, bboxes=bboxes, class_labels=class_ids)
                aug_image  = result["image"]
                aug_bboxes = [[max(0.0, min(1.0, v)) for v in bb] for bb in result["bboxes"]]
                aug_labels = list(result["class_labels"])
            except Exception:
                aug_image, aug_bboxes, aug_labels = image.copy(), bboxes, class_ids

            suffix   = f"_aug_{pipeline_name}_{i}"
            out_name = img_path.stem + suffix + img_path.suffix
            cv2.imwrite(str(img_output / out_name), cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR))
            write_yolo_label(lbl_output / (img_path.stem + suffix + ".txt"), aug_labels, aug_bboxes)

            for cid in aug_labels:
                class_after[cid] = class_after.get(cid, 0) + 1

            log["augmentations"].append({"original": img_path.name, "pipeline": pipeline_name})

    # ── Guardar log ──
    log["class_distribution_before"] = class_before
    log["class_distribution_after"]  = class_after
    log["total_augmented"] = len(list(img_output.iterdir()))

    with open(output_dir / "augmentation_log.json", "w") as f:
        json.dump(log, f, indent=2)

    print(f"\n✅ Aumentación completada")
    print(f"   Originales  : {log['total_original']:,}")
    print(f"   Total final : {log['total_augmented']:,}")

    _plot_distribution(class_before, class_after, output_dir)


def _plot_distribution(before: dict, after: dict, output_dir: Path):
    all_classes = sorted(set(list(before.keys()) + list(after.keys())))
    before_vals = [before.get(c, 0) for c in all_classes]
    after_vals  = [after.get(c, 0)  for c in all_classes]
    x = np.arange(len(all_classes))

    fig, ax = plt.subplots(figsize=(max(12, len(all_classes) * 0.5), 6))
    ax.bar(x - 0.2, before_vals, 0.4, label="Antes",   color="#4C72B0", alpha=0.8)
    ax.bar(x + 0.2, after_vals,  0.4, label="Después", color="#DD8452", alpha=0.8)
    ax.set_xlabel("Clase ID")
    ax.set_ylabel("Instancias")
    ax.set_title("Distribución de Clases: Antes vs Después de Aumentación")
    ax.set_xticks(x)
    ax.set_xticklabels([str(c) for c in all_classes], rotation=90, fontsize=6)
    ax.legend()
    ax.grid(axis="y", alpha=0.4)
    plt.tight_layout()
    plt.savefig(str(output_dir / "class_distribution.png"), dpi=150)
    plt.close()
    print(f"   Gráfica     : {output_dir / 'class_distribution.png'}")


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir",  default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--multiplier", type=int, default=DEFAULT_MULTIPLIER)
    parser.add_argument("--seed",       type=int, default=SEED)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    augment_dataset(args.input_dir, args.output_dir, args.multiplier, args.seed)
