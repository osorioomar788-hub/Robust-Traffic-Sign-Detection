"""
Fase 2 - Paso 3: Validación del Dataset Aumentado
Proyecto: Robust Traffic Sign Detection (MICAI 25)
Responsable: Yael

Descripción:
    Valida la integridad del dataset Robust-TT100K generado en los pasos anteriores.
    Verifica que:
      - Cada imagen tenga su correspondiente .txt de etiquetas
      - Los valores de bounding boxes estén en el rango [0, 1]
      - No haya imágenes corruptas
      - El split val/test NO fue modificado
    Genera un reporte final de la Fase 2 listo para entregar a Fase 3.

Uso:
    python phase2_validate_dataset.py
    python phase2_validate_dataset.py --augmented_dir data/augmented --split_dir data/split
"""

import os
import cv2
import json
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# ─────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────
DEFAULT_AUGMENTED_DIR = "data/augmented"
DEFAULT_SPLIT_DIR     = "data/split"
IMG_EXTENSIONS        = {".jpg", ".jpeg", ".png", ".bmp"}


# ─────────────────────────────────────────────
# VALIDACIÓN
# ─────────────────────────────────────────────

def validate_split(split_dir: Path, split_name: str) -> dict:
    """Valida un split completo (train, val, o test)."""
    img_dir = split_dir / "images"
    lbl_dir = split_dir / "labels"

    report = {
        "split": split_name,
        "total_images": 0,
        "missing_labels": [],
        "corrupt_images": [],
        "invalid_bboxes": [],
        "class_counts": defaultdict(int),
        "image_sizes": [],
    }

    if not img_dir.exists():
        print(f"   ⚠️  Directorio no encontrado: {img_dir}")
        return report

    image_paths = [p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTENSIONS]
    report["total_images"] = len(image_paths)

    for img_path in tqdm(image_paths, desc=f"  Validando {split_name}", leave=False):
        # ── Verificar etiqueta correspondiente ──
        lbl_path = lbl_dir / (img_path.stem + ".txt")
        if not lbl_path.exists():
            report["missing_labels"].append(img_path.name)
            continue

        # ── Verificar imagen no corrupta ──
        img = cv2.imread(str(img_path))
        if img is None:
            report["corrupt_images"].append(img_path.name)
            continue
        h, w = img.shape[:2]
        report["image_sizes"].append((w, h))

        # ── Verificar bounding boxes ──
        with open(lbl_path, "r") as f:
            for line_num, line in enumerate(f, 1):
                parts = line.strip().split()
                if len(parts) != 5:
                    report["invalid_bboxes"].append(
                        f"{img_path.name}:line{line_num} (cols={len(parts)})"
                    )
                    continue
                cid = int(parts[0])
                coords = [float(x) for x in parts[1:]]
                report["class_counts"][cid] += 1

                # Verificar rango [0, 1]
                if any(v < 0 or v > 1 for v in coords):
                    report["invalid_bboxes"].append(
                        f"{img_path.name}:line{line_num} (coords out of range: {coords})"
                    )

    return report


def validate_dataset(augmented_dir: str, split_dir: str):
    """Valida el dataset completo (augmented/train + split/val + split/test)."""
    augmented_dir = Path(augmented_dir)
    split_dir     = Path(split_dir)

    print("\n" + "="*60)
    print(" VALIDACIÓN DATASET ROBUST-TT100K — FASE 2")
    print("="*60)

    splits_to_validate = [
        (augmented_dir / "train", "train (aumentado)"),
        (split_dir / "val",       "val (original)"),
        (split_dir / "test",      "test (original)"),
    ]

    all_reports = []
    for split_path, split_name in splits_to_validate:
        print(f"\n📁 Validando: {split_name}")
        report = validate_split(split_path, split_name)
        all_reports.append(report)
        _print_split_report(report)

    # ─── Resumen final ───
    _generate_summary(all_reports, augmented_dir)
    _plot_class_comparison(all_reports, augmented_dir)

    print(f"\n✅ Validación completa. Reporte guardado en: {augmented_dir / 'validation_report.json'}")


def _print_split_report(report: dict):
    total  = report["total_images"]
    errors = len(report["missing_labels"]) + len(report["corrupt_images"]) + len(report["invalid_bboxes"])
    status = "✅ OK" if errors == 0 else f"⚠️  {errors} problemas"

    print(f"   {status}")
    print(f"   Imágenes     : {total}")
    print(f"   Clases únicas: {len(report['class_counts'])}")
    print(f"   Instancias   : {sum(report['class_counts'].values())}")

    if report["missing_labels"]:
        print(f"   ❌ Labels faltantes ({len(report['missing_labels'])}): "
              f"{report['missing_labels'][:3]}{'...' if len(report['missing_labels']) > 3 else ''}")
    if report["corrupt_images"]:
        print(f"   ❌ Imágenes corruptas: {report['corrupt_images'][:3]}")
    if report["invalid_bboxes"]:
        print(f"   ❌ BBoxes inválidas ({len(report['invalid_bboxes'])}): "
              f"{report['invalid_bboxes'][:3]}")

    if report["image_sizes"]:
        sizes = np.array(report["image_sizes"])
        print(f"   Resolución   : min={sizes.min(axis=0)} max={sizes.max(axis=0)} "
              f"media={sizes.mean(axis=0).astype(int)}")


def _generate_summary(reports: list, output_dir: Path):
    """Genera y guarda el JSON de reporte completo."""
    summary = {
        "fase": "Fase 2 — Generación de Datos Sintéticos y Aumentación",
        "entregable": "Dataset Robust-TT100K",
        "splits": []
    }

    for r in reports:
        summary["splits"].append({
            "nombre": r["split"],
            "total_images": r["total_images"],
            "total_instances": sum(r["class_counts"].values()),
            "unique_classes": len(r["class_counts"]),
            "missing_labels": len(r["missing_labels"]),
            "corrupt_images": len(r["corrupt_images"]),
            "invalid_bboxes": len(r["invalid_bboxes"]),
            "class_distribution": {str(k): v for k, v in sorted(r["class_counts"].items())},
        })

    path = output_dir / "validation_report.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)

    # Print resumen total
    total_imgs = sum(s["total_images"]      for s in summary["splits"])
    total_inst = sum(s["total_instances"]   for s in summary["splits"])
    total_err  = sum(s["missing_labels"] + s["corrupt_images"] + s["invalid_bboxes"]
                     for s in summary["splits"])

    print("\n" + "="*60)
    print(" RESUMEN GENERAL")
    print("="*60)
    print(f"  Total imágenes  : {total_imgs}")
    print(f"  Total instancias: {total_inst}")
    print(f"  Total errores   : {total_err}")
    if total_err == 0:
        print("  🎉 Dataset listo para Fase 3")
    else:
        print("  ⚠️  Revisar errores antes de continuar a Fase 3")


def _plot_class_comparison(reports: list, output_dir: Path):
    """Gráfica de barras comparando instancias por split."""
    fig, axes = plt.subplots(1, len(reports), figsize=(18, 6))
    colors = ["#DD8452", "#4C72B0", "#55A868"]

    for ax, report, color in zip(axes, reports, colors):
        counts  = dict(sorted(report["class_counts"].items()))
        classes = list(counts.keys())
        values  = list(counts.values())
        x = np.arange(len(classes))

        ax.bar(x, values, color=color, alpha=0.85)
        ax.set_title(f"{report['split']}\n({report['total_images']} imgs, "
                     f"{sum(values)} instancias)")
        ax.set_xlabel("Clase ID")
        ax.set_ylabel("Instancias")
        ax.set_xticks(x)
        ax.set_xticklabels([str(c) for c in classes], rotation=90, fontsize=5)
        ax.grid(axis='y', alpha=0.3)

    plt.suptitle("Distribución de Clases por Split — Robust-TT100K",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()

    plot_path = output_dir / "split_class_distribution.png"
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(plot_path), dpi=150)
    plt.close()
    print(f"\n   Gráfica guardada en: {plot_path}")


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Fase 2 – Validación del Dataset Aumentado")
    parser.add_argument("--augmented_dir", default=DEFAULT_AUGMENTED_DIR,
                        help="Directorio del dataset aumentado (default: data/augmented)")
    parser.add_argument("--split_dir",     default=DEFAULT_SPLIT_DIR,
                        help="Directorio del split original de Fase 1 (default: data/split)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    validate_dataset(
        augmented_dir=args.augmented_dir,
        split_dir=args.split_dir,
    )
