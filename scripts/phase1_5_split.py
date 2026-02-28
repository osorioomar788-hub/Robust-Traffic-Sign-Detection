"""
Fase 1.5: Split Estratificado del Dataset Procesado
Proyecto: Robust Traffic Sign Detection (MICAI 25)
Responsable: Yael

Descripción:
    Toma los datos ya procesados por la Fase 1 (data/processed/) y los
    distribuye en los splits finales (data/split/) listos para la Fase 2.

    Estrategia:
        processed/train/ → split/train/ (70%) + split/val/ (30%)
        processed/test/  → split/test/  (100%, intacto)

Uso:
    python phase1_5_split.py
"""

import os
import shutil
import random
from pathlib import Path
from tqdm import tqdm

# ─────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────
PROCESSED_DIR = Path("data/processed")
SPLIT_DIR     = Path("data/split")
VAL_RATIO     = 0.30   # 30% de train → val
SEED          = 42
IMG_EXTENSIONS = {".jpg", ".jpeg", ".png"}

random.seed(SEED)


def get_image_label_pairs(images_dir: Path, labels_dir: Path) -> list:
    """Retorna lista de (img_path, lbl_path) donde ambos existen."""
    pairs = []
    for img_path in images_dir.iterdir():
        if img_path.suffix.lower() not in IMG_EXTENSIONS:
            continue
        lbl_path = labels_dir / (img_path.stem + ".txt")
        if lbl_path.exists():
            pairs.append((img_path, lbl_path))
        else:
            # Incluir imágenes sin etiqueta (fondo/negativo)
            pairs.append((img_path, None))
    return pairs


def copy_pairs(pairs: list, dest_img_dir: Path, dest_lbl_dir: Path, desc: str):
    """Copia pares imagen+etiqueta al destino."""
    dest_img_dir.mkdir(parents=True, exist_ok=True)
    dest_lbl_dir.mkdir(parents=True, exist_ok=True)

    for img_path, lbl_path in tqdm(pairs, desc=desc):
        shutil.copy2(img_path, dest_img_dir / img_path.name)
        if lbl_path:
            shutil.copy2(lbl_path, dest_lbl_dir / (img_path.stem + ".txt"))


def run_split():
    print("=" * 55)
    print(" SPLIT ESTRATIFICADO — Fase 1.5")
    print("=" * 55)

    # ── Obtener pares de processed/train ──
    train_imgs = PROCESSED_DIR / "train" / "images"
    train_lbls = PROCESSED_DIR / "train" / "labels"
    pairs_train = get_image_label_pairs(train_imgs, train_lbls)
    random.shuffle(pairs_train)

    # ── Dividir en train/val ──
    n_val   = int(len(pairs_train) * VAL_RATIO)
    n_train = len(pairs_train) - n_val

    final_train = pairs_train[:n_train]
    final_val   = pairs_train[n_train:]

    # ── Obtener pares de processed/test ──
    test_imgs = PROCESSED_DIR / "test" / "images"
    test_lbls = PROCESSED_DIR / "test" / "labels"
    final_test = get_image_label_pairs(test_imgs, test_lbls)

    print(f"\n  Split calculado:")
    print(f"    train : {len(final_train):,} imágenes ({100-VAL_RATIO*100:.0f}%)")
    print(f"    val   : {len(final_val):,} imágenes ({VAL_RATIO*100:.0f}%)")
    print(f"    test  : {len(final_test):,} imágenes (100% de processed/test)")
    print(f"    TOTAL : {len(final_train)+len(final_val)+len(final_test):,} imágenes\n")

    # ── Copiar a data/split/ ──
    copy_pairs(final_train, SPLIT_DIR/"train"/"images", SPLIT_DIR/"train"/"labels", "  Copiando train")
    copy_pairs(final_val,   SPLIT_DIR/"val"/"images",   SPLIT_DIR/"val"/"labels",   "  Copiando val  ")
    copy_pairs(final_test,  SPLIT_DIR/"test"/"images",  SPLIT_DIR/"test"/"labels",  "  Copiando test ")

    print("\n✅ Split completado.")
    print(f"   Resultado en: {SPLIT_DIR.absolute()}")
    print("\n▶  Ahora ejecuta: python scripts/phase2_classic_augmentation.py")


if __name__ == "__main__":
    run_split()
