"""
Phase 1 — Retroactive dense remapping of YOLO class IDs.

Fixes audit finding F-3 / D-2 / B-2 of docs/audits/2026-04-17-phase4-audit.md:
    on-disk labels under data/split/{train,val,test}/labels/*.txt carry 143
    distinct class IDs but scattered in the range [0, 213] (non-dense).
    Ultralytics requires label IDs densely packed in [0, nc-1].

This script builds the canonical original_id -> dense_id mapping from the set
of IDs actually present on disk, emits it to data/processed/class_id_map.json
(augmented with TT100K class names pulled from the pre-fix dataset.yaml when
available), and rewrites every .txt label file in place.

Idempotent: if every scanned label already uses IDs in [0, N-1] with the
expected density, the script exits without writing.

Usage:
    python scripts/phase1_remap_labels.py --dry-run
    python scripts/phase1_remap_labels.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_MAP_SOURCE_GLOBS = (
    # Source of truth for the class-ID space (audit §6.3).
    # Using train-only enforces the architectural contract nc=143. Val/test
    # contain IDs absent from train — annotations referring to those classes
    # are dropped at rewrite time (see REWRITE_GLOBS below).
    "data/split/train/labels/*.txt",
)

DEFAULT_REWRITE_GLOBS = (
    "data/split/train/labels/*.txt",
    "data/split/val/labels/*.txt",
    "data/split/test/labels/*.txt",
    "data/augmented/train/labels/*.txt",
)

CLASS_MAP_JSON = PROJECT_ROOT / "data" / "processed" / "class_id_map.json"
PRE_FIX_DATASET_YAML = PROJECT_ROOT / "data" / "processed" / "dataset.yaml"


def _iter_label_files(globs: Iterable[str]) -> list[Path]:
    files: list[Path] = []
    for g in globs:
        files.extend((PROJECT_ROOT).glob(g))
    return sorted(files)


def _collect_original_ids(files: Iterable[Path]) -> set[int]:
    ids: set[int] = set()
    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                try:
                    ids.add(int(parts[0]))
                except ValueError:
                    continue
    return ids


def _load_pre_fix_names() -> dict[int, str]:
    """Read the (pre-fix) dataset.yaml names map, original_id -> class_name.

    Used only to enrich class_id_map.json with human-readable labels; if the
    file is absent or malformed the mapping is emitted without names.
    """
    if not PRE_FIX_DATASET_YAML.exists():
        return {}
    try:
        import yaml
    except ImportError:
        return {}
    with open(PRE_FIX_DATASET_YAML, "r", encoding="utf-8") as f:
        d = yaml.safe_load(f) or {}
    raw = d.get("names", {}) or {}
    out: dict[int, str] = {}
    if isinstance(raw, dict):
        for k, v in raw.items():
            try:
                out[int(k)] = str(v)
            except (TypeError, ValueError):
                continue
    elif isinstance(raw, list):
        for i, v in enumerate(raw):
            out[i] = str(v)
    return out


def build_dense_map(original_ids: set[int]) -> dict[int, int]:
    """original_id -> dense_id, stable (sorted-ascending)."""
    return {orig: dense for dense, orig in enumerate(sorted(original_ids))}


def remap_labels_dense(
    map_source_globs: Iterable[str] = DEFAULT_MAP_SOURCE_GLOBS,
    rewrite_globs: Iterable[str] = DEFAULT_REWRITE_GLOBS,
    dry_run: bool = False,
    class_map_path: Path = CLASS_MAP_JSON,
    verbose: bool = True,
) -> dict[int, int]:
    """
    Build a dense [0, N-1] class-ID map from *map_source_globs* (the source
    of truth — typically train labels) and rewrite every file matched by
    *rewrite_globs* in place using that map.

    Lines in the rewrite set whose class ID does NOT appear in the source map
    are dropped (and counted per-file); this occurs when val/test or
    augmented splits reference classes absent from train. A per-split drop
    count is printed for operator review.

    Idempotent: if the source labels are already dense-packed and the rewrite
    set has no out-of-map IDs, no files are touched.

    Importable from scripts.phase1_data_acquisition as a pipeline step
    (audit F-3 part 2).
    """
    source_files = _iter_label_files(map_source_globs)
    rewrite_files = _iter_label_files(rewrite_globs)
    if not source_files:
        if verbose:
            print(f"[remap] no map-source files matched: {list(map_source_globs)}")
        return {}

    original_ids = _collect_original_ids(source_files)
    if not original_ids:
        if verbose:
            print("[remap] map-source files contain no class IDs")
        return {}

    n = len(original_ids)
    max_id = max(original_ids)
    dense_map = build_dense_map(original_ids)
    names = _load_pre_fix_names()

    if verbose:
        preview = list(dense_map.items())[:5]
        print(f"[remap] map source:        {len(source_files)} files")
        print(f"[remap] map size:          {n} classes")
        print(f"[remap] original max ID:   {max_id}")
        print(f"[remap] dense target max:  {n - 1}")
        print(f"[remap] first 5 pairs:     {preview}")
        print(f"[remap] rewrite targets:   {len(rewrite_files)} files")

    # Idempotency: source already dense AND no rewrite-set IDs outside the map.
    rewrite_ids = _collect_original_ids(rewrite_files)
    already_dense = max_id == n - 1 and original_ids == set(range(n))
    no_overflow = rewrite_ids.issubset(original_ids)
    if already_dense and no_overflow:
        if verbose:
            print(f"[remap] already dense-packed ({n} classes) — nothing to do")
        if not class_map_path.exists():
            _write_class_map(dense_map, names, class_map_path, dry_run, verbose)
        return dense_map

    out_of_map = rewrite_ids - original_ids
    if out_of_map and verbose:
        print(
            f"[remap] WARNING: {len(out_of_map)} IDs present in rewrite set "
            f"but absent from map source — lines will be dropped: "
            f"{sorted(out_of_map)}"
        )

    if dry_run:
        if verbose:
            print("[remap] DRY RUN — no files written")
        return dense_map

    per_dir_dropped: dict[str, int] = {}
    total_dropped = 0

    for fp in rewrite_files:
        with open(fp, "r", encoding="utf-8") as f:
            lines = f.readlines()
        out_lines: list[str] = []
        dropped_here = 0
        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue
            try:
                orig = int(parts[0])
            except ValueError:
                out_lines.append(line.rstrip("\n"))
                continue
            if orig not in dense_map:
                dropped_here += 1
                continue
            parts[0] = str(dense_map[orig])
            out_lines.append(" ".join(parts))
        with open(fp, "w", encoding="utf-8") as f:
            f.write("\n".join(out_lines))
            if out_lines:
                f.write("\n")
        if dropped_here:
            key = str(fp.parent.relative_to(PROJECT_ROOT)).replace("\\", "/")
            per_dir_dropped[key] = per_dir_dropped.get(key, 0) + dropped_here
            total_dropped += dropped_here

    _write_class_map(dense_map, names, class_map_path, dry_run=False, verbose=verbose)

    if verbose:
        print(f"[remap] rewrote {len(rewrite_files)} label files in place")
        if total_dropped:
            print(f"[remap] dropped {total_dropped} annotation lines (out-of-map IDs):")
            for d, c in sorted(per_dir_dropped.items()):
                print(f"    {d}: {c}")
        print(f"[remap] class map written to {class_map_path}")
    return dense_map


def _write_class_map(
    dense_map: dict[int, int],
    names: dict[int, str],
    path: Path,
    dry_run: bool,
    verbose: bool,
) -> None:
    entries = [
        {
            "original_id": orig,
            "dense_id": dense,
            "name": names.get(orig, f"class_{orig}"),
        }
        for orig, dense in sorted(dense_map.items(), key=lambda kv: kv[1])
    ]
    if dry_run:
        if verbose:
            print(f"[remap] DRY RUN — would write {len(entries)} entries to {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)


def _cli() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1] if __doc__ else "")
    p.add_argument("--dry-run", action="store_true", help="report without writing")
    args = p.parse_args()
    remap_labels_dense(dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(_cli())
