# Changelog

Project-level changelog for Robust Traffic Sign Detection (MICAI 2025).

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
the project does not yet adopt SemVer, so entries are grouped by **audit
round** + **fix item** rather than by released version.

---

## 2026-05-04 — Phase 4 audit closure (round 1)

Closes the 21 findings catalogued in [`docs/audits/2026-04-17-phase4-audit.md`](audits/2026-04-17-phase4-audit.md). Tier 1 (BLOCKER + HIGH) is fully closed; Tier 2/3 (MEDIUM/LOW + hygiene) is mostly closed, with `F-7` (dataset-fetch script) and `CONTRIBUTING.md` deferred to backlog per PI direction.

### Added
* `scripts/phase4_focal_loss.py` — `FocalBCE` module + `register_custom_loss()` monkey-patch (γ=2.0, α=0.25). Replaces the 1264-LOC orphan `scripts/loss_focal_micai.py` with a 60-LOC in-place patch (audit **F-1**, **F-2**, **A-1**, **A-3**).
* `scripts/phase1_remap_labels.py` — idempotent dense remap of TT100K class IDs from sparse `[0, 231]` to `[0, 142]`. Train labels are the source of truth; out-of-map IDs in val/test/aug are dropped per-line (audit **F-3**).
* `scripts/phase4_validate.py` — Phase-4 smoke test analogous to `phase3_architecture_design.py:main`. Asserts `criterion.bce` is `FocalBCE`, runs forward+backward on synthetic batch, writes `results/phase4_loss_validation.txt` (audit **F-8**, **H-4**).
* `LICENSE` — canonical AGPL-3.0 text (audit **F-9**, **A-2**).
* `README.md` — License, Attribution, and Reproducibility sections citing Ultralytics and Lin et al. 2017 (audit **A-1**, **A-2**, **A-3**).
* `requirements.txt` — pinned `torch==2.4.1`, `torchvision==0.19.1`, `ultralytics==8.3.40` so a fresh checkout can install the training stack (audit **F-5**, **B-4**).
* `data/split/*/labels/*.txt` — committed (images remain ignored by size). `.gitignore` updated accordingly (audit **H-2**, scope-limited to labels only per PI).

### Changed
* `train.py` — rewritten as an argparse-driven driver. Loads the custom YAML (not `yolov8n.pt`), calls `register_custom_modules()` AND `register_custom_loss()` in the correct order before importing YOLO, drops the silent `except ImportError`, and exposes device/workers/epochs/batch/imgsz/seed/gamma/alpha/project/name (audit **F-6**, **B-3**, **M-1**, **M-2**, **M-3**).
* `scripts/phase1_data_acquisition.py` — calls `remap_labels_dense()` immediately before writing `dataset.yaml`, so any future re-run of preprocessing produces dense labels by construction (audit **F-3** part 2).
* `data/processed/dataset.yaml` — rewritten with `nc=143` and 143 dense TT100K class names. The previous file declared `nc=232` while the architecture (`models/configs/yolov8_custom.yaml`) declared `nc=143`, a head-vs-data dimension mismatch (audit **F-4**, **H-1**).
* `requirements.txt` — removed duplicate `opencv-python-headless==4.10.0.84` (kept `opencv-python==4.13.0.90`) so cv2 import resolution is deterministic (audit **L-1**).
* `.gitignore` — removed blanket `data/split/` entry; replaced with `data/split/*/images/` so labels can be tracked (audit **H-2**).

### Removed
* `scripts/loss_focal_micai.py` — 1264-LOC near-verbatim copy of `ultralytics/utils/loss.py`. Replaced by the in-place monkey-patch in `scripts/phase4_focal_loss.py` (audit **F-2**, **A-4**).

### Historical note (cannot be retroactively rewritten)

Commit `99056f0f` ("clean requirements, fix dataset paths, update gitignore and backup focal loss") *also* mutated `nc` from 221 → 232 in `data/processed/dataset.yaml`, an architectural change that the message did not advertise. Force-pushing to rewrite history was rejected by PI; this entry exists so that future readers correlating `git log` with the audit findings can see the discrepancy without spelunking the diff (audit **H-3**, **M-4**, **F-10**).

**Going forward, commits should be atomic and single-intent.** A `chore:` commit that touches `requirements.txt` should not also flip a hyperparameter in `dataset.yaml` — the latter belongs in a separate `feat:` or `fix:` commit with a message that explains the dimension change. (`CONTRIBUTING.md` formalising this is in the backlog.)

### Audit follow-ups (2026-05-04 same-day)
* `data/processed/dataset.yaml` — path resolution failed under Windows when 
  Ultralytics `settings.json` had `datasets_dir` pointing elsewhere; reset 
  required (`yolo settings datasets_dir=''`) and yaml path normalized.
* `scripts/phase4_validate.py` — `model.model(dummy_batch)` failed with 
  `AttributeError: 'dict' object has no attribute 'box'` because `self.hyp` 
  is normally inflated by Trainer; manual `get_cfg(DEFAULT_CFG)` added.
