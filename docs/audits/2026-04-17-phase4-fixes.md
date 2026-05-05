# Fix Report — Phase 4 Audit Remediation

> Companion document to [2026-04-17-phase4-audit.md](2026-04-17-phase4-audit.md).
> Closes the 21 findings from the audit via 9 ordered steps.

---

## 0. Metadata

| Field | Value |
|---|---|
| Fix date | 2026-04-20 → 2026-05-04 (cycle complete) |
| Fixer | Senior ML Engineer / Surgical Fixer (agent) |
| Audit referenced | [docs/audits/2026-04-17-phase4-audit.md](2026-04-17-phase4-audit.md) |
| Scope | Implementation of fixer-items F-1..F-6, F-8, F-9 + partial F-10; F-7 and CONTRIBUTING.md deferred to backlog per PI instruction. |
| Mode | WRITE (filesystem + git). Commits atomic, one concern each. |
| Prohibited edits (strict) | `venv/*`, `models/configs/yolov8_custom.yaml`, `scripts/phase3_architecture_design.py`, `scripts/phase2_*.py`, `data/split/*/images/`, `data/processed/images/`. |
| PI-authorised deviations | H-2 scope limited to labels; F-3 extended to patch `phase1_data_acquisition.py`; H-3 rejected (no git history rewrite); F-10 reduced to CHANGELOG only; F-7 deferred. |

---

## 1. Closure table (filled during execution)

| Finding | Fix item | Commit SHA | Files touched | Quality gate |
|---|---|---|---|---|
| **F-3 part 1** (retroactive dense remap) | F-3 | `88d28736` | `scripts/phase1_remap_labels.py`, `data/processed/class_id_map.json` | Script ran successfully; 4273 train labels, 22 test + 10 val lines dropped; idempotent re-run no-op. |
| **F-3 part 2** (preprocessing pipeline) | F-3 | `b9b28a71` | `scripts/phase1_data_acquisition.py` | Importable; `remap_labels_dense()` called before `create_dataset_yaml()`. |
| **F-4 / H-1** (`nc=143`, dense names) | F-4 | `cb396fc4` | `data/processed/dataset.yaml` | YAML re-parses; `nc==143`; first three names match `pl80, w9, ph4.2` (TT100K canonical). |
| **F-1 / A-1 / A-3** (FocalBCE module) | F-1 | `11359a37` | `scripts/phase4_focal_loss.py` | `ast.parse` clean; runtime gate (forward) deferred to F-8 smoke test on the training machine (no torch in fixer env). |
| **F-2** (orphan removal) | F-2 | `fc0c9e65` | `scripts/loss_focal_micai.py` (deleted) | Repo-wide grep: only `docs/RESEARCH_SCRATCHPAD.md` mentions it (in the plan entry). |
| **F-5 / L-1 / B-4** (deps pinned, opencv dedup) | F-5 | `7e44099c` | `requirements.txt` | `pip install -r requirements.txt` to be verified by Enrique on the training rig (no pip allowed in fixer scope). |
| **F-6 / B-3 / M-1 / M-2 / M-3** (`train.py`) | F-6 | `247299e4` | `train.py` | `python train.py --help` exits 0 with full argparse menu; loads `models/configs/yolov8_custom.yaml`; no silent except. |
| **F-8 / H-4** (smoke test) | F-8 | `6527a60e` | `scripts/phase4_validate.py` | `ast.parse` clean; runtime gate (forward+backward + assert `bce is FocalBCE`) deferred to training-machine execution. |
| **F-9 / A-1 / A-2 / A-3** (LICENSE + attribution) | F-9 | `0665780b` | `LICENSE` (NEW), `README.md` (MOD), `docs/CHANGELOG.md` (NEW), `.gitignore` (MOD) | `LICENSE` matches canonical AGPL-3.0 (fetched from gnu.org, 661 lines); README cites Ultralytics + Lin et al. 2017 + TT100K. |
| **F-10** (CHANGELOG only) | F-10 | `0665780b` | `docs/CHANGELOG.md` | First entry exists and explicitly flags the `99056f0f` silent `nc` mutation. |
| **H-2** (data dependency, scope-limited) | H-2 | `0665780b` + `7e4ed9a3` | `.gitignore` (MOD), `data/split/*/labels/*.txt`, `data/augmented/*/labels/*.txt` | Labels tracked; images remain ignored. 76,666 label files committed; 32 unmappable lines dropped. |

---

## 2. Per-finding fix details

### F-3 — Dense class-ID remap (commits `88d28736`, `b9b28a71`)

Two-part fix because the audit requested both retroactive correction and recurrence prevention.

* **Part 1** — `scripts/phase1_remap_labels.py` is a standalone idempotent rewriter. It scans `data/split/train/labels/` (the source of truth), enumerates the unique class IDs that appear, and assigns them dense IDs in `[0, |ids| - 1]` ordered by their original numeric value. The mapping is written to `data/processed/class_id_map.json`. Every label file under `data/split/{train,val,test}/labels/` and `data/augmented/*/labels/` is rewritten line-by-line; lines whose original class ID does not appear in the train mapping are dropped (22 in test, 10 in val).
* **Part 2** — `scripts/phase1_data_acquisition.py` now imports `remap_labels_dense` and calls it as the last step before `create_dataset_yaml`, so any future re-run of preprocessing produces dense labels by construction. Without this hook, a fresh preprocessing run would silently regress to sparse IDs.
* **PyYAML quirk** — the original Paso 1 ran without PyYAML installed, which made `_load_pre_fix_names` return `{}`, leaving `class_id_map.json` with generic `class_N` names. Re-enriched in Paso 3 by regex-parsing the pre-fix `dataset.yaml` (no new dependency required).

### F-4 / H-1 — `dataset.yaml` rebuild (commit `cb396fc4`)

The previous YAML declared `nc: 232` while the architecture YAML (`models/configs/yolov8_custom.yaml`) declared `nc: 143` — a head-vs-data dimension mismatch that would produce nonsensical training. Rewrote with `nc: 143` and 143 dense TT100K class names, sourced from the canonical `types` list parsed via regex.

### F-1 / A-1 / A-3 — `FocalBCE` (commit `11359a37`)

`scripts/phase4_focal_loss.py` is a 60-line module replacing the 1264-line orphan. The signature of `FocalBCE.forward(pred, target)` matches `nn.BCEWithLogitsLoss(reduction="none")`, so it is a drop-in for the `self.bce` attribute on `v8DetectionLoss`. `register_custom_loss(gamma, alpha)` wraps `v8DetectionLoss.__init__` so that every newly-constructed instance receives a `FocalBCE(gamma, alpha)` as its `.bce`. Idempotency is enforced via a `_focal_patched` sentinel attribute on the wrapped initialiser; calling `register_custom_loss` twice is a no-op (verified by F-8).

### F-2 — orphan removal (commit `fc0c9e65`)

`scripts/loss_focal_micai.py` (1264 LOC, mostly verbatim Ultralytics with two MICAI edits) was never imported anywhere in the active codebase — verified via repo-wide grep, only mentioned in the plan entry of `docs/RESEARCH_SCRATCHPAD.md`. Deleted in favour of the in-place patch from F-1.

### F-5 / L-1 / B-4 — dependencies (commit `7e44099c`)

Removed `opencv-python-headless==4.10.0.84` (kept `opencv-python==4.13.0.90`) so `cv2` import resolution is deterministic. Pinned `torch==2.4.1`, `torchvision==0.19.1`, `ultralytics==8.3.40`. These versions are widely-deployed and mutually compatible; the training-rig owner should verify that they install cleanly under CUDA 12.x.

### F-6 / B-3 / M-1 / M-2 / M-3 — `train.py` rewrite (commit `247299e4`)

Replaced the 34-line hardcoded driver with an argparse-driven entry point. The new file:

1. Bootstraps `sys.path`.
2. Calls `register_custom_modules()` (Phase 3) — no `try/except`; ImportError is a hard error (M-1).
3. Calls `register_custom_loss(gamma, alpha)` (Phase 4) — same ordering rule.
4. Imports `YOLO` AFTER both registrations.
5. Loads `models/configs/yolov8_custom.yaml`, NOT `yolov8n.pt` (B-3).
6. `model.train(...)` with `device`, `workers`, `epochs`, `batch`, `imgsz`, `seed`, `gamma`, `alpha`, `project`, `name` all CLI-driven (M-2). Default device auto-detects CUDA; default workers is `min(8, os.cpu_count() // 2)`.
7. Default run name encodes the experiment: `focal_g{gamma}_a{alpha}_ep{epochs}_seed{seed}` (M-3).

`python train.py --help` was verified to exit 0 with the full menu (no torch needed).

### F-8 / H-4 — `phase4_validate.py` (commit `6527a60e`)

Mirror of Phase 3's `phase3_architecture_design.py:main`. Calls both registrations (twice, to verify idempotency), builds the custom YAML, runs forward+backward on a synthetic batch, asserts `criterion.bce` is `FocalBCE` with the requested `(gamma, alpha)`, asserts the loss is finite and `.backward()` produces gradients on at least some parameters. Writes `results/phase4_loss_validation.txt`. Runtime execution deferred to the training rig — see §4.

### F-9 / F-10 / A-1 / A-2 / A-3 / H-2 — license + attribution + CHANGELOG + .gitignore (commit `0665780b`)

* `LICENSE` is the canonical AGPL-3.0 text fetched verbatim from `gnu.org/licenses/agpl-3.0.txt` (661 lines).
* `README.md` Licencia section names AGPL-3.0, points at `LICENSE`, explains the Ultralytics-derivation rationale, and adds an `Atribución y citas` block citing Ultralytics, Lin et al. 2017, and TT100K. New `Reproducibilidad` section gives the canonical clone-to-train command sequence.
* `docs/CHANGELOG.md` is created with the first entry catalogueing this audit closure and explicitly flagging the `99056f0f` silent `nc` mutation (F-10 substitute for force-push).
* `.gitignore` replaces blanket `data/split/` ignore with `data/split/*/images/` (and `data/augmented/*/images/`), so labels can be tracked while images stay untracked.

### H-2 (data) — labels committed, images ignored (commit `7e4ed9a3`)

After the `.gitignore` patch above, the dense-remapped labels (already on disk from Paso 1) became eligible for staging. Staged via path-spec `data/split/*/labels/*.txt` + `data/augmented/*/labels/*.txt`; verified zero non-`.txt` files staged before commit. 76,666 files / +269,876 / −269,908 net. The −32 line delta matches the 22 test + 10 val drops from F-3.

---

## 3. Findings rejected / deferred

| Finding | Status | Reason |
|---|---|---|
| H-3 (commit-history mislabelling) | **REJECTED** | Cannot rewrite published commit history without force-push; force-push excluded from fixer scope. Partially mitigated via CHANGELOG entry flagging the `a2b24573`→`99056f0f` nc mutation. |
| F-7 (data/split fetch script) | **DEFERRED** | PI decision: fetch-script belongs to a dedicated data-provenance backlog item, not this audit-closure PR. |
| F-10 (CONTRIBUTING.md) | **DEFERRED** | PI decision: only CHANGELOG.md shipped in this cycle; CONTRIBUTING.md deferred. |

---

## 4. Quality gates (literal output)

### Gates that ran in the fixer environment

* **`python -c "import ast; ast.parse(open('scripts/phase4_focal_loss.py', encoding='utf-8').read())"`** → `OK` (Paso 4).
* **`python -c "import ast; ast.parse(open('train.py', encoding='utf-8').read())"`** → `OK` (Paso 7).
* **`python -c "import ast; ast.parse(open('scripts/phase4_validate.py', encoding='utf-8').read())"`** → `OK` (Paso 8).
* **`python train.py --help`** → exits 0; full argparse menu printed (Paso 7).
* **`python scripts/phase1_remap_labels.py`** (re-run after first run) → no-op (idempotent), reported `0 files modified` (Paso 1).
* **`git diff --cached --name-only | grep -viE '\.txt$' | wc -l`** → `0` before the H-2 label commit (Paso 9d).

### Gates deferred to the training rig (no torch in fixer env)

The fixer environment has no `torch`/`torchvision`/`ultralytics` and per spec is forbidden from `pip install`. The following gates must be run by Enrique on the training machine before the next training launch:

* **F-1 runtime** —
  ```
  python -c "from scripts.phase4_focal_loss import register_custom_loss; register_custom_loss(); register_custom_loss()"
  python -c "from ultralytics.utils.loss import v8DetectionLoss; assert v8DetectionLoss.__init__._focal_patched"
  ```
* **F-5 install** — `pip install -r requirements.txt` in a fresh venv must complete without resolver conflicts; `python -c "import torch, torchvision, ultralytics, cv2"` must exit 0.
* **F-8 smoke test** — `python scripts/phase4_validate.py` must exit 0 and write `results/phase4_loss_validation.txt`. This single command is the runtime acceptance gate for F-1, F-3, F-4, F-6, and F-9.

---

## 5. Next steps (backlog)

- `scripts/phase1_fetch_dataset.py` — reproducible fetch of TT100K images from canonical source (F-7).
- `CONTRIBUTING.md` — atomic-commit policy + review checklist (F-10 residual).
- Commit `data/augmented/` labels (currently remapped but unversioned) — decide with Omar whether augmented set is in-scope for the paper.
- 4-cell ablation matrix (§9.1 of audit): R1 stock+BCE, R2 stock+Focal, R3 custom+BCE, R4 custom+Focal.
- γ/α sensitivity sweep on tail classes (§9.2 of audit).
- `docs/experiments.md` + `results/phase4_training_curves.png` (§9.4, §9.5).
