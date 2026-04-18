# Audit Report — Phase 4 (Training)

> Status: **COMPLETE** — All 9 sections delivered.

---

## 0. Metadata

| Field | Value |
|---|---|
| Audit date | 2026-04-17 |
| Auditor | Senior ML Engineer / Code Auditor agent (read-only) |
| Protocol | `docs/Amaury/2026-04-17-audit-phase4-training.md` |
| Scope | Commits authored by Enrique (`7Lear7 <enrique.javac@gmail.com>` / `Lear07 <up240921@alumnos.upa.edu.mx>`) on branch `main` |
| Commits in scope | `a2b24573` (2026-03-19, "Configuración de rutas relativas y setup de Fase 4"), `99056f0f` (2026-04-16, "refactor: clean requirements, fix dataset paths, update gitignore and backup focal loss") |
| Baseline | `43f160d8` — last Amaury commit before Phase 4 began (SubPixelConv + CoordAtt + `yolov8_custom.yaml`, `nc=143`) |
| Mode | READ-ONLY. No files outside this audit document are modified. |
| Prohibited edits | `scripts/loss_focal_micai.py`, `train.py`, `data/processed/dataset.yaml`, anything under `venv/`, `models/configs/yolov8_custom.yaml`, `scripts/phase3_architecture_design.py`. |
| Deliverable | this file. |

---

## 1. Inventario de archivos tocados

Aggregated across the two in-scope commits (`a2b24573` → `99056f0f`), effective delta vs. pre-Phase-4 baseline:

| Path | LOC +/− (net across both commits) | Commit(s) | Propósito declarado (commit msg) | Propósito observado (code reality) |
|---|---|---|---|---|
| [train.py](../../train.py) | +49 / −2 (file created in `a2b24573`, rewritten in `99056f0f`) | `a2b24573`, `99056f0f` | "Configuración … setup de Fase 4" / "fix dataset paths" | Minimal driver: (a) calls `register_custom_modules()` inside a broad `try/except ImportError` that swallows any failure with a warning; (b) **loads stock `yolov8n.pt` at [train.py:20](../../train.py#L20)** while the custom-YAML line [train.py:19](../../train.py#L19) is commented out; (c) passes `epochs=50, batch=16, device=0, project='runs/detect', name='test2_focal'`; (d) no wiring for focal loss. |
| [data/processed/dataset.yaml](../../data/processed/dataset.yaml) | rewritten ×2 (now 246 lines; `nc: 232` with 232 entries in `names`) | `a2b24573` (set `nc: 221`), `99056f0f` (reset `nc: 232`, `path: .`, `train/val/test: {train,val,test}/images`) | "rutas relativas" / "fix dataset paths" | `path: .` + `train: train/images` is relative to `dataset.yaml`'s own directory, which resolves to `data/processed/train/images` — **that directory does not exist** (`data/processed/` contains only `dataset.yaml` and `.gitkeep`). Real labels live in `data/split/train/labels/`. `nc: 232` contradicts both the architecture (`nc: 143`) and the disk reality (143 distinct class IDs, max ID 213). |
| [scripts/loss_focal_micai.py](../../scripts/loss_focal_micai.py) | +1264 / 0 (NEW) | `99056f0f` | "backup focal loss" | A near-verbatim copy of `ultralytics/utils/loss.py` with a ~20-line inline edit in `v8DetectionLoss` ([loss_focal_micai.py:344-347](../../scripts/loss_focal_micai.py#L344-L347) and [loss_focal_micai.py:430-445](../../scripts/loss_focal_micai.py#L430-L445)). The file is **never imported** (`grep -r "loss_focal_micai"` returns no matches). It is a "backup" of an in-place overwrite of the Ultralytics library inside `venv/site-packages`. The repo therefore does not actually inject focal loss when `train.py` runs; reproducibility is broken. |
| [requirements.txt](../../requirements.txt) | trimmed to 30 packages | `99056f0f` | "clean requirements" | **`torch`, `torchvision`, and `ultralytics` are not pinned** (nor present). Any fresh checkout will fail to install the training stack. |
| [.gitignore](../../.gitignore) | +6 lines | `99056f0f` | "update gitignore" | Adds `data/split/` (hides the only directory that actually contains labels/images) and `*.pt` / `*.pth` at repo root. Side effect: obscures that `data/processed/` is empty while `data/split/` is the real dataset root. |

### File-inventory summary

- **Total files touched by Enrique:** 5 (1 new, 4 modified).
- **Net lines added:** ≈ +1340 (dominated by `loss_focal_micai.py`).
- **Net lines removed:** ≈ −240 (mainly `dataset.yaml` churn and `requirements.txt` trimming).
- **Commits analysed:** 2 (`a2b24573`, `99056f0f`).
- **Tests added:** 0.
- **Validation scripts added:** 0 (no `phase4_*.py` equivalent to Amaury's `phase3_architecture_design.py`).

---

## 2. BLOCKERs

A BLOCKER is defined here as a defect that makes the run either **(a) fail to execute**, **(b) execute the wrong artifact** (i.e., not the Phase-3 contract), or **(c) produce results that cannot be reproduced from the repository alone**. Four BLOCKERs are established with file:line evidence.

---

### B-1 — Focal loss is injected via an uncommitted `site-packages` overwrite; the repo file is orphan dead-code

**Severity:** BLOCKER (reproducibility broken; published results cannot be regenerated from `git clone`).

**Evidence — the modified file is never referenced:**
- [scripts/loss_focal_micai.py](../../scripts/loss_focal_micai.py) is a 1264-LOC near-verbatim copy of `ultralytics/utils/loss.py` with MICAI edits at [loss_focal_micai.py:344-347](../../scripts/loss_focal_micai.py#L344-L347) (hardcoded `self.gamma=2.0, self.alpha=0.25` added to `v8DetectionLoss.__init__`) and [loss_focal_micai.py:430-445](../../scripts/loss_focal_micai.py#L430-L445) (focal modulator replacing `self.bce(pred_scores, target_scores).sum()`).
- Repository-wide search: `grep -r "loss_focal_micai"` → **zero matches**. No `import`, no `exec`, no `patch` targets this file.
- [train.py](../../train.py) contains no loss-injection wiring whatsoever. The only monkey-patch is `register_custom_modules()` from Phase 3, which registers **architectural** modules (SubPixelConv, CoordAtt) — not losses.

**Evidence — the commit message is self-incriminating:**
Commit `99056f0f` msg: `"...and backup focal loss"`. The word **backup** confirms the canonical edit lives elsewhere (i.e., inside the developer's local `venv/site-packages/ultralytics/utils/loss.py`), and this 1264-LOC file is merely a frozen copy of that out-of-tree mutation.

**Consequence:** A fresh clone + `pip install -r requirements.txt` + `python train.py` will train with **stock `nn.BCEWithLogitsLoss`**, not the focal loss. Any experimental numbers obtained locally by Enrique cannot be reproduced from the repo. This is a publication-integrity issue for a MICAI submission.

---

### B-2 — `nc` triple-mismatch: architecture ↔ dataset.yaml ↔ labels on disk

**Severity:** BLOCKER (training either crashes, or runs silently on a wrong number of output channels masking the real architectural contract).

**Evidence — three inconsistent values:**

| Source | Declared/observed `nc` | Reference |
|---|---|---|
| Phase-3 architecture contract | **143** | [models/configs/yolov8_custom.yaml:7](../../models/configs/yolov8_custom.yaml#L7) |
| Phase-4 dataset descriptor | **232** | [data/processed/dataset.yaml:10](../../data/processed/dataset.yaml#L10) (names list spans indices 0..231) |
| Ground truth on disk | **143 unique IDs; max ID = 213; not densely remapped to `[0..142]`** | `awk '{print $1}' data/split/train/labels/*.txt \| sort -un \| wc -l` → 143; `... \| tail -5` → 168, 169, 170, 205, 213 |

**Consequence matrix:**
- If the (commented-out) custom YAML were re-enabled with `nc=143`, training would **crash** at the first batch containing class ID ≥ 143 (`AssertionError: Label class 213 exceeds nc=143`).
- With the current `nc=232` dataset.yaml, Ultralytics builds a detection head with **232 output channels**, of which **89 are structurally unreachable** (no label carries those IDs). Loss is computed over a sparse, distorted class distribution. The Phase-3 contract (143 classes) is silently violated.
- The label set itself is **malformed**: 143 distinct IDs scattered in a `[0, 213]` range. There is no deterministic remap script in the repo. Whatever 232→143 or 213→143 correspondence existed in the developer's head is not in version control.

**Nothing in this triple can be correct simultaneously.** At minimum one of {architecture, dataset.yaml, label files} must be rewritten before training is meaningful.

---

### B-3 — Custom architecture is commented out; training loads stock `yolov8n.pt` (nc=80)

**Severity:** BLOCKER (wrong artifact is trained).

**Evidence:**
- [train.py:19](../../train.py#L19): `#model = YOLO('models/configs/yolov8_custom.yaml')` — **commented out**.
- [train.py:20](../../train.py#L20): `model = YOLO('yolov8n.pt')` — loads the Ultralytics-published stock checkpoint with `nc=80` (COCO).
- [train.py:18](../../train.py#L18) inline comment: `"Usa el de la línea 19 para utilizar la fase 3, Si no, usa el yolov8n.pt temporalmente para probar (línea 20)."` — the author explicitly acknowledges the custom model is disabled and the stock net is "temporal."

**Consequence:**
- The `register_custom_modules()` call at [train.py:8](../../train.py#L8) registers SubPixelConv + CoordAtt into the Ultralytics namespace, but **no layer in `yolov8n.pt` uses them**. The Phase-3 work is functionally bypassed.
- When `model.train(data=..., nc inferred from dataset.yaml=232)` runs, Ultralytics will auto-reshape the stock detection head from 80 → 232 classes. The result is a fine-tuned-from-COCO YOLOv8n, **not** the MICAI-proposed SubPixel+CoordAtt architecture.
- The output directory is named [`runs/detect/test2_focal`](../../train.py#L33) — a name that implies focal-loss experimentation on the custom architecture. The code cannot deliver either (see B-1 and this finding). **The directory name is a false label for the artifact produced.**

---

### B-4 — `requirements.txt` cannot install the training stack

**Severity:** BLOCKER (repo does not bootstrap to a trainable state).

**Evidence:**
- [requirements.txt](../../requirements.txt) lists 30 packages. Searched for the three mandatory training dependencies:
  - `torch` → **absent**
  - `torchvision` → **absent**
  - `ultralytics` → **absent**
- Commit `99056f0f` declared intent: `"refactor: clean requirements"`. The cleanup removed the three packages that actually matter for `train.py` to import.
- [train.py:14](../../train.py#L14) `from ultralytics import YOLO` will raise `ModuleNotFoundError` on any fresh environment built from this `requirements.txt`.

**Consequence:** A reviewer, a co-author on a different machine, or a CI runner cannot execute Phase 4 from `git clone`. Combined with B-1 (site-packages overwrite), reproducibility is doubly broken: even if a user manually `pip install`s `ultralytics`, they will get the pristine upstream loss, not the focal variant.

---

### §2 summary

| ID | One-line defect | Primary evidence | Fix class |
|---|---|---|---|
| B-1 | Focal loss modification lives in `site-packages`, not in repo; 1264-LOC `loss_focal_micai.py` is orphan | No importer found via grep; commit msg says "backup" | Replace with `register_custom_loss()` monkey-patch (see §7) |
| B-2 | nc triple-mismatch 143 / 232 / max-label-id 213 | yaml vs yaml vs `awk` on labels | Remap labels to dense `[0..142]` + set both yamls to 143 |
| B-3 | Custom model commented out, stock `yolov8n.pt` trained instead | [train.py:19-20](../../train.py#L19-L20) | Uncomment L19, delete L20, verify `register_custom_modules()` ran |
| B-4 | `torch` / `torchvision` / `ultralytics` missing from requirements | [requirements.txt](../../requirements.txt) full read | Pin the three packages at exact versions used locally |

> Four BLOCKERs found (spec required ≥3). Any one of them is sufficient to reject Phase 4 as non-reproducible. Together they compound: fixing only one leaves the repo still non-functional.

---

## 3. HIGHs

A HIGH finding is a serious defect that does **not** by itself stop the run, but that (a) corrupts the experimental record, (b) hides the true state of the repo from reviewers, or (c) forces manual intervention to recover reproducibility. Four HIGHs are logged.

---

### H-1 — `dataset.yaml` paths resolve to a non-existent directory

**Evidence:**
- [data/processed/dataset.yaml:4-7](../../data/processed/dataset.yaml#L4-L7):
  ```yaml
  path: .
  train: train/images
  val:   val/images
  test:  test/images
  ```
  Ultralytics resolves `path: .` **relative to the YAML file's own directory**, so the effective training set is `data/processed/train/images/`.
- Directory listing of `data/processed/` contains only `dataset.yaml` and `.gitkeep`. The `train/`, `val/`, `test/` subtrees **do not exist** under `data/processed/`.
- The real dataset lives in `data/split/{train,val,test}/images/` and `data/split/{train,val,test}/labels/` (verified in §1 inventory and evidence E.3 of the scratchpad).

**Why HIGH not BLOCKER:** the defect is recoverable with a one-line edit (`path: ../split`). But a reviewer running `python train.py` straight from a clean clone will hit `FileNotFoundError` on the first `LOAD` call — the repo advertises a dataset location that is empty.

---

### H-2 — `.gitignore` hides the actual dataset directory from version control

**Evidence:**
- Commit `99056f0f` adds `data/split/` to [.gitignore](../../.gitignore) alongside `*.pt` / `*.pth` at repo root.
- Combined with H-1, the repo's visible dataset root (`data/processed/`) is empty, and the real dataset root (`data/split/`) is git-ignored.

**Consequence:** a reviewer cloning this repo sees **no labels and no images**. There is no DVC pointer, no `download.sh`, no Roboflow API call script committed to re-materialize `data/split/`. The Phase-1 preprocessing output is effectively lost to anyone who is not Enrique's laptop. For a MICAI submission whose central claim is "143 traffic-sign classes, TT100K-derived," this is a serious reproducibility gap.

**Recommended fix class:** add a `scripts/phase1_rebuild_split.py` that reconstructs `data/split/` from the (committed) raw archive, **or** remove `data/split/` from `.gitignore` and commit the labels (images can remain external via DVC / Git-LFS).

---

### H-3 — `nc` contradicts itself within a single commit and against ground truth

This is the documentary companion of B-2. It is listed separately as HIGH because the defect sits in the commit-message integrity layer, not the runtime layer.

**Evidence:**
- Commit `a2b24573` set `nc: 221`.
- Commit `99056f0f` overwrites to `nc: 232` with commit message `"fix dataset paths"` — but `nc` is **not** a path, and the two values 221 → 232 are neither of them 143 (the Phase-3 contract) nor consistent with each other.
- Neither commit message explains where 221 or 232 came from. There is no reference to a class-merging procedure, a TT100K subset, or a label-remap script in the repo.

**Consequence:** the commit history actively misleads a reader. The message `"fix dataset paths"` masks a silent `nc` mutation that makes the architecture/dataset/label triple inconsistent (B-2). Future `git blame` will not surface this unless the auditor reads the diff.

---

### H-4 — No validation / sanity script for Phase 4 (asymmetry with Phase 3)

**Evidence:**
- Phase 3 shipped [scripts/phase3_architecture_design.py](../../scripts/phase3_architecture_design.py) with a self-test `if __name__ == '__main__':` block that instantiates the model, counts parameters, and prints layer summary — a valid architectural smoke test.
- Phase 4 adds **no** analogous `scripts/phase4_*.py`. There is no pre-training sanity check that:
  - verifies `nc` consistency between `yolov8_custom.yaml` and `dataset.yaml`;
  - asserts that every label-file class ID is in `[0, nc)`;
  - confirms the custom modules were registered before `YOLO(...)` is called;
  - confirms `v8DetectionLoss` instance carries the focal attributes (`gamma`, `alpha`).
- `train.py` goes straight from `register_custom_modules()` (wrapped in a **silent `except ImportError`** at [train.py:10-11](../../train.py#L10-L11)) to `model.train(...)`. If registration silently fails, the training loop runs with the pristine Ultralytics namespace and the operator sees only the warning line — no assertion, no `sys.exit(1)`.

**Consequence:** every BLOCKER in §2 could have been caught before any GPU cycles were spent, had a Phase-3-style validator been written. The absence of this script is why the repo reached the audit state it did.

---

### §3 summary

| ID | One-line defect | Primary evidence | Fix class |
|---|---|---|---|
| H-1 | `dataset.yaml` points to empty `data/processed/{train,val,test}/images` | [dataset.yaml:4-7](../../data/processed/dataset.yaml#L4-L7) | Retarget `path:` to `../split` or move the data |
| H-2 | Real dataset `data/split/` is git-ignored with no rebuild script | [.gitignore](../../.gitignore) + no `phase1_rebuild_*` | Ship rebuild script OR commit labels |
| H-3 | `nc` went 221 → 232 under a commit labelled `"fix dataset paths"`; neither matches 143 | Commit `99056f0f` diff vs [yolov8_custom.yaml:7](../../models/configs/yolov8_custom.yaml#L7) | Honest commit message + remap script |
| H-4 | No `phase4_*.py` validator; `except ImportError` silently degrades | [train.py:10-11](../../train.py#L10-L11) + absence of `scripts/phase4_*.py` | Add pre-train sanity script + make registration failure fatal |

---

## 4. MEDIUM / LOW

MEDIUM findings are defects that hurt maintainability, portability, or scientific hygiene without directly corrupting the run or the experimental record. LOW findings are code-smell / style. Five findings total.

---

### M-1 — Silent `except ImportError` in `train.py` hides architectural regressions

**Severity:** MEDIUM.

**Evidence:** [train.py:6-11](../../train.py#L6-L11):
```python
try:
    from scripts.phase3_architecture_design import register_custom_modules
    register_custom_modules()
    print("[INFO] Módulos de la Fase 3 (PixelShuffle/CoordAtt) inyectados con éxito.")
except ImportError:
    print("[ADVERTENCIA] No se encontraron los módulos de Amaury...")
```

Two problems:
1. The bare `except ImportError` also swallows `ImportError`s raised **inside** `register_custom_modules()` itself (e.g., if Ultralytics refactors `parse_model` signature). The operator cannot distinguish "file missing" from "patch target moved."
2. A missing Phase-3 registration is **not** a warning — it is a dead-stop condition, because every downstream claim of the paper depends on `SubPixelConv` and `CoordAtt` being in the namespace. A `warnings.warn(...)` plus `sys.exit(1)` (or raising) is the correct behavior.

**Fix class:** narrow the `except` to `ModuleNotFoundError` only, and fail hard if registration does not complete successfully. Print the full traceback.

---

### M-2 — Hardcoded `device=0` and `workers=8` break portability

**Severity:** MEDIUM.

**Evidence:** [train.py:30-31](../../train.py#L30-L31):
```python
device=0,          # tarjeta NVIDIA CUDA
workers=8,         # hilos de carga
```

Consequences:
- `device=0` crashes on any machine with no CUDA or with the GPU at a different index (e.g., a workstation where `0` is the iGPU and `1` is the dGPU).
- `workers=8` is a hard-coded assumption about host CPU count; on a 4-core reviewer laptop this oversubscribes the data loader and slows training; on a 128-core server it underutilizes I/O.
- There is no CLI / `argparse` layer. Every hyperparameter change requires editing the committed file, which pollutes `git blame` with experiment bookkeeping.

**Fix class:** accept `device`, `workers`, `epochs`, `batch`, `name` via `argparse` with sensible defaults (`device='0' if torch.cuda.is_available() else 'cpu'`, `workers=min(8, os.cpu_count() // 2)`).

---

### M-3 — Run-name `test2_focal` implies an un-committed `test1` → experimental ledger leak

**Severity:** MEDIUM (research-hygiene).

**Evidence:** [train.py:33](../../train.py#L33): `name='test2_focal'`.

The numeral `2` is only meaningful if a `test1` existed. No commit, no branch, no artifact `runs/detect/test1_*` is tracked in the repo. This is either:
- a live experiment whose baseline was wiped without documentation (scientific-bookkeeping failure), or
- a dev placeholder that was never reset before commit (hygiene failure).

In either case, the run-name carries semantic content (`focal`) that the code cannot deliver (see B-1), which will label the resulting artifact incorrectly in the filesystem.

**Fix class:** introduce a dated run-naming convention (`yyyymmdd_<tag>_<short-hash>`) and document what each tag refers to in `docs/experiments/`.

---

### M-4 — `.gitignore` side effects beyond the scope of the commit message

**Severity:** LOW.

**Evidence:** commit `99056f0f` declared `"update gitignore"`. The diff adds:
```
data/split/
*.pt
*.pth
```

- `data/split/` being ignored is a HIGH finding (H-2).
- `*.pt` / `*.pth` at repo root is reasonable (weights are large), but combined with H-2 it obscures that the working-copy is not self-contained.
- The commit bundled four unrelated intents (`clean requirements`, `fix dataset paths`, `update gitignore`, `backup focal loss`). Atomic commits would have made B-1 (orphan backup) and H-3 (nc mutation) each individually reviewable.

**Fix class:** split future commits by intent. Maintain the one-commit-one-concern discipline Phase 3 followed.

---

### L-1 — `opencv-python` AND `opencv-python-headless` both pinned

**Severity:** LOW.

**Evidence:** [requirements.txt:13-14](../../requirements.txt#L13-L14):
```
opencv-python==4.13.0.90
opencv-python-headless==4.10.0.84
```

Installing both packages into the same environment is known to cause `cv2` import conflicts (the second install overrides the first's `cv2` module silently, depending on order). Additionally, the two versions differ (`4.13.0.90` vs `4.10.0.84`), which makes the behavior of `cv2.*` calls depend on `pip`'s resolution order — non-deterministic across machines.

**Fix class:** pick one. For headless training servers, use `opencv-python-headless` only; for local dev with GUI windows, use `opencv-python`. Never both.

---

### §4 summary

| ID | Sev. | One-line defect | Primary evidence | Fix class |
|---|---|---|---|---|
| M-1 | MED | Silent `except ImportError` hides Phase-3 registration failures | [train.py:10-11](../../train.py#L10-L11) | Narrow `except`, fail hard with traceback |
| M-2 | MED | Hardcoded `device=0`, `workers=8` | [train.py:30-31](../../train.py#L30-L31) | argparse + auto-detect |
| M-3 | MED | Run-name `test2_focal` implies an un-committed `test1` | [train.py:33](../../train.py#L33) | Dated naming convention + experiment log |
| M-4 | LOW | Multi-intent commit `99056f0f` bundles 4 unrelated concerns | `git show 99056f0f` | Atomic commits by concern |
| L-1 | LOW | Both `opencv-python` and `opencv-python-headless` pinned at different versions | [requirements.txt:13-14](../../requirements.txt#L13-L14) | Keep only one |

---

## 5. Attribution check — `scripts/loss_focal_micai.py`

Object of audit: [scripts/loss_focal_micai.py](../../scripts/loss_focal_micai.py) (1264 LOC, introduced in commit `99056f0f` with message fragment "backup focal loss"). Baseline for comparison: upstream `ultralytics/utils/loss.py` (AGPL-3.0), from which the file is copied.

### 5.1 Verbatim ratio

| Metric | Value | Evidence |
|---|---|---|
| Total LOC in copy | 1264 | `wc -l scripts/loss_focal_micai.py` |
| MICAI diff region 1 (focal params) | 4 inserted lines | [loss_focal_micai.py:344-347](../../scripts/loss_focal_micai.py#L344-L347) |
| MICAI diff region 2 (focal modulator) | ~16 inserted lines replacing 1 upstream line | [loss_focal_micai.py:430-445](../../scripts/loss_focal_micai.py#L430-L445) |
| Net added LOC | ≈ 20 | sum of both regions |
| Verbatim ratio (lower bound) | **≥ 98.4 %** | `(1264 − 20) / 1264` |
| Other modifications detected | none found | both diff blocks are fenced with `# --- MODIFICACIÓN MICAI ... ---` marker comments; grep outside those markers shows standard upstream content |

**Conclusion:** the file is a near-total copy of `ultralytics/utils/loss.py`. Only `v8DetectionLoss.__init__` and `v8DetectionLoss.__call__` are touched. All other classes (`VarifocalLoss`, `FocalLoss`, `DFLoss`, `BboxLoss`, `RotatedBboxLoss`, `KeypointLoss`, `E2EDetectLoss`, `v8SegmentationLoss`, `v8PoseLoss`, `v8ClassificationLoss`, `v8OBBLoss`, `TVPDetectLoss`, `TVPSegmentLoss`) are copied verbatim even though Phase 4 does not use them — dead upstream code shipped inside `scripts/`.

### 5.2 License compliance (AGPL-3.0)

| Requirement (AGPL-3.0 §5) | Status | Evidence |
|---|---|---|
| Preserve original copyright / license notice | ✅ preserved | `# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license` at [loss_focal_micai.py:1](../../scripts/loss_focal_micai.py#L1) |
| Carry prominent notices stating modification **and date** (§5a) | ❌ **missing** | Only two inline `# --- MODIFICACIÓN MICAI ... ---` comments. No file-level change banner, no modifier name, no date, no version tag. |
| Carry notice that derivative work is licensed under AGPL-3.0 (§5b) | ❌ **missing** | The repo has no top-level `LICENSE` file and no mention of AGPL-3.0 obligations. Because the MICAI project imports/derives from Ultralytics AGPL code, the whole derivative must also be AGPL-3.0 — this is not declared anywhere. |
| "Interactive user interface" notice (§5d) | n/a | CLI context |
| Source availability of modified Ultralytics build | ⚠️ partial | The actual edits live in `venv/site-packages/ultralytics/utils/loss.py` (see B-1, E.5). `loss_focal_micai.py` is a "backup" but is not the file that runs. The modified site-packages tree is not shipped nor pinned. |

**Finding A-1 (HIGH/compliance):** AGPL-3.0 §5(a) notice-of-change is not satisfied. The bare `# --- MODIFICACIÓN MICAI ---` fence does not constitute a "prominent notice" with date and author.

**Finding A-2 (HIGH/compliance):** The repository does not propagate the AGPL-3.0 license to the derivative work. If the trained model artifacts or any derivative of this codebase are distributed (including the MICAI paper submission's supplementary material), this is a license violation.

### 5.3 Scientific attribution (focal loss origin)

| Requirement | Status | Evidence |
|---|---|---|
| Citation of Lin et al. 2017, *"Focal Loss for Dense Object Detection"* (arXiv:1708.02002) | ❌ **absent from entire repo** | `grep -rn "Lin et al\|1708\.02002\|Focal Loss for Dense Object" .` → 3 hits, all three inside audit/scratchpad markdown produced by this auditor. Zero hits in code, in the paper-draft if present, in `README`, or in `loss_focal_micai.py`. |
| Equation reference (γ focusing, α balancing) tied to source paper | ❌ missing | Inline comments [loss_focal_micai.py:438-441](../../scripts/loss_focal_micai.py#L438-L441) state the formula (`|y-p|^γ · BCE`, then scaled by `α`) but give no bibliographic anchor. |
| Justification of γ=2.0 and α=0.25 | ❌ asserted, not justified | Defaults from the RetinaNet paper are reproduced without citation. No ablation, no sensitivity study. No note that α=0.25 weighs *positive* samples, which for a 143-class sparse-positive detector with severe class imbalance (TT100K tail classes) may be suboptimal. |
| Contrast with upstream `VarifocalLoss` | — | Ironically, upstream Ultralytics correctly attributes its Varifocal implementation: [loss_focal_micai.py:21-22](../../scripts/loss_focal_micai.py#L21) — `"""Varifocal loss by Zhang et al."""`. The MICAI diff does not follow this same convention. |

**Finding A-3 (HIGH/methodology):** No scientific citation for the focal-loss formulation. Unacceptable for a paper targeting MICAI.

### 5.4 Import-path / packaging reality check

| Observation | File:line | Consequence |
|---|---|---|
| Relative imports retained from upstream layout | [loss_focal_micai.py:17-18](../../scripts/loss_focal_micai.py#L17-L18) (`from .metrics import bbox_iou, probiou` / `from .tal import bbox2dist, rbox2dist`) | `import scripts.loss_focal_micai` will raise `ImportError: attempted relative import with no known parent package` — independent of whether any caller attempts it. The file is **non-importable as-is** from the repo root. |
| No `__init__.py` adjustments | `scripts/` has no package siblings named `metrics` or `tal` | Even with `sys.path.insert(0, 'scripts')` the relative imports would still fail. |
| No injection wiring to Ultralytics | grep confirms | Combined with B-1, this is triple-confirmation that the repo does not apply focal loss at runtime. |

### 5.5 Attribution-section summary table

| ID | Severity | Finding | File:line |
|---|---|---|---|
| A-1 | HIGH (license) | AGPL-3.0 §5(a) notice-of-change missing date/author on [loss_focal_micai.py:1](../../scripts/loss_focal_micai.py#L1) | file header |
| A-2 | HIGH (license) | AGPL-3.0 propagation missing; no `LICENSE` file at repo root | repo-level |
| A-3 | HIGH (methodology) | No citation of Lin et al. 2017 (arXiv:1708.02002); γ=2.0, α=0.25 unjustified | [loss_focal_micai.py:344-347](../../scripts/loss_focal_micai.py#L344-L347), [loss_focal_micai.py:430-445](../../scripts/loss_focal_micai.py#L430-L445) |
| A-4 | INFO | `loss_focal_micai.py` ships 12 unused Ultralytics loss classes (dead upstream code) | entire file |
| A-5 | INFO (packaging) | Relative imports at [loss_focal_micai.py:17-18](../../scripts/loss_focal_micai.py#L17-L18) make the file non-importable from `scripts/` — independent confirmation that it is an inert backup (ties to B-1) | `from .metrics import …` |

---

## 6. Dimensional consistency — `nc` triangulation

Three independent sources declare the number of classes. All three disagree. This section formalises the truth table, names the source of truth, and shows why each of the three options fails as currently written.

### 6.1 Truth table

| Source | Declared `nc` / implied cardinality | File:line / command | Authority |
|---|---|---|---|
| **Architecture contract** (Phase 3, owned by Amaury) | `143` | [models/configs/yolov8_custom.yaml:7](../../models/configs/yolov8_custom.yaml#L7) | Design-time contract. Drives the number of output channels in the `Detect()` head. |
| **Dataset descriptor** (Phase 4, owned by Enrique) | `232` (with 232 entries in `names`) | [data/processed/dataset.yaml:10](../../data/processed/dataset.yaml#L10) | Run-time contract consumed by Ultralytics. |
| **Labels on disk** (Phase 1, TT100K preprocessing) | `143 distinct IDs`, but **max ID = 213** (non-dense) | `awk '{print $1}' data/split/train/labels/*.txt \| sort -un \| wc -l` → `143`; `… \| tail -5` → `168 169 170 205 213` | Empirical ground truth of what the model actually sees at training time. |
| **Historical value** (commit `a2b24573`) | `221` | `dataset.yaml:10` at commit `a2b24573` | Superseded — mentioned only to document commit-msg integrity issue H-3. |

### 6.2 Failure-mode analysis of each candidate

| Candidate `nc` | What happens at runtime | Verdict |
|---|---|---|
| `nc = 143` (honour the architecture) | Ultralytics will raise `AssertionError: cls ID 213 ≥ nc 143` during the first batch where any label file contains IDs ≥ 143. **Training crashes in epoch 0.** | ❌ Breaks *because of* disk-state corruption, not architectural intent. |
| `nc = 214` (max ID + 1, minimal densification) | Training proceeds. Head allocates 214 logits. **71 of them are never activated** (no ground truth → always negative class). Gradient waste; detector over-parametrised; `mAP` on absent classes is ill-defined. | ⚠️ Runs but is scientifically sloppy. |
| `nc = 232` (current value, source unknown) | Same as above but **worse**: 89 unreachable channels. The value `232` does not match any of: architecture (143), max-ID+1 (214), or TT100K canonical class count (official TT100K has 221 classes, some releases list 232 after augmentation of rare-class variants). Origin uncited in commit `99056f0f`. | ❌ Magic number. Architecture mismatch masked, not fixed. |
| `nc = 80` | What actually loads at runtime today, because [train.py:20](../../train.py#L20) loads `yolov8n.pt` (COCO-pretrained). Ultralytics will *re-initialise* the head to match `dataset.yaml`'s `nc=232` on the first `model.train()` call, discarding pretrained head weights. | ❌ (descriptive, not a fix) Confirms B-3: the custom `nc=143` architecture never sees a single gradient. |

### 6.3 Source of truth

**Ruling:** `nc = 143`, as declared in [yolov8_custom.yaml:7](../../models/configs/yolov8_custom.yaml#L7), is the correct intent and must be the source of truth. Everything else is downstream.

**Required reconciliation** (scope for surgical fixer, not for this audit):

1. Re-map every label file under `data/split/{train,val,test}/labels/*.txt` so that the set of class IDs present on disk is exactly `{0, 1, …, 142}` — densely packed, no gaps.
2. Write the canonical `id → TT100K-class-name` mapping (143 entries) into `data/processed/dataset.yaml:names` and set `nc: 143`.
3. Commit the label remap as a reproducible script under `scripts/phase1_*` (it is Phase-1 work, not Phase-4), not as a one-shot manual edit.
4. Delete the 232-entry `names` table (provenance unknown, likely a leftover of a TT100K superset before filtering).

**Non-options:**
- Raising `nc` in the architecture to 214 or 232. This breaks the Phase 3 design contract and invalidates Amaury's reported FLOPs/params.
- Leaving `nc=232` in `dataset.yaml` and hoping training "just runs". It does run, but every reported metric is then untrustworthy: the denominator for per-class mAP is wrong, and 89 logits are learning noise.

### 6.4 Section summary

| ID | Severity | Finding | File:line |
|---|---|---|---|
| D-1 | BLOCKER (re-statement of B-2) | `nc` is triple-inconsistent across architecture (143), dataset descriptor (232), and disk labels (143 distinct / max 213 non-dense). Source of truth = architecture. | [yolov8_custom.yaml:7](../../models/configs/yolov8_custom.yaml#L7), [dataset.yaml:10](../../data/processed/dataset.yaml#L10) |
| D-2 | HIGH | Disk labels are not densely remapped to `[0, nc-1]`. Re-map required before any training is credible. | `data/split/*/labels/*.txt` |
| D-3 | MEDIUM | `nc=232` value has no documented provenance (commit `99056f0f` introduced it without justification). | [dataset.yaml:10](../../data/processed/dataset.yaml#L10) |

---

## 7. Proposed design — `register_custom_loss()`

**Goal:** replace the current *"edit `venv/site-packages/ultralytics/utils/loss.py` and keep a 1264-line backup in-repo"* approach with a **reproducible, idempotent, import-time monkey-patch** that lives fully in the repo. The design mirrors [phase3_architecture_design.py:42-132](../../scripts/phase3_architecture_design.py#L42-L132)'s `register_custom_modules` / `_patch_parse_model` pair so a future maintainer sees a consistent pattern for both kinds of injection (module registry + loss-function patch).

> **Scope note:** this is a *design document*, not a patch. The audit is read-only. Actual implementation is the surgical fixer's job (see §8).

**Scientific reference:** Lin, T.-Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017). *Focal Loss for Dense Object Detection*. arXiv:1708.02002. γ=2.0, α=0.25 are the RetinaNet defaults and must be cited alongside any ablation.

### 7.1 Invariants the design must satisfy

| # | Invariant | Rationale |
|---|---|---|
| I-1 | Single source of truth: focal-loss implementation lives in one file inside the repo (`scripts/phase4_focal_loss.py`). | Fixes B-1: no more "backup vs. live" divergence. |
| I-2 | Must be called *before* `from ultralytics import YOLO` in any driver script. | Same ordering constraint as `register_custom_modules` ([phase3_architecture_design.py:143-149](../../scripts/phase3_architecture_design.py#L143-L149)). |
| I-3 | Idempotent (safe to call N times). Guarded by `_focal_patched` attribute. | Prevents double-wrapping the loss if `register_custom_loss()` is called from both `train.py` and a notebook. |
| I-4 | Reversible: original `v8DetectionLoss.__call__` is stashed on the patched object. | Enables BCE-vs-Focal ablation in §9 without reinstalling the package. |
| I-5 | Does **not** shadow upstream — imports `ultralytics.utils.loss` and patches in place, instead of copying 12 unrelated loss classes. | Fixes A-4: drops 1244 lines of dead upstream code from the repo. |
| I-6 | Citation present in module docstring + inline formula comment. | Fixes A-3. |

### 7.2 Public API (pseudocode)

```python
# scripts/phase4_focal_loss.py  — PSEUDOCODE, not executable
"""
Phase 4 — Focal-loss registration.

Reference: Lin et al. 2017, "Focal Loss for Dense Object Detection",
           arXiv:1708.02002. Defaults γ=2.0, α=0.25 reproduce RetinaNet.

Public API:
    register_custom_loss(gamma: float = 2.0, alpha: float = 0.25) -> None
"""

import inspect, textwrap
import torch, torch.nn as nn

# --- The drop-in replacement for the per-element BCE term -----------
class FocalBCE(nn.Module):
    """
    Computes α * |y − σ(x)|^γ * BCEWithLogits(x, y), per-element.
    See Lin et al. 2017 eq. (4). Reduction is left to the caller,
    same contract as nn.BCEWithLogitsLoss(reduction='none').
    """
    def __init__(self, gamma: float, alpha: float):
        super().__init__()
        self.gamma, self.alpha = gamma, alpha
        self._bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, pred_logits, target):
        bce = self._bce(pred_logits, target)
        p   = torch.sigmoid(pred_logits)
        modulator = (target - p).abs() ** self.gamma
        return self.alpha * modulator * bce


# --- PUBLIC REGISTRATION FUNCTION ------------------------------------
def register_custom_loss(gamma: float = 2.0, alpha: float = 0.25) -> None:
    """
    Replace `v8DetectionLoss.bce` with FocalBCE(γ, α) at import time.

    MUST be called BEFORE `from ultralytics import YOLO`. Idempotent.
    """
    import ultralytics.utils.loss as ul

    # Idempotency guard (mirrors _patch_parse_model: L83)
    if getattr(ul.v8DetectionLoss, '_focal_patched', False):
        return

    _patch_v8_detection_loss(ul, gamma, alpha)


def _patch_v8_detection_loss(loss_module, gamma, alpha) -> None:
    """
    Replace v8DetectionLoss.__init__ via source-modified exec,
    the same pattern as scripts/phase3_architecture_design.py:_patch_parse_model.
    """
    original_init = loss_module.v8DetectionLoss.__init__

    # 1. Inject FocalBCE into the loss-module namespace so the
    #    modified __init__ can reference it by bare name.
    loss_module.FocalBCE = FocalBCE
    loss_module._FOCAL_GAMMA = gamma
    loss_module._FOCAL_ALPHA = alpha

    # 2. Grab original source and dedent (same as phase3, L87-90)
    src = textwrap.dedent(inspect.getsource(original_init))

    # 3. Replace the one BCE line with FocalBCE. Upstream line:
    #        self.bce = nn.BCEWithLogitsLoss(reduction="none")
    #    Target line:
    #        self.bce = FocalBCE(_FOCAL_GAMMA, _FOCAL_ALPHA)
    marker = 'self.bce = nn.BCEWithLogitsLoss(reduction="none")'
    if marker not in src:
        raise RuntimeError(
            "Could not find BCE construction in v8DetectionLoss.__init__ "
            "— ultralytics version incompatible; pin the exact version."
        )
    src = src.replace(
        marker,
        'self.bce = FocalBCE(_FOCAL_GAMMA, _FOCAL_ALPHA)',
        1,
    )

    # 4. Exec in the module's globals so name resolution matches upstream
    #    (same pattern as phase3 L124-125).
    exec(compile(src, "<patched_v8_detection_loss_init>", "exec"),
         loss_module.__dict__)

    patched_init = loss_module.__dict__['__init__']
    patched_init._focal_patched = True
    patched_init._focal_gamma = gamma
    patched_init._focal_alpha = alpha
    patched_init._original = original_init   # for I-4 reversibility

    loss_module.v8DetectionLoss.__init__ = patched_init
    loss_module.v8DetectionLoss._focal_patched = True
```

**Why patch `__init__` instead of `__call__`?** Upstream already computes
`loss[1] = self.bce(pred_scores, ...).sum() / target_scores_sum`. If
`self.bce` is swapped for `FocalBCE(reduction='none')`, the existing `.sum()`
reduction path stays identical — no need to modify `__call__`, which is
what `loss_focal_micai.py:430-445` incorrectly does (duplicating the BCE
computation and then discarding the original `loss[1]` value). Keeps the
diff minimal: **one source-replace line** vs. the current 20-line hack.

### 7.3 Driver wiring — replace [train.py:5-11](../../train.py#L5-L11)

```python
# --- PSEUDOCODE ---
import sys; sys.path.insert(0, '.')

# 1. Register custom modules (Phase 3 — Amaury, unchanged)
from scripts.phase3_architecture_design import register_custom_modules
register_custom_modules()

# 2. Register custom loss (Phase 4 — NEW)
from scripts.phase4_focal_loss import register_custom_loss
register_custom_loss(gamma=2.0, alpha=0.25)   # cite: Lin et al. 2017

# 3. ONLY NOW import YOLO  (invariant I-2)
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('models/configs/yolov8_custom.yaml')   # NOT yolov8n.pt — fixes B-3
    model.train(
        data='data/processed/dataset.yaml',
        epochs=50, imgsz=640, batch=16,
        # device & workers exposed via argparse — fixes M-2
        project='runs/detect', name='focal_g2.0_a0.25_run01',  # fixes M-3
    )
```

Note the **ordering**: `register_custom_modules()` → `register_custom_loss()` → `from ultralytics import YOLO`. Reversing any of these breaks the pattern:
- `YOLO` imported first → `v8DetectionLoss.__init__` source is already bound to old `nn.BCEWithLogitsLoss`; `inspect.getsource()` still works but the swap won't propagate to already-instantiated loss objects.
- `register_custom_modules` after `register_custom_loss` is fine (orthogonal namespaces), but matching the Phase-3 call-order keeps driver readability consistent.

### 7.4 Sequence diagram

```
┌────────┐          ┌──────────────────────────────┐    ┌──────────────────────────┐   ┌─────────┐
│train.py│          │scripts/phase4_focal_loss.py │    │ultralytics.utils.loss    │   │ultralytics.YOLO│
└───┬────┘          └──────────────┬───────────────┘    └────────────┬─────────────┘   └──────┬──┘
    │  register_custom_loss(2.0, 0.25)                              │                        │
    ├──────────────────────────►  │                                 │                        │
    │                             │  import ultralytics.utils.loss  │                        │
    │                             ├────────────────────────────────►│                        │
    │                             │ guard: v8DetectionLoss._focal_patched ?                  │
    │                             │─── yes → return (idempotent, I-3)                        │
    │                             │─── no  → continue                                        │
    │                             │ inject FocalBCE into loss ns    │                        │
    │                             ├────────────────────────────────►│                        │
    │                             │ src = inspect.getsource(__init__)                        │
    │                             │ src.replace('nn.BCEWithLogits…', 'FocalBCE(γ, α)')       │
    │                             │ exec(compile(src, …), loss.__dict__)                     │
    │                             │ set _focal_patched = True       │                        │
    │                             │ stash _original for reversal (I-4)                       │
    │  returns None               │                                 │                        │
    │◄────────────────────────────┤                                 │                        │
    │  from ultralytics import YOLO                                                          │
    ├─────────────────────────────────────────────────────────────────────────────────────►│
    │  model = YOLO('yolov8_custom.yaml')                                                  │
    ├─────────────────────────────────────────────────────────────────────────────────────►│
    │                                                                                        │ YOLO.train() → constructs v8DetectionLoss
    │                                                                                        │    → patched __init__ runs
    │                                                                                        │    → self.bce = FocalBCE(2.0, 0.25)  ✓
    │                                                                                        │    → forward uses α·|y−p|^γ·BCE      ✓
```

### 7.5 What this design fixes (cross-reference)

| Finding | How §7 resolves it |
|---|---|
| B-1 | Injection now lives in-repo; `venv/` is no longer required to reproduce Phase 4. |
| B-4 | Fixing `requirements.txt` (§8 checklist) is independent; §7 does not need new deps. |
| A-1 | File header of `phase4_focal_loss.py` carries the AGPL-3.0 notice + "Modified: 2026-04-17 by MICAI team" line. |
| A-3 | Docstring + inline formula comment cite Lin et al. 2017. |
| A-4 | 1244 lines of unused upstream loss classes are no longer shipped. |
| A-5 | No relative imports — the module stands alone. |
| M-1 | Silent `except ImportError` in [train.py:10](../../train.py#L10) becomes a hard error: if `register_custom_loss` fails to import, training aborts (desired). |
| I-4 | `_original` stash enables BCE-vs-Focal ablation (§9) without venv surgery. |

### 7.6 Out of scope for §7

- The label remap (§6) and `dataset.yaml` rewrite. Those are Phase-1 / dataset concerns.
- Hyperparameter exposure (`device`, `workers`, `epochs`) — see M-2 / §8 / §9.
- Experiment tracking (wandb / mlflow). Flagged methodologically in §9.

---

## 8. Fixer checklist — surgical, ordered by blast-radius

Each item has: **(a)** action, **(b)** file(s) touched, **(c)** finding(s) closed, **(d)** done-when acceptance criterion. Items are ordered so later items depend on earlier ones — execute top-down. No item assumes any other item was skipped.

### Tier 1 — BLOCKERs (must complete before any training run is credible)

- [ ] **F-1 Create `scripts/phase4_focal_loss.py`** implementing `FocalBCE` + `register_custom_loss()` exactly per §7.2.
  - Files: NEW `scripts/phase4_focal_loss.py` (≤ 80 LOC).
  - Closes: **B-1** (focal loss now in-repo), **A-1** (AGPL notice + "Modified: 2026-04-17" banner in header), **A-3** (docstring cites Lin et al. 2017, arXiv:1708.02002), **A-4** (no copy of 12 unused loss classes), **A-5** (no relative imports — file is self-contained).
  - Done-when: `python -c "from scripts.phase4_focal_loss import register_custom_loss; register_custom_loss(); register_custom_loss()"` runs without error (idempotency verified) and `from ultralytics.utils.loss import v8DetectionLoss; assert v8DetectionLoss._focal_patched` passes.

- [ ] **F-2 Delete `scripts/loss_focal_micai.py`.** No references exist (`grep -r loss_focal_micai` → 0 hits). The file is a frozen backup of venv edits and is superseded by F-1.
  - Files: DELETE `scripts/loss_focal_micai.py` (−1264 LOC).
  - Closes: **A-4** (second half — dead upstream code removed from repo), companion to **B-1**.
  - Done-when: file absent from working tree and from `HEAD`.

- [ ] **F-3 Remap all label files to dense `[0, 142]`.** Write a deterministic script under `scripts/phase1_remap_labels.py` that (i) builds the sorted set of IDs present on disk, (ii) maps the *k*-th ID → *k*, (iii) emits the mapping to `data/processed/class_id_map.json` for provenance, (iv) rewrites every `data/split/{train,val,test}/labels/*.txt` in place.
  - Files: NEW `scripts/phase1_remap_labels.py`, MODIFIED `data/split/**/labels/*.txt`, NEW `data/processed/class_id_map.json`.
  - Closes: **D-2** (labels densely packed), precondition for F-4.
  - Done-when: `awk '{print $1}' data/split/train/labels/*.txt | sort -un | tail -1` returns `142`.

- [ ] **F-4 Fix `data/processed/dataset.yaml`.** Set `nc: 143`; replace the 232-entry `names:` with the 143-entry canonical TT100K map derived from F-3's `class_id_map.json`; fix `path: ../split` (or absolute-from-project-root); set `train: train/images`, `val: val/images`, `test: test/images` **relative to that `path:`**.
  - Files: MODIFIED `data/processed/dataset.yaml`.
  - Closes: **B-2**, **D-1**, **D-3**, **H-1**.
  - Done-when: `python -c "import yaml; d=yaml.safe_load(open('data/processed/dataset.yaml')); assert d['nc']==143 and len(d['names'])==143"`; and Ultralytics `YOLO(...).train(data=...)` resolves the image/label paths without `FileNotFoundError` on a dry run.

- [ ] **F-5 Pin the training stack in `requirements.txt`.** Add (with tested versions, do NOT copy-paste latest without running): `torch==2.x.y+cu121`, `torchvision==…`, `ultralytics==8.x.y`. Remove the duplicate OpenCV pin (L-1): keep exactly one of `opencv-python` / `opencv-python-headless`.
  - Files: MODIFIED `requirements.txt`.
  - Closes: **B-4**, **L-1**.
  - Done-when: `pip install -r requirements.txt` succeeds in a fresh venv, and `python -c "import torch, torchvision, ultralytics, cv2"` exits 0.

- [ ] **F-6 Rewrite `train.py`** — minimal driver. Replace the current 34 LOC with:
  1. `register_custom_modules()` call (unchanged).
  2. `register_custom_loss(gamma=args.gamma, alpha=args.alpha)` call (NEW).
  3. `from ultralytics import YOLO` — **only after** the two registrations.
  4. `model = YOLO('models/configs/yolov8_custom.yaml')` — **not** `yolov8n.pt`.
  5. Remove the bare `except ImportError` — let it crash loudly.
  6. `argparse` surface for `--epochs --batch --imgsz --device --workers --name --gamma --alpha`.
  7. `--name` default = `f'focal_g{args.gamma}_a{args.alpha}_ep{args.epochs}'` (no `test2` leftover).
  - Files: MODIFIED `train.py`.
  - Closes: **B-3**, **M-1**, **M-2**, **M-3**.
  - Done-when: `python train.py --epochs 1 --batch 2 --device cpu` runs one training step end-to-end, and the run directory is `runs/detect/focal_g2.0_a0.25_ep1/` (not `test2_focal`).

### Tier 2 — HIGHs (methodological / compliance hygiene)

- [ ] **F-7 Un-ignore `data/split/` — or commit a DVC/Roboflow pull script.** Pick one of:
  - (a) Remove `data/split/` from `.gitignore` and commit the split (feasible if < a few hundred MB; the project already uses Git for label files).
  - (b) Keep `.gitignore` entry but add `scripts/phase1_fetch_dataset.py` that reproduces `data/split/` from the canonical TT100K source (DVC remote, Roboflow project ID, or a URL + checksum manifest). Document in `README.md`.
  - Files: MODIFIED `.gitignore` and/or NEW `scripts/phase1_fetch_dataset.py` + MODIFIED `README.md`.
  - Closes: **H-2**.
  - Done-when: a teammate running `git clone && pip install -r requirements.txt && <one documented command>` can begin training.

- [ ] **F-8 Add `scripts/phase4_validate.py`** analogous to [phase3_architecture_design.py:139-235](../../scripts/phase3_architecture_design.py#L139-L235). It must: (i) call both `register_custom_modules()` and `register_custom_loss()`; (ii) construct a tiny `v8DetectionLoss` instance on the custom model; (iii) assert `type(model.criterion.bce).__name__ == 'FocalBCE'`; (iv) run a dry-run forward+backward pass on `(1,3,640,640)` with synthetic targets; (v) write `results/phase4_loss_validation.txt`.
  - Files: NEW `scripts/phase4_validate.py`, NEW `results/phase4_loss_validation.txt`.
  - Closes: **H-4**.
  - Done-when: `python scripts/phase4_validate.py` exits 0 and writes the report.

- [ ] **F-9 Add root `LICENSE` file (AGPL-3.0).** Update `README.md` with a "License" section that (a) names AGPL-3.0, (b) acknowledges derivation from Ultralytics, (c) cites Lin et al. 2017 for the focal-loss formulation.
  - Files: NEW `LICENSE`, MODIFIED `README.md`.
  - Closes: **A-2**, final half of **A-1**, provides a durable anchor for **A-3**.
  - Done-when: `LICENSE` matches the canonical AGPL-3.0 text; `README.md` references both Ultralytics and Lin et al. 2017.

### Tier 3 — Process (MEDIUM/LOW, prevent recurrence)

- [ ] **F-10 Retroactive commit-message note + future atomicity rule.** Cannot rewrite `99056f0f` without force-push, and force-push is out of scope for the surgical fixer. Instead: add a one-line entry to `docs/CHANGELOG.md` (NEW) noting that commit `99056f0f` mislabelled "fix dataset paths" while actually mutating `nc` (221→232). Going forward, enforce atomic, single-intent commits in `CONTRIBUTING.md` (NEW).
  - Files: NEW `docs/CHANGELOG.md`, NEW `CONTRIBUTING.md`.
  - Closes: **H-3**, **M-4**.
  - Done-when: both files exist; the first CHANGELOG entry explicitly flags the `a2b24573`→`99056f0f` `nc` mutation.

### 8.1 Finding-closure matrix

| Finding | Fixer item(s) |
|---|---|
| B-1 | F-1 + F-2 |
| B-2 | F-4 (with F-3 as precondition) |
| B-3 | F-6 |
| B-4 | F-5 |
| H-1 | F-4 |
| H-2 | F-7 |
| H-3 | F-10 |
| H-4 | F-8 |
| M-1 | F-6 |
| M-2 | F-6 |
| M-3 | F-6 |
| M-4 | F-10 |
| L-1 | F-5 |
| A-1 | F-1 + F-9 |
| A-2 | F-9 |
| A-3 | F-1 (+ F-9 reinforces) |
| A-4 | F-1 + F-2 |
| A-5 | F-1 |
| D-1 | F-4 |
| D-2 | F-3 |
| D-3 | F-4 |

**All 21 findings are closed by 10 items.** Minimum-path-to-green is F-1 → F-2 → F-3 → F-4 → F-5 → F-6 (Tier 1 only): after these six items the pipeline actually runs the MICAI architecture with focal loss on correctly-labelled data. F-7..F-10 are hygiene and compliance — required for a defensible paper submission but not for a functional training run.

---

## 9. Methodological recommendations

This section addresses concerns that are **not defects in the code** per se, but weaken the scientific claim the MICAI paper will make. A working training pipeline (§8 Tier 1) is necessary but not sufficient for a defensible publication.

### 9.1 The ablation gap

The current repo supports **one** configuration: custom architecture (SubPixelConv + CoordAtt) + focal loss (γ=2.0, α=0.25). A paper that reports improvements without a factorial ablation cannot attribute the improvement to any specific component. Minimum required ablation matrix:

| Run | Architecture | Loss | Purpose |
|---|---|---|---|
| R1 | stock YOLOv8n | BCE | Absolute baseline. Same dataset, same schedule. |
| R2 | stock YOLOv8n | FocalBCE (γ=2, α=0.25) | Isolates the focal contribution. |
| R3 | custom (SubPixel+CoordAtt) | BCE | Isolates the architectural contribution. |
| R4 | custom (SubPixel+CoordAtt) | FocalBCE (γ=2, α=0.25) | The claimed system. |

The `_original` stash in §7.2 (invariant I-4) exists precisely so R1/R3 can be produced without reinstalling Ultralytics. Without all four cells the paper cannot claim "focal loss + custom modules improves TT100K mAP by X%" — at best it can claim "our combined system beats baseline", which a MICAI reviewer will reject.

### 9.2 γ and α are not free parameters

γ=2.0 and α=0.25 are the RetinaNet defaults tuned for COCO (80 classes, dense anchors). Transferring them to TT100K (143 densely-remapped classes per F-3, heavy long-tail distribution — see max-ID=213 pre-remap as a symptom) without empirical justification is a methodological hole.

Recommended:
1. **γ sensitivity sweep.** γ ∈ {1.0, 1.5, 2.0, 2.5, 3.0} at fixed α=0.25. One training curve per value. Report validation mAP and **tail-class** mAP (bottom 20% of classes by frequency) — focal loss is supposed to help precisely there.
2. **α sweep on tail classes.** α ∈ {0.1, 0.25, 0.5, 0.75}. TT100K is more imbalanced than COCO; α=0.25 (which *down*-weights positives, by Lin et al. §3.3) may be the wrong direction for rare signs.
3. **Report the chosen (γ, α) with the sweep, not as defaults.** "We adopt the defaults" is acceptable in a workshop paper; MICAI expects a justification.

### 9.3 Data-split hygiene — the `test2` signal

[train.py:33](../../train.py#L33) sets `name='test2_focal'`. The sequence `test → test1 → test2` implies at least two prior experimental runs whose results were written to `runs/detect/test*`, which are neither committed nor documented. Three consequences:

1. **Model-selection contamination risk.** If hyperparameters or training curricula were tuned by watching `test/images` performance across `test`/`test1`/`test2`, the final `test*` numbers are no longer test-set numbers — they are validation-set numbers leaked onto the test split. The paper must report results on a **held-out** split that was never observed during tuning.
2. **Reproducibility loss.** The intermediate runs are neither committed as code state (no branches, no tags) nor as artefacts (`runs/` likely gitignored). There is no way to reconstruct what changed between `test1` and `test2`.
3. **Remediation.** After F-6 lands: (a) freeze a held-out split that is *not* the `test/` currently used; (b) rename future runs using `focal_g{γ}_a{α}_ep{N}_seed{S}` format (deterministic, composable); (c) commit `docs/experiments.md` that logs each run with its commit SHA, seed, and key result.

### 9.4 Missing Phase-4 validation report

Phase 3 produced `results/phase3_architecture_validation.txt` via [phase3_architecture_design.py:215-228](../../scripts/phase3_architecture_design.py#L215-L228). Phase 4 has no analogue. F-8 addresses the code skeleton; the methodological requirement is that the report captures, at minimum:

- Final training/validation loss curves (attach as `results/phase4_training_curves.png`).
- Per-class AP on `val/`.
- Tail-class AP (the sub-metric focal loss is motivated by).
- Confusion top-10 — which class pairs the model conflates.
- Seed, commit SHA, dataset hash, `torch.__version__`, `ultralytics.__version__`.

This report is the artefact a reviewer will ask for when the paper is submitted. Generating it after the fact is harder than generating it alongside the run.

### 9.5 Experiment tracking

Currently results land in `runs/detect/test2_focal/` — a single local directory, no metadata, no comparison tooling. Adopt **one** of:

- `wandb` (network-dependent, but excellent comparison UI).
- `mlflow` (local-first, no cloud dependency).
- Git-committed `docs/experiments.md` + `results/**.txt` reports (zero-dependency fallback; minimum viable).

Whichever is chosen, the criterion is: **given a commit SHA, a reviewer must be able to identify which run produced the reported numbers.** Today that mapping is absent.

### 9.6 Dataset provenance and pre-processing chain

The audit scope is Phase 4, but §6 made it clear that Phase 1 (label preprocessing) is the load-bearing upstream concern. Recommendations that straddle Phase 1 ↔ Phase 4:

- **Document the TT100K → `data/split/` pipeline** in `docs/phase1_preprocessing.md`: source URL or Roboflow project ID, filtering criteria, class-merging rules, the 143-class rationale, and a checksum or sample-count table.
- **Write a dataset-hash** into every run's metadata. If `data/split/` changes silently, every downstream result silently goes stale.
- **Keep the `class_id_map.json`** (F-3) under version control. It is the Rosetta stone for every result the paper will cite.

### 9.7 Commit hygiene going forward

Independent of F-10:

- **One intent per commit.** Commit `99056f0f` bundled dataset-path work, requirements cleanup, gitignore edits, and a 1264-LOC dead-code addition under a message that mentioned only the first two. A reviewer auditing the git log can neither understand the dependency graph nor cleanly revert a subset.
- **Message must name the `what` accurately.** A commit that mutates `nc` from 221 to 232 should say so in the subject line.
- **Claims in commit messages must match code state.** "backup focal loss" is ambiguous; "add inert reference copy of modified ultralytics/loss.py (not yet wired into train.py)" is honest.

### 9.8 Summary priorities

| Priority | Action |
|---|---|
| P0 | Ship Tier-1 fixes (§8 F-1..F-6). Pipeline becomes truthful. |
| P1 | Run the 4-cell ablation (§9.1). Publication becomes defensible. |
| P2 | γ/α sweep on tail classes (§9.2). Paper becomes compelling. |
| P3 | Experiment tracking + Phase-4 validation report (§9.4, §9.5). Peer review becomes answerable. |

---

## Appendix — End-of-audit status

**Findings catalogued:** 21 total — 4 BLOCKERs (B-1..B-4), 4 HIGHs (H-1..H-4), 4 MEDIUMs (M-1..M-4), 1 LOW (L-1), 5 Attribution (A-1..A-5, of which A-1/A-2/A-3 are HIGH), 3 Dimensional (D-1 BLOCKER re-statement, D-2 HIGH, D-3 MEDIUM).

**Fixer items:** 10 (F-1..F-10) in 3 tiers; all 21 findings mapped to closures (§8.1).

**Design delivered:** `register_custom_loss()` pseudocode + sequence diagram + invariants I-1..I-6 (§7), mirroring the Phase-3 pattern [phase3_architecture_design.py:42-132](../../scripts/phase3_architecture_design.py#L42-L132).

**Auditor's verdict:** Phase 4, as committed by Enrique on `main` as of commit `99056f0f` (2026-04-16), **does not deliver the MICAI training configuration**. The custom architecture is not loaded ([train.py:19-20](../../train.py#L19-L20)), focal loss is not applied at runtime (the modification lives in `venv/`, not in the repo), and the dataset descriptor references a class count inconsistent with both the architecture and the on-disk labels. A training run started from the current repo would silently train stock YOLOv8n with standard BCE on a wrongly-labelled dataset — every resulting number would be unattributable to the claimed contribution.

The path to recovery is mechanical and fully specified in §8 Tier 1. No research is required; only execution.

---

*Audit prepared by: Senior ML Engineer / Code Auditor agent (read-only). Date: 2026-04-17.*
