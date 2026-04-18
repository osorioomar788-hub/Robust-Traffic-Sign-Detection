# Research Scratchpad — Phase 4 Forensic Audit

**Auditor:** Claude (read-only agent, role: Senior ML Engineer / Code Auditor)
**Date:** 2026-04-17
**Deliverable:** `docs/audits/2026-04-17-phase4-audit.md`
**Protocol:** `docs/Amaury/2026-04-17-audit-phase4-training.md`

---

## Git State Snapshot (captured at audit start)

```
Branch: main (up to date with origin/main)
Untracked: audit_dump.txt, docs/Amaury/2026-04-17-audit-phase4-training.md
Last 3 commits:
  99056f0f  7Lear7  2026-04-16  refactor: clean requirements, fix dataset paths, update gitignore and backup focal loss
  a2b24573  Lear07  2026-03-19  Configuración de rutas relativas y setup de Fase 4
  43f160d8  Amaury  (earlier)   feat: Introduce custom SubPixelConv and CoordAtt modules ...
```

Scope: **2 commits** authored by Enrique/Quique (`7Lear7`, `Lear07`). Nothing else in scope.

---

## Micro-Steps (one per audit section)

- [x] **Step 0 — Metadata** — Frozen in audit §0.
- [x] **Step 1 — Inventory** — Table of files touched × commit × LOC × declared vs observed purpose. (THIS STEP)
- [x] **Step 2 — BLOCKERs** — Done. 4 BLOCKERs logged (B-1 orphan focal-loss, B-2 nc triple-mismatch, B-3 stock model loaded, B-4 missing torch/ultralytics/torchvision in requirements).
- [x] **Step 3 — HIGHs** — Done. 4 HIGHs logged (H-1 dataset paths, H-2 gitignore hides split, H-3 nc commit-msg integrity, H-4 no phase4 validator).
- [x] **Step 4 — MEDIUM / LOW** — Done. 5 findings (M-1 silent except, M-2 hardcoded device/workers, M-3 test2 naming leak, M-4 multi-intent commit, L-1 opencv duplication).
- [x] **Step 5 — Attribution Check** — Done. Verbatim ≥98.4%. A-1 AGPL §5(a) notice-of-change missing date/author. A-2 no LICENSE file propagating AGPL-3.0. A-3 Lin et al. 2017 uncited, γ/α unjustified. A-4 12 unused loss classes shipped. A-5 relative imports (L17-18) make the file non-importable (ties B-1).
- [x] **Step 6 — Dimensional Consistency** — Done. Truth table. Source of truth = architecture (`nc=143`). D-1 triple-mismatch, D-2 labels not densely remapped (max ID 213), D-3 `nc=232` provenance undocumented.
- [x] **Step 7 — Proposed design: `register_custom_loss()`** — Done. Pseudocode + sequence diagram. Mirror: idempotency guard (`_focal_patched`), inject FocalBCE into loss ns, `inspect.getsource()` + source-replace + `exec(compile(...))` on `v8DetectionLoss.__init__` (one-line swap: `nn.BCEWithLogitsLoss` → `FocalBCE`). Stashes `_original` for ablation. Invariants I-1..I-6. Cites Lin et al. 2017.
- [x] **Step 8 — Checklist for Surgical Fixer** — Done. 10 items (F-1..F-10) in 3 tiers. Closure matrix proves all 21 findings mapped. Minimum-path-to-green = F-1..F-6 (Tier 1).
- [x] **Step 9 — Methodological recommendations** — Done. 9.1 4-cell ablation matrix. 9.2 γ/α sweep rationale. 9.3 test1/test2 leakage remediation. 9.4 missing Phase-4 report. 9.5 experiment tracking. 9.6 dataset provenance. 9.7 commit hygiene. 9.8 P0..P3 priorities. Verdict: pipeline does not deliver MICAI config as committed.

---

## Evidence Gathered So Far (raw, for later consolidation)

### E.1 — Commit `a2b24573` stats
```
data/processed/dataset.yaml | 5  +---
train.py                    | 34 +---
```
Declared: "Configuración de rutas relativas y setup de Fase 4". Scaffolded train.py v1 with device='cpu', epochs=1, batch=4.

### E.2 — Commit `99056f0f` stats
```
.gitignore                  |    6 +
data/processed/dataset.yaml |  456 (massive rewrite)
requirements.txt            |    2 (cleaned to 30 packages, no torch/ultralytics pinned)
scripts/loss_focal_micai.py | 1264 NEW (copy of ultralytics/utils/loss.py w/ edits)
train.py                    |   17 (device=0, epochs=50, name='test2_focal', project='runs/detect')
```
Declared: "refactor: clean requirements, fix dataset paths, update gitignore and backup focal loss".
Observed: introduces a dead-code 1264-line copy of Ultralytics loss module, changes nc from 221→232 (neither matches architecture's 143), and train.py still loads `yolov8n.pt` (not custom model) at line 20 with line 19 commented out.

### E.3 — Ground-truth class count from disk
```
$ awk '{print $1}' data/split/train/labels/*.txt | sort -un | wc -l
143
$ awk '{print $1}' data/split/train/labels/*.txt | sort -un | tail -5
168 / 169 / 170 / 205 / 213
```
**143 distinct class IDs, but max ID = 213.** Labels are NOT densely remapped to `[0..142]`.
Implication: training with `nc=143` will crash (`cls=213 ≥ nc=143`). With `nc=232` it runs silently but allocates 89 unused output channels — architecture mismatch masked, not fixed.

### E.4 — Architecture ground truth
- `models/configs/yolov8_custom.yaml:7` → `nc: 143` (Phase 3 contract, owned by Amaury).

### E.5 — Loss injection surface
- `scripts/loss_focal_micai.py:344-347` — hardcoded `self.gamma=2.0, self.alpha=0.25` added to `v8DetectionLoss.__init__`.
- `scripts/loss_focal_micai.py:430-445` — custom focal modulator replaces `self.bce(pred_scores,...).sum()`.
- **The file is never imported.** `grep -r "loss_focal_micai"` returns nothing. `train.py` does not wire it into Ultralytics. It is a "backup" (per commit message) of modifications made directly to `site-packages/ultralytics/utils/loss.py` — i.e. the *real* change is in venv and is not reproducible from the repo.

### E.6 — Custom-model line commented out
- `train.py:19` → `#model = YOLO('models/configs/yolov8_custom.yaml')`
- `train.py:20` → `model = YOLO('yolov8n.pt')` (stock pretrained, `nc=80`)
Any claim of "Phase 4 trained the MICAI architecture with focal loss" is unsupported by this code path.

---

## Files Read (context_injection)
- `docs/Amaury/PHASE3_FINAL_REPORT.md` ✓
- `scripts/phase3_architecture_design.py` ✓ (pattern of reference for §7)
- `scripts/loss_focal_micai.py` ✓ (object of audit)
- `train.py` ✓
- `models/configs/yolov8_custom.yaml` ✓
- `data/processed/dataset.yaml` ✓
- `requirements.txt` ✓

## Files NOT Opened (per prohibition)
- `venv/**` — never
- `data/augmented/**` — out of scope
- any `phaseN_*.py` other than phase3 — out of scope

---

## Next Action
Step 2: Extract BLOCKERs with file:line evidence. Expected hits:
1. BLOCKER — Focal loss injected via site-packages overwrite, not via repo code (evidence: E.5).
2. BLOCKER — `data/processed/dataset.yaml:10` declares `nc: 232`; architecture demands `nc: 143`; labels carry IDs up to 213 (evidence: E.3, E.4).
3. BLOCKER — `train.py:19-20` loads `yolov8n.pt` instead of the custom architecture (evidence: E.6).
