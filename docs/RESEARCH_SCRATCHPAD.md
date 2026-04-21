# Phase 4 Audit — Surgical Fix Scratchpad

**Spec vinculante:** [audits/2026-04-17-phase4-audit.md](audits/2026-04-17-phase4-audit.md)
**Fix report:** [audits/2026-04-17-phase4-fixes.md](audits/2026-04-17-phase4-fixes.md)
**Fixer:** Senior ML Engineer / Surgical Fixer (agent)
**Fecha inicio:** 2026-04-20

(Supersedes previous scratchpad used during read-only audit phase —
historical content preserved in `docs/audits/2026-04-17-phase4-audit.md`.)

---

## Estado de pasos

- [x] **Paso 0** — Init scratchpad + fix-report skeleton.
- [ ] **Paso 1** — Retroactive dense remap of class IDs to `[0, 142]` (F-3 part 1).
- [ ] **Paso 2** — Integrate `remap_labels_dense()` into `scripts/phase1_data_acquisition.py` (F-3 part 2).
- [ ] **Paso 3** — Rebuild `data/processed/dataset.yaml` with `nc=143` dense names (F-4, H-1).
- [ ] **Paso 4** — Create `scripts/phase4_focal_loss.py` (F-1, A-1, A-3).
- [ ] **Paso 5** — `git rm scripts/loss_focal_micai.py` (F-2, A-4, A-5).
- [ ] **Paso 6** — Pin torch / torchvision / ultralytics; dedupe opencv (F-5, L-1).
- [ ] **Paso 7** — Rewrite `train.py` with argparse + proper registration (F-6, M-1, M-2, M-3, B-3).
- [ ] **Paso 8** — `scripts/phase4_validate.py` smoke test (H-4, F-8).
- [ ] **Paso 9** — LICENSE + README + CHANGELOG + commit labels (A-1, A-2, A-3, H-2, part of F-10).

---

## Deviations vs audit spec (authorised by PI)

1. **H-2** scope limited: commit labels only, NOT images.
2. **F-3** extended: also patch `scripts/phase1_data_acquisition.py` so the bug does not recur.
3. **H-3** (git history rewrite) **REJECTED**.
4. **F-10** reduced: CHANGELOG.md only; CONTRIBUTING.md to backlog.
5. **F-7** (fetch script) to backlog.

---

## Reality snapshot (captured 2026-04-20)

- `data/split/{train,val,test}/labels/` → **present**, 4273 / 1830 / 3067 files, ~12 MB total.
- `data/split/*/images/` → **empty** (0 bytes). No images to worry about for H-2.
- `data/augmented/train/labels/` → 67509 files, 95 MB. Remapped by Paso 1 but NOT committed (scope limit).
- `data/augmented/**/images/` → not present.
- Audit-confirmed: 143 distinct class IDs in labels, max ID = 213.

---

## Next step details

**Paso 1** — write `scripts/phase1_remap_labels.py`:
- Walk `data/split/{train,val,test}/labels/*.txt` and `data/augmented/**/labels/*.txt`.
- Build sorted set of unique IDs → dense map `original_id → 0..N-1`.
- Rewrite each `.txt` in place with new IDs.
- Emit `data/processed/class_id_map.json` `[{"original_id": int, "dense_id": int, "name": str}, ...]`.
- Idempotency: if max ID already ≤ 142 and set size == 143, skip with a log line.
- Argparse `--dry-run`.

Then: run `--dry-run`, report counts, **STOP for user confirm** before real run.
