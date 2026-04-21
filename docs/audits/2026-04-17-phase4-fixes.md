# Fix Report — Phase 4 Audit Remediation

> Companion document to [2026-04-17-phase4-audit.md](2026-04-17-phase4-audit.md).
> Closes the 21 findings from the audit via 9 ordered steps.

---

## 0. Metadata

| Field | Value |
|---|---|
| Fix date | 2026-04-20 |
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
| _(populated in §9.5 at end of fix cycle)_ | | | | |

---

## 2. Per-finding fix details

_(populated after each step)_

---

## 3. Findings rejected / deferred

| Finding | Status | Reason |
|---|---|---|
| H-3 (commit-history mislabelling) | **REJECTED** | Cannot rewrite published commit history without force-push; force-push excluded from fixer scope. Partially mitigated via CHANGELOG entry flagging the `a2b24573`→`99056f0f` nc mutation. |
| F-7 (data/split fetch script) | **DEFERRED** | PI decision: fetch-script belongs to a dedicated data-provenance backlog item, not this audit-closure PR. |
| F-10 (CONTRIBUTING.md) | **DEFERRED** | PI decision: only CHANGELOG.md shipped in this cycle; CONTRIBUTING.md deferred. |

---

## 4. Quality gates (literal output)

_(populated after each step)_

---

## 5. Next steps (backlog)

- `scripts/phase1_fetch_dataset.py` — reproducible fetch of TT100K images from canonical source (F-7).
- `CONTRIBUTING.md` — atomic-commit policy + review checklist (F-10 residual).
- Commit `data/augmented/` labels (currently remapped but unversioned) — decide with Omar whether augmented set is in-scope for the paper.
- 4-cell ablation matrix (§9.1 of audit): R1 stock+BCE, R2 stock+Focal, R3 custom+BCE, R4 custom+Focal.
- γ/α sensitivity sweep on tail classes (§9.2 of audit).
- `docs/experiments.md` + `results/phase4_training_curves.png` (§9.4, §9.5).
