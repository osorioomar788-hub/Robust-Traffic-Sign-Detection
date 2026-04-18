# Protocol P4-AUDIT-01 — Forensic Audit de Phase 4
Autor: Amaury (Lead) + Agent 0 (PI)  |  Modo SRLC: AUDIT (propuesto)  |  Fases: F7, F8

## Research Question
¿Los entregables de Phase 4 (training) son reproducibles, atribuibles
y consistentes con la arquitectura de Phase 3, al punto de habilitar
el entrenamiento completo sin bloquear Phase 5 (Nataly)?

## Hipótesis de trabajo
H0: El código entregado ejecuta pero contiene ≥1 BLOCKER de reproducibilidad
    (overwrite de site-packages) y ≥1 inconsistencia dimensional (nc 143 vs 232).
H1: Además, al menos un fragmento de código carece de atribución académica.

## Metodología — patrón Auditor → Fixer
Dos Workers con un checkpoint humano obligatorio entre ellos.

### Worker A — Forensic Auditor (read-only)
Output único: /docs/audits/2026-04-17-phase4-audit.md con formato contractual:

1. Tabla "archivos tocados":
   | path | LOC añadidas | commit | propósito declarado | propósito observado |

2. Hallazgos con etiqueta [BLOCKER | HIGH | MEDIUM | LOW], cada uno con:
   - Evidencia: file:line + snippet de ≤15 líneas
   - Hipótesis de causa
   - Fix propuesta en PSEUDOCÓDIGO (no código real)
   - Tiempo estimado de fix

3. Attribution Check — buscar fingerprints de:
   - Import patterns típicos de MMDetection (`from mmdet.models.losses`)
   - Variables/nombres característicos de Ultralytics upstream
   - Docstrings con fraseos de tutoriales conocidos

4. Dimensional Consistency Check:
   - Contar class_ids únicos en labels reales de Omar
   - Cross-check contra yolov8_custom.yaml (nc=143)
   - Cross-check contra dataset.yaml de Quique (nc=232)
   - Reportar la triple-inconsistencia si existe

5. Checklist accionable para el Fixer (marcables).

### Checkpoint humano
Amaury lee el audit, etiqueta cada hallazgo como FIX-NOW | BACKLOG | REJECT.
Solo lo etiquetado FIX-NOW entra al scope del Worker B.

### Worker B — Surgical Fixer
NEXT_PROMPT aparte, generado después del checkpoint.

## Decisiones metodológicas (con justificación)
D1. Scope temporal: SOLO commits de Quique (via git log --author).
    → Razón: una batalla a la vez. No contaminar con bugs heredados de Phase 1-3.

D2. NO ejecutar train.py durante el audit.
    → Razón: análisis estático primero. Si corres el training antes de arreglar
      el overwrite, tus resultados quedan contaminados por site-packages modificado.

D3. Evidencia file:line obligatoria en TODO hallazgo. Cero "creo que".
    → Razón: un audit sin trazabilidad no es audit, es opinión.

D4. Attribution como first-class citizen.
    → Razón: MICAI pide reproducibilidad y los reviewers preguntan origen del código.

## Métricas de éxito del audit
- Cobertura: ≥90% de archivos nuevos/modificados con ≥1 observación.
- Precisión: 0 hallazgos "speculative" (todos con file:line).
- Accionabilidad: toda fix de BLOCKER implementable en <30 min de Fixer.