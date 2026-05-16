# Phase 4 — Focal Loss Implementation
**Owner**: Enrique
**Status**: ⏳ pending re-implementation
**Reference**: `docs/audits/2026-04-17-phase4-audit.md`
**State doc**: `docs/PROJECT_STATE_2026-05-04.md`

---

## Contexto

Quique — hicimos un audit forense de tu push de Phase 4 porque me
preocupó lo del overwrite a `venv/site-packages`. El audit encontró 21
issues; cerré los 18 estructurales con un agente para no bloquear a
Naty con Phase 5. Pero la Focal Loss específicamente es **tu fase**,
y quiero que la re-implementes con tu criterio usando el audit como
referencia, no como spec rígida.

El sistema actualmente entrena con **BCE estándar de Ultralytics** sobre
la arquitectura custom (SubPixel + CoordAtt) y 143 clases bien indexadas.
El smoke test pasa. Tu trabajo es agregar Focal Loss encima sin romper
nada de lo que ya quedó.

---

## Lo que YA quedó arreglado (no lo toques, está en main)

| Issue del audit | Cómo se arregló | Archivo |
|---|---|---|
| nc 232 → 143 | Reescribí `dataset.yaml` con 143 nombres densos | `data/processed/dataset.yaml` |
| Labels sparse [0,213] → densos [0,142] | Script de remapeo retroactivo + integrado al pipeline de Omar | `scripts/phase1_remap_labels.py` + `scripts/phase1_data_acquisition.py` |
| Modelo custom comentado en train.py | Reescribí `train.py` con argparse, modelo custom cargado | `train.py` |
| `requirements.txt` sin torch/torchvision/ultralytics | Pineados a versiones reales del venv | `requirements.txt` |
| Path `dataset.yaml` no resolvía | `path: ./data/split` + reset de Ultralytics settings | `data/processed/dataset.yaml` |
| Sin validator de Phase 4 | Smoke test sintético | `scripts/phase4_validate.py` |
| Sin LICENSE / sin citas | AGPL-3.0 + README con BibTeX | `LICENSE`, `README.md` |

---

## Lo que TE TOCA implementar

### Tarea principal — `scripts/phase4_focal_loss.py` (NEW)

Una función `register_custom_loss(gamma, alpha)` que monkey-patchee
`v8DetectionLoss` de Ultralytics para que use Focal Loss en lugar de
BCE estándar.

**Invariantes no-negociables del equipo** (decisión del audit, no opinable):

1. **Cero modificación a `venv/site-packages`**. Todo en runtime, en
   el repo. Sigue el patrón de mi `scripts/phase3_architecture_design.py:register_custom_modules`.
2. **Atribución obligatoria**: docstring con cita formal de Lin et al. 2017
   (arXiv:1708.02002) + nota AGPL-3.0 + fecha de modificación.
3. **Idempotente**: si se llama dos veces, la segunda no hace nada
   (guard flag `_focal_patched`).
4. **Reversible**: el `__init__` original se guarda para poder hacer
   ablation BCE vs Focal sin reinstalar.
5. **Falla fast**: si Ultralytics cambia su API y el patch no encuentra
   su target, `RuntimeError` con mensaje claro.

**Libertad de diseño tuya**:
- Cómo implementas `FocalBCE` (la fórmula es la del paper, pero tú
  decides si subclase de `nn.Module`, si funcional, etc.)
- Parámetros default de `gamma` y `alpha` (defaults del paper son
  2.0 y 0.25, pero justifica tu elección)
- Si quieres explorar Varifocal o IoU-aware variants — adelante, pero
  documenta por qué.

### Tarea secundaria — modificar `train.py`

Agregar la llamada a `register_custom_loss` antes de `from ultralytics import YOLO`.
Hay un comentario marcador `# TODO [Enrique]: insert register_custom_loss(args.gamma, args.alpha) here`
en `train.py`. El resto del orden de imports lo respetas porque es
invariante de Phase 3 también.

Agregar args `--gamma` y `--alpha` al argparse, con defaults sensatos
y ayuda que explique los rangos comunes.

### Tarea terciaria — modificar `scripts/phase4_validate.py`

Agregar al final del smoke test un assert:
```python
assert type(criterion.bce).__name__ == 'FocalBCE'
```
(o el nombre que le pongas a tu clase). Esto es la firma de que tu
patch se aplicó correctamente. Hay un comentario marcador
`# TODO [Enrique]: re-add focal loss assertion here` en el archivo.

---

## Lo que TE TOCA experimentar (para el paper MICAI)

Esto NO es bug-fix, es ciencia. Para que el paper sea defendible
necesitamos esta evidencia empírica, y solo tú puedes generarla porque
implica decisiones sobre la loss:

### Ablation 4-cell (prioridad P0)
Cuatro corridas, mismo seed, mismo schedule, mismo split:

| Run | Arquitectura | Loss |
|---|---|---|
| R1 | vanilla YOLOv8n | BCE |
| R2 | vanilla YOLOv8n | Focal (γ, α tuyas) |
| R3 | custom (Phase 3) | BCE |
| R4 | custom (Phase 3) | Focal (γ, α tuyas) |

Sin las 4 corridas no podemos defender en el paper qué porcentaje de
mejora viene de la arquitectura vs qué porcentaje viene de la loss.
Un reviewer de MICAI puede pedir esto literal — sin las 4 celdas el
paper se rebota.

### γ/α sweep (prioridad P1)
γ ∈ {1.0, 1.5, 2.0, 2.5, 3.0} fijando α=0.25.
α ∈ {0.1, 0.25, 0.5, 0.75} fijando γ=2.0.

Reportar mAP global Y mAP tail-class (bottom-20% por frecuencia). La
hipótesis del paper es que Focal Loss ayuda en tail-class — necesitamos
evidencia directa o ajustar la hipótesis.

### Justificación documentada
`docs/Enrique/PHASE4_DESIGN_NOTES.md` (lo creas tú) explicando:
- Por qué los γ/α que escogiste
- Qué viste en el sweep
- Si encontraste algo inesperado (tail-class peor con focal, por
  ejemplo, sería interesante reportar)

---

## Cómo probar que tu implementación funciona

Mismo smoke test que ya usamos:

```bash
# 1. Activar venv y configurar Ultralytics (post-clone setup)
.venv\Scripts\activate
yolo settings datasets_dir=''

# 2. Smoke test sintético
python scripts/phase4_validate.py
# Debe imprimir ALL CHECKS PASSED y assertar que criterion.bce es tu FocalBCE

# 3. Smoke test funcional con 1 epoch en CPU
python train.py --epochs 1 --batch 2 --imgsz 320 --device cpu --workers 0 \
                --gamma 2.0 --alpha 0.25 --name focal_smoke_test

# Debe correr 1 epoch completo sin crash
```

Si los dos smoke tests pasan, mándame el output y hacemos code review
juntos antes del merge.

---

## Recursos

- **Audit completo** (la spec del re-design): `docs/audits/2026-04-17-phase4-audit.md`
  - §7 tiene mi diseño propuesto en pseudocódigo — úsalo como referencia
    si quieres, o ignóralo y haz el tuyo
- **Patrón de monkey-patch**: `scripts/phase3_architecture_design.py:register_custom_modules`
- **State doc del proyecto**: `docs/PROJECT_STATE_2026-05-04.md`
- **Paper original**: Lin et al. 2017, *Focal Loss for Dense Object Detection*, arXiv:1708.02002

---

## Filosofía del handoff

El sistema actual entrena con BCE. Tu Focal Loss tiene que ser **mejor
en métricas observables** que BCE en al menos la mAP de tail-class, o
si no, el paper tiene que justificar por qué la incluimos. No es
suficiente que "funcione" — tiene que ganar.

Si te trabas en algo, no resuelvas solo: el audit y este TODO son la
spec, pero las decisiones científicas (γ, α, ablation) las discutimos
en equipo. Y si encuentras algo en el audit que crees que está mal,
también lo discutimos — el audit no es ley sagrada, es la mejor
hipótesis que tuvimos en 24 horas.

**Plazo sugerido**: una semana para implementación + 2-3 semanas para
ablations. Si los plazos no encajan con lo que tienes encima, hablamos.

— Amaury

## 2. Justificación de Diseño de Software (Arquitectura de FocalBCE)
Siguiendo los requerimientos de diseño, la implementación de la pérdida se estructuró tomando las siguientes decisiones arquitectónicas:

* **Enfoque Orientado a Objetos (Subclase de `nn.Module`):** Se decidió implementar `FocalBCE` como una clase de PyTorch en lugar de una aproximación puramente funcional. Esto garantiza la compatibilidad "drop-in" con la arquitectura interna de Ultralytics, la cual espera que el atributo `self.bce` de la clase `v8DetectionLoss` sea un objeto instanciado con su propio método `forward`. Esto permite que el *monkey-patch* sea limpio e imperceptible para el resto del pipeline de entrenamiento.
* **Elección de Variante (Standard vs. Varifocal/IoU-aware):** Para esta primera fase experimental, se optó por implementar la Focal Loss estándar (Lin et al., 2017) descartando momentáneamente variantes más complejas como Varifocal Loss. La justificación empírica es establecer una **línea base (baseline)** sólida para el paper de MICAI. Introducir variaciones combinadas (arquitectura custom + variante de loss compleja) oscurecería la matriz de ablación. Si la Focal Loss estándar demuestra superioridad en las tail-classes, la exploración de Varifocal quedará como trabajo futuro.