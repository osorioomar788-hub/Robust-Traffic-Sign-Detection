# 📦 Fase 2 — Generación de Datos Sintéticos y Aumentación
**Proyecto:** Detector Robusto de Señales de Tráfico Multi-escala  
**Conferencia:** 25th MICAI  
**Responsable:** Yael  
**Entregable para:** Amaury (Fase 3 — Ingeniería de Arquitectura)

---

## ¿Qué hace esta fase?

Toma el dataset TT100K procesado en la Fase 1 y lo transforma en el dataset **Robust-TT100K**, optimizado para entrenar un modelo resistente a condiciones climáticas adversas (lluvia, niebla, noche) y con clases balanceadas.

---

## 📊 Resultados Finales

| Split | Imágenes | Instancias | Clases únicas |
|-------|----------|------------|---------------|
| `train` (aumentado) | **67,509** | 246,377 | 143 |
| `val` (original) | 1,830 | 4,884 | 120 |
| `test` (original) | 3,067 | 8,181 | 136 |
| **TOTAL** | **72,406** | **259,442** | — |

> Partimos de **9,170 imágenes originales** → incremento de **~8x**  
> Errores de validación: **0**

---

## 🗂️ Estructura de Archivos Generados

```
data/
├── split/                        ← Datos originales divididos (NO modificar)
│   ├── train/images/ + labels/   ← 4,273 imágenes originales de entrenamiento
│   ├── val/images/   + labels/   ← 1,830 imágenes de validación
│   └── test/images/  + labels/   ← 3,067 imágenes de prueba
│
└── augmented/                    ← Dataset Robust-TT100K (usar esto para Fase 3)
    └── train/
        ├── images/               ← 67,509 imágenes (originales + sintéticas)
        ├── labels/               ← Etiquetas YOLO correspondientes
        ├── augmentation_log.json ← Log de qué pipeline se aplicó a cada imagen
        ├── balancing_log.json    ← Log del balanceo de clases
        ├── class_distribution.png    ← Gráfica antes/después de aumentación
        ├── class_balancing.png       ← Gráfica antes/después de balanceo
        └── validation_report.json   ← Reporte de integridad del dataset
```

> ⚠️ **Importante para Amaury:** Los splits `val/` y `test/` están en `data/split/`, NO en `data/augmented/`. Esto es intencional para mantener la evaluación con datos 100% reales.

---

## 📋 Formato de las Etiquetas (YOLO)

Cada imagen tiene su archivo `.txt` correspondiente con el mismo nombre. Formato por línea:

```
<class_id> <x_center> <y_center> <width> <height>
```

Todos los valores están **normalizados entre 0 y 1**. Ejemplo:
```
45 0.523438 0.401234 0.089062 0.076543
12 0.751200 0.623100 0.045000 0.038000
```

El mapeo de `class_id` → nombre de señal está en:
```
data/processed/dataset.yaml
```

---

## ⚙️ Configuración para el dataset.yaml de YOLO

Para la Fase 3, el `dataset.yaml` debe apuntar a estas rutas:

```yaml
path: ./data
train: augmented/train/images     # ← Dataset aumentado con 67,509 imágenes
val:   split/val/images           # ← Validación original (1,830 imágenes)
test:  split/test/images          # ← Prueba original (3,067 imágenes)

nc: 143   # número de clases en entrenamiento
```

> El `nc` exacto y los nombres de clases los puedes obtener de `data/processed/dataset.yaml` generado en la Fase 1.

---

## 🔄 Pipeline de Aumentación Aplicado

Cada imagen original de `train` generó **3 versiones sintéticas** usando estos pipelines:

| Pipeline | Condición simulada | Transformaciones principales |
|----------|--------------------|------------------------------|
| `general` | Variaciones de luz | Brillo, contraste, ruido gaussiano, rotación leve |
| `rain`    | Lluvia severa      | Motion blur, oscurecimiento, líneas de lluvia |
| `fog`     | Niebla densa       | Overlay blanco, reducción de contraste, blur |
| `night`   | Baja iluminación   | Oscurecimiento severo, tinte azulado, ruido |

> Las **bounding boxes se ajustan automáticamente** en cada transformación. No hay reetiquetado manual.

---

## ⚖️ Balanceo de Clases (Long-Tail)

Se detectaron **120 clases minoritarias** con menos de 500 instancias. El script de balanceo generó imágenes adicionales hasta alcanzar el umbral mínimo por clase.

- **Antes del balanceo:** 17,092 imágenes de train
- **Después del balanceo:** 67,509 imágenes de train
- **Umbral aplicado:** 500 instancias mínimas por clase

Detalle completo en: `data/augmented/train/balancing_log.json`

---

## 🚀 Lo que necesitas para la Fase 3

1. **Usar** `data/augmented/train/` como conjunto de entrenamiento
2. **Usar** `data/split/val/` para validación durante el entrenamiento
3. **Reservar** `data/split/test/` para la evaluación final (Fase 6)
4. **Configurar** el `dataset.yaml` con las rutas indicadas arriba
5. **No modificar** `data/split/` — esos datos deben permanecer puros

---

## 📜 Scripts de esta fase

| Script | Descripción |
|--------|-------------|
| `scripts/phase1_5_split.py` | División estratificada 70/15/15 |
| `scripts/phase2_classic_augmentation.py` | Aumentación con 4 pipelines climáticos |
| `scripts/phase2_class_balancing.py` | Balanceo de clases minoritarias |
| `scripts/phase2_validate_dataset.py` | Validación de integridad del dataset |

---

## 📦 Dependencias usadas

```
albumentations==1.3.1
opencv-python>=4.9.0
numpy>=1.26.0
matplotlib>=3.8.0
tqdm>=4.66.0
```

---

*Fase 2 completada ✅ — Dataset Robust-TT100K listo para ingestión en la red neuronal.*
