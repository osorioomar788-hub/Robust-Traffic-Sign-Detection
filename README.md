# Robust Traffic Sign Detection

Proyecto académico desarrollado para la 25th Mexican International Conference on Artificial Intelligence (MICAI), enfocado en la detección robusta de señales de tránsito utilizando técnicas de Inteligencia Artificial.

---

## 📌 Requisitos previos

Antes de comenzar, asegúrate de tener instalado:

* **Python 3.11.x**
* **pip** (incluido con Python)
* **Sistema operativo Windows**

---

## 📁 Estructura del proyecto

```
Robust-Traffic-Sign-Detection/
│
├── data/               # Datos del proyecto (datasets)
├── notebooks/          # Análisis y pruebas (opcional)
├── scripts/            # Scripts auxiliares del proyecto
├── venv/               # Entorno virtual de Python
├── requirements.txt    # Dependencias del proyecto
└── train.py            # Punto de entrada principal
```

---

## ⚙️ Configuración del entorno

### 1️⃣ Clonar o copiar el proyecto

Coloca la carpeta del proyecto en la ubicación deseada de tu computadora.

---

### 2️⃣ Crear el entorno virtual (solo una vez)

Desde la carpeta raíz del proyecto:

```bat
python -m venv venv
```

---

### 3️⃣ Activar el entorno virtual

En **CMD**:

```bat
venv\Scripts\activate
```

Si el entorno se activó correctamente, la terminal mostrará:

```text
(venv)
```

---

### 4️⃣ Instalar dependencias

Con el entorno virtual activo:

```bat
pip install -r requirements.txt
```

Espera a que el proceso termine sin errores.

---

## ▶️ Ejecución del proyecto

Desde la carpeta raíz y con el entorno virtual activo:

```bat
python train.py
```

Salida esperada:

```text
Entorno listo, entrenamiento iniciando...
```

---

## ℹ️ Notas importantes

* El proyecto se encuentra en una etapa inicial.
* Actualmente no es necesario descargar datasets adicionales.
* La carpeta `data/` se utilizará en etapas posteriores del desarrollo.
* El archivo `train.py` funciona como prueba de configuración del entorno.

---

## 👥 Trabajo en equipo

Cada integrante del equipo debe repetir los pasos de configuración del entorno para garantizar compatibilidad y reproducibilidad del proyecto.

---

## 📄 Licencia

Este proyecto se distribuye bajo la **GNU Affero General Public License v3.0 (AGPL-3.0)**. El texto íntegro de la licencia se encuentra en el archivo [`LICENSE`](LICENSE) en la raíz del repositorio.

La elección de AGPL-3.0 se debe a que el pipeline de entrenamiento se construye sobre [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics), también bajo AGPL-3.0. Al integrar y modificar `ultralytics.utils.loss.v8DetectionLoss` mediante el monkey-patch en [`scripts/phase4_focal_loss.py`](scripts/phase4_focal_loss.py), este repositorio constituye un *trabajo derivado* en el sentido de la licencia y por lo tanto debe redistribuirse bajo los mismos términos (audit A-2).

### Atribución y citas

* **Ultralytics YOLOv8** — Jocher, G. *et al.* (2023). *Ultralytics YOLOv8*. AGPL-3.0. <https://github.com/ultralytics/ultralytics>
* **Focal Loss** — Lin, T.-Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017). *Focal Loss for Dense Object Detection*. arXiv:1708.02002. La formulación `FL(p_t) = -α_t · (1-p_t)^γ · log(p_t)` con γ=2.0 y α=0.25 implementada en [`scripts/phase4_focal_loss.py`](scripts/phase4_focal_loss.py) es directa de este trabajo (audit A-1, A-3).
* **TT100K** — Zhu, Z., Liang, D., Zhang, S., Huang, X., Li, B., & Hu, S. (2016). *Traffic-Sign Detection and Classification in the Wild*. CVPR. Dataset utilizado para el entrenamiento (143 clases densas tras el remap; ver [`data/processed/dataset.yaml`](data/processed/dataset.yaml)).

---

## 🔁 Reproducibilidad

Para entrenar desde un *checkout* limpio:

```bat
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python scripts/phase4_validate.py        # smoke test (~30 s)
python train.py --epochs 50 --batch 16   # entrenamiento completo
```

Las imágenes de `data/split/*/images/` **no se versionan** por tamaño (ver `.gitignore`); las etiquetas YOLO sí. El dataset original puede obtenerse de la fuente canónica de TT100K. Los identificadores de clase ya se encuentran densamente empaquetados en `[0, 142]` por el script idempotente [`scripts/phase1_remap_labels.py`](scripts/phase1_remap_labels.py).
