# Detector Robusto de Señales de Tráfico
## 25th MICAI - Mexican International Conference on Artificial Intelligence

### Descripción del Proyecto
Desarrollo de un Detector Robusto de Señales de Tráfico Multi-escala en Tiempo Real mediante Técnicas de Deep Learning y Domain Adaptation.

---

## 📁 Estructura del Proyecto

```
ROBUST-TRAFFIC-SIGN-DETECTION/
├── data/                          # Directorio de datos (NO subir a git)
│   ├── raw/                       # Datos originales descargados
│   │   ├── images/
│   │   │   ├── train/
│   │   │   └── test/
│   │   ├── annotations_json/
│   │   │   └── annotations.json
│   │   └── split/
│   └── processed/                 # Datos procesados en formato YOLO
│       ├── train/
│       │   ├── images/
│       │   └── labels/
│       ├── val/
│       │   ├── images/
│       │   └── labels/
│       ├── test/
│       │   ├── images/
│       │   └── labels/
│       └── dataset.yaml
│
├── scripts/                       # Scripts de procesamiento
│   ├── phase1_data_acquisition.py      # Fase 1 (Omar)
│   ├── phase2_synthetic_generation.py   # Fase 2 (Yael)
│   ├── phase3_architecture_design.py    # Fase 3 (Amaury)
│   ├── phase4_training.py              # Fase 4 (Enrique)
│   ├── phase5_edge_deployment.py       # Fase 5 (Nataly)
│   └── phase6_validation.py            # Fase 6 (Gabriela)
│
├── models/                        # Modelos y configuraciones
│   ├── configs/                   # Archivos .yaml de arquitectura
│   └── weights/                   # Pesos entrenados (.pt, .onnx, .trt)
│
├── notebooks/                     # Jupyter notebooks para experimentación
│   ├── 01_data_exploration.ipynb
│   ├── 02_augmentation_tests.ipynb
│   └── 03_results_analysis.ipynb
│
├── results/                       # Resultados y visualizaciones
│   ├── metrics/                   # Tablas de métricas
│   ├── plots/                     # Gráficas y visualizaciones
│   └── comparisons/               # Comparaciones con SOTA
│
├── venv/                         # Entorno virtual (NO subir a git)
├── .gitignore
├── requirements.txt
├── README.md
└── train.py                      # Script principal de entrenamiento
```

---

## 🚀 Configuración Inicial

### 1. Clonar el Repositorio
```bash
cd ~/projects
git clone <tu-repo>
cd ROBUST-TRAFFIC-SIGN-DETECTION
```

### 2. Crear Entorno Virtual
```bash
python3 -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

### 3. Instalar Dependencias
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Estructura de requirements.txt
```txt
# Deep Learning Framework
torch>=2.0.0
torchvision>=0.15.0

# YOLO
ultralytics>=8.0.0

# Computer Vision
opencv-python>=4.8.0
opencv-contrib-python>=4.8.0

# Image Processing & Augmentation
albumentations>=1.3.0
Pillow>=10.0.0

# Generative AI (Fase 2)
diffusers>=0.21.0
transformers>=4.30.0
accelerate>=0.20.0
controlnet-aux>=0.0.3

# Data Processing
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.14.0

# Progress Bars
tqdm>=4.65.0

# File Handling
pycocotools>=2.0.6
pyyaml>=6.0

# Export & Deployment (Fase 5)
onnx>=1.14.0
onnxruntime>=1.15.0

# Utilities
requests>=2.31.0
```

---

## 📊 Fase 1: Adquisición y Preprocesamiento de Datos
**Responsable: Omar**

### Dónde Colocar el Código
```
scripts/phase1_data_acquisition.py
```

### Cómo Ejecutar
```bash
# Activar entorno virtual
source venv/bin/activate

# Ejecutar script de Fase 1
python scripts/phase1_data_acquisition.py
```

### Pasos Detallados

#### 1. Descargar el Dataset TT100K
- Visita: https://cg.cs.tsinghua.edu.cn/traffic-sign/
- Descarga:
  - `train.zip` (~40GB)
  - `test.zip` (~10GB)
  - `annotations.json`

#### 2. Organizar los Archivos
```bash
# Crear estructura de directorios
mkdir -p data/raw/images/train
mkdir -p data/raw/images/test
mkdir -p data/raw/annotations_json

# Extraer archivos
unzip train.zip -d data/raw/images/train/
unzip test.zip -d data/raw/images/test/
cp annotations.json data/raw/annotations_json/
```

#### 3. Ejecutar la Conversión
```bash
python scripts/phase1_data_acquisition.py
```

#### 4. Verificar la Salida
```bash
# Ver estructura creada
tree data/processed -L 2

# Verificar archivos YAML
cat data/processed/dataset.yaml

# Verificar algunas etiquetas
head data/processed/train/labels/00001.txt
```

---

## 🎨 Fase 2: Generación de Datos Sintéticos
**Responsable: Yael**

### Recursos Necesarios
- Stable Diffusion con ControlNet
- GPU con al menos 8GB VRAM

### Dónde Colocar el Código
```
scripts/phase2_synthetic_generation.py
notebooks/02_augmentation_tests.ipynb
```

---

## 🏗️ Fase 3: Ingeniería de Arquitectura
**Responsable (Lead): Amaury**

### Dónde Colocar el Código
```
scripts/phase3_architecture_design.py
models/configs/yolov8_custom.yaml
```

### Modificaciones Clave
- Integración de Convolución Sub-píxel (PixelShuffle)
- Inserción de bloques de Coordinate Attention

---

## 🎯 Fase 4: Entrenamiento
**Responsable: Enrique**

### Dónde Colocar el Código
```
scripts/phase4_training.py
train.py
```

---

## 📱 Fase 5: Despliegue en Borde
**Responsable: Nataly**

### Dónde Colocar el Código
```
scripts/phase5_edge_deployment.py
```

---

## 📈 Fase 6: Validación y Benchmarking
**Responsable: Gabriela**

### Dónde Colocar el Código
```
scripts/phase6_validation.py
notebooks/03_results_analysis.ipynb
```

---

## 🔧 Comandos Útiles

### Verificar Instalación de CUDA
```bash
python -c "import torch; print(f'CUDA disponible: {torch.cuda.is_available()}')"
python -c "import torch; print(f'Versión CUDA: {torch.version.cuda}')"
```

### Verificar Ultralytics (YOLO)
```bash
yolo checks
```

### Entrenar Modelo Base (Ejemplo)
```bash
yolo detect train data=data/processed/dataset.yaml model=yolov8n.pt epochs=100 imgsz=640
```

---

## 📝 Notas Importantes

### Control de Versiones (.gitignore)
```gitignore
# Entorno virtual
venv/
env/

# Datos (muy pesados)
data/raw/
data/processed/

# Modelos entrenados
models/weights/*.pt
models/weights/*.onnx
models/weights/*.trt

# Cache de Python
__pycache__/
*.pyc
*.pyo

# Jupyter
.ipynb_checkpoints/

# IDE
.vscode/
.idea/

# Sistema
.DS_Store
Thumbs.db
```

### Buenas Prácticas
1. **Commits frecuentes**: Hacer commit después de cada subtarea completada
2. **Documentación**: Comentar código complejo
3. **Logs**: Guardar logs de entrenamiento
4. **Backup**: Hacer backup de modelos entrenados
5. **Versionado**: Usar tags de git para versiones importantes

---

## 📚 Recursos de Referencia

### Documentación Oficial
- [Ultralytics YOLOv8](https://docs.ultralytics.com/)
- [PyTorch](https://pytorch.org/docs/stable/index.html)
- [HuggingFace Diffusers](https://huggingface.co/docs/diffusers/index)

### Papers Clave
- [YOLO Series](https://arxiv.org/abs/2304.00501)
- [Focal Loss](https://arxiv.org/abs/1708.02002)
- [ControlNet](https://arxiv.org/abs/2302.05543)
- [Coordinate Attention](https://arxiv.org/abs/2103.02907)

---

## 👥 Equipo

| Fase | Responsable | Actividad |
|------|-------------|-----------|
| 1 | Omar | Adquisición y Preprocesamiento |
| 2 | Yael | Generación de Datos Sintéticos |
| 3 | Amaury | Ingeniería de Arquitectura |
| 4 | Enrique | Entrenamiento y Optimización |
| 5 | Nataly | Despliegue en Borde |
| 6 | Gabriela | Validación y Benchmarking |

---

## 📧 Contacto
Para dudas o sugerencias sobre el proyecto, contactar al equipo del 25th MICAI.
