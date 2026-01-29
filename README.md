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

Proyecto con fines académicos y de investigación.
