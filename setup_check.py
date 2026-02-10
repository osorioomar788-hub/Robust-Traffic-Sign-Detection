"""
Script de Prueba Inicial - Verificar que YOLO funciona correctamente
Úsalo ANTES de empezar con la Fase 1 para asegurar que todo está instalado
"""

import sys
import subprocess

def check_installation():
    """Verifica que todas las dependencias estén instaladas"""
    
    print("=" * 60)
    print("VERIFICANDO INSTALACIÓN")
    print("=" * 60)
    
    # Verificar Python
    print(f"\n✓ Python Version: {sys.version}")
    
    # Verificar PyTorch
    try:
        import torch
        print(f"✓ PyTorch Version: {torch.__version__}")
        print(f"✓ CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"✓ CUDA Version: {torch.version.cuda}")
            print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("✗ PyTorch NO instalado - Ejecuta: pip install torch torchvision")
        return False
    
    # Verificar Ultralytics (YOLO)
    try:
        import ultralytics
        print(f"✓ Ultralytics Version: {ultralytics.__version__}")
    except ImportError:
        print("✗ Ultralytics NO instalado - Ejecuta: pip install ultralytics")
        return False
    
    # Verificar OpenCV
    try:
        import cv2
        print(f"✓ OpenCV Version: {cv2.__version__}")
    except ImportError:
        print("✗ OpenCV NO instalado - Ejecuta: pip install opencv-python")
        return False
    
    # Verificar otras dependencias
    dependencies = {
        'numpy': 'NumPy',
        'pandas': 'Pandas',
        'matplotlib': 'Matplotlib',
        'tqdm': 'tqdm'
    }
    
    for module, name in dependencies.items():
        try:
            __import__(module)
            print(f"✓ {name} instalado")
        except ImportError:
            print(f"✗ {name} NO instalado")
    
    return True


def test_yolo_basic():
    """Prueba básica de YOLO con una imagen de ejemplo"""
    
    print("\n" + "=" * 60)
    print("PRUEBA BÁSICA DE YOLO")
    print("=" * 60)
    
    try:
        from ultralytics import YOLO
        import requests
        from PIL import Image
        from io import BytesIO
        
        print("\nDescargando imagen de prueba...")
        # Descargar imagen de ejemplo (un auto en la calle)
        url = "https://ultralytics.com/images/bus.jpg"
        response = requests.get(url)
        img = Image.open(BytesIO(response.content))
        img.save("test_image.jpg")
        print("✓ Imagen descargada: test_image.jpg")
        
        print("\nCargando modelo YOLOv8 nano (descarga automática)...")
        model = YOLO('yolov8n.pt')  # Se descarga automáticamente la primera vez
        print("✓ Modelo cargado")
        
        print("\nRealizando inferencia...")
        results = model("test_image.jpg")
        
        print("✓ Inferencia completada")
        print(f"  Objetos detectados: {len(results[0].boxes)}")
        
        # Guardar resultado
        results[0].save("test_result.jpg")
        print("✓ Resultado guardado: test_result.jpg")
        
        print("\n¡YOLO está funcionando correctamente!")
        return True
        
    except Exception as e:
        print(f"\n✗ Error en la prueba de YOLO: {str(e)}")
        return False


def create_project_structure():
    """Crea la estructura básica de carpetas del proyecto"""
    
    print("\n" + "=" * 60)
    print("CREANDO ESTRUCTURA DE PROYECTO")
    print("=" * 60)
    
    import os
    
    directories = [
        "data/raw/images/train",
        "data/raw/images/test",
        "data/raw/annotations_json",
        "data/processed/train/images",
        "data/processed/train/labels",
        "data/processed/val/images",
        "data/processed/val/labels",
        "data/processed/test/images",
        "data/processed/test/labels",
        "scripts",
        "models/configs",
        "models/weights",
        "notebooks",
        "results/metrics",
        "results/plots",
        "results/comparisons"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Creado: {directory}")
    
    print("\n✓ Estructura de proyecto creada correctamente")


def show_next_steps():
    """Muestra los siguientes pasos a seguir"""
    
    print("\n" + "=" * 60)
    print("PRÓXIMOS PASOS")
    print("=" * 60)
    
    print("""
1. DESCARGAR DATASET TT100K:
   - Visita: https://cg.cs.tsinghua.edu.cn/traffic-sign/
   - Descarga: train.zip, test.zip, annotations.json
   - Extrae en: data/raw/

2. EJECUTAR FASE 1 (Omar):
   cd scripts
   python phase1_data_acquisition.py

3. VERIFICAR CONVERSIÓN:
   - Revisa: data/processed/
   - Verifica etiquetas en formato YOLO
   - Usa Roboflow para visualización

4. CONTINUAR CON FASE 2 (Yael):
   - Generación de datos sintéticos
   - Stable Diffusion + ControlNet

5. GIT:
   git init
   git add .
   git commit -m "Estructura inicial del proyecto"

DOCUMENTACIÓN:
- README.md: Guía completa del proyecto
- requirements.txt: Dependencias necesarias
- .gitignore: Archivos a ignorar en git

¿DUDAS?
- Revisa el README.md
- Consulta la documentación de YOLO: https://docs.ultralytics.com/
""")


def main():
    """Función principal"""
    
    print("""
╔══════════════════════════════════════════════════════════════╗
║  DETECTOR ROBUSTO DE SEÑALES DE TRÁFICO                      ║
║  25th MICAI - Setup Inicial                                  ║
╚══════════════════════════════════════════════════════════════╝
""")
    
    # Paso 1: Verificar instalación
    if not check_installation():
        print("\n⚠ Por favor instala las dependencias faltantes y vuelve a ejecutar")
        print("Ejecuta: pip install -r requirements.txt")
        return
    
    # Paso 2: Prueba básica de YOLO
    if not test_yolo_basic():
        print("\n⚠ Hay problemas con YOLO, revisa la instalación")
        return
    
    # Paso 3: Crear estructura del proyecto
    create_project_structure()
    
    # Paso 4: Mostrar próximos pasos
    show_next_steps()
    
    print("\n" + "=" * 60)
    print("✓ SETUP INICIAL COMPLETADO")
    print("=" * 60)


if __name__ == "__main__":
    main()
