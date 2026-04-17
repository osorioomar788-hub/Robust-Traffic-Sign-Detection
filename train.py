import sys
# Esto asegura que Python encuentre tus carpetas locales
sys.path.insert(0, '.') 

# 1. INYECTAR LA ARQUITECTURA (El Monkey-Patch)
try:
    from scripts.phase3_architecture_design import register_custom_modules
    register_custom_modules()
    print("[INFO] Módulos de la Fase 3 (PixelShuffle/CoordAtt) inyectados con éxito.")
except ImportError:
    print("[ADVERTENCIA] No se encontraron los módulos de Amaury. Verifica si la carpeta 'models' y los scripts de la Fase 3 están en esta rama.")

# 2. IMPORTAR LA LIBRERÍA DE YOLO
from ultralytics import YOLO

if __name__ == '__main__':
    # 3. CARGAR EL MODELO BASE O PERSONALIZADO
    # Usa el de la línea 19 para utilizar la fase 3, Si no, usa el yolov8n.pt temporalmente para probar (línea 20).
    #model = YOLO('models/configs/yolov8_custom.yaml') 
    model = YOLO('yolov8n.pt') 

    # 4. CONFIGURACIÓN DE HIPERPARÁMETROS (Tu zona de control)
    print("Iniciando entrenamiento de la Fase 4...")
    
    model.train(
        data='data/processed/dataset.yaml', # La ruta al mapa de datos
        epochs=50,                          # Cantidad de veces que la red verá TODO el dataset
        imgsz=640,                          # Resolución a la que se redimensionarán las imágenes
        batch=16,                           # Cuántas imágenes procesa a la vez (Ajustar según tu GPU)
        device=0,                           # Indica que use tu tarjeta de video NVIDIA (CUDA)
        workers=8,                          # Cuántos hilos de tu procesador ayudan a cargar imágenes
        project='runs/detect',              # EL CANDADO: Obliga a guardar los resultados aquí
        name='test2_focal'                  # Nombre de la carpeta donde se guardarán tus resultados
    )