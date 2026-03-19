import sys
# Esto asegura que Python encuentre tus carpetas locales
sys.path.insert(0, '.') 

# 1. INYECTAR LA ARQUITECTURA DE AMAURY (El Monkey-Patch)
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
    # Si Amaury ya subió su archivo, usa el de la línea 19. Si no, usa el yolov8n.pt temporalmente para probar.
    #model = YOLO('models/configs/yolov8_custom.yaml') 
    model = YOLO('yolov8n.pt') 

    # 4. CONFIGURACIÓN DE HIPERPARÁMETROS (Tu zona de control)
    print("Iniciando entrenamiento de la Fase 4...")
    
    model.train(
        data='data/processed/dataset.yaml', # La ruta al mapa de datos de Omar
        epochs=1,                           # Cantidad de veces que la red verá TODO el dataset   100
        imgsz=640,                          # Resolución a la que se redimensionarán las imágenes
        batch=4,                            # Cuántas imágenes procesa a la vez (Ajustar según tu GPU)  16
        device='cpu',                       # 0 indica que use tu tarjeta de video NVIDIA (CUDA)
        workers=4,                          # Cuántos hilos de tu procesador ayudan a cargar imágenes  8
        name='micai_phase4_run1'            # Nombre de la carpeta donde se guardarán tus resultados
    )