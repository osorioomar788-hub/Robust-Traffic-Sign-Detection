"""
Fase 1: Adquisición y Preprocesamiento de Datos
Responsable: Omar
Proyecto: Detector Robusto de Señales de Tráfico
25th MICAI
"""

import os
import json
import shutil
import requests
from pathlib import Path
from typing import Dict, List, Tuple
import cv2
import numpy as np
from tqdm import tqdm

from scripts.phase1_remap_labels import remap_labels_dense


class TT100KDatasetPreprocessor:
    """
    Clase para gestionar la descarga, conversión y preprocesamiento del dataset TT100K
    """
    
    def __init__(self, base_dir: str = "./data"):
        """
        Inicializa el preprocesador
        
        Args:
            base_dir: Directorio base donde se guardará el dataset
        """
        self.base_dir = Path(base_dir)
        self.raw_dir = self.base_dir / "raw"
        self.processed_dir = self.base_dir / "processed"
        
        # Crear estructura de directorios
        self._create_directory_structure()
        
    def _create_directory_structure(self):
        """Crea la estructura de directorios necesaria"""
        dirs = [
            self.raw_dir / "annotations_json",
            self.raw_dir / "images",
            self.raw_dir / "split",
            self.processed_dir / "train" / "images",
            self.processed_dir / "train" / "labels",
            self.processed_dir / "val" / "images",
            self.processed_dir / "val" / "labels",
            self.processed_dir / "test" / "images",
            self.processed_dir / "test" / "labels",
        ]
        
        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)
            
        print(f"✓ Estructura de directorios creada en: {self.base_dir}")
    
    def download_dataset(self):
        """
        Descarga el dataset TT100K
        NOTA: Debido a las restricciones de acceso, este método proporciona instrucciones
        """
        print("\n=== INSTRUCCIONES PARA DESCARGAR TT100K ===")
        print("1. Visita: https://cg.cs.tsinghua.edu.cn/traffic-sign/")
        print("2. Descarga los siguientes archivos:")
        print("   - train.zip (imágenes de entrenamiento)")
        print("   - test.zip (imágenes de prueba)")
        print("   - annotations.json (anotaciones)")
        print(f"3. Extrae las imágenes en: {self.raw_dir / 'images'}")
        print(f"4. Coloca annotations.json en: {self.raw_dir / 'annotations_json'}")
        print("\nEstructura esperada:")
        print(f"{self.raw_dir}/")
        print("├── images/")
        print("│   ├── train/")
        print("│   │   ├── 00000.jpg")
        print("│   │   ├── 00001.jpg")
        print("│   │   └── ...")
        print("│   └── test/")
        print("│       └── ...")
        print("└── annotations_json/")
        print("    └── annotations.json")
        
    def load_annotations(self, annotations_path: str = None) -> Dict:
        """
        Carga el archivo de anotaciones JSON
        
        Args:
            annotations_path: Ruta al archivo annotations.json
            
        Returns:
            Diccionario con las anotaciones
        """
        if annotations_path is None:
            annotations_path = self.raw_dir / "annotations_json" / "annotations.json"
        
        print(f"\nCargando anotaciones desde: {annotations_path}")
        
        with open(annotations_path, 'r', encoding='utf-8') as f:
            annotations = json.load(f)
            
        print(f"✓ Anotaciones cargadas: {len(annotations.get('imgs', {}))} imágenes")
        return annotations
    
    def get_class_mapping(self, annotations: Dict) -> Tuple[Dict[str, int], Dict[int, str]]:
        """
        Extrae el mapeo de clases del dataset TT100K
        
        Args:
            annotations: Diccionario de anotaciones
            
        Returns:
            Tupla con (class_to_id, id_to_class)
        """
        types = annotations.get('types', [])
        
        # Crear mapeo: nombre_clase -> id (empezando desde 0)
        class_to_id = {class_name: idx for idx, class_name in enumerate(types)}
        id_to_class = {idx: class_name for class_name, idx in class_to_id.items()}
        
        print(f"\n✓ Clases detectadas: {len(class_to_id)}")
        print(f"Ejemplos: {list(class_to_id.keys())[:5]}")
        
        return class_to_id, id_to_class
    
    def convert_bbox_to_yolo(self, bbox: Dict, img_width: int, img_height: int) -> Tuple[float, float, float, float]:
        """
        Convierte bounding box del formato TT100K a formato YOLO
        
        TT100K format: {'xmin': x, 'ymin': y, 'xmax': x2, 'ymax': y2}
        YOLO format: [x_center, y_center, width, height] (normalizados 0-1)
        
        Args:
            bbox: Diccionario con coordenadas de la caja
            img_width: Ancho de la imagen
            img_height: Alto de la imagen
            
        Returns:
            Tupla (x_center, y_center, width, height) normalizada
        """
        xmin = bbox['xmin']
        ymin = bbox['ymin']
        xmax = bbox['xmax']
        ymax = bbox['ymax']
        
        # Calcular centro y dimensiones
        x_center = (xmin + xmax) / 2.0
        y_center = (ymin + ymax) / 2.0
        width = xmax - xmin
        height = ymax - ymin
        
        # Normalizar por dimensiones de la imagen
        x_center_norm = x_center / img_width
        y_center_norm = y_center / img_height
        width_norm = width / img_width
        height_norm = height / img_height
        
        # Asegurar que los valores estén en el rango [0, 1]
        x_center_norm = max(0.0, min(1.0, x_center_norm))
        y_center_norm = max(0.0, min(1.0, y_center_norm))
        width_norm = max(0.0, min(1.0, width_norm))
        height_norm = max(0.0, min(1.0, height_norm))
        
        return x_center_norm, y_center_norm, width_norm, height_norm
    
    def convert_annotations_to_yolo(self, annotations: Dict, class_to_id: Dict, 
                                   split: str = 'train'):
        """
        Convierte las anotaciones del formato JSON a formato YOLO .txt
        
        Args:
            annotations: Diccionario de anotaciones
            class_to_id: Mapeo de nombre de clase a ID
            split: 'train', 'val', o 'test'
        """
        imgs_data = annotations.get('imgs', {})
        
        print(f"\n=== Convirtiendo anotaciones para split: {split} ===")
        
        converted_count = 0
        skipped_count = 0
        
        for img_id, img_info in tqdm(imgs_data.items(), desc=f"Procesando {split}"):
            # Verificar que la imagen pertenece al split correcto
            if img_info.get('path', '').startswith(split):
                
                # Ruta de la imagen
                img_path = self.raw_dir / "images" / img_info['path']
                
                if not img_path.exists():
                    skipped_count += 1
                    continue
                
                # Obtener dimensiones de la imagen
                img = cv2.imread(str(img_path))
                if img is None:
                    skipped_count += 1
                    continue
                    
                img_height, img_width = img.shape[:2]
                
                # Nombre del archivo de etiquetas
                img_name = img_path.stem
                label_path = self.processed_dir / split / "labels" / f"{img_name}.txt"
                
                # Copiar imagen al directorio procesado
                dest_img_path = self.processed_dir / split / "images" / img_path.name
                shutil.copy2(img_path, dest_img_path)
                
                # Procesar objetos en la imagen
                objects = img_info.get('objects', [])
                
                yolo_annotations = []
                for obj in objects:
                    category = obj.get('category')
                    bbox = obj.get('bbox')
                    
                    if category not in class_to_id or not bbox:
                        continue
                    
                    class_id = class_to_id[category]
                    
                    # Convertir bbox a formato YOLO
                    x_center, y_center, width, height = self.convert_bbox_to_yolo(
                        bbox, img_width, img_height
                    )
                    
                    # Formato YOLO: class_id x_center y_center width height
                    yolo_annotations.append(
                        f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
                    )
                
                # Guardar archivo de etiquetas
                with open(label_path, 'w') as f:
                    f.write('\n'.join(yolo_annotations))
                
                converted_count += 1
        
        print(f"✓ Conversión completada:")
        print(f"  - Imágenes convertidas: {converted_count}")
        print(f"  - Imágenes omitidas: {skipped_count}")
    
    def create_data_splits(self, annotations: Dict, 
                          train_ratio: float = 0.7,
                          val_ratio: float = 0.15,
                          test_ratio: float = 0.15):
        """
        Realiza la segregación estratificada de los datos
        
        Args:
            annotations: Diccionario de anotaciones
            train_ratio: Proporción para entrenamiento
            val_ratio: Proporción para validación
            test_ratio: Proporción para prueba
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            "Las proporciones deben sumar 1.0"
        
        print("\n=== Creando splits de datos ===")
        print(f"Train: {train_ratio*100}%, Val: {val_ratio*100}%, Test: {test_ratio*100}%")
        
        # En TT100K, las imágenes ya están divididas en train y test
        # Aquí tomaremos una parte del train para validation
        
        # Este método es simplificado - en producción deberías hacer
        # una estratificación por clases
        
    def create_dataset_yaml(self, class_to_id: Dict):
        """
        Crea el archivo dataset.yaml para YOLO
        
        Args:
            class_to_id: Mapeo de nombre de clase a ID
        """
        yaml_content = f"""# TT100K Dataset Configuration
# Proyecto: Detector Robusto de Señales de Tráfico - 25th MICAI

path: {self.processed_dir.absolute()}
train: train/images
val: val/images
test: test/images

# Número de clases
nc: {len(class_to_id)}

# Nombres de las clases
names:
"""
        
        # Agregar nombres de clases en orden por ID
        id_to_class = {v: k for k, v in class_to_id.items()}
        for i in range(len(class_to_id)):
            yaml_content += f"  {i}: {id_to_class[i]}\n"
        
        yaml_path = self.processed_dir / "dataset.yaml"
        with open(yaml_path, 'w', encoding='utf-8') as f:
            f.write(yaml_content)
        
        print(f"\n✓ Archivo dataset.yaml creado en: {yaml_path}")
    
    def validate_conversion(self, sample_size: int = 5):
        """
        Valida la conversión visualizando algunas muestras aleatorias
        
        Args:
            sample_size: Número de muestras a visualizar
        """
        print(f"\n=== Validando conversión (mostrando {sample_size} ejemplos) ===")
        
        train_imgs = list((self.processed_dir / "train" / "images").glob("*.jpg"))
        
        if len(train_imgs) == 0:
            print("⚠ No se encontraron imágenes para validar")
            return
        
        samples = np.random.choice(train_imgs, min(sample_size, len(train_imgs)), replace=False)
        
        for img_path in samples:
            label_path = self.processed_dir / "train" / "labels" / f"{img_path.stem}.txt"
            
            if not label_path.exists():
                continue
            
            img = cv2.imread(str(img_path))
            height, width = img.shape[:2]
            
            # Leer anotaciones YOLO
            with open(label_path, 'r') as f:
                lines = f.readlines()
            
            print(f"\n{img_path.name}:")
            print(f"  Dimensiones: {width}x{height}")
            print(f"  Número de objetos: {len(lines)}")
            
            # Mostrar primeras 3 anotaciones
            for i, line in enumerate(lines[:3]):
                parts = line.strip().split()
                print(f"  Objeto {i+1}: clase={parts[0]}, bbox=({parts[1]}, {parts[2]}, {parts[3]}, {parts[4]})")


def main():
    """
    Función principal para ejecutar la Fase 1
    """
    print("=" * 60)
    print("FASE 1: ADQUISICIÓN Y PREPROCESAMIENTO DE DATOS")
    print("Responsable: Omar")
    print("Proyecto: Detector Robusto de Señales de Tráfico")
    print("=" * 60)
    
    # Inicializar preprocesador
    preprocessor = TT100KDatasetPreprocessor(base_dir="./data")
    
    # Paso 1: Instrucciones de descarga
    preprocessor.download_dataset()
    
    print("\n" + "=" * 60)
    print("PAUSA: Por favor descarga el dataset antes de continuar")
    print("Presiona Enter cuando hayas completado la descarga...")
    print("=" * 60)
    input()
    
    # Paso 2: Cargar anotaciones
    try:
        annotations = preprocessor.load_annotations()
    except FileNotFoundError:
        print("\n⚠ ERROR: No se encontró el archivo annotations.json")
        print("Verifica que hayas descargado y colocado el archivo correctamente")
        return
    
    # Paso 3: Obtener mapeo de clases
    class_to_id, id_to_class = preprocessor.get_class_mapping(annotations)
    
    # Paso 4: Convertir anotaciones a formato YOLO
    print("\nIniciando conversión a formato YOLO...")
    
    # Convertir cada split
    for split in ['train', 'test']:
        preprocessor.convert_annotations_to_yolo(annotations, class_to_id, split)
    
    # Paso 4b: Dense-pack class IDs to [0, nc-1] across all splits (audit F-3).
    # TT100K's `types` list carries 232 entries but only ~143 classes actually
    # appear in data; Ultralytics requires densely packed IDs. Train split is
    # the source of truth; val/test annotations referring to train-absent
    # classes are dropped. Idempotent — no-op if already dense.
    print("\n=== Dense-packing class IDs (audit F-3) ===")
    remap_labels_dense(
        map_source_globs=(f"{preprocessor.processed_dir}/train/labels/*.txt",),
        rewrite_globs=(
            f"{preprocessor.processed_dir}/train/labels/*.txt",
            f"{preprocessor.processed_dir}/val/labels/*.txt",
            f"{preprocessor.processed_dir}/test/labels/*.txt",
        ),
    )

    # Paso 5: Crear archivo dataset.yaml
    preprocessor.create_dataset_yaml(class_to_id)
    
    # Paso 6: Validar conversión
    preprocessor.validate_conversion(sample_size=5)
    
    print("\n" + "=" * 60)
    print("✓ FASE 1 COMPLETADA")
    print("=" * 60)
    print(f"\nDirectorio de salida: {preprocessor.processed_dir}")
    print("\nPróximos pasos:")
    print("1. Revisar las imágenes y etiquetas en el directorio 'processed'")
    print("2. Usar Roboflow u otra herramienta para verificar visualmente las bounding boxes")
    print("3. Proceder con la Fase 2 (Generación de Datos Sintéticos)")


if __name__ == "__main__":
    main()
