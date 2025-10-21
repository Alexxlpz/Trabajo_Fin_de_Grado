from ultralytics import YOLO
import os

YAML_FILE = 'hojas.yaml'
PROJECT_NAME = 'TFG_deteccion_hojas'
RUN_NAME = 'yolov12_hoja_sana_run1'
if __name__ == '__main__':
    print("hello world!")

    #cargo YOLO12 al sistema
    model = YOLO('yolo12n.pt')

    print(f"Iniciando entrenamiento con el archivo de configuración: {YAML_FILE}")

    results = model.train(
        data=YAML_FILE,
        epochs=100,
        imgsz=640,
        batch=-1, # Usa el tamaño de batch máximo y seguro segun la GPU
        name=RUN_NAME,
        project=PROJECT_NAME,
        device=0,
        workers=8
        # Parámetros de ajuste fino para objetos pequeños (aunque YOLOv12 ya es bueno)
        # lr0=0.01,
        # patience=50 Detiene el entrenamiento si no hay mejora después de 50 épocas
    )

    print("Entrenamiento completado :))).")
