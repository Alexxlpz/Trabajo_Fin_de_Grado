from ultralytics import YOLO
# 'yolov12_Leaves_Detector_run5' para entrenamiento con 500 epocas del dataset normal
YAML_FILE = '../../config/yolo.yaml'
PROJECT_NAME = '../../data/models/CNN'
RUN_NAME = 'yolov12_Leaves_Detector_run_6(with-sam3)'

# script para entrenar un modelo de YOLO
if __name__ == '__main__':

    #cargo YOLO12 al sistema
    model = YOLO('../../data/models/yolo12n.pt')

    print(f"iniciando entrenamiento con el archivo de configuración: {YAML_FILE}")

    results = model.train(
        data=YAML_FILE,
        epochs=400,
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
