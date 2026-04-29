from ultralytics import RTDETR
# 'yolov12_Leaves_Detector_run5' para entrenamiento con 500 epocas del dataset normal
WEIGHTS = '/../data/models/rtdetr-l.pt'
YAML_FILE = '../config/yolo.yaml'
PROJECT_NAME = '/../data/models/CNN'
RUN_NAME = 'rtdetr_run1'
if __name__ == '__main__':
    print("hello world!")

    model = RTDETR(WEIGHTS)

    print(f"Iniciando entrenamiento con el archivo de configuración: {YAML_FILE}")

    results = model.train(
        data=YAML_FILE,
        epochs=100,
        imgsz=640,
        batch=-1, # Usa el tamaño de batch máximo y seguro segun la GPU
        name=RUN_NAME,
        project=PROJECT_NAME,
        device=0,
        workers=12,
        patience=50,
        amp=True
        # Parámetros de ajuste fino para objetos pequeños (aunque YOLOv12 ya es bueno)
        # lr0=0.01,
        # patience=50 Detiene el entrenamiento si no hay mejora después de 50 épocas
    )

    print("Entrenamiento completado :))).")