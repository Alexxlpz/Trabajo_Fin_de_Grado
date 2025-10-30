from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
import os

# ruta del YOLO entrenado
YOLO_MODEL_PATH = 'C:/Users/alexl/PycharmProjects/Trabajo_Fin_de_Grado/leavesDetection/Data/code/TFG_deteccion_hojas/yolov12_hoja_sana_run1/weights/best.pt'

# ruta de la imagen a predicie
IMAGE_SOURCE = 'C:/Users/alexl/PycharmProjects/Trabajo_Fin_de_Grado/leavesDetection/Data/dataset/predictionImages/imagenReal.jpg'

# donde guardamos la imagen con la prediccion hecha
OUTPUT_IMAGE_PATH = 'C:/Users/alexl/PycharmProjects/Trabajo_Fin_de_Grado/leavesDetection/Data/dataset/Resultados_Inferencia/inferencia_hojas_sanas/resultado_teselado/'

detection_model = AutoDetectionModel.from_pretrained(
    model_type='ultralytics',  # Tipo de modelo: YOLO
    model_path=YOLO_MODEL_PATH,
    confidence_threshold=0.3,  # Umbral de confianza al principio
    device="cuda:0"  # para usar la grafica
)
print("Modelo YOLO cargado exitosamente.")

# Ponemos que los teselados sean 640x640 ya que a yolo lo entrenemos asi y se le hara mas facil predecir de esa manera
result = get_sliced_prediction(
    IMAGE_SOURCE,
    detection_model,
    slice_height=640,          # Altura de cada teselado
    slice_width=640,           # Anchura de cada teselado
    overlap_height_ratio=0.2,  # Para el solapamiento vertical
    overlap_width_ratio=0.2    # Para el solapamiento horizontal
)

object_prediction_list = result.object_prediction_list

print(f"total objetos detectados: {len(object_prediction_list)}")
'''
for pred in object_prediction_list:
    bbox = pred.bbox # Obtiene la caja delimitadora de YOLO ([xmin, ymin, xmax, ymax])
    score = pred.score.value
    category = pred.category.name

    print(f"------- Clase: {category}, Confianza: {score:.2f}, BBox: {bbox.to_xywh()}")
'''
os.makedirs(OUTPUT_IMAGE_PATH, exist_ok=True)
result.export_visuals(export_dir=OUTPUT_IMAGE_PATH, file_name="prediccion_teselada")

print("Proceso finalizado!!!")