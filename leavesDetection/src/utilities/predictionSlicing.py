from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
import os
from src.utilities.makePlot import make_plot

def predictionSlicing(imageName: str, plots: bool = False):
    teseledImagenName = f"teselado_{imageName}"

    # ruta del YOLO entrenado
    YOLO_MODEL_PATH = 'C:/Users/alexl/PycharmProjects/Trabajo_Fin_de_Grado/leavesDetection/data/models/CNN/yolov12_Leaves_Detector_run2/weights/best.pt'

    # ruta de la imagen a predicie
    IMAGE_SOURCE = f'C:/Users/alexl/PycharmProjects/Trabajo_Fin_de_Grado/leavesDetection/data/cache/{imageName}'

    # donde guardamos la imagen con la prediccion hecha
    OUTPUT_IMAGE_PATH = 'C:/Users/alexl/PycharmProjects/Trabajo_Fin_de_Grado/leavesDetection/data/inference_results'

    MAX_HEIGHT = 640
    MAX_WIDTH = 640

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
        slice_height=640,  # Altura de cada teselado
        slice_width=640,  # Anchura de cada teselado
        overlap_height_ratio=0.2,  # Para el solapamiento vertical
        overlap_width_ratio=0.2  # Para el solapamiento horizontal
    )

    object_prediction_list = result.object_prediction_list

    print(f"total objetos detectados: {len(object_prediction_list)}")

    if plots:
        # hacemos un  grafico para ver la confianza de las predicciones tras filtrar
        confidentList = [pred.score.value for pred in result.object_prediction_list]
        make_plot(confidentList, "confianza_predicciones_teselado_antes_del_filtrado.png")

    validObjects = []
    purgated = 0
    for pred in object_prediction_list:
        if pred.bbox.maxx - pred.bbox.minx < MAX_WIDTH and pred.bbox.maxy - pred.bbox.miny < MAX_HEIGHT:  # In order to try to filter, we only accept predictions smaller than the slice size (640x640)
            validObjects.append(pred)
        else:
            purgated += 1
    print(f"Total de objetos purgados: {purgated}")
    # actualizar la lista de predicciones que usan las visualizaciones/exportaciones
    result.object_prediction_list = validObjects
    result.prediction_list = validObjects

    if plots:
        # hacemos un  grafico para ver la confianza de las predicciones tras filtrar
        confidentList = [pred.score.value for pred in result.object_prediction_list]
        make_plot(confidentList, "confianza_predicciones_teselado_tras_filtrado.png")

    '''
    for pred in object_prediction_list:
        bbox = pred.bbox # Obtiene la caja delimitadora de YOLO ([xmin, ymin, xmax, ymax])
        score = pred.score.value
        category = pred.category.name

        print(f"------- Clase: {category}, Confianza: {score:.2f}, BBox: {bbox.to_xywh()}")
    '''
    os.makedirs(OUTPUT_IMAGE_PATH, exist_ok=True)
    result.export_visuals(export_dir=OUTPUT_IMAGE_PATH, file_name=teseledImagenName)

    print("Proceso finalizado!!!")
    os.remove(IMAGE_SOURCE)

    return teseledImagenName, len(validObjects)