import base64

import cv2
import numpy as np
from PIL import Image, ExifTags
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
import os
import time

from src.backend.repository.image_repo import create_image_with_direction
from src.utilities.classifyObject import classifyObject
from src.utilities.makePlot import make_plot

# ruta del YOLO entrenado
YOLO_MODEL_PATH = 'C:/Users/alexl/PycharmProjects/Trabajo_Fin_de_Grado/leavesDetection/data/models/CNN/yolov12_Leaves_Detector_run_6(with-sam3)/weights/best.pt'

# donde guardamos la imagen con la prediccion hecha
OUTPUT_IMAGE_PATH = 'C:/Users/alexl/PycharmProjects/Trabajo_Fin_de_Grado/leavesDetection/data/inference_results'

CACHE_PATH = 'C:/Users/alexl/PycharmProjects/Trabajo_Fin_de_Grado/leavesDetection/data/cache'

_MODEL = None

def get_model():
    """
    Carga y cachea el modelo. Lanza FileNotFoundError si no existe el archivo.
    """
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    if not os.path.exists(YOLO_MODEL_PATH):
        raise FileNotFoundError(f"Modelo no encontrado en `{YOLO_MODEL_PATH}`")

    try:
        _MODEL = AutoDetectionModel.from_pretrained(
            model_type='ultralytics',  # Tipo de modelo: YOLO
            model_path=YOLO_MODEL_PATH,
            confidence_threshold=0.3,  # Umbral de confianza al principio
            device="cuda:0"  # para usar la grafica
        )
    except Exception as e:
        raise RuntimeError(f"Error cargando el modelo desde `{YOLO_MODEL_PATH}`: {e}")
    return _MODEL

# ---------------------------------------------------------------------------------------------------------------------

def predictionSlicing(imageName: str, plots: bool = False):
    # ruta de la imagen a predicie
    IMAGE_SOURCE = os.path.join(CACHE_PATH, imageName)

    detection_model = get_model()
    print("Modelo YOLO cargado exitosamente.")

    filtered_pred_list = prediction(IMAGE_SOURCE, detection_model, plots)

    encodeImage, healthy, sick = cropAndClasify(filtered_pred_list, IMAGE_SOURCE, CACHE_PATH)

    save_database_entry(
        IMAGE_SOURCE = IMAGE_SOURCE,
        healthy = healthy,
        sick = sick
    )

    #os.makedirs(OUTPUT_IMAGE_PATH, exist_ok=True)
    #result.export_visuals(export_dir=OUTPUT_IMAGE_PATH, file_name=teseledImagenName)

    #with open(OUTPUT_IMAGE_PATH+'/'+teseledImagenName+'.png', "rb") as image:
        #encodeImage = base64.b64encode(image.read()).decode('utf-8')

    print("Proceso finalizado!!!")
    os.remove(IMAGE_SOURCE)

    return encodeImage, len(filtered_pred_list)

def save_database_entry(IMAGE_SOURCE, healthy=0, sick=0):
    lat, lon, timestamp = metainfo_collect(IMAGE_SOURCE)

    img = create_image_with_direction(
        path=IMAGE_SOURCE,
        latitude=float(lat) if lat is not None else None,
        longitude=float(lon) if lon is not None else None,
        num_sick=int(sick),
        num_healthy=int(healthy),
        upload_date=timestamp
    )

    print(img)



def prediction(IMAGE_SOURCE, detection_model, plots: bool = False):

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

    if plots: # guardados en /data/plots
        # hacemos un  grafico para ver la confianza de las predicciones tras filtrar
        confidentList = [pred.score.value for pred in result.object_prediction_list]
        make_plot(confidentList, "confianza_predicciones_teselado_antes_del_filtrado.png")

    #filtrado
    validObjects = sorted(object_prediction_list, key=lambda x: x.score.value, reverse=True)
    bestItems = validObjects[:10]

    if plots: # guardados en /data/plots
        # hacemos un  grafico para ver la confianza de las predicciones tras filtrar
        confidentList = [pred.score.value for pred in result.object_prediction_list]
        make_plot(confidentList, "confianza_predicciones_teselado_tras_filtrado.png")


    return bestItems


def cropAndClasify(predictions, image, cache_folder):
    output_folder = os.path.join(cache_folder, "crops")
    os.makedirs(output_folder, exist_ok=True)

    img = load_image_fix_orientation(image)
    if img is None:
        raise FileNotFoundError(f"Imagen no encontrada: {image}")


    saved = 0
    healthy = 0
    sick = 0
    for i, object_prediction in enumerate(predictions):

        crop, coords = cropObject(img, object_prediction)

        #filename = f"crop_{i}.jpg"
        #out_path = os.path.join(output_folder, filename)
        #ok = cv2.imwrite(out_path, crop, [cv2.IMWRITE_JPEG_QUALITY, int(100)]) #realmente no es necesario guardar el
                                                                                # crop, le pasamos directamente el array
                                                                                #al clasificador y que se encargue el
                                                                                #otro archivo
        if classifyObject(crop):
            #pintamos cuadro verde
            cv2.rectangle(img, (coords[0], coords[1]), (coords[2], coords[3]), (0, 255, 0), 2)
            healthy += 1
        else:
            #pintamos cuadro rojo
            cv2.rectangle(img, (coords[0], coords[1]), (coords[2], coords[3]), (0, 0, 255), 2)
            sick += 1

    print("se han guardado "+str(saved)+" crops")

    # Guardar la imagen anotada y devolverla en Base64
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    annotated_filename = f"annotated_{timestamp}.png"
    annotated_path = os.path.join(output_folder, annotated_filename)
    ok = cv2.imwrite(annotated_path, img, [cv2.IMWRITE_JPEG_QUALITY, int(90)])

    if not ok:
        raise RuntimeError(f"No se pudo guardar la imagen anotada en {annotated_path}")

    with open(annotated_path, "rb") as f:
        annotated_b64 = base64.b64encode(f.read()).decode('utf-8')

    return annotated_b64, healthy, sick



def cropObject(img, object_prediction):
    bbox = object_prediction.bbox.to_xyxy()
    x1, y1, x2, y2 = [int(coord) for coord in bbox]
    h, w = img.shape[:2]

    x1 = max(0, min(x1, w - 1))
    x2 = max(1, min(x2, w))
    y1 = max(0, min(y1, h - 1))
    y2 = max(1, min(y2, h))

    if x2 <= x1 or y2 <= y1:
        print(f"skipeamos el siguiente bbox: ({x1},{y1},{x2},{y2})")
        return -1  # Indicador de bbox no válido

    #print(x1, y1, x2, y2)

    # (0,0) <------------------- (x1,y1)
    #
    #
    #
    # (x2,y2) ------------------> (100,400)

    crop = img[y1:y2, x1:x2].copy()

    return crop, (x1, y1, x2, y2)

def metainfo_collect(IMAGE_SOURCE):
    try:
        img = Image.open(IMAGE_SOURCE)
        exif_data = img._getexif()
        if exif_data is not None:
            gps_info = exif_data.get(34853)  # GPSInfo tag
            if gps_info is not None:
                lat = gps_info.get(2)  # Latitude
                lon = gps_info.get(4)  # Longitude
                timestamp = exif_data.get(36867)  # DateTimeOriginal
                return lat, lon, timestamp
    except Exception as e:
        print(f"Error al extraer metainformación: {e}")
    return None, None, None

# Funcion desarrollada ya que a veces las imagenes se guardaban con una orientacion erronea (orientada hacia los lados
# o hacia abajo) y nosotros queremos que esten bien orientadas para una mejor prediccion
def load_image_fix_orientation(path):
    img = Image.open(path)
    try:
        exif = img._getexif()
        if exif is not None:
            # obtener la clave para 'Orientation'
            orientation_key = next((k for k, v in ExifTags.TAGS.items() if v == 'Orientation'), None)
            if orientation_key is not None:
                orientation = exif.get(orientation_key)
                if orientation == 2:
                    img = img.transpose(Image.FLIP_LEFT_RIGHT)
                elif orientation == 3:
                    img = img.transpose(Image.ROTATE_180)
                elif orientation == 4:
                    img = img.transpose(Image.FLIP_TOP_BOTTOM)
                elif orientation == 5:
                    img = img.transpose(Image.TRANSPOSE)
                elif orientation == 6:
                    img = img.transpose(Image.ROTATE_270)
                elif orientation == 7:
                    img = img.transpose(Image.TRANSVERSE)
                elif orientation == 8:
                    img = img.transpose(Image.ROTATE_90)
    except Exception:
        # si falla EXIF, seguimos con la imagen original
        pass

    arr = np.array(img)
    # RGB -> BGR para cv2
    if arr.ndim == 3 and arr.shape[2] == 3:
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    elif arr.ndim == 3 and arr.shape[2] == 4:
        # conservar alpha si existe: RGBA -> BGRA
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGRA)
    else:
        bgr = arr  # ESCALA DE GRISES O FORMATO DESCONOCIDO, DEVOLVEMOS COMO ESTA
    return bgr