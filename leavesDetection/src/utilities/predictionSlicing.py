import base64
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ExifTags
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
import os
import time

from src.backend.repository.classes.Crop import Crop
from src.backend.repository.image_repo import create_image_with_direction
from src.utilities import HitTimer
from src.utilities.classifyObject import classifyObject
from src.utilities.makePlot import make_plot
from src.utilities.metainfoCollect import metainfo_collect

# ruta del YOLO entrenado
YOLO_MODEL_PATH = '../../data/models/YOLO_run6.pt'

# donde guardamos la imagen con la prediccion hecha
OUTPUT_IMAGE_PATH = '../../data/inference_results'

CACHE_PATH = '../../data/cache'

DEBUG_PATH = '../../data/inference_results/debug'

_MODEL = None # lo guardamos de manera global para no tener que cargarlo cada vez que se llama a la función de
# prediccion, lo cargamos solo la primera vez y luego se reutiliza. Asi es bastante mas rápida la predicción

def get_model():
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    if not os.path.exists(YOLO_MODEL_PATH):
        raise FileNotFoundError(f"modelo no encontrado en `{YOLO_MODEL_PATH}`")

    try:
        _MODEL = AutoDetectionModel.from_pretrained(
            model_type='ultralytics',  # tipo de modelo: YOLO
            model_path=YOLO_MODEL_PATH,
            confidence_threshold=0.3,  # umbral de confianza al principio
            device="cuda:0"  # para usar la grafica
        )
    except Exception as e:
        raise RuntimeError(f"error cargando el modelo desde `{YOLO_MODEL_PATH}`: {e}")
    return _MODEL

# ---------------------------------------------------------------------------------------------------------------------

def look_clock(timer: Optional[HitTimer] = None, label: str = ""):
    if timer is not None:
        elapsed = timer.hit()
        print(f'{label}: "{elapsed*1000} milisegundos" desde el paso anterior')

def predictionSlicing(image_name: str, session: int, debug_mode: bool = False, hit_timer:  Optional[HitTimer] = None):
    # ruta de la imagen a predicie
    image_source = os.path.join(CACHE_PATH, image_name)
    threshold = 0.85

    detection_model = get_model()
    print("modelo YOLO cargado exitosamente.")

    filtered_pred_list = prediction(image_source, detection_model, threshold, debug_mode, hit_timer)

    encode_image, annotated_path, healthy, sick, crop_list = cropAndClasify(filtered_pred_list, image_source, CACHE_PATH, debug_mode, hit_timer)

    look_clock(hit_timer, "antes_de_guardar_resultados")
    if session != -1:
        save_database_entry(
            image_source= annotated_path,
            session=session,
            healthy = healthy,
            sick = sick,
            crop_list = crop_list
        )
    look_clock(hit_timer, "despues_de_guardar_resultados")

    #os.makedirs(OUTPUT_IMAGE_PATH, exist_ok=True)
    #results.export_visuals(export_dir=OUTPUT_IMAGE_PATH, file_name=teseledImagenName)

    #with open(OUTPUT_IMAGE_PATH+'/'+teseledImagenName+'.png', "rb") as image:
        #encodeImage = base64.b64encode(image.read()).decode('utf-8')

    print("Proceso finalizado!!!")
    os.remove(image_source)

    return encode_image, len(filtered_pred_list)

def save_database_entry(image_source, session:int, healthy=0, sick=0, crop_list=None):
    if crop_list is None:
        crop_list = []

    lat, lon, timestamp = metainfo_collect(image_source)

    print(f"Metainformación procesada - Lat: {lat}, Lon: {lon}, Fecha: {timestamp}")

    img = create_image_with_direction(
        path=image_source,
        latitude=lat,
        longitude=lon,
        user_id=session,
        num_sick=int(sick),
        num_healthy=int(healthy),
        upload_date=timestamp,  # Ahora el formato es compatible con SQL
        model_path=YOLO_MODEL_PATH,
        crop_list=crop_list
    )

    print(img)



def prediction(image_source, detection_model, threshold: float = 0.85, debug_mode: bool = False, hit_timer: Optional[HitTimer] = None):

    # Ponemos que los teselados sean 640x640 ya que a yolo lo entrenemos asi y se le hara mas facil predecir de esa manera

    look_clock(hit_timer, "antes_de_localizar_hojas")
    result = get_sliced_prediction(
        image_source,
        detection_model,
        slice_height=640,  # Altura de cada teselado
        slice_width=640,  # Anchura de cada teselado
        overlap_height_ratio=0.2,  # Para el solapamiento vertical
        overlap_width_ratio=0.2  # Para el solapamiento horizontal
    )
    look_clock(hit_timer, "despues_de_localizar_hojas")

    object_prediction_list = result.object_prediction_list

    if debug_mode:
        tiled_basename = f"YOLO_PREDICTION_{os.path.splitext(os.path.basename(image_source))[0]}"
        try:
            result.export_visuals(export_dir=DEBUG_PATH, file_name=tiled_basename)
            print(f"visuales exportados en `{DEBUG_PATH}` con prefijo `{tiled_basename}`")
        except Exception as e:
            print(f"error exportando visuales: {e}")

        confident_list = [pred.score.value for pred in result.object_prediction_list]
        make_plot(confident_list, "confianza_predicciones_teselado_antes_del_filtrado_testing.png")

    print(f"total objetos detectados: {len(object_prediction_list)}")

    look_clock(hit_timer, "antes_de_filtrar")

    #filtrado
    valid_objects = [pred for pred in object_prediction_list if pred.score.value >= threshold]
    valid_objects = sorted(valid_objects, key=lambda x: x.score.value, reverse=True)
    best_items = valid_objects[:10]
    result.object_prediction_list = best_items

    look_clock(hit_timer, "despues_de_filtrar")

    if debug_mode:
        #Guardamos lo que ha visto el modelo filtrado
        tiled_basename = f"YOLO_PREDICTION_{os.path.splitext(os.path.basename(image_source))[0]}_FILTERED"
        try:
            result.export_visuals(export_dir=DEBUG_PATH, file_name=tiled_basename)
            print(f"visuales exportados en `{DEBUG_PATH}` con prefijo `{tiled_basename}`")
        except Exception as e:
            print(f"error exportando visuales: {e}")

        # hacemos un  grafico para ver la confianza de las predicciones tras filtrar
        confidentList = [pred.score.value for pred in result.object_prediction_list]
        make_plot(confidentList, "confianza_predicciones_teselado_despues_del_filtrado_testing.png")


    return best_items


def cropAndClasify(predictions, image, cache_folder, debug_mode: bool = False, hit_timer: Optional[HitTimer] = None):
    output_folder = os.path.join(cache_folder, "results")
    crops_output_folder = os.path.join(cache_folder, "crops")

    os.makedirs(output_folder, exist_ok=True)

    img = load_image_fix_orientation(image)
    if img is None:
        raise FileNotFoundError(f"Imagen no encontrada: {image}")

    # Guardar la imagen anotada y devolverla en Base64
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    annotated_filename = f"annotated_{timestamp}.jpg"
    annotated_path = os.path.join(output_folder, annotated_filename)

    saved = 0
    healthy = 0
    sick = 0
    crop_list = []


    look_clock(hit_timer, "antes_de_crop_and_classify")
    for i, object_prediction in enumerate(predictions):

        crop, coords = cropObject(img, object_prediction)

        if classifyObject(crop):
            #pintamos cuadro verde
            cv2.rectangle(img, (coords[0], coords[1]), (coords[2], coords[3]), (0, 255, 0), 2)
            healthy += 1
            status = "healthy"
        else:
            #pintamos cuadro rojo
            cv2.rectangle(img, (coords[0], coords[1]), (coords[2], coords[3]), (0, 0, 255), 2)
            sick += 1
            status = "diseased"

        filename = f"{annotated_filename}_crop_{i}.jpg"
        out_path = os.path.join(crops_output_folder, filename)
        cv2.imwrite(out_path, crop, [cv2.IMWRITE_JPEG_QUALITY, int(100)])
        crop_list.append(Crop(status=status, path=out_path))

        if debug_mode:
                # guardamos el crop para debuggear el clasificador
                filename = f"crop_{i}.jpg"
                out_path = os.path.join(DEBUG_PATH, filename)
                cv2.imwrite(out_path, crop, [cv2.IMWRITE_JPEG_QUALITY, int(100)])

        saved += 1

    print("se han guardado "+str(saved)+" crops")
    look_clock(hit_timer, f"termina de clasificar las {saved} hojas")

    ok = cv2.imwrite(annotated_path, img, [cv2.IMWRITE_JPEG_QUALITY, int(90)])

    if not ok:
        raise RuntimeError(f"no se pudo guardar la imagen anotada en {annotated_path}")

    if debug_mode:
        debug_annotated_path = os.path.join(DEBUG_PATH, annotated_filename)
        cv2.imwrite(debug_annotated_path, img, [cv2.IMWRITE_JPEG_QUALITY, int(90)])


    if not ok:
        raise RuntimeError(f"no se pudo guardar la imagen anotada en {annotated_path}")

    with open(annotated_path, "rb") as f:
        annotated_b64 = base64.b64encode(f.read()).decode('utf-8')

    return annotated_b64, annotated_path, healthy, sick, crop_list



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