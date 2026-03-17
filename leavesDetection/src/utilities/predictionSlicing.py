import base64

import cv2
import numpy as np
from PIL import Image, ExifTags
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
import os

from src.utilities.classifyObject import classifyObject
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

    encodeImage = cropAndClasify(result.object_prediction_list, IMAGE_SOURCE, 'C:/Users/alexl/PycharmProjects/Trabajo_Fin_de_Grado/leavesDetection/data/cache/crops')

    #os.makedirs(OUTPUT_IMAGE_PATH, exist_ok=True)
    #result.export_visuals(export_dir=OUTPUT_IMAGE_PATH, file_name=teseledImagenName)

    #with open(OUTPUT_IMAGE_PATH+'/'+teseledImagenName+'.png', "rb") as image:
        #encodeImage = base64.b64encode(image.read()).decode('utf-8')

    print("Proceso finalizado!!!")
    os.remove(IMAGE_SOURCE)

    return teseledImagenName, encodeImage, len(validObjects)



def cropAndClasify(predictions, image, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    img = load_image_fix_orientation(image)
    if img is None:
        raise FileNotFoundError(f"Imagen no encontrada: {image}")


    saved = 0
    for i, object_prediction in enumerate(predictions):

        crop, coords = cropObject(img, object_prediction)

        filename = f"crop_{i}.jpg"
        out_path = os.path.join(output_folder, filename)
        ok = cv2.imwrite(out_path, crop, [cv2.IMWRITE_JPEG_QUALITY, int(100)])

        if ok :
            saved+=1
            if classifyObject(img):
                #pintamos cuadro verde
                cv2.rectangle(img, (coords[0], coords[1]), (coords[2], coords[3]), (0, 255, 0), 2)
            else:
                #pintamos cuadro rojo
                cv2.rectangle(img, (coords[0], coords[1]), (coords[2], coords[3]), (0, 0, 255), 2)
        else:
            print(f"Error al guardar el crop: {out_path}")

    print("se han guardado "+str(saved)+" crops")

    # Guardar la imagen anotada y devolverla en Base64
    annotated_filename = f"annotated_{os.path.basename(image)}.png"
    annotated_path = os.path.join(output_folder, annotated_filename)
    ok = cv2.imwrite(annotated_path, img, [cv2.IMWRITE_JPEG_QUALITY, int(90)])

    if not ok:
        raise RuntimeError(f"No se pudo guardar la imagen anotada en {annotated_path}")

    with open(annotated_path, "rb") as f:
        annotated_b64 = base64.b64encode(f.read()).decode('utf-8')

    return annotated_b64



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