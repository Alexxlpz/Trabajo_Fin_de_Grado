import os

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_resnet


SAVE_MODEL_PATH = '../../data/models/'
MODEL_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'models', 'discriminador_hojas_ResNet50v2.h5')
)

_MODEL = None  # cache del modelo, para evitar cargarlo múltiples veces si classifyObject se llama varias veces
                # en la misma ejecución

def get_model():
    """
    Carga y cachea el modelo. Lanza FileNotFoundError si no existe el archivo.
    """
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"modelo no encontrado en `{MODEL_PATH}`. Coloca el .h5 en esa ruta o pasa `model_path` "
                                f"a `classifyObject`.")

    try:
        _MODEL = tf.keras.models.load_model(MODEL_PATH)
    except Exception as e:
        raise RuntimeError(f"error cargando el modelo desde `{MODEL_PATH}`: {e}")
    return _MODEL

# -----------------------------------------------------------------------------------------------------------------------

def classifyObject(img_array) -> bool:
    model = get_model()
    img_lista = cargar_y_preparar_imagen_cv2(img_array)
    prediccion = model.predict(img_lista)
    resultado_numerico = prediccion[0][0]

    print(f"\n--- RESULTADO ---")
    print(f"valor numérico crudo: {resultado_numerico:.4f}")

    # Usamos 0.5 como punto de corte.
    if resultado_numerico < 0.5:
        confianza = (1 - resultado_numerico) * 100
        print(f"Diagnóstico: 🍂 ENFERMA")
        print(f"Seguridad: {confianza:.2f}%")
        return False
    else:
        confianza = resultado_numerico * 100
        print(f"Diagnóstico: 🌿 SANA")
        print(f"Seguridad: {confianza:.2f}%")
        return True


def cargar_y_preparar_imagen_cv2(img_array, target_size=(224, 224), from_bgr=True):
    if img_array is None or not isinstance(img_array, np.ndarray):
        raise ValueError("se esperaba un ndarray válido en `img_array`")

    if img_array.ndim == 2: # 2 es que esta en escala de grises, asi que lo converrtimos a 3 canales (BGR)
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)

    if img_array.ndim == 3 and img_array.shape[2] == 4: # si tiene canal alpha, convertir RGBA -> RGB (descartar alpha)
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGRA2BGR)


    if not from_bgr: # si la fuente es RGB y necesitamos BGR para el preprocess 'caffe' convertir RGB -> BGR
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    # redimensionamos a target_size (224x224) usando interpolación lineal
    resized = cv2.resize(img_array, (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR)

    # convertir a float32 y expandir batch
    arr = np.expand_dims(resized.astype(np.float32), axis=0)

    # aplicar preprocess específico de ResNet50 (modo 'caffe' espera BGR)
    preprocessed = preprocess_resnet(arr)

    return preprocessed

if __name__ == '__main__':
    # prueba de carga del modelo
    try:
        model = get_model()
        print("Modelo cargado exitosamente.")
    except Exception as e:
        print(f"Error al cargar el modelo: {e}")