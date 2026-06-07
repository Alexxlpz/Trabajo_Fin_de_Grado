from ultralytics import YOLO
import os

MODELO_ENTRENADO = '../../data/models/CNN/yolov12_Leaves_Detector_run1/weights/best.pt'

CARPETA_RAIZ_RESULTADOS = '../../data/inference_results'

IMAGEN_A_DETECTAR1 = '../../data/dataset/dummy_images/imagenReal2.jpeg'
IMAGEN_A_DETECTAR2 = '../../data/dataset/dummy_images/image (21).JPG'
IMAGEN_REAL = '../../data/dataset/dummy_images/imagenReal.JPG'


# Este script carga un modelo YOLO entrenado y realiza predicciones simples, sin teselar, sobre una imagen,
# mostrando los resultados en la consola y guardando la imagen con las detecciones.
if __name__ == '__main__':
    # cargar el modelo YOLO entrenado
    model = YOLO(MODELO_ENTRENADO)
    print(f"Iniciando predicción en {IMAGEN_REAL}")

    results = model.predict(
        source=IMAGEN_A_DETECTAR1,
        conf=0.25,
        save=True,
        device=0,
        project=CARPETA_RAIZ_RESULTADOS,
        name='prueba_A',
        exist_ok=True
    )

    print("Predicción completada. Revisando resultados...")
    result = results[0]

    if result.boxes:
        # imprime el número de detecciones encontradas
        print(f"\nNúmero total de objetos detectados: {len(result.boxes)}")

        # iterar sobre cada detección
        for i, box in enumerate(result.boxes):
            clase_id = int(box.cls)
            confianza = float(box.conf)
            coordenadas = box.xyxy[0].tolist()  # Coordenadas [x1, y1, x2, y2]

            # obtener el nombre de la clase
            nombre_clase = model.names[clase_id]

            print(f"Detección {i + 1}:")
            print(f"  Clase: {nombre_clase}")
            print(f"  Confianza: {confianza:.2f}")
            print(f"  Coordenadas (xyxy): {coordenadas}")

    else:
        print("\nNo se detectaron objetos en la imagen con el umbral de confianza dado.")

    print(f"\nImagen con detecciones guardada en: {result.save_dir}")