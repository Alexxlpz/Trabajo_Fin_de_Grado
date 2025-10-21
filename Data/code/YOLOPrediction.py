from ultralytics import YOLO
import os

# 1. ESPECIFICA LA RUTA A TU MEJOR MODELO
# Reemplaza 'Ruta/a/tu/proyecto' con la ruta real donde se guardó el modelo.
# El archivo se llama generalmente 'best.pt'.
MODELO_ENTRENADO = 'C:/Users/alexl/PycharmProjects/Trabajo_Fin_de_Grado/Data/code/TFG_deteccion_hojas/yolov12_hoja_sana_run1/weights/best.pt'

CARPETA_RAIZ_RESULTADOS = 'C:/Users/alexl/PycharmProjects/Trabajo_Fin_de_Grado/Data/dataset/Resultados_Inferencia'

# 2. ESPECIFICA LA RUTA DE LA IMAGEN A PROCESAR
IMAGEN_A_DETECTAR1 = 'C:/Users/alexl/PycharmProjects/Trabajo_Fin_de_Grado/Data/dataset/predictionImages/image (2).JPG'
IMAGEN_A_DETECTAR2 = 'C:/Users/alexl/PycharmProjects/Trabajo_Fin_de_Grado/Data/dataset/predictionImages/image (21).JPG'
IMAGEN_REAL = 'C:/Users/alexl/PycharmProjects/Trabajo_Fin_de_Grado/Data/dataset/predictionImages/imagenReal.JPG'

if __name__ == '__main__':
    # Cargar el modelo YOLO entrenado
    model = YOLO(MODELO_ENTRENADO)
    print(f"Iniciando predicción en {IMAGEN_REAL}")

    results = model.predict(
        source=IMAGEN_REAL,
        conf=0.25,
        save=True,
        device=0,
        project=CARPETA_RAIZ_RESULTADOS,
        name='inferencia_hojas_sanas',
        exist_ok=True
    )

    print("✅ Predicción completada. Revisando resultados...")
    result = results[0]

    if result.boxes:
        # Imprime el número de detecciones encontradas
        print(f"\nNúmero total de objetos detectados: {len(result.boxes)}")

        # Iterar sobre cada detección
        for i, box in enumerate(result.boxes):
            clase_id = int(box.cls)
            confianza = float(box.conf)
            coordenadas = box.xyxy[0].tolist()  # Coordenadas [x1, y1, x2, y2]

            # Obtener el nombre de la clase
            nombre_clase = model.names[clase_id]

            print(f"Detección {i + 1}:")
            print(f"  Clase: {nombre_clase}")
            print(f"  Confianza: {confianza:.2f}")
            print(f"  Coordenadas (xyxy): {coordenadas}")

    else:
        print("\nNo se detectaron objetos en la imagen con el umbral de confianza dado.")

    print(f"\nImagen con detecciones guardada en: {result.save_dir}")