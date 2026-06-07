import os
import cv2

# Función para dibujar recuadros a partir de etiquetas YOLO, con búsqueda recursiva en subcarpetas
# se utilizo para ver como de bien estaban las etiquetas generadas por sam3, y para revisar el dataset antes de entrenar
# el modelo
def dibujar_recuadros_yolo_recursivo(ruta_imagenes, ruta_etiquetas, ruta_destino):
    extensiones = (".jpg", ".jpeg", ".png", ".bmp")
    contador_imagenes = 0

    print("Iniciando búsqueda recursiva de imágenes...")

    for dirpath, _, filenames in os.walk(ruta_imagenes):
        ruta_relativa = str(os.path.relpath(dirpath, ruta_imagenes))

        for nombre_img in filenames:
            if not nombre_img.lower().endswith(extensiones):
                continue

            camino_img = os.path.join(dirpath, nombre_img)

            nombre_txt = os.path.splitext(nombre_img)[0] + ".txt"

            if ruta_relativa == ".":
                camino_txt = os.path.join(ruta_etiquetas, nombre_txt)
                carpeta_salida_final = ruta_destino
            else:
                camino_txt = os.path.join(ruta_etiquetas, ruta_relativa, nombre_txt)
                carpeta_salida_final = os.path.join(ruta_destino, ruta_relativa)

            if not os.path.exists(camino_txt):
                ruta_alerta = (
                    nombre_img
                    if ruta_relativa == "."
                    else os.path.join(ruta_relativa, nombre_img)
                )
                print(
                    f"Advertencia: No se encontró etiqueta para {ruta_alerta}"
                )
                continue

            img = cv2.imread(str(camino_img))
            if img is None:
                print(f"Error al leer la imagen {nombre_img}")
                continue

            alto, ancho, _ = img.shape

            with open(camino_txt, "r") as f:
                lineas = f.readlines()

            for linea in lineas:
                partes = linea.strip().split()
                if len(partes) < 5:
                    continue

                id_clase = partes[0]
                x_centro = float(partes[1])
                y_centro = float(partes[2])
                w_bbox = float(partes[3])
                h_bbox = float(partes[4])

                x1 = int((x_centro - w_bbox / 2) * ancho)
                y1 = int((y_centro - h_bbox / 2) * alto)
                x2 = int((x_centro + w_bbox / 2) * ancho)
                y2 = int((y_centro + h_bbox / 2) * alto)

                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    img,
                    f"Clase {id_clase}",
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                )

            if not os.path.exists(carpeta_salida_final):
                os.makedirs(carpeta_salida_final)

            camino_salida = os.path.join(carpeta_salida_final, nombre_img)
            cv2.imwrite(str(camino_salida), img)
            contador_imagenes += 1

    print(f"\n¡Proceso completado con éxito!")
    print(f"Se procesaron {contador_imagenes} imágenes y se guardaron en: {ruta_destino}")

if __name__ == "__main__":
    CARPETA_IMAGENES = '../../data/dataset/images/'
    CARPETA_ETIQUETAS = '../../data/dataset/labels/'
    CARPETA_RESULTADO = '../../data/dataset/yoloTrainingImages/'

    # Ejecutar la función
    dibujar_recuadros_yolo_recursivo(
        CARPETA_IMAGENES, CARPETA_ETIQUETAS, CARPETA_RESULTADO
    )