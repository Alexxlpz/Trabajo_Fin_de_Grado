import os

CLASE_ID = 0
X_CENTRO = 0.5000
Y_CENTRO = 0.5000
ANCHO = 0.8500
ALTO = 0.8500

ETIQUETA_LINEA = f"{CLASE_ID} {X_CENTRO:.4f} {Y_CENTRO:.4f} {ANCHO:.4f} {ALTO:.4f}\n"

#IMAGENES_DIR = '../dataset/images/Pepper,_bell___healthy'
#LABELS_DIR = '../dataset/labels/Pepper,_bell___healthy'

IMAGENES_DIR = '../dataset/images/Pepper,_bell___Bacterial_spot'
LABELS_DIR = '../dataset/labels/Pepper,_bell___Bacterial_spot'

# makedirs es para hacer la carpeta si no existe
os.makedirs(LABELS_DIR, exist_ok=True)
archivos_generados = 0

#Iteraram0os sobre todas las imágenes
for filename in os.listdir(IMAGENES_DIR):
    if filename.endswith(('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')):
        base_name = os.path.splitext(filename)[0]
        label_filename = base_name + '.txt'
        label_filepath = os.path.join(LABELS_DIR, label_filename)

        # Escribimos la etiqueta fija en el archivo
        with open(label_filepath, 'w') as f:
            f.write(ETIQUETA_LINEA)
            archivos_generados += 1

#print(f"Se han generado {archivos_generados} archivos de etiquetas en {LABELS_DIR}")
