from rembg import remove
from PIL import Image
import os

pathInput = '../Trabajo_Fin_de_Grado/data/dataset/images/Pepper,_bell___healthy/'
pathOutput = '../Trabajo_Fin_de_Grado/data/dataset/images_withoutBG/Pepper,_bell___healthy'

for image in os.listdir(pathInput):
    if image.endswith(".png") or image.endswith(".jpg") or image.endswith(".jpeg"):
        imagen = Image.open(pathInput+image)
        output = remove(imagen)
        output.save(pathOutput)