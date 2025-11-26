from rembg import remove
from PIL import Image
import os

pathInput = '../Trabajo_Fin_de_Grado/Data/dataset/images/Pepper,_bell___healthy/'
pathOutput = '../Trabajo_Fin_de_Grado/Data/code/ImageCreation/ImagesWithoutBackground'

for image in os.listdir(pathInput):
    if image.endswith(".png") or image.endswith(".jpg") or image.endswith(".jpeg"):
        imagen = Image.open(pathInput+image)
        output = remove(imagen)
        output.save(pathOutput)