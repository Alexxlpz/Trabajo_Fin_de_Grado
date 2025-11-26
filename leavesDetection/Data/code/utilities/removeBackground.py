from rembg import remove
from PIL import Image
import os

script_dir = os.path.dirname(os.path.abspath(__file__))

pathInput = os.path.abspath(os.path.join(script_dir, '..', '..', '..', 'Data', 'dataset', 'images', 'Pepper,_bell___healthy'))
pathOutput = os.path.abspath(os.path.join(script_dir, '..', '..', '..', 'Data', 'dataset', 'images_withoutBG', 'Pepper,_bell___healthy'))
pathOutputPNG = os.path.abspath(os.path.join(script_dir, '..', '..', '..', 'Data', 'dataset', 'imagesPNG', 'Pepper,_bell___healthy__PNG'))

pathInput2 = os.path.abspath(os.path.join(script_dir, '..', '..', '..', 'Data', 'dataset', 'images', 'Pepper,_bell___Bacterial_spot'))
pathOutput2 = os.path.abspath(os.path.join(script_dir, '..', '..', '..', 'Data', 'dataset', 'images_withoutBG', 'Pepper,_bell___Bacterial_spot'))
pathOutputPNG2 = os.path.abspath(os.path.join(script_dir, '..', '..', '..', 'Data', 'dataset', 'imagesPNG', 'Pepper,_bell___Bacterial_spot__PNG'))

os.makedirs(pathOutput, exist_ok=True)
os.makedirs(pathOutput2, exist_ok=True)
os.makedirs(pathOutputPNG, exist_ok=True)
os.makedirs(pathOutputPNG2, exist_ok=True)

#PASAMOS TODAS LAS IMAGENES A PNG Y LO ALOJAMOS EN OTRA CARPETA
for image_name in os.listdir(pathInput):
    if image_name.lower().endswith(('.jpg', '.jpeg')): #convertir a png primero para que haya transparencia en en fondo de las imagenes

        input_path = os.path.join(pathInput, image_name)

        base_name = os.path.splitext(image_name)[0]  # Obtener el nombre sin extensión
        output_filename = base_name + '.png'
        output_path = os.path.join(pathOutputPNG, output_filename)

        try:
            img = Image.open(input_path)
            img = img.convert('RGBA')
            # Guardar como PNG
            img.save(output_path)
            print(f"Convertido: '{image_name}' a '{output_filename}'")

        except Exception as e:
            print(f"Error al procesar '{image_name}': {e}")

for image_name in os.listdir(pathInput2):
    if image_name.lower().endswith(('.jpg', '.jpeg')): #convertir a png primero para que haya transparencia en en fondo de las imagenes

        input_path = os.path.join(pathInput2, image_name)

        base_name = os.path.splitext(image_name)[0]  # Obtener el nombre sin extensión
        output_filename = base_name + '.png'
        output_path = os.path.join(pathOutputPNG2, output_filename)

        try:
            img = Image.open(input_path)
            img = img.convert('RGBA')
            # Guardar como PNG
            img.save(output_path)
            print(f"Convertido: '{image_name}' a '{output_filename}'")

        except Exception as e:
            print(f"Error al procesar '{image_name}': {e}")


for image_name in os.listdir(pathOutputPNG):
    if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
        input_path = os.path.join(pathOutputPNG, image_name)
        output_img = remove(Image.open(input_path))
        output_path = os.path.join(pathOutput, image_name)

        base, ext = os.path.splitext(image_name)
        ext = ext.lower()

        if output_img.mode == 'RGBA':
            output_img.save(os.path.join(pathOutput, base + '.png'))
        else:
            output_img.save(os.path.join(pathOutput, image_name))


        print(f"Processed {image_name}")
print("Background removal from healthy completed.")

for image_name in os.listdir(pathOutputPNG2):
    if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
        input_path = os.path.join(pathOutputPNG2, image_name)
        output_img = remove(Image.open(input_path))
        output_path = os.path.join(pathOutput2, image_name)

        base, ext = os.path.splitext(image_name)
        ext = ext.lower()

        if output_img.mode == 'RGBA':
            output_img.save(os.path.join(pathOutput2, base + '.png'))
        else:
            output_img.save(os.path.join(pathOutput2, image_name))


        print(f"Processed {image_name}")
print("Background removal from healthy completed.")