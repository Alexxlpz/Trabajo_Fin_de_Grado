from rembg import remove
from PIL import Image
import os

#pathInput = '../../data/dataset/images/healthy_leaves/'
#pathOutput = '../../data/dataset/images_withoutBG/healthy_leaves'

pathInput = '../../data/dataset/images/diseased_leaves/'
pathOutput = '../../data/dataset/images_withoutBG/diseased_leaves'

os.makedirs(pathOutput, exist_ok=True)

for filename in os.listdir(pathInput):
    if filename.lower().endswith((".png", ".jpg", ".jpeg")):
        input_path = os.path.join(pathInput, filename)
        output_path = os.path.join(pathOutput, filename)

        try:
            image = Image.open(input_path).convert("RGBA")
            sin_fondo = remove(image)

            black = Image.new("RGBA", sin_fondo.size, (0, 0, 0, 255))
            final = Image.alpha_composite(black, sin_fondo)

            final_rgb = final.convert("RGB")

            extension = os.path.splitext(filename)[1].lower()
            if extension in ('.jpg', '.jpeg'):
                final_rgb.save(output_path, format='JPEG', quality=95)
            else:
                final_rgb.save(output_path)

            print(f"Guardado: {output_path}")
        except Exception as e:
            print(f"Error procesando {input_path}: {e}")