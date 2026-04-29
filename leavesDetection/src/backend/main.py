from fastapi import FastAPI, Body
from fastapi.concurrency import run_in_threadpool
import base64
from src.backend.repository.database import init_db

# para el teselado y la clasificacion de yolo
import os
from src.utilities.predictionSlicing import predictionSlicing


app = FastAPI()

@app.on_event("startup")
def on_startup():
    init_db()

@app.post("/analyze")
async def analyzeImage(imageb64: str = Body(..., embed=True)):
    name = await run_in_threadpool(recibir, imageb64)
    # dentro guardaremos informacion para la base de datos.
    encodeImage, cont = await run_in_threadpool(predictionSlicing, name, True)

    return {"message": "OK", "number": cont, "imageb64": encodeImage}

def recibir(imageb64: str):
    image_data = base64.b64decode(imageb64)
    name = "imagen_decodificada"

    # Guardar como archivo
    os.makedirs(r"data/cache", exist_ok=True)
    with open(f"data/cache/{name}", "wb") as f:
        f.write(image_data)
        print("imagen_decodificada.png guardada correctamente")

    return name