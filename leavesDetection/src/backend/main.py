import base64
# para el teselado y la clasificacion de yolo
import os
import time

from fastapi import FastAPI, Body, Depends
from fastapi.concurrency import run_in_threadpool
from sqlalchemy.orm import Session

from src.backend.Schemas.LoginSchema import LoginSchema, RegisterSchema
from src.backend.repository.database import init_db, get_db
from src.backend.repository.user_repo import authenticate_and_get_recent_paths, \
    register_authenticate_and_get_recent_paths
from src.utilities import HitTimer as HitTimerModule
from src.utilities.predictionSlicing import predictionSlicing

app = FastAPI()

@app.on_event("startup")
def on_startup():
    init_db()

@app.post("/analyze")
async def analyzeImage(imageb64: str = Body(..., embed=True),
                       user_id: int = Body(..., embed=True)):

    name = await run_in_threadpool(recibir, imageb64)

    # dentro guardaremos información para la base de datos.
    encode_image, cont = await run_in_threadpool(predictionSlicing, name, user_id)
    return {
        "message": "OK",
        "leaf_count": cont,
        "image_base64": encode_image
    }

@app.post("/analyzeDebugg")
async def analyzeImageDebugg(imageb64: str = Body(..., embed=True),
                             user_id: int = Body(..., embed=True),
                             sent_time: float = Body(..., embed=True)):

    server_arrival_time = time.time() * 1000

    timer = HitTimerModule.HitTimer(name="analyze")

    elapsed = timer.hit()
    print(f'desde el start: "{elapsed*1000} milisegundos" desde el paso anterior')
    name = await run_in_threadpool(recibir, imageb64)

    elapsed = timer.hit()
    print(f'una vez que recibe pasan: "{elapsed*1000} milisegundos" desde el paso anterior')

    # dentro guardaremos información para la base de datos.
    encode_image, cont = await run_in_threadpool(predictionSlicing, name, user_id, True, timer)

    server_processing_time = time.time() * 1000
    if sent_time:
        tiempo_ida_servidor = server_arrival_time - sent_time
        print(f"tiempo de ida (sujeto a desfase de reloj del móvil): {tiempo_ida_servidor:.2f} ms")

    return {
        "message": "OK",
        "leaf_count": cont,
        "image_base64": encode_image,
        "server_arrival_time": server_arrival_time,
        "server_processing_time": server_processing_time
    }

@app.post("/login")
async def login(data: LoginSchema, db: Session = Depends(get_db)):

    message, recent_list, user = authenticate_and_get_recent_paths(db, data.email, data.password)
    return {"message": message, "recent_list": recent_list, "user": user}

@app.post("/register")
async def register(data: RegisterSchema, db: Session = Depends(get_db)):
    message, recent_list, user = register_authenticate_and_get_recent_paths(db, data.email, data.password, data.username)
    return {"message": message, "recent_list": recent_list, "user": user}

def recibir(imageb64: str):
    image_data = base64.b64decode(imageb64)
    name = "imagen_decodificada"

    # Guardar como archivo
    os.makedirs(r"data/cache", exist_ok=True)
    with open(f"data/cache/{name}", "wb") as f:
        f.write(image_data)
        print("imagen_decodificada.png guardada correctamente")

    return name