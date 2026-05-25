from fastapi import FastAPI, Body, Depends
from fastapi.concurrency import run_in_threadpool
import base64

from sqlalchemy.orm import Session

from src.backend.Schemas.LoginSchema import LoginSchema
from src.backend.repository.database import init_db, get_db

# para el teselado y la clasificacion de yolo
import os

from src.backend.repository.user_repo import authenticate_and_get_recent_paths, register_authenticate_and_get_recent_paths
from src.utilities.predictionSlicing import predictionSlicing

class RegisterSchema(LoginSchema):
    username: str
app = FastAPI()

@app.on_event("startup")
def on_startup():
    init_db()

@app.post("/analyze")
async def analyzeImage(imageb64: str = Body(..., embed=True), user_id: int = Body(..., embed=True)):
    name = await run_in_threadpool(recibir, imageb64)
    # dentro guardaremos informacion para la base de datos.
    encodeImage, cont = await run_in_threadpool(predictionSlicing, name, user_id, False)

    return {"message": "OK", "leaf_count": cont, "image_base64": encodeImage}

@app.post("/login")
async def login(data: LoginSchema, db: Session = Depends(get_db)):

    message, recent_list, user = authenticate_and_get_recent_paths(db, data.email, data.password)
    return {"message": message, "recent_list": recent_list, "user": user} #  cambiar el nombre, messahe no es muy representativo

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