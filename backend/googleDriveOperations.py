import io

from fastapi import FastAPI, Response, Body, HTTPException, File, UploadFile
from googleapiclient.discovery import build
from google.oauth2 import service_account
from googleapiclient.http import MediaFileUpload, MediaIoBaseUpload
import os
from fastapi.middleware.cors import CORSMiddleware
from environs import Env

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # dominios que pueden acceder a la API, poner http://localhost:mi puerto
    allow_credentials=True, # si se permiten cookies
    allow_methods=["*"],# métodos HTTP permitidos
    allow_headers=["*"],# headers permitidos
)
env = Env()
env.read_env()

SERVICE_ACCOUNT_FILE = env('GOOGLE_SERVICE_ACCOUNT_FILE')
SCOPES = ['https://www.googleapis.com/auth/drive']
credentials = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
drive_service = build('drive', 'v3', credentials=credentials)

TARGET_FOLDER_ID = env('GOOGLE_DRIVE_FOLDER_ID')  # ID de la carpeta en drive donde se subiran los archivos


path = "/v1/drive" # http://0.0.0.0:8000/v1/drive

#guarda una imagen en drive
@app.post(path+"/upload")
async def upload_file_to_drive(file: UploadFile = File(...)):

    file_content = await file.read()

    media_body = MediaIoBaseUpload(
        io.BytesIO(file_content),
        mimetype=file.content_type,
        resumable=True
    )

    file_metadata = {
        'name': file.filename,
        'parents': [TARGET_FOLDER_ID] if TARGET_FOLDER_ID else []
    }

    try:
        uploaded_file = drive_service.files().create(
            body=file_metadata,
            media_body=media_body,
            fields='id, webContentLink, mimeType'
        ).execute()

        file_id = uploaded_file.get('id')
        link = uploaded_file.get('webContentLink')

        return {
            "message": "Archivo subido exitosamente",
            "file_name": file.filename,
            "file_id": file_id,
            "file_link": link
        }

    except Exception as e:
        print(f"Error al subir a Google Drive: {e}")
        raise HTTPException(status_code=500, detail="error al subir el archivo a Drive")