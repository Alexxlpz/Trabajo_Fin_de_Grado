@echo off
title Entorno de Desarrollo
echo Configurando entorno del proyecto...

set /p EXPO_USER="Usuario de Expo: "
set /p EXPO_PASS="Contrasena de Expo: "

echo.
echo Revisando base de datos PostgreSQL...
docker start leaves-db >nul 2>&1
if %errorlevel% neq 0 (
    docker run --name leaves-db -e POSTGRES_PASSWORD=postgres-database -p 5432:5432 -d postgres
)

echo Iniciando FastAPI (se abrira en una nueva ventana)...
start "Servidor FastAPI" cmd /k "call leavesDetection\.venv\Scripts\activate.bat && cd leavesDetection && python -m uvicorn src.backend.main:app --host 0.0.0.0 --port 8000 --reload || echo. && echo ERROR FATAL: El servidor FastAPI se ha detenido. Revisa el fallo arriba. && echo." 

echo Preparando entorno de Expo...
cd mobileApp

echo Iniciando sesion en Expo...
call npx expo login -u %EXPO_USER% -p %EXPO_PASS%

echo Iniciando servidor Expo con tunnel...
echo (Presiona Ctrl+C en esta ventana para detener Expo)
call npx expo start -c