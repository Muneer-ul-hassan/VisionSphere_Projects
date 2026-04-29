@echo off
echo Cleaning up old processes...
taskkill /F /IM python.exe /T 2>NUL
taskkill /F /IM node.exe /T 2>NUL

echo Starting VisionSphere Astra Backend...
start "Astra Backend - DO NOT CLOSE" cmd /k "cd backend && call venv\Scripts\activate.bat && python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --log-level info"

echo Starting VisionSphere Astra Frontend...
start "Astra Frontend - DO NOT CLOSE" cmd /k "cd frontend && npm start"

echo Both services are starting up in separate windows. 
echo Please wait ~60 seconds for the models to load and the browser to launch automatically!
exit
