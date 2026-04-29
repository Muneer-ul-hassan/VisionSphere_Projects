@echo off
echo ========================================
echo   Starting VisionSphere Astra
echo ========================================
echo.

:: Start backend in a new window
echo Starting backend server...
start "Astra Backend" cmd /k "cd backend && venv\Scripts\activate && python -m app.main"

:: Wait a moment for backend to start
timeout /t 3 /nobreak >nul

:: Start frontend in a new window
echo Starting frontend...
start "Astra Frontend" cmd /k "cd frontend && npm start"

echo.
echo ========================================
echo   VisionSphere Astra is starting!
echo ========================================
echo.
echo - Backend: http://localhost:8000
echo - Frontend: http://localhost:3000
echo - API Docs: http://localhost:8000/docs
echo.
echo Check the terminal windows for status.
echo.
pause
