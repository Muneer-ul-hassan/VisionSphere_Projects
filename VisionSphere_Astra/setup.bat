@echo off
echo ========================================
echo   VisionSphere Astra - Setup Script
echo   Optimized for: i5-8350U, 16GB RAM
echo ========================================
echo.

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH
    echo Please install Python 3.9+ from https://python.org
    pause
    exit /b 1
)

:: Check Node.js
node --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Node.js is not installed or not in PATH
    echo Please install Node.js 16+ from https://nodejs.org
    pause
    exit /b 1
)

echo [1/6] Setting up Python backend...
cd backend

:: Create virtual environment
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

:: Activate and install dependencies
echo Installing Python dependencies...
call venv\Scripts\activate.bat

:: Upgrade pip first
python -m pip install --upgrade pip

:: Install requirements
pip install -r requirements.txt

cd ..

echo.
echo [2/6] Setting up React frontend...
cd frontend

:: Install Node dependencies
echo Installing Node dependencies...
call npm install

cd ..

echo.
echo [3/6] Checking Ollama...
ollama --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Ollama is not installed
    echo Please download from: https://ollama.ai
    pause
    exit /b 1
)

echo Ollama is installed!
echo.
echo [4/6] Checking for required model...
ollama list | findstr /i "qwen2.5:0.5b" >nul 2>&1
if errorlevel 1 (
    echo Downloading qwen2.5:0.5b model...
    echo This is a very fast, lightweight model for your CPU.
    echo.
    ollama pull qwen2.5:0.5b
) else (
    echo qwen2.5:0.5b model already installed!
)

echo.
echo [5/6] Creating data directories...
if not exist "backend\data" mkdir backend\data
if not exist "backend\data\memory_db" mkdir backend\data\memory_db

echo.
echo [6/6] Checking Tailscale (for iPhone access)...
tailscale --version >nul 2>&1
if errorlevel 1 (
    echo [INFO] Tailscale not found
    echo Install from: https://tailscale.com (free)
    echo This enables iPhone access from anywhere
) else (
    echo Tailscale is installed!
    echo Your Tailscale IP:
    tailscale ip
    echo.
    echo Use this IP to access Astra from iPhone Safari
)

echo.
echo ========================================
echo   Setup Complete!
echo ========================================
echo.
echo To start VisionSphere Astra:
echo.
echo   1. Make sure Ollama is running:
echo      ollama serve
echo.
echo   2. Start the backend:
echo      cd backend
echo      venv\Scripts\activate
echo      python -m app.main
echo.
echo   3. Start the frontend (new terminal):
echo      cd frontend
echo      npm start
echo.
echo   OR use the quick start script:
echo      .\start.bat
echo.
echo ========================================
echo   iPhone Access:
echo ========================================
echo.
echo   Method 1: Tailscale (works anywhere)
echo   - Install Tailscale on iPhone (App Store)
echo   - Sign in with same account
echo   - Open Safari: http://YOUR-TAILSCALE-IP:8000
echo   - Share -^ Add to Home Screen
echo.
echo   Method 2: Local network (home only)
echo   - Find your local IP: ipconfig
echo   - iPhone (same Wi-Fi): http://LOCAL-IP:8000
echo.
echo ========================================
echo.
pause
