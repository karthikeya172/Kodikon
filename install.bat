@echo off
REM Kodikon Project Installation Script
REM This script sets up the Python environment with all required packages

echo.
echo ╔════════════════════════════════════════════════════════════╗
echo ║  Kodikon Project - Python Environment Setup               ║
echo ╚════════════════════════════════════════════════════════════╝
echo.

REM Create virtual environment
echo [1/4] Creating Python virtual environment...
python -m venv venv
if errorlevel 1 (
    echo ERROR: Failed to create virtual environment
    exit /b 1
)

REM Activate virtual environment
echo [2/4] Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment
    exit /b 1
)

REM Upgrade pip
echo [3/4] Upgrading pip, setuptools, and wheel...
python -m pip install --upgrade pip setuptools wheel -q
if errorlevel 1 (
    echo WARNING: Failed to upgrade pip (continuing anyway)
)

REM Install requirements
echo [4/4] Installing project dependencies...
echo This may take 5-10 minutes on first install...
echo.

REM Install core packages first (faster)
python -m pip install numpy pillow pyyaml python-dotenv -q
python -m pip install requests tqdm colorama -q

REM Install ML frameworks (larger downloads)
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu -q
python -m pip install ultralytics onnxruntime -q

REM Install API and async packages
python -m pip install fastapi uvicorn pydantic aiohttp websockets -q

REM Install data science packages
python -m pip install opencv-python scipy scikit-learn scikit-image -q

REM Install testing packages
python -m pip install pytest pytest-asyncio pytest-cov -q

REM Install dev tools
python -m pip install black flake8 mypy isort -q

echo.
echo ╔════════════════════════════════════════════════════════════╗
echo ║  ✓ Installation Complete!                                 ║
echo ╚════════════════════════════════════════════════════════════╝
echo.
echo Next steps:
echo   1. Backend:  python backend/server.py
echo   2. Frontend: cd frontend ^&^& npm install ^&^& npm run dev
echo.
pause
