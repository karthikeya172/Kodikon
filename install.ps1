# Kodikon Project Installation Script
# This script sets up the Python environment with all required packages

Write-Host ""
Write-Host "╔════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║  Kodikon Project - Python Environment Setup                ║" -ForegroundColor Cyan
Write-Host "╚════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

# Create virtual environment
Write-Host "[1/4] Creating Python virtual environment..." -ForegroundColor Yellow
python -m venv venv
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to create virtual environment" -ForegroundColor Red
    exit 1
}

# Activate virtual environment
Write-Host "[2/4] Activating virtual environment..." -ForegroundColor Yellow
& .\venv\Scripts\Activate.ps1
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to activate virtual environment" -ForegroundColor Red
    exit 1
}

# Upgrade pip
Write-Host "[3/4] Upgrading pip, setuptools, and wheel..." -ForegroundColor Yellow
python -m pip install --upgrade pip setuptools wheel -q
if ($LASTEXITCODE -ne 0) {
    Write-Host "WARNING: Failed to upgrade pip (continuing anyway)" -ForegroundColor Yellow
}

# Install requirements
Write-Host "[4/4] Installing project dependencies..." -ForegroundColor Yellow
Write-Host "This may take 5-10 minutes on first install..." -ForegroundColor Gray
Write-Host ""

# Install core packages first (faster)
Write-Host "  • Installing core packages..." -ForegroundColor Cyan
python -m pip install numpy pillow pyyaml python-dotenv -q
python -m pip install requests tqdm colorama -q

# Install ML frameworks (larger downloads)
Write-Host "  • Installing PyTorch and ML frameworks..." -ForegroundColor Cyan
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu -q
python -m pip install ultralytics onnxruntime -q

# Install API and async packages
Write-Host "  • Installing backend API packages..." -ForegroundColor Cyan
python -m pip install fastapi uvicorn pydantic aiohttp websockets -q

# Install data science packages
Write-Host "  • Installing data science packages..." -ForegroundColor Cyan
python -m pip install opencv-python scipy scikit-learn scikit-image -q

# Install testing packages
Write-Host "  • Installing testing packages..." -ForegroundColor Cyan
python -m pip install pytest pytest-asyncio pytest-cov -q

# Install dev tools
Write-Host "  • Installing development tools..." -ForegroundColor Cyan
python -m pip install black flake8 mypy isort -q

Write-Host ""
Write-Host "╔════════════════════════════════════════════════════════════╗" -ForegroundColor Green
Write-Host "║  ✓ Installation Complete!                                 ║" -ForegroundColor Green
Write-Host "╚════════════════════════════════════════════════════════════╝" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  1. Backend:  python backend/server.py"
Write-Host "  2. Frontend: cd frontend && npm install && npm run dev"
Write-Host ""
