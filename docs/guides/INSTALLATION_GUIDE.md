# 📦 Kodikon Installation Commands

## Quick Start - Complete Installation Command

Copy and paste this entire command into PowerShell (after activating venv):

```powershell
pip install numpy pillow pyyaml python-dotenv requests tqdm colorama; pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu; pip install ultralytics onnxruntime; pip install fastapi uvicorn pydantic aiohttp websockets; pip install opencv-python scipy scikit-learn scikit-image; pip install pytest pytest-asyncio pytest-cov; pip install black flake8 mypy isort
```

## Step-by-Step Installation Guide

### Step 1: Create Virtual Environment
```powershell
python -m venv venv
```

### Step 2: Activate Virtual Environment
```powershell
# For PowerShell:
.\venv\Scripts\Activate.ps1

# For Command Prompt (CMD):
.\venv\Scripts\activate.bat
```

### Step 3: Upgrade pip (Optional but Recommended)
```powershell
python -m pip install --upgrade pip setuptools wheel
```

### Step 4: Install Packages in Batches

#### Batch 1: Core Data Science Packages (Fast)
```powershell
pip install numpy pillow pyyaml python-dotenv requests tqdm colorama
```

#### Batch 2: PyTorch (Large - ~500MB)
```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

#### Batch 3: YOLO and ONNX (Medium)
```powershell
pip install ultralytics onnxruntime
```

#### Batch 4: Backend API Packages
```powershell
pip install fastapi uvicorn pydantic aiohttp websockets
```

#### Batch 5: Computer Vision Packages
```powershell
pip install opencv-python scipy scikit-learn scikit-image
```

#### Batch 6: Testing Packages
```powershell
pip install pytest pytest-asyncio pytest-cov
```

#### Batch 7: Development Tools
```powershell
pip install black flake8 mypy isort
```

---

## Automated Installation Scripts

### Option A: PowerShell Script (Recommended)
```powershell
.\install.ps1
```

### Option B: Batch Script (CMD)
```cmd
install.bat
```

---

## Verify Installation

Check if all packages installed correctly:

```powershell
# List installed packages
pip list

# Test PyTorch
python -c "import torch; print('PyTorch version:', torch.__version__)"

# Test OpenCV
python -c "import cv2; print('OpenCV version:', cv2.__version__)"

# Test YOLO
python -c "from ultralytics import YOLO; print('YOLO ready')"

# Test FastAPI
python -c "import fastapi; print('FastAPI version:', fastapi.__version__)"
```

---

## Troubleshooting

### If pip install fails:
1. Make sure you're in the virtual environment (prompt should show `(venv)`)
2. Update pip: `python -m pip install --upgrade pip`
3. Try installing one package at a time instead of all together
4. Check your internet connection (some packages are large)

### If torch won't install:
Use the CPU version (faster for development):
```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### If you get permission errors:
```powershell
# Run PowerShell as Administrator
```

---

## Next Steps After Installation

### 1. Install Frontend Dependencies
```powershell
cd frontend
npm install
```

### 2. Start Backend Server
```powershell
python backend/server.py
```

### 3. Start Frontend Development Server (in new terminal)
```powershell
cd frontend
npm run dev
```

### 4. Download YOLO Models
```powershell
python scripts/download_models.py
```

---

## Package Summary

| Category | Packages |
|----------|----------|
| **Core Data** | numpy, pillow, scipy, scikit-image, scikit-learn |
| **Computer Vision** | opencv-python |
| **ML/DL** | torch, torchvision, ultralytics, onnxruntime |
| **Backend API** | fastapi, uvicorn, pydantic |
| **Async/Network** | aiohttp, websockets |
| **Utilities** | pyyaml, python-dotenv, requests, tqdm, colorama |
| **Testing** | pytest, pytest-asyncio, pytest-cov |
| **Development** | black, flake8, mypy, isort |

---

**Total Installation Size:** ~2.5 GB (mostly PyTorch)
**Installation Time:** 5-15 minutes depending on internet speed
