@echo off
echo ========================================================
echo DepthWizard - Standalone Deployment (SIH 2026 - SIH26175)
echo ========================================================
echo Checking for virtual environment...
if not exist ".venv" (
    echo Creating Python virtual environment...
    python -m venv .venv
)

echo Activating environment...
call .venv\Scripts\activate.bat

echo Installing requirements...
pip install numpy rasterio pillow fastapi uvicorn python-multipart torch transformers

echo Starting API Server and 3D Visualization Platform...
start http://127.0.0.1:8000
python -m depthwizard --data runs --weights C:\Users\gargm\Desktop\DepthWizard_Model_v2_Rollback.pt

pause
