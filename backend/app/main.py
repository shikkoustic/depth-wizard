"""
DepthWizard - Main FastAPI Application.
Unified Web Application serving REST APIs and the Three.js 3D Geospatial Viewer.
"""

import os
import sys

# Ensure backend root is on sys.path
BASE_BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_BACKEND not in sys.path:
    sys.path.insert(0, BASE_BACKEND)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.api.endpoints import router as api_router

app = FastAPI(
    title="DepthWizard - Single-View Satellite Elevation & 3D Viewer",
    description="SIH26175 AI-powered DSM estimation and interactive 3D geospatial digital twin.",
    version="2.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API endpoints
app.include_router(api_router, prefix="/api")

# Static files directory setup
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_OUTPUTS_DIR = os.path.join(BACKEND_DIR, "..", "static")
FRONTEND_DIR = os.path.join(BACKEND_DIR, "..", "..", "frontend")

os.makedirs(os.path.join(STATIC_OUTPUTS_DIR, "outputs"), exist_ok=True)
os.makedirs(FRONTEND_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory=STATIC_OUTPUTS_DIR), name="static")
app.mount("/assets", StaticFiles(directory=os.path.join(FRONTEND_DIR, "assets") if os.path.exists(os.path.join(FRONTEND_DIR, "assets")) else FRONTEND_DIR), name="assets")


@app.get("/")
def serve_index():
    """Serves the main 3D visualization platform."""
    index_file = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"message": "DepthWizard API running. Frontend index.html not yet initialized."}


if __name__ == "__main__":
    import uvicorn
    print("Starting DepthWizard server on http://localhost:8000 ...")
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=False)
