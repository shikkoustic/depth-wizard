"""
DepthWizard - One-Command Launcher.
Starts the FastAPI backend and serves the Three.js 3D Geospatial Viewer.
Automatically launches the browser at http://localhost:8000
"""

import os
import sys
import webbrowser
import threading
import time

# Add backend to sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.join(BASE_DIR, "backend")
sys.path.insert(0, BACKEND_DIR)

from app.processing.generate_samples import generate_benchmark_samples


def open_browser():
    time.sleep(1.5)
    url = "http://localhost:8000"
    print(f"\n[DepthWizard] Opening browser at {url} ...")
    webbrowser.open(url)


def main():
    print("=" * 70)
    print("  SIH26175 DepthWizard - Monocular Satellite Elevation & 3D Viewer")
    print("  Unified Software Suite for ISRO Evaluation")
    print("=" * 70)

    # 1. Ensure sample benchmark data is present
    sample_dir = os.path.join(BASE_DIR, "sample_data")
    if not os.path.exists(os.path.join(sample_dir, "sample_urban_georef.tif")):
        print("[Launcher] Generating bundled benchmark test tiles...")
        generate_benchmark_samples(sample_dir)

    # 2. Launch browser in background thread
    threading.Thread(target=open_browser, daemon=True).start()

    # 3. Start Uvicorn Server
    import uvicorn
    print("\n[Launcher] Starting backend server on http://localhost:8000 (Press Ctrl+C to stop)...")
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=False, app_dir=BACKEND_DIR)


if __name__ == "__main__":
    main()
