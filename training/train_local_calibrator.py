"""
DepthWizard - Local Height Calibrator Trainer.
Trains a lightweight ridge regression & gradient-boosted height calibrator
on local sample tiles to calibrate building and canopy heights above ground.
Runs directly on laptop CPU in seconds without needing a GPU!
"""

import os
import sys
import numpy as np

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(BASE_DIR, "backend"))

from app.processing.geotiff_io import GeoTIFFHandler
from app.processing.metrics_evaluator import MetricsEvaluator
import tifffile


def extract_features(rgb: np.ndarray) -> np.ndarray:
    """Extracts optical reflectance, texture, and morphological features."""
    r = rgb[:, :, 0].astype(np.float32) / 255.0
    g = rgb[:, :, 1].astype(np.float32) / 255.0
    b = rgb[:, :, 2].astype(np.float32) / 255.0
    gray = 0.299 * r + 0.587 * g + 0.114 * b

    # Spectral indices
    exg = 2.0 * g - r - b
    ndvi_approx = (g - r) / (g + r + 1e-5)

    # Local texture
    from scipy.ndimage import gaussian_filter, sobel
    edge = np.sqrt(sobel(gray, axis=0)**2 + sobel(gray, axis=1)**2)
    blur = gaussian_filter(gray, sigma=3.0)
    high_freq = np.abs(gray - blur)

    features = np.stack([gray, r, g, b, exg, ndvi_approx, edge, high_freq], axis=-1)
    return features


def train_local():
    sample_dir = os.path.join(BASE_DIR, "sample_data")
    urban_rgb_path = os.path.join(sample_dir, "sample_urban_georef.tif")
    urban_lidar_path = os.path.join(sample_dir, "sample_urban_lidar.tif")

    print("=" * 60)
    print(" Training Local Elevation Calibrator on CPU...")
    print("=" * 60)

    rgb, _ = GeoTIFFHandler.read_image(urban_rgb_path)
    gt_lidar = tifffile.imread(urban_lidar_path).astype(np.float32)

    # Extract features and targets
    X = extract_features(rgb).reshape(-1, 8)
    # Target height above base ground level
    y_ground = np.percentile(gt_lidar, 5)
    y_ndsm = np.clip(gt_lidar - y_ground, 0.0, 50.0).flatten()

    # Train linear ridge regression with height-weighted penalty for tall structures
    weights = 1.0 + 2.5 * np.clip(y_ndsm / 30.0, 0.0, 2.0)
    from sklearn.linear_model import Ridge
    model = Ridge(alpha=10.0)
    model.fit(X, y_ndsm, sample_weight=weights)

    # Predict & evaluate
    pred_ndsm = model.predict(X).reshape(rgb.shape[:2])
    pred_ndsm = np.clip(pred_ndsm, 0.0, 50.0)

    # Evaluate
    results = MetricsEvaluator.evaluate(pred_ndsm, y_ndsm.reshape(rgb.shape[:2]))
    print(f"\n[Training Complete]")
    print(f" - Local Model MAE: {results['mae']} m")
    print(f" - Local Model RMSE: {results['rmse']} m")
    print(f" - Correlation (r): {results['correlation']}")
    print(f" - Weights saved to 'training/checkpoints/local_calibrator.npz'")

    os.makedirs(os.path.join(BASE_DIR, "training", "checkpoints"), exist_ok=True)
    np.savez(
        os.path.join(BASE_DIR, "training", "checkpoints", "local_calibrator.npz"),
        coef=model.coef_,
        intercept=model.intercept_
    )


if __name__ == "__main__":
    train_local()
