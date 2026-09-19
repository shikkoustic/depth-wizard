"""
DepthWizard - Training Pipeline on Real Hugging Face GAMUS Dataset.
Trains on real satellite optical RGB imagery and paired LiDAR metric heights (AGL)
downloaded directly from https://huggingface.co/datasets/earthflow/GAMUS
Evaluates on unseen real GAMUS test tiles.
"""

import os
import sys
import glob
import h5py
import numpy as np
from scipy.ndimage import gaussian_filter, sobel
from sklearn.linear_model import Ridge
from sklearn.ensemble import ExtraTreesRegressor

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(BASE_DIR, "backend"))

from app.processing.metrics_evaluator import MetricsEvaluator


def extract_dense_features(rgb: np.ndarray) -> np.ndarray:
    """
    Extracts multi-scale optical, spectral, and structural features from satellite imagery.
    """
    h, w = rgb.shape[:2]
    r = rgb[:, :, 0].astype(np.float32) / 255.0
    g = rgb[:, :, 1].astype(np.float32) / 255.0
    b = rgb[:, :, 2].astype(np.float32) / 255.0
    gray = 0.299 * r + 0.587 * g + 0.114 * b

    # Spectral indices
    exg = 2.0 * g - r - b  # Vegetation canopy index
    ndvi_approx = (g - r) / (g + r + 1e-5)

    # Multi-scale edge & gradient features
    gx = sobel(gray, axis=1)
    gy = sobel(gray, axis=0)
    edge_fine = np.sqrt(gx**2 + gy**2)

    # Multi-scale structural blur & contrast
    blur_3 = gaussian_filter(gray, sigma=1.5)
    blur_7 = gaussian_filter(gray, sigma=4.0)
    blur_15 = gaussian_filter(gray, sigma=8.0)

    hf_fine = np.abs(gray - blur_3)
    hf_coarse = np.abs(gray - blur_7)
    context = blur_7 - blur_15

    # Shadow & roof contrast
    shadow_mask = np.clip((0.22 - gray) / 0.22, 0.0, 1.0)
    roof_salience = np.clip((gray - 0.55) / 0.45, 0.0, 1.0)

    features = np.stack([
        gray, r, g, b, exg, ndvi_approx,
        edge_fine, hf_fine, hf_coarse, context,
        shadow_mask, roof_salience
    ], axis=-1)

    return features


def load_gamus_split(data_dir: str, split: str = "train", subsample_step: int = 4):
    """
    Loads real GAMUS .h5 samples from disk.
    Downsamples pixels by subsample_step to fit smoothly in memory while preserving full spatial diversity.
    """
    split_dir = os.path.join(data_dir, split)
    rgb_files = sorted(glob.glob(os.path.join(split_dir, "*_RGB.h5")))

    all_X = []
    all_y = []

    print(f"Loading {len(rgb_files)} real GAMUS {split} tiles...")
    for rgb_p in rgb_files:
        agl_p = rgb_p.replace("_RGB.h5", "_AGL.h5")
        if not os.path.exists(agl_p):
            continue

        with h5py.File(rgb_p, "r") as f_rgb, h5py.File(agl_p, "r") as f_agl:
            rgb = np.array(f_rgb["image"])
            agl = np.array(f_agl["image"]).astype(np.float32)

        # Clean invalid values (NaN, Inf, negative heights)
        agl = np.nan_to_num(agl, nan=0.0, posinf=60.0, neginf=0.0)
        agl = np.clip(agl, 0.0, 70.0)

        feats = extract_dense_features(rgb)

        # Subsample grid to train efficiently across all tiles
        sub_feats = feats[::subsample_step, ::subsample_step].reshape(-1, feats.shape[-1])
        sub_agl = agl[::subsample_step, ::subsample_step].flatten()

        all_X.append(sub_feats)
        all_y.append(sub_agl)

    if not all_X:
        raise RuntimeError(f"No GAMUS {split} data found in {split_dir}")

    X = np.vstack(all_X)
    y = np.concatenate(all_y)
    return X, y


def train_and_evaluate_gamus():
    data_dir = os.path.abspath(os.path.join(BASE_DIR, "data", "gamus"))
    chk_dir = os.path.join(BASE_DIR, "training", "checkpoints")
    os.makedirs(chk_dir, exist_ok=True)

    print("=" * 70)
    print(" SIH26175 DepthWizard - Training on Real Hugging Face GAMUS Dataset")
    print("=" * 70)

    # 1. Load Real Training Data
    X_train, y_train = load_gamus_split(data_dir, split="train", subsample_step=4)
    print(f"Training dataset size: {X_train.shape[0]:,} real pixel samples across {X_train.shape[1]} features.")
    print(f"Ground Truth LiDAR Heights: Min = {np.min(y_train):.2f}m | Mean = {np.mean(y_train):.2f}m | Max = {np.max(y_train):.2f}m")

    # 2. Tall-Building Weighted Loss Penalty:
    # Penalize tall structures exponentially to address the 8.8m error
    sample_weights = 1.0 + 3.0 * np.clip(y_train / 25.0, 0.0, 2.5)

    print("\nTraining Height Estimator with Tall-Building Weighted Loss...")
    model = Ridge(alpha=5.0)
    model.fit(X_train, y_train, sample_weight=sample_weights)

    # 3. Load Real Unseen GAMUS Test Data
    X_test, y_test = load_gamus_split(data_dir, split="test", subsample_step=2)
    print(f"\nEvaluating on {X_test.shape[0]:,} unseen real GAMUS test pixels...")

    y_pred = model.predict(X_test)
    y_pred = np.clip(y_pred, 0.0, 70.0)

    # 4. Accuracy Evaluation
    mae = float(np.mean(np.abs(y_pred - y_test)))
    rmse = float(np.sqrt(np.mean((y_pred - y_test)**2)))
    from scipy.stats import pearsonr
    corr, _ = pearsonr(y_test, y_pred)

    print("\n" + "=" * 70)
    print(" OFFICIAL TEST ACCURACY ON REAL HUGGING FACE GAMUS DATASET")
    print("=" * 70)
    print(f" - Test MAE (Mean Absolute Error):     {mae:.2f} meters")
    print(f" - Test RMSE (Root Mean Square Error): {rmse:.2f} meters")
    print(f" - Test Pearson Correlation (r):       {corr:.3f}")
    print(f" - Evaluated Pixels:                   {len(y_test):,}")

    # Per-class breakdown
    ground_mask = y_test < 2.0
    veg_mask = (y_test >= 2.0) & (y_test < 10.0)
    bldg_mask = y_test >= 10.0

    print("\n[Per-Landscape Breakdown on Real LiDAR]")
    if np.any(ground_mask):
        print(f" • Ground / Sparse (<2m):   MAE = {np.mean(np.abs(y_pred[ground_mask] - y_test[ground_mask])):.2f} m")
    if np.any(veg_mask):
        print(f" • Vegetation / Trees (2-10m): MAE = {np.mean(np.abs(y_pred[veg_mask] - y_test[veg_mask])):.2f} m")
    if np.any(bldg_mask):
        print(f" • Urban Buildings (>10m):    MAE = {np.mean(np.abs(y_pred[bldg_mask] - y_test[bldg_mask])):.2f} m (Weighted Loss Applied)")

    # Save Checkpoint
    chk_file = os.path.join(chk_dir, "gamus_real_model.npz")
    np.savez(chk_file, coef=model.coef_, intercept=model.intercept_)
    print(f"\n[Model Checkpoint Saved]: {chk_file}")
    print("=" * 70)

    return {
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "correlation": round(corr, 3)
    }


if __name__ == "__main__":
    train_and_evaluate_gamus()
