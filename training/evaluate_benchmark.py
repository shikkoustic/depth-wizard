"""
DepthWizard - Evaluation Benchmark Suite (Person A).
Evaluates predicted DSMs against LiDAR ground truth on Urban, Sparse, Hilly, and Forest tiles.
Outputs comparative accuracy metrics (RMSE, MAE, Pearson r) against baseline methods.
"""

import os
import sys
import numpy as np

# Add backend to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend")))
from app.processing.geotiff_io import GeoTIFFHandler
from app.processing.dem_calibrator import DEMCalibrator
from app.processing.metrics_evaluator import MetricsEvaluator
from app.models.depth_estimator import DepthEstimator


def run_benchmark():
    sample_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "sample_data"))
    urban_img = os.path.join(sample_dir, "sample_urban_georef.tif")
    urban_lidar = os.path.join(sample_dir, "sample_urban_lidar.tif")

    print("=" * 70)
    print(" SIH26175 DepthWizard - Accuracy Benchmark & Validation")
    print("=" * 70)

    # 1. Load optical image and metadata
    rgb, meta = GeoTIFFHandler.read_image(urban_img)
    _, gt_meta = GeoTIFFHandler.read_image(urban_lidar)

    import tifffile
    gt_lidar = tifffile.imread(urban_lidar)

    # 2. Run DepthWizard model
    estimator = DepthEstimator()
    ndsm = estimator.predict_ndsm(rgb, target_max_height=38.0)

    # 3. Fuse Ground DEM + nDSM
    ground_dem = DEMCalibrator.get_ground_dem(meta.get("bounds"), rgb.shape[:2], meta.get("crs"))
    pred_dsm, stats = DEMCalibrator.fuse_dsm(ndsm, ground_dem, is_georeferenced=meta["is_georeferenced"])

    # 4. Evaluate Metrics
    results = MetricsEvaluator.evaluate(pred_dsm, gt_lidar)

    print(f"\n[Overall Accuracy Results]")
    print(f" - MAE (Mean Absolute Error):     {results['mae']} meters")
    print(f" - RMSE (Root Mean Square Error): {results['rmse']} meters")
    print(f" - Pearson Correlation (r):       {results['correlation']}")
    print(f" - Valid Evaluated Pixels:        {results['valid_pixel_count']:,}")

    print("\n[Per-Landscape Breakdown]")
    for k, v in results["breakdown"].items():
        print(f" • {k.replace('_', ' ').title()}: MAE = {v['mae']} m | RMSE = {v['rmse']} m ({v['pixel_percentage']}% of scene)")

    print("\n[Comparison with Hackathon Baselines]")
    print(f"{'Method':<45} | {'MAE (m)':<8} | {'RMSE (m)':<8} | {'Corr (r)':<8}")
    print("-" * 75)
    for b in results["baselines"]:
        print(f"{b['method']:<45} | {b['mae']:<8.2f} | {b['rmse']:<8.2f} | {b['correlation']:<8.2f}")
    print("-" * 75)

    return results


if __name__ == "__main__":
    run_benchmark()
