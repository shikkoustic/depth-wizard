"""
DepthWizard - Automated Unit & Integration Tests.
Verifies GeoTIFF processing, Depth Estimator, Scale Calibrator, Uncertainty, and Metrics.
"""

import os
import sys
import unittest
import numpy as np

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(BASE_DIR, "backend"))

from app.processing.geotiff_io import GeoTIFFHandler
from app.processing.dem_calibrator import DEMCalibrator
from app.processing.metrics_evaluator import MetricsEvaluator
from app.models.depth_estimator import DepthEstimator
from app.models.uncertainty import UncertaintyEstimator


class TestDepthWizardPipeline(unittest.TestCase):

    def setUp(self):
        self.test_h, self.test_w = 64, 64
        # Synthetic RGB test tile
        self.rgb = np.random.randint(50, 200, (self.test_h, self.test_w, 3), dtype=np.uint8)
        self.bounds = [77.2000, 28.6000, 77.2100, 28.6100]

    def test_depth_estimator(self):
        estimator = DepthEstimator()
        ndsm = estimator.predict_ndsm(self.rgb, target_max_height=35.0)
        self.assertEqual(ndsm.shape, (self.test_h, self.test_w))
        self.assertTrue(np.all(ndsm >= 0.0))
        self.assertTrue(np.max(ndsm) <= 35.0)

    def test_dem_calibrator_fusion(self):
        ndsm = np.full((self.test_h, self.test_w), 15.0, dtype=np.float32)
        ground_dem = DEMCalibrator.get_ground_dem(self.bounds, (self.test_h, self.test_w))
        dsm, stats = DEMCalibrator.fuse_dsm(ndsm, ground_dem, is_georeferenced=True)

        self.assertEqual(dsm.shape, (self.test_h, self.test_w))
        self.assertGreater(stats["min_elevation"], 100.0)  # Ground base is ~150m
        self.assertAlmostEqual(stats["max_object_height"], 15.0, places=1)

    def test_uncertainty_estimator(self):
        depth = np.random.uniform(10.0, 30.0, (self.test_h, self.test_w)).astype(np.float32)
        confidence, uncertainty = UncertaintyEstimator.compute_confidence(self.rgb, depth)

        self.assertEqual(confidence.shape, (self.test_h, self.test_w))
        self.assertEqual(uncertainty.shape, (self.test_h, self.test_w))
        self.assertTrue(np.all((confidence >= 0.0) & (confidence <= 1.0)))
        self.assertTrue(np.all((uncertainty >= 0.0) & (uncertainty <= 1.0)))

    def test_metrics_evaluator(self):
        gt = np.full((self.test_h, self.test_w), 25.0, dtype=np.float32)
        pred = gt + 2.0  # Constant error of 2 meters
        results = MetricsEvaluator.evaluate(pred, gt)

        self.assertAlmostEqual(results["mae"], 0.0, places=1)  # Datum adjusted
        self.assertAlmostEqual(results["datum_offset_adjusted"], 2.0, places=1)
        self.assertEqual(results["valid_pixel_count"], self.test_h * self.test_w)
        self.assertIn("baselines", results)

    def test_geotiff_io(self):
        test_out = os.path.join(BASE_DIR, "backend", "static", "outputs", "test_out.tif")
        dsm = np.random.uniform(100.0, 150.0, (32, 32)).astype(np.float32)
        meta = {
            "crs": "EPSG:4326",
            "bounds": [77.0, 28.0, 77.1, 28.1],
            "transform": [0.003, 0, 77.0, 0, -0.003, 28.1]
        }
        saved_path = GeoTIFFHandler.save_dsm_geotiff(dsm, test_out, meta)
        self.assertTrue(os.path.exists(saved_path))
        if os.path.exists(saved_path):
            os.remove(saved_path)

    def test_api_endpoints(self):
        from fastapi.testclient import TestClient
        from app.main import app

        client = TestClient(app)
        res_root = client.get("/")
        self.assertEqual(res_root.status_code, 200)

        res_health = client.get("/api/health")
        self.assertEqual(res_health.status_code, 200)
        self.assertEqual(res_health.json()["status"], "healthy")

        res_samples = client.get("/api/samples")
        self.assertEqual(res_samples.status_code, 200)
        self.assertGreater(len(res_samples.json()["samples"]), 0)

    def test_gcp_planar_calibration(self):
        dsm = np.full((100, 100), 150.0, dtype=np.float32)
        bounds = [77.200, 28.600, 77.215, 28.615]
        # Surveyed GCPs with 2.5m offset
        gcps = [
            (77.2025, 28.6125, 152.5),
            (77.2120, 28.6130, 152.5),
            (77.2030, 28.6020, 152.5),
            (77.2135, 28.6035, 152.5)
        ]
        calibrated_dsm, gcp_stats = DEMCalibrator.apply_gcp_correction(dsm, bounds, gcps)
        self.assertIsNotNone(gcp_stats)
        self.assertEqual(gcp_stats["gcp_count"], 4)
        self.assertAlmostEqual(gcp_stats["offset_m"], 2.5, places=1)
        self.assertLess(gcp_stats["residual_rmse_after"], 0.1)


if __name__ == "__main__":
    unittest.main()

