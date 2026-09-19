"""
DepthWizard - Uncertainty & Confidence Estimation Module.
Computes a pixel-wise confidence map indicating model certainty in height predictions.
Highlights shadow zones, complex occlusions, and steep building edges.
"""

from typing import Tuple
import numpy as np
from scipy.ndimage import sobel, gaussian_filter


class UncertaintyEstimator:
    @staticmethod
    def compute_confidence(
        rgb_image: np.ndarray,
        predicted_depth: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes uncertainty and confidence maps.
        Returns:
            confidence_map: ndarray float32 in [0.0, 1.0], 1.0 = highly certain.
            uncertainty_map: ndarray float32 in [0.0, 1.0], 1.0 = high error risk.
        """
        # 1. Height gradient discontinuity (steep step changes have higher uncertainty)
        grad_y = sobel(predicted_depth, axis=0)
        grad_x = sobel(predicted_depth, axis=1)
        depth_grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        norm_depth_grad = np.clip(depth_grad_mag / (np.percentile(depth_grad_mag, 95) + 1e-5), 0, 1)

        # 2. Photometric / shadow contrast uncertainty from RGB
        if rgb_image.ndim == 3:
            gray = np.mean(rgb_image, axis=2).astype(np.float32)
        else:
            gray = rgb_image.astype(np.float32)

        # Very dark pixels (deep cast shadows) typically suffer higher height error
        shadow_risk = np.clip((40.0 - gray) / 40.0, 0, 1)

        # 3. Local structural variance
        blurred_gray = gaussian_filter(gray, sigma=2.0)
        local_var = np.abs(gray - blurred_gray)
        norm_local_var = np.clip(local_var / (np.percentile(local_var, 95) + 1e-5), 0, 1)

        # Weighted combination for overall uncertainty
        uncertainty = 0.5 * norm_depth_grad + 0.3 * shadow_risk + 0.2 * norm_local_var
        uncertainty = gaussian_filter(uncertainty, sigma=1.0)
        uncertainty = np.clip(uncertainty, 0.0, 1.0).astype(np.float32)

        confidence = (1.0 - uncertainty).astype(np.float32)
        return confidence, uncertainty
