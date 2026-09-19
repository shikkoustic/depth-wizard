"""
DepthWizard - GeoTIFF and Remote Sensing Image I/O Module.
Supports reading and writing georeferenced GeoTIFFs and standard optical images (PNG/JPG).
Uses rasterio with tifffile/Pillow fallbacks.
"""

import os
from typing import Dict, Any, Tuple, Optional
import numpy as np
from PIL import Image

try:
    import rasterio
    from rasterio.transform import from_bounds
    from rasterio.crs import CRS
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

try:
    import tifffile
    HAS_TIFFFILE = True
except ImportError:
    HAS_TIFFFILE = False


class GeoTIFFHandler:
    @staticmethod
    def read_image(file_path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reads any satellite image (GeoTIFF, PNG, JPG, etc.).
        Returns:
            rgb_array: uint8 ndarray of shape (H, W, 3)
            metadata: dict containing georeferencing information
        """
        ext = os.path.splitext(file_path)[1].lower()
        is_tiff = ext in [".tif", ".tiff", ".geotiff"]

        if is_tiff and HAS_RASTERIO:
            try:
                with rasterio.open(file_path) as src:
                    meta = {
                        "is_georeferenced": bool(src.crs is not None),
                        "crs": str(src.crs) if src.crs else None,
                        "bounds": list(src.bounds) if src.bounds else None,
                        "transform": list(src.transform) if src.transform else None,
                        "width": src.width,
                        "height": src.height,
                        "count": src.count,
                        "nodata": src.nodata,
                    }

                    # Read RGB bands
                    if src.count >= 3:
                        bands = src.read([1, 2, 3])  # (3, H, W)
                        rgb = np.transpose(bands, (1, 2, 0))
                    elif src.count == 1:
                        band = src.read(1)
                        rgb = np.stack([band, band, band], axis=-1)
                    else:
                        bands = src.read()
                        rgb = np.transpose(bands[:3], (1, 2, 0))

                    # Normalize to uint8 if float or 16-bit
                    if rgb.dtype != np.uint8:
                        if np.issubdtype(rgb.dtype, np.floating):
                            min_v, max_v = np.nanmin(rgb), np.nanmax(rgb)
                            if max_v > min_v:
                                rgb = ((np.nan_to_num(rgb) - min_v) / (max_v - min_v) * 255).astype(np.uint8)
                            else:
                                rgb = np.zeros(rgb.shape, dtype=np.uint8)
                        elif rgb.dtype == np.uint16:
                            rgb = (rgb / 256).astype(np.uint8)
                        else:
                            rgb = rgb.astype(np.uint8)

                    return rgb, meta
            except Exception as e:
                pass  # Fall back to Pillow/tifffile

        if is_tiff and HAS_TIFFFILE:
            try:
                data = tifffile.imread(file_path)
                meta = {
                    "is_georeferenced": False,
                    "crs": None,
                    "bounds": None,
                    "transform": None,
                    "width": data.shape[1] if data.ndim >= 2 else 0,
                    "height": data.shape[0] if data.ndim >= 2 else 0,
                    "count": data.shape[2] if data.ndim == 3 else 1,
                    "nodata": None,
                }
                if data.ndim == 2:
                    rgb = np.stack([data, data, data], axis=-1)
                elif data.ndim == 3:
                    rgb = data[:, :, :3]
                else:
                    rgb = np.zeros((256, 256, 3), dtype=np.uint8)

                if rgb.dtype != np.uint8:
                    rgb = ((rgb - np.min(rgb)) / (np.max(rgb) - np.min(rgb) + 1e-6) * 255).astype(np.uint8)
                return rgb, meta
            except Exception:
                pass

        # Standard non-georeferenced image (PNG, JPG) via PIL
        img = Image.open(file_path).convert("RGB")
        rgb = np.array(img, dtype=np.uint8)
        meta = {
            "is_georeferenced": False,
            "crs": None,
            "bounds": None,
            "transform": None,
            "width": rgb.shape[1],
            "height": rgb.shape[0],
            "count": 3,
            "nodata": None,
        }
        return rgb, meta

    @staticmethod
    def save_dsm_geotiff(
        dsm_array: np.ndarray,
        output_path: str,
        meta: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Saves DSM array (Float32 in meters) as a standard GeoTIFF.
        Preserves CRS and Affine transform if present.
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        h, w = dsm_array.shape[:2]
        dsm_f32 = dsm_array.astype(np.float32)

        if HAS_RASTERIO:
            crs_val = meta.get("crs") if meta else None
            transform_val = meta.get("transform") if meta else None

            if transform_val is not None and len(transform_val) >= 6:
                from rasterio.transform import Affine
                trans = Affine(
                    transform_val[0], transform_val[1], transform_val[2],
                    transform_val[3], transform_val[4], transform_val[5]
                )
            else:
                bounds = meta.get("bounds") if meta else None
                if bounds:
                    trans = from_bounds(bounds[0], bounds[1], bounds[2], bounds[3], w, h)
                else:
                    trans = from_bounds(0, 0, w, h, w, h)

            # Replace any NaNs or Infs with standard nodata value
            dsm_clean = np.nan_to_num(dsm_f32, nan=-9999.0, posinf=-9999.0, neginf=-9999.0)

            try:
                try:
                    crs_obj = CRS.from_string(crs_val) if crs_val else CRS.from_epsg(4326)
                except Exception:
                    crs_obj = CRS.from_epsg(4326)

                with rasterio.open(
                    output_path,
                    "w",
                    driver="GTiff",
                    height=h,
                    width=w,
                    count=1,
                    dtype=rasterio.float32,
                    crs=crs_obj,
                    transform=trans,
                    nodata=-9999.0
                ) as dst:
                    dst.write(dsm_clean, 1)

                return output_path
            except Exception as e:
                # If rasterio fails, fall back gracefully to tifffile
                if HAS_TIFFFILE:
                    tifffile.imwrite(output_path, dsm_clean)
                    return output_path
                raise e

        elif HAS_TIFFFILE:
            dsm_clean = np.nan_to_num(dsm_f32, nan=-9999.0, posinf=-9999.0, neginf=-9999.0)
            tifffile.imwrite(output_path, dsm_clean)
            return output_path
        else:
            # Fallback: Save as 16-bit PNG if no tiff library
            dsm_clean = np.nan_to_num(dsm_f32, nan=0.0)
            norm = ((dsm_clean - np.min(dsm_clean)) / (np.max(dsm_clean) - np.min(dsm_clean) + 1e-6) * 65535).astype(np.uint16)
            Image.fromarray(norm).save(output_path.replace(".tif", ".png"))
            return output_path

