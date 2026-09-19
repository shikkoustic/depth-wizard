"""
DepthWizard - Sample Dataset Generator.
Generates realistic benchmark test pairs:
1. Georeferenced optical satellite GeoTIFF (Urban landscape)
2. Co-registered Ground Truth LiDAR DSM (GeoTIFF)
3. Non-georeferenced optical satellite image (PNG/JPG)
4. Hilly terrain GeoTIFF
"""

import os
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


def generate_benchmark_samples(output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    H, W = 512, 512

    # --- 1. Generate Synthetic Urban Scene with Realistic LiDAR DSM ---
    print("[SampleGen] Synthesizing Urban Satellite & LiDAR Benchmark Pair...")
    np.random.seed(42)

    # Base topography (ground DEM with slight undulation ~150m)
    x = np.linspace(0, 1, W)
    y = np.linspace(0, 1, H)
    xx, yy = np.meshgrid(x, y)
    ground_dem = 150.0 + (xx * 12.0) - (yy * 8.0) + np.sin(xx * 6.28) * 3.0

    # Ground truth object heights (nDSM)
    ndsm_gt = np.zeros((H, W), dtype=np.float32)
    rgb = np.zeros((H, W, 3), dtype=np.uint8)

    # Ground texture: Roads and bare earth / grass
    rgb[:, :] = [110, 130, 95]  # Greenish ground

    # Add Road Grid
    for r in range(40, H, 100):
        rgb[r:r+14, :] = [60, 62, 65]
        ndsm_gt[r:r+14, :] = 0.1  # flat road
    for c in range(40, W, 100):
        rgb[:, c:c+14] = [60, 62, 65]
        ndsm_gt[:, c:c+14] = 0.1

    # Place Buildings of varying heights (low-rise ~8m to high-rise ~35m)
    buildings = [
        # (r, c, h, w, height_m, color)
        (70, 70, 55, 65, 28.5, [190, 185, 175]),   # High-rise commercial
        (70, 170, 60, 50, 18.0, [210, 195, 180]),  # Mid-rise office
        (70, 260, 50, 60, 32.0, [170, 180, 195]),  # Tall glass tower
        (70, 370, 55, 70, 22.0, [205, 170, 160]),  # Residential block
        (180, 70, 65, 55, 14.0, [220, 215, 205]),
        (180, 170, 55, 60, 38.0, [180, 190, 210]), # Skyscraper
        (180, 260, 60, 65, 12.5, [225, 210, 190]),
        (180, 370, 65, 50, 26.0, [195, 190, 185]),
        (280, 70, 50, 60, 9.0,  [210, 205, 195]),
        (280, 170, 65, 55, 16.0, [200, 185, 175]),
        (280, 260, 55, 60, 29.0, [185, 195, 205]),
        (280, 370, 60, 65, 15.5, [215, 210, 200]),
        (380, 70, 65, 60, 24.0, [200, 195, 190]),
        (380, 170, 50, 55, 11.0, [220, 215, 205]),
        (380, 260, 60, 60, 34.0, [175, 185, 200]),
        (380, 370, 55, 65, 20.0, [205, 195, 185]),
    ]

    for (r, c, bh, bw, height_val, bcolor) in buildings:
        # Building roof
        rgb[r:r+bh, c:c+bw] = bcolor
        ndsm_gt[r:r+bh, c:c+bw] = height_val
        # Building shadow on Southeast side
        s_len = int(height_val * 0.5)
        for sr in range(r, r + bh):
            for sc in range(c + bw, min(W, c + bw + s_len)):
                if ndsm_gt[sr, sc] < 1.0:
                    rgb[sr, sc] = (rgb[sr, sc] * 0.45).astype(np.uint8)

    # Add Tree clusters (vegetation ~5-12m)
    for _ in range(45):
        tr = np.random.randint(20, H - 30)
        tc = np.random.randint(20, W - 30)
        radius = np.random.randint(5, 12)
        tree_h = np.random.uniform(4.0, 11.0)
        for dr in range(-radius, radius):
            for dc in range(-radius, radius):
                if dr*dr + dc*dc <= radius*radius:
                    nr, nc = tr + dr, tc + dc
                    if 0 <= nr < H and 0 <= nc < W and ndsm_gt[nr, nc] < 1.0:
                        rgb[nr, nc] = [np.random.randint(30, 55), np.random.randint(90, 135), np.random.randint(30, 50)]
                        ndsm_gt[nr, nc] = tree_h * (1.0 - (dr*dr + dc*dc) / (radius*radius * 1.5))

    # Absolute LiDAR DSM = Ground DEM + nDSM
    lidar_dsm_gt = (ground_dem + ndsm_gt).astype(np.float32)

    # Geographic metadata for georeferenced GeoTIFF (e.g. New Delhi urban district)
    # Bounding box: [77.2000, 28.6000, 77.2150, 28.6150] in EPSG:4326
    bounds = [77.2000, 28.6000, 77.2150, 28.6150]

    # Save Urban Optical GeoTIFF
    urban_geotiff_path = os.path.join(output_dir, "sample_urban_georef.tif")
    urban_lidar_path = os.path.join(output_dir, "sample_urban_lidar.tif")

    if HAS_RASTERIO:
        trans = from_bounds(bounds[0], bounds[1], bounds[2], bounds[3], W, H)
        crs = CRS.from_epsg(4326)

        # Write 3-band RGB GeoTIFF
        with rasterio.open(
            urban_geotiff_path, "w", driver="GTiff",
            height=H, width=W, count=3, dtype=rasterio.uint8,
            crs=crs, transform=trans
        ) as dst:
            dst.write(np.transpose(rgb, (2, 0, 1)))

        # Write 1-band LiDAR Float32 GeoTIFF
        with rasterio.open(
            urban_lidar_path, "w", driver="GTiff",
            height=H, width=W, count=1, dtype=rasterio.float32,
            crs=crs, transform=trans, nodata=-9999.0
        ) as dst:
            dst.write(lidar_dsm_gt, 1)
    elif HAS_TIFFFILE:
        tifffile.imwrite(urban_geotiff_path, rgb)
        tifffile.imwrite(urban_lidar_path, lidar_dsm_gt)

    # Save non-georeferenced JPG/PNG
    sample_jpg_path = os.path.join(output_dir, "sample_drone_rgb.jpg")
    Image.fromarray(rgb).save(sample_jpg_path, quality=95)

    # --- 2. Generate Hilly Landscape GeoTIFF ---
    print("[SampleGen] Synthesizing Hilly / Mountainous Terrain Sample...")
    hilly_dem = 450.0 + np.sin(xx * 4.0) * 85.0 + np.cos(yy * 3.5) * 65.0 + np.sin((xx + yy) * 5.0) * 35.0
    hilly_rgb = np.zeros((H, W, 3), dtype=np.uint8)
    norm_h = (hilly_dem - np.min(hilly_dem)) / (np.max(hilly_dem) - np.min(hilly_dem))
    hilly_rgb[:, :, 0] = (80 + norm_h * 110).astype(np.uint8)  # Earthy brown
    hilly_rgb[:, :, 1] = (95 + norm_h * 90).astype(np.uint8)   # Forest green
    hilly_rgb[:, :, 2] = (50 + norm_h * 60).astype(np.uint8)

    hilly_geotiff_path = os.path.join(output_dir, "sample_hilly_georef.tif")
    hilly_lidar_path = os.path.join(output_dir, "sample_hilly_lidar.tif")

    if HAS_RASTERIO:
        hilly_bounds = [78.1000, 30.2000, 78.1250, 30.2250]
        hilly_trans = from_bounds(hilly_bounds[0], hilly_bounds[1], hilly_bounds[2], hilly_bounds[3], W, H)
        with rasterio.open(
            hilly_geotiff_path, "w", driver="GTiff",
            height=H, width=W, count=3, dtype=rasterio.uint8,
            crs=crs, transform=hilly_trans
        ) as dst:
            dst.write(np.transpose(hilly_rgb, (2, 0, 1)))

        with rasterio.open(
            hilly_lidar_path, "w", driver="GTiff",
            height=H, width=W, count=1, dtype=rasterio.float32,
            crs=crs, transform=hilly_trans, nodata=-9999.0
        ) as dst:
            dst.write(hilly_dem.astype(np.float32), 1)
    elif HAS_TIFFFILE:
        tifffile.imwrite(hilly_geotiff_path, hilly_rgb)
        tifffile.imwrite(hilly_lidar_path, hilly_dem.astype(np.float32))

    print(f"[SampleGen] Generated 4 benchmark sample datasets in '{output_dir}'.")
    return {
        "urban_geotiff": urban_geotiff_path,
        "urban_lidar": urban_lidar_path,
        "drone_jpg": sample_jpg_path,
        "hilly_geotiff": hilly_geotiff_path
    }


if __name__ == "__main__":
    generate_benchmark_samples("sample_data")
