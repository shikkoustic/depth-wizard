"""Georeferencing helpers: grid definition, resampling onto a grid, GeoTIFF writing."""
import os
for _k, _v in (("GDAL_HTTP_TIMEOUT", "60"), ("GDAL_HTTP_MAX_RETRY", "4"), ("GDAL_HTTP_RETRY_DELAY", "2"), ("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")):
    os.environ.setdefault(_k, _v)
import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine
from rasterio.warp import reproject, Resampling, calculate_default_transform, transform_bounds


def metric_grid(src_crs, src_transform, width, height, res=None):
    """Return (crs, transform, width, height) of a metric (UTM) grid covering the source raster.
    If the source is already projected in metres and res is None, the source grid is kept."""
    src_crs = CRS.from_user_input(src_crs)
    if src_crs.is_projected and res is None:
        return src_crs, src_transform, width, height
    if src_crs.is_projected:
        dst_crs = src_crs
    else:
        l, b, r, t = rasterio.transform.array_bounds(height, width, src_transform)
        lon, lat = (l + r) / 2, (b + t) / 2
        dst_crs = CRS.from_epsg((32600 if lat >= 0 else 32700) + int((lon + 180) // 6) + 1)
    tr, w, h = calculate_default_transform(src_crs, dst_crs, width, height,
                                           *rasterio.transform.array_bounds(height, width, src_transform),
                                           resolution=res)
    return dst_crs, tr, w, h


def resample(src, src_transform, src_crs, dst_transform, dst_crs, shape, method="bilinear", src_nodata=None):
    """Resample array src (C,H,W or H,W) onto the destination grid. Returns float32 with NaN for nodata."""
    two_d = src.ndim == 2
    s = src[None] if two_d else src
    out = np.full((s.shape[0], *shape), np.nan, np.float32)
    for i in range(s.shape[0]):
        reproject(s[i].astype(np.float32), out[i], src_transform=src_transform, src_crs=src_crs,
                  dst_transform=dst_transform, dst_crs=dst_crs, resampling=getattr(Resampling, method),
                  src_nodata=src_nodata, dst_nodata=np.nan)
    return out[0] if two_d else out


def read_onto_grid(path, dst_transform, dst_crs, shape, method="bilinear", band=1):
    """Read a raster file (local path or /vsicurl URL) resampled onto a destination grid.
    Only the window covering the destination is read."""
    with rasterio.open(path) as src:
        h, w = shape
        bounds = rasterio.transform.array_bounds(h, w, dst_transform)
        b = transform_bounds(dst_crs, src.crs, *bounds, densify_pts=21)
        win = rasterio.windows.from_bounds(*b, transform=src.transform)
        pad = 2  # margin so bilinear sampling at the edges has neighbours
        c0, r0 = int(np.floor(win.col_off)) - pad, int(np.floor(win.row_off)) - pad
        win = rasterio.windows.Window(c0, r0, int(np.ceil(win.col_off + win.width)) + pad - c0,
                                      int(np.ceil(win.row_off + win.height)) + pad - r0)
        data = src.read(band, window=win, boundless=True, fill_value=src.nodata if src.nodata is not None else np.nan)
        return resample(data, src.window_transform(win), src.crs, dst_transform, dst_crs, shape,
                        method=method, src_nodata=src.nodata)


def write_geotiff(path, arr, transform, crs, nodata=np.nan, description=None):
    arr = np.asarray(arr, np.float32)
    a = arr[None] if arr.ndim == 2 else arr
    prof = dict(driver="GTiff", width=a.shape[2], height=a.shape[1], count=a.shape[0], dtype="float32",
                crs=crs, transform=transform, nodata=nodata, compress="deflate", predictor=3,
                tiled=True, blockxsize=256, blockysize=256)
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(a)
        if description:
            for i, d in enumerate([description] if isinstance(description, str) else description):
                dst.set_band_description(i + 1, d)
