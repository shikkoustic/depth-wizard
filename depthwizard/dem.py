"""Bare-earth terrain (DTM) for an image footprint.

Default source: FABDEM v1.2 (Copernicus GLO-30 with forests and buildings removed; 1 arc-second,
heights in metres above the EGM2008 geoid; University of Bristol, non-commercial licence), read over
HTTPS with windowed COG reads. Fallback: Copernicus GLO-30 *surface* model from AWS open data (it
contains buildings/forest, so adding an nDSM on top double-counts them; flagged in the output).
A local DEM file can always be supplied instead (offline use).
"""
import math
import numpy as np
from rasterio.transform import array_bounds
from rasterio.warp import transform_bounds
from .geo import read_onto_grid

FABDEM = "/vsicurl/https://huggingface.co/buckets/links-ads/fabdem/resolve/tiles/{grp}/{tile}_FABDEM_V1-2.tif"
COP30 = "/vsicurl/https://copernicus-dem-30m.s3.amazonaws.com/{name}/{name}.tif"


def _ns(v): return f"{'N' if v >= 0 else 'S'}{abs(v):02d}"
def _ew(v): return f"{'E' if v >= 0 else 'W'}{abs(v):03d}"


def _fabdem_url(lat, lon):
    la, lo = math.floor(lat / 10) * 10, math.floor(lon / 10) * 10
    return FABDEM.format(grp=f"{_ns(la)}{_ew(lo)}-{_ns(la + 10)}{_ew(lo + 10)}_FABDEM_V1-2", tile=f"{_ns(lat)}{_ew(lon)}")


def _cop30_url(lat, lon):
    name = f"Copernicus_DSM_COG_10_{'N' if lat >= 0 else 'S'}{abs(lat):02d}_00_{'E' if lon >= 0 else 'W'}{abs(lon):03d}_00_DEM"
    return COP30.format(name=name)


def _tiles(dst_transform, dst_crs, shape):
    b = transform_bounds(dst_crs, "EPSG:4326", *array_bounds(*shape, dst_transform), densify_pts=21)
    return [(la, lo) for la in range(math.floor(b[1]), math.floor(b[3]) + 1) for lo in range(math.floor(b[0]), math.floor(b[2]) + 1)]


def _mosaic(url_fn, dst_transform, dst_crs, shape):
    out = np.full(shape, np.nan, np.float32)
    for la, lo in _tiles(dst_transform, dst_crs, shape):
        try:
            a = read_onto_grid(url_fn(la, lo), dst_transform, dst_crs, shape, method="bilinear")
        except Exception as e:  # missing tile (open ocean) or network error
            if "404" in str(e) or "does not exist" in str(e) or "No such file" in str(e):
                continue
            raise
        out = np.where(np.isfinite(a), a, out)
    return out


def terrain(dst_transform, dst_crs, shape, dem_path=None):
    """Return (dtm float32 HxW, info dict)."""
    if dem_path:
        return read_onto_grid(str(dem_path), dst_transform, dst_crs, shape), dict(
            source="user-supplied DEM", kind="as provided", double_count_risk="unknown")
    errors = []
    for name, fn, kind in (("FABDEM v1.2 (bare earth, EGM2008)", _fabdem_url, "bare-earth"),
                           ("Copernicus GLO-30 (surface model, EGM2008)", _cop30_url, "surface")):
        try:
            d = _mosaic(fn, dst_transform, dst_crs, shape)
            if np.isfinite(d).mean() > 0.5:
                return d, dict(source=name, kind=kind, double_count_risk=kind == "surface")
            errors.append(f"{name}: no coverage")
        except Exception as e:
            errors.append(f"{name}: {type(e).__name__}: {e}")
    raise RuntimeError("could not obtain terrain: " + " | ".join(errors))


def coarse_ndsm(dst_transform, dst_crs, shape):
    """Copernicus (surface) minus FABDEM (bare earth): a 30 m estimate of mean above-ground height."""
    return _mosaic(_cop30_url, dst_transform, dst_crs, shape) - _mosaic(_fabdem_url, dst_transform, dst_crs, shape)
