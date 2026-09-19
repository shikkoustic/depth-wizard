"""Fetch a co-registered benchmark site from Microsoft Planetary Computer (no account needed).

For a lon/lat bounding box it writes, into bench/sites/<name>/:
  naip.tif        NAIP aerial RGB (0.6 m, the model input)
  lidar_dsm.tif   USGS 3DEP LiDAR digital surface model (2 m)
  lidar_dtm.tif   USGS 3DEP LiDAR bare-earth terrain (2 m)
  lidar_hag.tif   USGS 3DEP LiDAR height above ground = nDSM (2 m)
  site.json       provenance (item ids, dates, bbox)

All rasters are cropped (not resampled) from Cloud-Optimized GeoTIFFs with windowed reads,
so memory use stays small. Alignment to a common grid happens at evaluation time.

Example:
  python bench/fetch_site.py --name smoky_forest --bbox -83.52 35.60 -83.50 35.62
"""
import argparse, json, os, urllib.request
import numpy as np
import rasterio
from rasterio.windows import from_bounds
from rasterio.warp import transform_bounds

STAC = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
_tokens = {}


def _post(url, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=120))


def sign(href, collection):
    if collection not in _tokens:
        _tokens[collection] = json.load(urllib.request.urlopen(
            f"https://planetarycomputer.microsoft.com/api/sas/v1/token/{collection}", timeout=60))["token"]
    return f"{href}?{_tokens[collection]}"


def search(collection, bbox, **kw):
    body = {"collections": [collection], "bbox": bbox, "limit": 100, **kw}
    return _post(STAC, body)["features"]


def covering(items, bbox):
    """Items whose footprint bbox fully contains ours, newest first."""
    x0, y0, x1, y1 = bbox
    full = [f for f in items if f["bbox"][0] <= x0 and f["bbox"][1] <= y0 and f["bbox"][2] >= x1 and f["bbox"][3] >= y1]
    return sorted(full, key=lambda f: f["properties"].get("datetime") or f["properties"].get("start_datetime") or "", reverse=True)


def crop(href, collection, bbox, out_path, bands=None):
    with rasterio.open(sign(href, collection)) as src:
        b = transform_bounds("EPSG:4326", src.crs, *bbox, densify_pts=21)
        win = from_bounds(*b, transform=src.transform).round_offsets().round_lengths()
        data = src.read(bands, window=win) if bands else src.read(window=win)
        prof = src.profile.copy()
        prof.update(driver="GTiff", width=data.shape[-1], height=data.shape[-2], count=data.shape[0],
                    transform=src.window_transform(win), compress="deflate", tiled=True, blockxsize=256, blockysize=256)
        for k in ("photometric", "interleave"):
            prof.pop(k, None)
        with rasterio.open(out_path, "w", **prof) as dst:
            dst.write(data)
        return dict(crs=str(src.crs), res=list(src.res), shape=list(data.shape), nodata=src.nodata)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--bbox", nargs=4, type=float, required=True, metavar=("W", "S", "E", "N"))
    ap.add_argument("--terrain", default="", help="label: urban | sparse | hilly | forest")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "sites"))
    a = ap.parse_args()
    out = os.path.join(a.out, a.name); os.makedirs(out, exist_ok=True)
    meta = {"name": a.name, "bbox": a.bbox, "terrain": a.terrain, "layers": {}}

    naip = covering(search("naip", a.bbox), a.bbox)
    if not naip:
        raise SystemExit("no single NAIP item covers this bbox; shrink or move it")
    it = naip[0]
    meta["layers"]["naip"] = dict(item=it["id"], date=it["properties"]["datetime"],
                                  **crop(it["assets"]["image"]["href"], "naip", a.bbox, f"{out}/naip.tif", bands=[1, 2, 3]))

    for kind in ("dsm", "dtm", "hag"):
        col = f"3dep-lidar-{kind}"
        items = covering(search(col, a.bbox), a.bbox)
        if not items:
            raise SystemExit(f"no single {col} item covers this bbox; shrink or move it")
        it = items[0]
        meta["layers"][f"lidar_{kind}"] = dict(item=it["id"], date=it["properties"].get("datetime") or it["properties"].get("start_datetime"),
                                               **crop(it["assets"]["data"]["href"], col, a.bbox, f"{out}/lidar_{kind}.tif"))

    with rasterio.open(f"{out}/lidar_hag.tif") as s:
        h = s.read(1, masked=True).astype(np.float32)
        meta["hag_stats"] = dict(mean=float(h.mean()), p95=float(np.percentile(h.compressed(), 95)), max=float(h.max()))
    with rasterio.open(f"{out}/lidar_dtm.tif") as s:
        t = s.read(1, masked=True)
        meta["terrain_relief_m"] = float(t.max() - t.min())
    json.dump(meta, open(f"{out}/site.json", "w"), indent=1)
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
