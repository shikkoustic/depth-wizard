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
os.environ.setdefault("GDAL_HTTP_TIMEOUT", "60"); os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "4"); os.environ.setdefault("GDAL_HTTP_RETRY_DELAY", "2")
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
    """SAS-sign an asset URL. Tokens are per storage container, so cache by account/container, not collection."""
    if "blob.core.windows.net" not in href:
        return href
    acct, cont = href.split("//")[1].split(".")[0], href.split("blob.core.windows.net/")[1].split("/")[0]
    key = (acct, cont)
    if key not in _tokens:
        _tokens[key] = json.load(urllib.request.urlopen(
            f"https://planetarycomputer.microsoft.com/api/sas/v1/token/{acct}/{cont}", timeout=60))["token"]
    return f"{href}?{_tokens[key]}"


def search(collection, bbox, **kw):
    body = {"collections": [collection], "bbox": bbox, "limit": 100, **kw}
    return _post(STAC, body)["features"]


def covering(items, bbox):
    """Items whose footprint bbox fully contains ours, newest first."""
    x0, y0, x1, y1 = bbox
    full = [f for f in items if f["bbox"][0] <= x0 and f["bbox"][1] <= y0 and f["bbox"][2] >= x1 and f["bbox"][3] >= y1]
    return sorted(full, key=lambda f: f["properties"].get("datetime") or f["properties"].get("start_datetime") or "", reverse=True)


def fit_bbox(bbox):
    """Shift bbox (keeping its size) so that one NAIP item and one 3DEP item both contain it.
    Tiles are a few km across, so a ~1 km box near a tile edge only needs a small shift."""
    x0, y0, x1, y1 = bbox; w, h = x1 - x0, y1 - y0; cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    pt = [cx - 1e-4, cy - 1e-4, cx + 1e-4, cy + 1e-4]
    best = None
    naips = search("naip", pt)
    for li in search("3dep-lidar-dsm", pt):
        for ni in naips:
            ix0, iy0 = max(li["bbox"][0], ni["bbox"][0]), max(li["bbox"][1], ni["bbox"][1])
            ix1, iy1 = min(li["bbox"][2], ni["bbox"][2]), min(li["bbox"][3], ni["bbox"][3])
            # small inset: item bboxes are footprints and the edges can hold nodata
            ix0, iy0, ix1, iy1 = ix0 + 0.002, iy0 + 0.002, ix1 - 0.002, iy1 - 0.002
            if ix1 - ix0 < w or iy1 - iy0 < h:
                continue
            nx0 = min(max(x0, ix0), ix1 - w); ny0 = min(max(y0, iy0), iy1 - h)
            shift = abs(nx0 - x0) + abs(ny0 - y0)
            if best is None or shift < best[0]:
                best = (shift, [round(nx0, 6), round(ny0, 6), round(nx0 + w, 6), round(ny0 + h, 6)])
    if best is None:
        raise SystemExit("no NAIP/3DEP item pair can contain this bbox")
    return best[1]


def crop(href, collection, bbox, out_path, bands=None):
    with rasterio.open(sign(href, collection)) as src:
        b = transform_bounds("EPSG:4326", src.crs, *bbox, densify_pts=21)
        win = from_bounds(*b, transform=src.transform).round_offsets().round_lengths()
        win = win.intersection(rasterio.windows.Window(0, 0, src.width, src.height))
        if win.width < 10 or win.height < 10:
            raise SystemExit(f"{collection}: item raster does not cover the bbox")
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

    a.bbox = fit_bbox(a.bbox)
    meta["bbox"] = a.bbox
    naip = covering(search("naip", a.bbox), a.bbox)
    if not naip:
        raise SystemExit("no single NAIP item covers this bbox; shrink or move it")
    it = naip[0]
    meta["layers"]["naip"] = dict(item=it["id"], date=it["properties"]["datetime"],
                                  **crop(it["assets"]["image"]["href"], "naip", a.bbox, f"{out}/naip.tif", bands=[1, 2, 3]))

    dsm_id = None
    for kind in ("dsm", "dtm", "hag"):
        col = f"3dep-lidar-{kind}"
        items = covering(search(col, a.bbox), a.bbox)
        if dsm_id:  # use the same LiDAR project/tile for DTM and HAG as for the DSM
            same = [i for i in items if i["id"] == dsm_id.replace("-dsm-", f"-{kind}-")]
            items = same or items
        if not items:
            if kind == "dsm":
                raise SystemExit(f"no single {col} item covers this bbox; shrink or move it")
            continue
        last = None
        for it in items:  # some catalogue footprints are wrong: try candidates until one really covers the box
            try:
                info = crop(it["assets"]["data"]["href"], col, a.bbox, f"{out}/lidar_{kind}.tif")
                break
            except (SystemExit, rasterio.errors.WindowError, rasterio.errors.RasterioIOError) as e:
                last = e
        else:
            if kind == "dsm":
                raise SystemExit(f"{col}: no candidate item covers the bbox ({last})")
            print(f"warning: {col} unavailable here ({last}); will derive it", flush=True)
            continue
        if kind == "dsm":
            dsm_id = it["id"]
        meta["layers"][f"lidar_{kind}"] = dict(item=it["id"], date=it["properties"].get("datetime") or it["properties"].get("start_datetime"), **info)

    # DTM and HAG are redundant given the DSM (HAG = DSM - DTM): derive whichever one is missing
    have = {k: os.path.exists(f"{out}/lidar_{k}.tif") for k in ("dtm", "hag")}
    if not all(have.values()):
        if not any(have.values()):
            raise SystemExit("neither LiDAR DTM nor HAG covers this bbox")
        src_k, dst_k = ("hag", "dtm") if have["hag"] else ("dtm", "hag")
        with rasterio.open(f"{out}/lidar_dsm.tif") as d, rasterio.open(f"{out}/lidar_{src_k}.tif") as o:
            if d.shape != o.shape or d.transform != o.transform:
                raise SystemExit(f"cannot derive {dst_k}: DSM and {src_k} grids differ")
            a, b = d.read(1, masked=True).astype("float32"), o.read(1, masked=True).astype("float32")
            prof = d.profile
            with rasterio.open(f"{out}/lidar_{dst_k}.tif", "w", **prof) as w:
                w.write((a - b).filled(d.nodata if d.nodata is not None else -9999), 1)
        meta["layers"][f"lidar_{dst_k}"] = dict(derived=f"lidar_dsm - lidar_{src_k}")
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
