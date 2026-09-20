# Build an extra training set of US DOWNTOWN tiles (tall buildings) — adapted from naip_prep.py;
# ground under towers comes from FABDEM where the LiDAR DTM keeps building remnants.
# (original header:) Build an extra training set of rural / forest / hilly / arid tiles: NAIP RGB (0.6 m) + USGS 3DEP LiDAR
# (DSM - DTM, 2 m) from Microsoft Planetary Computer, resampled to the GAMUS training format
# (512 x 512 tiles at 0.66 m/px, above-ground height in metres). Purpose: GAMUS covers only three
# eastern US cities; the model over-predicts rural vegetation height (measured on our benchmark).
#
# Integrity rules:
#  - LiDAR tiles within 0.5 deg of any benchmark site are never used (the benchmark stays held out)
#  - split by LiDAR tile (not by image tile): whole regions are train, val or test
#  - quality checks drop broken catalogue entries (LiDAR DTM far from FABDEM, DSM-DTM inconsistent)
import os, json, time, random, math, urllib.request, traceback
from concurrent.futures import ThreadPoolExecutor
import numpy as np, rasterio
from rasterio.warp import transform_bounds, reproject, Resampling
from rasterio.windows import from_bounds

for k, v in (("GDAL_HTTP_TIMEOUT", "60"), ("GDAL_HTTP_MAX_RETRY", "4"), ("GDAL_HTTP_RETRY_DELAY", "2"),
             ("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR"), ("VSI_CACHE", "TRUE")):
    os.environ.setdefault(k, v)
OUT = os.environ.get("URBAN_OUT", "/kaggle/working"); T0 = time.time(); S = 512; GSD = 0.66
N_ITEMS, TILES_PER_ITEM = int(os.environ.get("NAIP_ITEMS", 90)), int(os.environ.get("NAIP_TILES", 14))
BENCH = [(-122.400, 37.790), (-80.005, 40.444), (-104.993, 39.746), (-97.185, 38.154), (-96.935, 40.884),
         (-98.275, 29.264), (-105.275, 39.979), (-105.525, 39.824), (-83.165, 37.414), (-83.510, 35.604),
         (-77.555, 41.324), (-72.715, 43.894), (-119.915, 39.034)]
STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"
random.seed(0); np.random.seed(0)
def log(*a): print(f"[{(time.time()-T0)/60:6.1f}m]", *a, flush=True)

def _retry(fn, tries=4):
    for k in range(tries):
        try:
            return fn()
        except Exception:
            if k == tries - 1:
                raise
            time.sleep(2 * (k + 1))

def post(url, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return _retry(lambda: json.load(urllib.request.urlopen(req, timeout=120)))  # the STAC API returns the odd 502
def get(url): return _retry(lambda: json.load(urllib.request.urlopen(url, timeout=120)))
_tok = {}
def sign(href):
    acct, cont = href.split("//")[1].split(".")[0], href.split("blob.core.windows.net/")[1].split("/")[0]
    if (acct, cont) not in _tok:
        _tok[(acct, cont)] = get(f"https://planetarycomputer.microsoft.com/api/sas/v1/token/{acct}/{cont}")["token"]
    return f"{href}?{_tok[(acct, cont)]}"

def near_bench(bb):
    cx, cy = (bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2
    return any(abs(cx - x) < 0.5 + (bb[2] - bb[0]) / 2 and abs(cy - y) < 0.5 + (bb[3] - bb[1]) / 2 for x, y in BENCH)

FAB = "/vsicurl/https://huggingface.co/buckets/links-ads/fabdem/resolve/tiles/{g}/{t}_FABDEM_V1-2.tif"
def ns(v): return f"{'N' if v >= 0 else 'S'}{abs(v):02d}"
def ew(v): return f"{'E' if v >= 0 else 'W'}{abs(v):03d}"
def fabdem_url(lat, lon):
    la, lo = math.floor(lat / 10) * 10, math.floor(lon / 10) * 10
    return FAB.format(g=f"{ns(la)}{ew(lo)}-{ns(la + 10)}{ew(lo + 10)}_FABDEM_V1-2", t=f"{ns(lat)}{ew(lon)}")


CITIES = {  # downtown centres with USGS 3DEP coverage on Planetary Computer (checked 2026-09-20)
    "houston": (-95.366, 29.758), "dallas": (-96.800, 32.780), "boston": (-71.058, 42.357), "charlotte": (-80.843, 35.227),
    "nashville": (-86.781, 36.162), "kansas_city": (-94.583, 39.100), "oklahoma_city": (-97.515, 35.468),
    "salt_lake": (-111.891, 40.763), "las_vegas": (-115.172, 36.114), "austin": (-97.743, 30.268), "detroit": (-83.046, 42.331),
    "st_louis": (-90.190, 38.627), "indianapolis": (-86.158, 39.768), "milwaukee": (-87.906, 43.039), "baltimore": (-76.612, 39.290),
    "new_orleans": (-90.071, 29.951), "tampa": (-82.458, 27.950), "jacksonville": (-81.656, 30.329), "san_diego": (-117.161, 32.716)}
SPLIT = {"dallas": "test", "charlotte": "test", "nashville": "val", "st_louis": "val"}
PER_CITY = int(os.environ.get("URBAN_TILES", 110)); RADIUS_DEG = 0.02  # ~2.2 km around the centre

def city_tiles(name):
    cx, cy = CITIES[name]
    out, reasons = [], {}
    rng = random.Random(sum(map(ord, name)))  # deterministic per city
    items = post(f"{STAC}/search", {"collections": ["3dep-lidar-dsm"], "bbox": [cx - RADIUS_DEG, cy - RADIUS_DEG, cx + RADIUS_DEG, cy + RADIUS_DEG], "limit": 20})["features"]
    for f in items:
        try:
            dtm_item = get(f"{STAC}/collections/3dep-lidar-dtm/items/{f['id'].replace('-dsm-', '-dtm-')}")
            hag_item = get(f"{STAC}/collections/3dep-lidar-hag/items/{f['id'].replace('-dsm-', '-hag-')}")
        except Exception as e:
            reasons["missing dtm/hag"] = reasons.get("missing dtm/hag", 0) + 1; continue
        with rasterio.open(sign(f["assets"]["data"]["href"])) as dsm, rasterio.open(sign(dtm_item["assets"]["data"]["href"])) as dtm, \
             rasterio.open(sign(hag_item["assets"]["data"]["href"])) as hag:
            same_grid = dsm.transform == dtm.transform and dsm.shape == dtm.shape and hag.transform == dsm.transform and hag.shape == dsm.shape
            def read_on(ds, win):  # read a layer on the DSM window grid (resampled if its own grid differs)
                if same_grid:
                    return ds.read(1, window=win, masked=True).filled(np.nan).astype(np.float32)
                out = np.full((win.height, win.width), np.nan, np.float32)
                reproject(rasterio.band(ds, 1), out, dst_transform=rasterio.windows.transform(win, dsm.transform), dst_crs=dsm.crs,
                          resampling=Resampling.bilinear, src_nodata=ds.nodata, dst_nodata=np.nan)
                return out
            tile_px = int(round(S * GSD / abs(dsm.transform.a)))
            cxy = rasterio.warp.transform("EPSG:4326", dsm.crs, [cx], [cy]); crow, ccol = dsm.index(cxy[0][0], cxy[1][0])
            rad = int(RADIUS_DEG * 111000 / abs(dsm.transform.a))
            tries = 0
            while len(out) < PER_CITY and tries < PER_CITY * 4:
                tries += 1
                r0 = min(max(0, crow + rng.randint(-rad, rad) - tile_px // 2), dsm.height - tile_px)
                c0 = min(max(0, ccol + rng.randint(-rad, rad) - tile_px // 2), dsm.width - tile_px)
                win = rasterio.windows.Window(c0, r0, tile_px, tile_px)
                a = dsm.read(1, window=win, masked=True).filled(np.nan).astype(np.float32)
                b = read_on(dtm, win); h = read_on(hag, win)
                if np.isnan(a).mean() > 0.01 or np.isnan(b).mean() > 0.3:
                    reasons["nodata"] = reasons.get("nodata", 0) + 1; continue
                wb = rasterio.windows.bounds(win, dsm.transform); wtr = rasterio.windows.transform(win, dsm.transform)
                geo = transform_bounds(dsm.crs, "EPSG:4326", *wb)
                fab = np.full(a.shape, np.nan, np.float32)
                with rasterio.open(fabdem_url(math.floor((geo[1] + geo[3]) / 2), math.floor((geo[0] + geo[2]) / 2))) as fb:
                    reproject(rasterio.band(fb, 1), fab, dst_transform=wtr, dst_crs=dsm.crs, resampling=Resampling.bilinear, dst_nodata=np.nan)
                off = float(np.nanmedian(b - fab))
                if np.isfinite(off) and abs(off) > 5:
                    # some older projects store elevations in US survey feet: check the ratio before giving up
                    ratio = float(np.nanmedian(b / np.where(np.abs(fab) > 20, fab, np.nan)))
                    if 3.1 < ratio < 3.45:
                        a, b, h = a / 3.2808399, b / 3.2808399, h / 3.2808399
                        off = float(np.nanmedian(b - fab))
                        reasons["converted from feet"] = reasons.get("converted from feet", 0) + 1
                if not np.isfinite(off) or abs(off) > 5:
                    reasons["DTM off FABDEM"] = reasons.get("DTM off FABDEM", 0) + 1; continue
                # robust ground: LiDAR DTM unless it sits > 4 m above FABDEM (building remnants under towers)
                ground = np.where(~np.isfinite(b) | (b - (fab + off) > 4), fab + off, b)  # also fill DTM holes under large roofs
                lift = (b + h) - a
                top = np.where(np.isfinite(lift) & (lift > 0) & (lift <= 40), b + h, a)  # same top-surface rule as the benchmark
                top = np.where(np.isfinite(top), top, a)
                nd = top - ground
                if np.nanmax(nd) > 450 or (nd < -2).mean() > 0.03:
                    reasons["implausible"] = reasons.get("implausible", 0) + 1; continue
                naips = post(f"{STAC}/search", {"collections": ["naip"], "bbox": list(geo), "limit": 20})["features"]
                naips = [n for n in naips if n["bbox"][0] <= geo[0] and n["bbox"][1] <= geo[1] and n["bbox"][2] >= geo[2] and n["bbox"][3] >= geo[3]]
                if not naips:
                    reasons["no naip"] = reasons.get("no naip", 0) + 1; continue
                ly = int((f["properties"].get("datetime") or f["properties"].get("start_datetime") or "2018")[:4])
                n = min(naips, key=lambda n: abs(int(n["properties"]["datetime"][:4]) - ly))
                dst_tr = rasterio.Affine(GSD, 0, wb[0], 0, -GSD, wb[3])
                with rasterio.open(sign(n["assets"]["image"]["href"])) as nai:
                    rgb = np.zeros((3, S, S), np.uint8)
                    for i in range(3):
                        reproject(rasterio.band(nai, i + 1), rgb[i], dst_transform=dst_tr, dst_crs=dsm.crs, resampling=Resampling.average)
                if (rgb.max(0) == 0).mean() > 0.01:
                    continue
                ndS = np.zeros((S, S), np.float32)
                reproject(nd, ndS, src_transform=wtr, src_crs=dsm.crs, dst_transform=dst_tr, dst_crs=dsm.crs, resampling=Resampling.bilinear)
                out.append((np.moveaxis(rgb, 0, -1), ndS, dict(city=name, naip=n["id"], lidar=f["id"], naip_year=int(n["properties"]["datetime"][:4]),
                                                              lidar_year=ly, p99=float(np.nanpercentile(nd, 99)))))
        if len(out) >= PER_CITY:
            break
    return name, out, reasons

tiles, rej = [], {}
with ThreadPoolExecutor(8) as ex:
    for name, out, reasons in ex.map(city_tiles, list(CITIES)):
        tiles += out
        for k, v in reasons.items(): rej[k] = rej.get(k, 0) + v
        log(f"{name}: {len(out)} tiles, p99 heights {[round(t[2]['p99']) for t in out][:8]}, rejected {reasons}")
n = len(tiles)
X = np.lib.format.open_memmap(f"{OUT}/urban_rgb.npy", "w+", np.uint8, (n, S, S, 3))
Y = np.lib.format.open_memmap(f"{OUT}/urban_agl.npy", "w+", np.float16, (n, S, S))
for i, (a, b, _) in enumerate(tiles):
    X[i] = a; Y[i] = np.clip(b, -5, 450)
X.flush(); Y.flush()
split = [SPLIT.get(t[2]["city"], "train") for t in tiles]
agl = np.asarray(Y, np.float32)
json.dump(dict(n=n, gsd=GSD, size=S, split=split, tiles=[t[2] for t in tiles], rejected=rej,
               frac_px_over_40m=float((agl > 40).mean()), frac_px_over_100m=float((agl > 100).mean()),
               runtime_min=(time.time() - T0) / 60), open(f"{OUT}/urban_meta.json", "w"), indent=1)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
order = np.argsort([-t[2]["p99"] for t in tiles])[:8]
fig, ax = plt.subplots(4, 4, figsize=(16, 16))
for k, i in enumerate(order):
    r, c = divmod(k, 2)
    ax[r, 2 * c].imshow(X[i]); ax[r, 2 * c].set_title(tiles[i][2]["city"])
    im = ax[r, 2 * c + 1].imshow(agl[i], vmin=0, vmax=150); ax[r, 2 * c + 1].set_title(f"nDSM p99 {tiles[i][2]['p99']:.0f} m")
for a_ in ax.ravel(): a_.axis("off")
plt.tight_layout(); plt.savefig(f"{OUT}/urban_montage.png", dpi=50)
log("DONE", n, "tiles; >40 m px:", float((agl > 40).mean()))
