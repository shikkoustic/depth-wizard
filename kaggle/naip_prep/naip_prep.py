# Build an extra training set of rural / forest / hilly / arid tiles: NAIP RGB (0.6 m) + USGS 3DEP LiDAR
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
OUT = os.environ.get("NAIP_OUT", "/kaggle/working"); T0 = time.time(); S = 512; GSD = 0.66
N_ITEMS, TILES_PER_ITEM = int(os.environ.get("NAIP_ITEMS", 220)), int(os.environ.get("NAIP_TILES", 18))
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

# ---- 1. discover LiDAR tiles spread over the contiguous US ----
items, seen, tries = [], set(), 0
while len(items) < N_ITEMS * 2 and tries < 2200:
    tries += 1
    lon, lat = random.uniform(-123.5, -68.0), random.uniform(26.0, 48.5)
    try:
        fs = post(f"{STAC}/search", {"collections": ["3dep-lidar-dsm"], "bbox": [lon, lat, lon + 0.6, lat + 0.6], "limit": 5})["features"]
    except Exception:
        continue
    for f in fs:
        if f["id"] in seen or near_bench(f["bbox"]):
            continue
        seen.add(f["id"]); items.append(f)
log("candidate LiDAR tiles", len(items), "after", tries, "searches")
random.shuffle(items)
items = items[:int(N_ITEMS * 1.5) + 2]  # the pool keeps working after we have enough, so keep it bounded

FAB = "/vsicurl/https://huggingface.co/buckets/links-ads/fabdem/resolve/tiles/{g}/{t}_FABDEM_V1-2.tif"
def ns(v): return f"{'N' if v >= 0 else 'S'}{abs(v):02d}"
def ew(v): return f"{'E' if v >= 0 else 'W'}{abs(v):03d}"
def fabdem_url(lat, lon):
    la, lo = math.floor(lat / 10) * 10, math.floor(lon / 10) * 10
    return FAB.format(g=f"{ns(la)}{ew(lo)}-{ns(la + 10)}{ew(lo + 10)}_FABDEM_V1-2", t=f"{ns(lat)}{ew(lon)}")

def process_item(f):
    """Returns (item_meta, [ (rgb uint8 512x512x3, ndsm float32 512x512) ... ]) or (reason, [])."""
    try:
        dtm_item = get(f"{STAC}/collections/3dep-lidar-dtm/items/{f['id'].replace('-dsm-', '-dtm-')}")
        hag_item = get(f"{STAC}/collections/3dep-lidar-hag/items/{f['id'].replace('-dsm-', '-hag-')}")
        with rasterio.open(sign(f["assets"]["data"]["href"])) as dsm, rasterio.open(sign(dtm_item["assets"]["data"]["href"])) as dtm, \
             rasterio.open(sign(hag_item["assets"]["data"]["href"])) as hag:
            if dsm.transform != dtm.transform or dsm.shape != dtm.shape or hag.transform != dsm.transform or hag.shape != dsm.shape:
                return ("dsm/dtm/hag grids differ", [])
            # item-level QC on a coarse overview: LiDAR bare earth must agree with FABDEM (catches mislabelled tiles)
            ov = dtm.read(1, out_shape=(64, 64), masked=True).filled(np.nan).astype(np.float32)
            tb = transform_bounds(dtm.crs, "EPSG:4326", *dtm.bounds)
            lat, lon = math.floor((tb[1] + tb[3]) / 2), math.floor((tb[0] + tb[2]) / 2)
            fab = np.full((64, 64), np.nan, np.float32)
            ov_tr = dtm.transform * rasterio.Affine.scale(dtm.width / 64, dtm.height / 64)
            with rasterio.open(fabdem_url(lat, lon)) as fb:
                reproject(rasterio.band(fb, 1), fab, dst_transform=ov_tr, dst_crs=dtm.crs, resampling=Resampling.bilinear, dst_nodata=np.nan)
            v = np.isfinite(ov) & np.isfinite(fab) & (ov > -500)
            if v.sum() < 500:
                return ("no overlap with FABDEM", [])
            off = float(np.median(ov[v] - fab[v]))
            if abs(off) > 5:
                return (f"LiDAR DTM off FABDEM by {off:.0f} m", [])
            tile_px = int(round(S * GSD / abs(dsm.transform.a)))  # 338 m at 2 m -> 169 px
            out, tries = [], 0
            while len(out) < TILES_PER_ITEM and tries < TILES_PER_ITEM * 3:
                tries += 1
                r0, c0 = random.randint(0, dsm.height - tile_px), random.randint(0, dsm.width - tile_px)
                win = rasterio.windows.Window(c0, r0, tile_px, tile_px)
                a = dsm.read(1, window=win, masked=True).filled(np.nan).astype(np.float32)
                b = dtm.read(1, window=win, masked=True).filled(np.nan).astype(np.float32)
                h = hag.read(1, window=win, masked=True).filled(np.nan).astype(np.float32)
                # label = top surface above ground: some DSM products under-record canopy tops, so take HAG where it
                # exceeds DSM - DTM by 0-40 m (larger gaps are HAG spikes) — same rule as the benchmark reference
                nd = a - b
                lift = h - nd
                nd = np.where(np.isfinite(lift) & (lift > 0) & (lift <= 40), h, nd)
                if np.isnan(nd).mean() > 0.01 or np.nanmax(np.abs(nd)) > 250:
                    continue
                if np.nanpercentile(nd, 2) > 1.5 or (nd < -1).mean() > 0.03:  # DSM/DTM inconsistent (e.g. DSM offset)
                    continue
                wb = rasterio.windows.bounds(win, dsm.transform)
                geo = transform_bounds(dsm.crs, "EPSG:4326", *wb)
                naips = post(f"{STAC}/search", {"collections": ["naip"], "bbox": list(geo), "limit": 20})["features"]
                naips = [n for n in naips if n["bbox"][0] <= geo[0] and n["bbox"][1] <= geo[1] and n["bbox"][2] >= geo[2] and n["bbox"][3] >= geo[3]]
                if not naips:
                    continue
                ly = int((f["properties"].get("datetime") or f["properties"].get("start_datetime") or "2018")[:4])
                n = min(naips, key=lambda n: abs(int(n["properties"]["datetime"][:4]) - ly))  # closest year to the LiDAR
                with rasterio.open(sign(n["assets"]["image"]["href"])) as nai:
                    rgb = np.zeros((3, S, S), np.uint8)
                    dst_tr = rasterio.Affine(GSD, 0, wb[0], 0, -GSD, wb[3])
                    for i in range(3):
                        reproject(rasterio.band(nai, i + 1), rgb[i], dst_transform=dst_tr, dst_crs=dsm.crs, resampling=Resampling.average)
                if (rgb.max(0) == 0).mean() > 0.01:
                    continue
                ndS = np.zeros((S, S), np.float32)
                reproject(nd, ndS, src_transform=rasterio.windows.transform(win, dsm.transform), src_crs=dsm.crs,
                          dst_transform=dst_tr, dst_crs=dsm.crs, resampling=Resampling.bilinear)
                out.append((np.moveaxis(rgb, 0, -1), ndS, dict(naip=n["id"], naip_year=int(n["properties"]["datetime"][:4]), lidar_year=ly,
                                                              lon=(geo[0] + geo[2]) / 2, lat=(geo[1] + geo[3]) / 2)))
            relief = float(np.nanmax(ov) - np.nanmin(ov))
            return (dict(item=f["id"], dtm_offset_vs_fabdem=off, relief_m=relief, lon=(tb[0] + tb[2]) / 2, lat=(tb[1] + tb[3]) / 2), out)
    except Exception as e:
        return (f"error {type(e).__name__}: {str(e)[:120]}", [])

kept, rejected, tiles = [], {}, []
with ThreadPoolExecutor(12) as ex:
    for res, out in ex.map(process_item, items):
        if isinstance(res, str):
            rejected[res.split(" ")[0] if res.startswith("error") else res[:40]] = rejected.get(res.split(" ")[0] if res.startswith("error") else res[:40], 0) + 1
            continue
        if not out:
            continue
        res["n_tiles"] = len(out); res["tile_index"] = list(range(len(tiles), len(tiles) + len(out)))
        kept.append(res); tiles += out
        log(f"item {len(kept)}: {res['item'][:40]} relief {res['relief_m']:.0f} m, {len(out)} tiles (total {len(tiles)})")
        if len(kept) >= N_ITEMS:
            break

n = len(tiles); log("tiles", n, "from", len(kept), "LiDAR tiles; rejected:", rejected)
X = np.lib.format.open_memmap(f"{OUT}/naip_rgb.npy", "w+", np.uint8, (n, S, S, 3))
Y = np.lib.format.open_memmap(f"{OUT}/naip_agl.npy", "w+", np.float16, (n, S, S))
for i, (a, b, _) in enumerate(tiles):
    X[i] = a; Y[i] = np.clip(b, -5, 250)
X.flush(); Y.flush()
# region-level split: 80 % / 10 % / 10 % of LiDAR tiles
order = list(range(len(kept))); random.Random(1).shuffle(order)
k1, k2 = int(0.8 * len(order)), int(0.9 * len(order))
split = {}
for rank, j in enumerate(order):
    s = "train" if rank < k1 else "val" if rank < k2 else "test"
    kept[j]["split"] = s
    for t in kept[j]["tile_index"]:
        split[t] = s
meta = dict(n=n, gsd=GSD, size=S, items=kept, rejected=rejected, split=[split[i] for i in range(n)],
            tiles=[t[2] for t in tiles], agl_mean=float(np.mean([np.nanmean(t[1]) for t in tiles])),
            runtime_min=(time.time() - T0) / 60)
json.dump(meta, open(f"{OUT}/naip_meta.json", "w"), indent=1)

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sel = np.linspace(0, n - 1, 8).astype(int)
fig, ax = plt.subplots(4, 4, figsize=(16, 16))
for k, i in enumerate(sel):
    r, c = divmod(k, 2)
    ax[r, 2 * c].imshow(X[i]); ax[r, 2 * c].set_title(f"{tiles[i][2]['lat']:.2f},{tiles[i][2]['lon']:.2f}")
    im = ax[r, 2 * c + 1].imshow(np.asarray(Y[i], np.float32), vmin=0, vmax=30); ax[r, 2 * c + 1].set_title("LiDAR DSM-DTM")
for a_ in ax.ravel(): a_.axis("off")
plt.tight_layout(); plt.savefig(f"{OUT}/naip_montage.png", dpi=50)
log("DONE")
