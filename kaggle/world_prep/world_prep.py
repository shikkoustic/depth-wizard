# Non-US training tiles from national open-data services, in our training format
# (512x512 at 0.66 m/px, above-ground height in metres):
#   FR  IGN LiDAR HD "MNH" (height above ground, 0.5 m) + BD ORTHO, served on the same grid by one WMS
#       -> no alignment work; Etalab Open Licence 2.0; rate limit 1 request/second.
#   CH  swisstopo swissSURFACE3D raster (DSM 0.5 m) - swissALTI3D (DTM 0.5 m), with SWISSIMAGE dop10 ortho;
#       all three share the same 1 km LV95 tile grid; keyless STAC; free use with attribution "©swisstopo".
# Splits are by 0.1-degree block so neighbouring tiles never straddle train/test.
import io, json, math, os, random, time, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor
import numpy as np

OUT = os.environ.get("WORLD_OUT", "/kaggle/working"); S, GSD = 512, 0.66
N_FR = int(os.environ.get("WORLD_FR", 900)); N_CH = int(os.environ.get("WORLD_CH", 700))
T0 = time.time(); random.seed(0); np.random.seed(0)
def log(*a): print(f"[{(time.time()-T0)/60:6.1f}m]", *a, flush=True)

def fetch(url, timeout=90, tries=3):
    for k in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                return r.read()
        except Exception as e:
            if k == tries - 1:
                raise
            time.sleep(1.5 * (k + 1))

def read_tif(buf):
    import rasterio
    with rasterio.MemoryFile(buf) as m, m.open() as ds:
        return ds.read()

# ---------------- France: one WMS, both layers, identical grid ----------------
GEOPF = ("https://data.geopf.fr/wms-r?SERVICE=WMS&VERSION=1.3.0&REQUEST=GetMap&LAYERS={layer}"
         "&FORMAT=image/geotiff&CRS=EPSG:2154&BBOX={bbox}&WIDTH={w}&HEIGHT={h}&STYLES=")
MNH = "IGNF_LIDAR-HD_MNH_ELEVATION.ELEVATIONGRIDCOVERAGE.LAMB93"
ORTHO = "HR.ORTHOIMAGERY.ORTHOPHOTOS"
FR_BOXES = [  # Lambert-93 ranges: plains, Alps, Pyrenees, Massif Central, Brittany coast, Paris basin
    (600000, 6700000, 1000000, 6900000), (900000, 6400000, 1050000, 6600000),
    (400000, 6200000, 700000, 6350000), (600000, 6350000, 800000, 6550000),
    (150000, 6750000, 400000, 6900000), (600000, 6850000, 700000, 6950000)]
_last = [0.0]
BIG = 2000  # one GetMap of 2000x2000 at 0.66 m covers 1320 m and yields up to 9 tiles (the service is slow
            # and rate-limited to 1 request/second, so fetch big and cut locally)
def fr_tile(_):
    x0, y0, x1, y1 = random.choice(FR_BOXES)
    span = BIG * GSD
    x = random.uniform(x0, x1 - span); y = random.uniform(y0, y1 - span)
    bbox = f"{x:.1f},{y:.1f},{x + span:.1f},{y + span:.1f}"  # data.geopf.fr expects easting-first for EPSG:2154
    out = []
    for layer in (MNH, ORTHO):
        while time.time() - _last[0] < 1.1:  # server allows 1 request/second
            time.sleep(0.2)
        _last[0] = time.time()
        try:
            out.append(fetch(GEOPF.format(layer=layer, bbox=bbox, w=BIG, h=BIG), timeout=180))
        except Exception:
            return []
    try:
        H = read_tif(out[0])[0].astype(np.float32); RGB = read_tif(out[1])[:3]
    except Exception:
        return []
    H = np.where(H < -100, np.nan, H)
    tiles_out = []
    step = (BIG - S) // 2
    for r in range(0, BIG - S + 1, step):
        for c in range(0, BIG - S + 1, step):
            h = H[r:r + S, c:c + S]; rgb = RGB[:, r:r + S, c:c + S]
            if not np.isfinite(h).all() or np.nanmax(h) <= 0.5 or np.nanmax(h) > 250:
                continue
            if np.nanpercentile(h, 2) > 1.5 or (rgb.max(0) == 0).mean() > 0.02:
                continue
            tiles_out.append((np.moveaxis(rgb.astype(np.uint8), 0, -1), h, dict(country="FR", x=round(x + c * GSD), y=round(y + r * GSD))))
    return tiles_out

# ---------------- Switzerland: STAC, three collections on one tile grid ----------------
STAC = "https://data.geo.admin.ch/api/stac/v0.9/collections"
CH_BOXES = [(6.0, 46.2, 7.5, 47.0), (7.5, 46.0, 9.0, 46.8), (8.0, 46.8, 9.5, 47.6),
            (6.5, 46.9, 8.0, 47.5), (8.8, 46.0, 9.4, 46.5)]  # Alps, Valais, east, plateau, Ticino
def ch_items(col, bbox, limit=20):
    u = f"{STAC}/{col}/items?bbox={','.join(str(round(b,4)) for b in bbox)}&limit={limit}"
    return json.loads(fetch(u))["features"]

def pick_asset(item, want):
    for k, v in item["assets"].items():
        if want in k and k.endswith(".tif"):
            return v["href"]
    for k, v in item["assets"].items():
        if v.get("href", "").endswith(".tif") and want in v["href"]:
            return v["href"]
    return None

def ch_tile(_):
    import rasterio
    from rasterio.warp import reproject, Resampling
    lon0, lat0, lon1, lat1 = random.choice(CH_BOXES)
    lon, lat = random.uniform(lon0, lon1), random.uniform(lat0, lat1)
    bb = (lon - 0.004, lat - 0.004, lon + 0.004, lat + 0.004)
    try:
        dsm_items = ch_items("ch.swisstopo.swisssurface3d-raster", bb, 5)
        if not dsm_items: return []
        it = dsm_items[0]; tile = it["id"].split("_")[-1]  # e.g. 2682-1246
        dsm_href = pick_asset(it, "_0.5_")
        alti = [x for x in ch_items("ch.swisstopo.swissalti3d", bb, 20) if tile in x["id"]]
        img = [x for x in ch_items("ch.swisstopo.swissimage-dop10", bb, 20) if tile in x["id"]]
        if not (dsm_href and alti and img): return []
        alti_href = pick_asset(alti[0], "_0.5_"); img_href = pick_asset(img[0], "_0.1_") or pick_asset(img[0], "_2.0_")
        if not (alti_href and img_href): return []
        with rasterio.MemoryFile(fetch(dsm_href)) as m1, m1.open() as dsm, \
             rasterio.MemoryFile(fetch(alti_href)) as m2, m2.open() as dtm, \
             rasterio.MemoryFile(fetch(img_href)) as m3, m3.open() as ort:
            span = S * GSD
            l, b, r, t = dsm.bounds
            if r - l < span or t - b < span: return []
            offs = [(dx, dy) for dx in (l + 10, r - span - 10) for dy in (b + 10, t - span - 10)]
            got = []
            for x, y in offs:
                tr = rasterio.Affine(GSD, 0, x, 0, -GSD, y + span)
                a = np.full((S, S), np.nan, np.float32); g = np.full((S, S), np.nan, np.float32)
                reproject(rasterio.band(dsm, 1), a, dst_transform=tr, dst_crs=dsm.crs, resampling=Resampling.bilinear, dst_nodata=np.nan)
                reproject(rasterio.band(dtm, 1), g, dst_transform=tr, dst_crs=dtm.crs, resampling=Resampling.bilinear, dst_nodata=np.nan)
                rgb = np.zeros((3, S, S), np.uint8)
                for i in range(3):
                    reproject(rasterio.band(ort, i + 1), rgb[i], dst_transform=tr, dst_crs=ort.crs, resampling=Resampling.average)
                h = a - g
                if not np.isfinite(h).all() or np.nanmax(h) > 250 or np.nanpercentile(h, 2) > 1.5 or (rgb.max(0) == 0).mean() > 0.02:
                    continue
                got.append((np.moveaxis(rgb, 0, -1), h.astype(np.float32), dict(country="CH", tile=tile, x=round(x), y=round(y))))
        return got
    except Exception:
        return []

tiles, stats = [], {}
def gather(fn, n, workers, tag):
    got, tried = 0, 0
    with ThreadPoolExecutor(workers) as ex:
        while got < n and tried < n * 6:
            batch = list(ex.map(fn, range(workers * 2)))
            tried += len(batch)
            for group in batch:
                for r in group or []:
                    if got < n:
                        tiles.append(r); got += 1
            log(tag, got, "/", n, f"({tried} attempts, {(time.time()-T0)/60:.0f} min)")
            if (time.time() - T0) / 60 > float(os.environ.get("WORLD_MAX_MIN", 210)): 
                log(tag, "time budget reached"); break
    stats[tag] = dict(kept=got, attempts=tried)

gather(fr_tile, N_FR, 2, "FR")   # 1 request/second limit -> few workers
gather(ch_tile, N_CH, 8, "CH")

n = len(tiles)
X = np.lib.format.open_memmap(f"{OUT}/world_rgb.npy", "w+", np.uint8, (n, S, S, 3))
Y = np.lib.format.open_memmap(f"{OUT}/world_agl.npy", "w+", np.float16, (n, S, S))
for i, (a, b, _) in enumerate(tiles):
    X[i] = a; Y[i] = np.clip(b, -5, 250)
X.flush(); Y.flush()
# split by coarse spatial block so neighbours stay in the same split
def block(m):
    return f"{m['country']}_{m['x'] // 10000}_{m['y'] // 10000}"  # ~10 km blocks; neighbours stay together
blocks = sorted({block(t[2]) for t in tiles}); rng = random.Random(1); rng.shuffle(blocks)
k1, k2 = int(0.8 * len(blocks)), int(0.9 * len(blocks))
split_of = {b: ("train" if i < k1 else "val" if i < k2 else "test") for i, b in enumerate(blocks)}
split = [split_of[block(t[2])] for t in tiles]
agl = np.asarray(Y, np.float32)
json.dump(dict(n=n, gsd=GSD, size=S, split=split, tiles=[t[2] for t in tiles], stats=stats,
               countries={c: sum(1 for t in tiles if t[2]["country"] == c) for c in ("FR", "CH")},
               frac_px_over_20m=float((agl > 20).mean()), agl_mean=float(agl.mean()),
               runtime_min=(time.time() - T0) / 60), open(f"{OUT}/world_meta.json", "w"), indent=1)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
top = list(np.argsort(-np.percentile(agl.reshape(max(n, 1), -1)[:, ::64], 99, axis=1))[:4])
sel = (top + list(np.random.choice(n, min(4, n), replace=False)))[:8] if n else []
fig, ax = plt.subplots(4, 4, figsize=(16, 16))
for k, i in enumerate(sel):
    r, c = divmod(k, 2)
    ax[r, 2 * c].imshow(X[i]); ax[r, 2 * c].set_title(json.dumps(tiles[i][2])[:40])
    ax[r, 2 * c + 1].imshow(agl[i], vmin=0, vmax=40); ax[r, 2 * c + 1].set_title("height above ground (m)")
for a_ in ax.ravel(): a_.axis("off")
plt.tight_layout(); plt.savefig(f"{OUT}/world_montage.png", dpi=50)
log("DONE", n, "tiles", stats)
