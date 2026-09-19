"""Replace the 'ref' layer of benchmark scenes with the corrected LiDAR top surface
(DTM + HAG where it lies 0-40 m above the 3DEP DSM product; the DSM elsewhere) — same rule as evaluate.py."""
import glob, json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from depthwizard.geo import read_onto_grid
from rasterio.transform import Affine
for sc in sorted(glob.glob("runs/scenes/bench_*")):
    site = f"bench/sites/{os.path.basename(sc)[6:]}"
    m = json.load(open(f"{sc}/scene.json")); W, H = m["width"], m["height"]
    tr, crs = Affine(*m["transform"]), m["crs"]
    dsm, dtm, hag = (read_onto_grid(f"{site}/lidar_{k}.tif", tr, crs, (H, W)) for k in ("dsm", "dtm", "hag"))
    top = dtm + hag; lift = top - dsm
    ref = np.where(np.isfinite(lift) & (lift > 0) & (lift <= 40), top, dsm).astype("<f4")
    ref.tofile(f"{sc}/ref.f32")
    v = ref[np.isfinite(ref)]
    m["layers"]["ref"]["stats"] = dict(min=float(v.min()), max=float(v.max()), mean=float(v.mean()),
                                       p2=float(np.percentile(v, 2)), p98=float(np.percentile(v, 98)))
    m["notes"] = [n for n in m.get("notes", []) if not n.startswith("reference")] + [
        "reference = USGS 3DEP LiDAR top surface (DSM, or DTM + HAG where the DSM product under-records tops); NAVD88"]
    json.dump(m, open(f"{sc}/scene.json", "w"), indent=1)
    print(os.path.basename(sc), "ref updated")
