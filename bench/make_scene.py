"""Build a viewer scene from a benchmark site (NAIP + LiDAR), optionally with a prediction.

  python bench/make_scene.py bench/sites/sf_downtown runs/scenes/sf_ref            # LiDAR DSM only
  python bench/make_scene.py bench/sites/sf_downtown runs/scenes/sf --pred dsm.tif  # prediction vs LiDAR
"""
import argparse, os, sys
import numpy as np, rasterio
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from depthwizard.geo import read_onto_grid
from depthwizard.scene import write_scene

ap = argparse.ArgumentParser()
ap.add_argument("site"); ap.add_argument("out")
ap.add_argument("--pred", help="predicted DSM GeoTIFF")
ap.add_argument("--res", type=float, default=1.0, help="scene grid resolution in metres")
a = ap.parse_args()

with rasterio.open(f"{a.site}/naip.tif") as s:
    rgb = np.moveaxis(s.read([1, 2, 3]), 0, -1)
    l, b, r, t = s.bounds; crs = s.crs
w, h = int((r - l) // a.res), int((t - b) // a.res)
tr = rasterio.transform.from_origin(l, t, a.res, a.res)
ref = read_onto_grid(f"{a.site}/lidar_dsm.tif", tr, crs, (h, w))
if a.pred:  # prediction vs LiDAR
    layers = {"dsm": read_onto_grid(a.pred, tr, crs, (h, w)), "ref": ref}
else:       # LiDAR only: shown as the surface itself, no accuracy comparison
    layers = {"dsm": ref}
# texture must cover exactly the grid extent: crop NAIP to w*res x h*res metres
px = s_res = rasterio.open(f"{a.site}/naip.tif").res[0]
rgb = rgb[: round(h * a.res / px), : round(w * a.res / px)]
title = os.path.basename(a.site) + ("" if a.pred else " — LiDAR DSM (ground truth, no prediction)")
notes = ["reference = USGS 3DEP LiDAR DSM (NAVD88)"] if a.pred else ["surface = USGS 3DEP LiDAR DSM (NAVD88); this scene shows measured data, not a prediction"]
m = write_scene(a.out, rgb, layers, (a.res, a.res), crs=crs, transform=tr, title=title, notes=notes)
print(m["width"], m["height"], {k: v["stats"] for k, v in m["layers"].items()})
