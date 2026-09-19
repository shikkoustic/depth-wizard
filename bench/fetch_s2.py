"""Cut a cloud-free Sentinel-2 true-colour GeoTIFF (10 m, 8-bit RGB) for a lon/lat box from
Microsoft Planetary Computer, plus the Copernicus GLO-30 DSM on the same footprint as a
(30 m, radar) reference. Used for India demo scenes; Sentinel-2 is far coarser than the
sub-metre imagery the height model is built for, so these scenes mainly show terrain.

  python bench/fetch_s2.py --name uttarakhand_mussoorie --bbox 78.03 30.44 78.08 30.48
"""
import argparse, json, os, sys
import numpy as np, rasterio
from rasterio.warp import transform_bounds
sys.path.insert(0, os.path.dirname(__file__))
from fetch_site import search, sign, crop  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True)
ap.add_argument("--bbox", nargs=4, type=float, required=True)
ap.add_argument("--dates", default="2023-10-01/2024-04-30")
ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "runs", "demos", "india"))
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
items = search("sentinel-2-l2a", a.bbox, datetime=a.dates, query={"eo:cloud_cover": {"lt": 3}})
x0, y0, x1, y1 = a.bbox
items = [i for i in items if i["bbox"][0] <= x0 and i["bbox"][1] <= y0 and i["bbox"][2] >= x1 and i["bbox"][3] >= y1]
if not items:
    raise SystemExit("no cloud-free Sentinel-2 scene fully covers the bbox")
it = min(items, key=lambda i: i["properties"]["eo:cloud_cover"])
info = crop(it["assets"]["visual"]["href"], "sentinel-2-l2a", a.bbox, f"{a.out}/{a.name}_s2_rgb.tif")
cop = search("cop-dem-glo-30", a.bbox)[0]
crop(cop["assets"]["data"]["href"], "cop-dem-glo-30", a.bbox, f"{a.out}/{a.name}_copernicus_dsm.tif")
meta = dict(name=a.name, bbox=a.bbox, s2_item=it["id"], date=it["properties"]["datetime"], cloud=it["properties"]["eo:cloud_cover"], **info)
json.dump(meta, open(f"{a.out}/{a.name}.json", "w"), indent=1)
print(json.dumps(meta))
