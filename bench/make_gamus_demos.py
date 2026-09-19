"""Turn the GAMUS test tiles saved by the prep kernel (demo_*.npz) into upload-ready demo inputs:
<id>.png (RGB, no georeference, 0.33 m/px) and <id>_lidar_agl.tif (LiDAR above-ground height, same grid)."""
import glob, os, sys
import numpy as np, rasterio
from PIL import Image
src, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
for f in sorted(glob.glob(f"{src}/demo_*.npz")):
    d = np.load(f); tid = os.path.basename(f)[5:-4]
    Image.fromarray(d["rgb"]).save(f"{out}/{tid}.png")
    a = d["agl"].astype(np.float32)
    with rasterio.open(f"{out}/{tid}_lidar_agl.tif", "w", driver="GTiff", width=a.shape[1], height=a.shape[0], count=1, dtype="float32") as dst:
        dst.write(a, 1)
    print(tid, d["rgb"].shape, "agl p50/p99 %.1f/%.1f" % tuple(np.nanpercentile(a, [50, 99])))
