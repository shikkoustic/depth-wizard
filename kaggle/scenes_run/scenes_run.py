# Build all viewer scenes on Kaggle (keeps the laptop free): LiDAR benchmark sites, GAMUS test tiles as
# PNG uploads, India Sentinel-2 demos, and a Sikkim hill town from Maxar Open Data (0.37 m).
# Inputs: depthwizard-code + depthwizard-bench-sites datasets, the GAMUS prep kernel (demo tiles) and a
# training kernel (checkpoint). Output: scenes/ (one folder per scene) ready to copy into the app's data dir.
import os, sys, glob, json, shutil, subprocess, urllib.request
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "rasterio"], check=True)
REPO = "/tmp/depth-wizard"; os.makedirs(REPO, exist_ok=True)
src = os.path.dirname(os.path.dirname(glob.glob("/kaggle/input/**/depthwizard/pipeline.py", recursive=True)[0]))  # dataset root
for d in ("depthwizard", "bench"):
    shutil.copytree(f"{src}/{d}", f"{REPO}/{d}", dirs_exist_ok=True)
sys.path.insert(0, REPO)
import numpy as np, rasterio
from PIL import Image
ck = sorted(glob.glob("/kaggle/input/**/best.pt", recursive=True))
os.environ["DEPTHWIZARD_WEIGHTS"] = os.environ.get("SCENE_WEIGHTS", ",".join(ck))
os.environ.setdefault("DEPTHWIZARD_THREADS", "4")
print("weights:", os.environ["DEPTHWIZARD_WEIGHTS"], flush=True)
from depthwizard.pipeline import process
from depthwizard.geo import read_onto_grid
OUT = "/kaggle/working/scenes"; os.makedirs(OUT, exist_ok=True)
report = {}

def run(tag, image, **kw):
    try:
        r = process(image, f"{OUT}/{tag}", progress=lambda m: None, **kw)
        report[tag] = r.get("vs_reference") or r.get("notes")
        print(tag, json.dumps(report[tag])[:200], flush=True)
    except Exception as e:
        import traceback; report[tag] = {"error": traceback.format_exc()[-600:]}; print(tag, "ERROR", report[tag]["error"], flush=True)
    json.dump(report, open("/kaggle/working/scenes_report.json", "w"), indent=1)

# 1. LiDAR benchmark sites (prediction vs LiDAR top surface)
sites = glob.glob("/kaggle/input/**/site.json", recursive=True)
for sp in sorted(sites):
    d = os.path.dirname(sp); name = os.path.basename(d)
    if name in ("nebraska_farmland", "boulder_foothills"):  # excluded: broken reference LiDAR
        continue
    site = json.load(open(sp))
    run(f"bench_{name}", f"{d}/naip.tif", reference=f"{d}/lidar_dsm.tif", title=f"{name} ({site.get('terrain','')})")

# 2. GAMUS test tiles as PNG uploads (relative DSM path), reference = LiDAR AGL
for npz in sorted(glob.glob("/kaggle/input/**/demo_*.npz", recursive=True)):
    tid = os.path.basename(npz)[5:-4]
    d = np.load(npz)
    Image.fromarray(d["rgb"]).save(f"/tmp/{tid}.png")
    a = d["agl"].astype(np.float32)
    with rasterio.open(f"/tmp/{tid}_agl.tif", "w", driver="GTiff", width=a.shape[1], height=a.shape[0], count=1, dtype="float32") as dst:
        dst.write(a, 1)
    run(f"gamus_{tid}", f"/tmp/{tid}.png", reference=f"/tmp/{tid}_agl.tif", gsd=0.33,
        title=f"GAMUS test tile {tid} (PNG, relative) vs LiDAR")

# 3. India: Sentinel-2 (10 m, terrain demo) and Maxar Open Data (0.37 m, real buildings)
sys.argv = ["x"]
for name, bbox in {"uttarakhand_mussoorie": [78.04, 30.44, 78.09, 30.48], "mumbai_south": [72.815, 18.915, 72.845, 18.945]}.items():
    try:
        subprocess.run([sys.executable, f"{REPO}/bench/fetch_s2.py", "--name", name, "--bbox", *map(str, bbox), "--out", "/tmp/india"], check=True)
        run(f"india_{name}", f"/tmp/india/{name}_s2_rgb.tif", reference=f"/tmp/india/{name}_copernicus_dsm.tif",
            title=f"India · {name.replace('_',' ')} (Sentinel-2 10 m; reference = Copernicus 30 m)")
    except Exception as e:
        report[f"india_{name}"] = {"error": repr(e)[:300]}
try:
    subprocess.run([sys.executable, f"{REPO}/bench/fetch_maxar.py", "--name", "sikkim_town", "--bbox", "88.365", "27.163", "88.385", "27.180"],
                   check=True, cwd=REPO)
    run("india_sikkim_town", f"{REPO}/runs/demos/india/sikkim_town_maxar_rgb.tif",
        title="India · Sikkim hill town (Maxar Open Data 0.37 m, CC BY-NC 4.0)")
except Exception as e:
    report["india_sikkim_town"] = {"error": repr(e)[:300]}

json.dump(report, open("/kaggle/working/scenes_report.json", "w"), indent=1)
shutil.make_archive("/kaggle/working/scenes", "zip", OUT)
shutil.rmtree(OUT)
print("DONE", list(report), flush=True)
