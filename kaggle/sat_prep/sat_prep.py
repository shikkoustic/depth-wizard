# Satellite training data with LiDAR heights: DFC2019 (WorldView-3; Jacksonville, Omaha) and Overhead Geopose
# Atlanta (WorldView; tall downtown), as repackaged by the SynRS3D project (github.com/JTRNEO/SynRS3D, MIT code;
# datasets CC-BY / CC-BY-SA). Converted to our training format: 512x512 tiles at 0.66 m/px, above-ground height (m).
#
# The repackaged files carry no pixel size, so the ground sampling distance is ESTIMATED from the data
# (median footprint of detached-house-sized buildings ~ 180 m2) and logged; tiles are rebuilt from their
# 512 px crops, resampled to 0.66 m and re-cut. Splits are by original scene (never by crop).
import os, re, sys, glob, json, time, random, zipfile, tarfile, subprocess
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "gdown", "rasterio", "scipy"], check=True)
import numpy as np, gdown, rasterio
from scipy import ndimage
import torch, torch.nn.functional as F

OUT, S, GSD = "/kaggle/working", 512, 0.66
MAX_TILES = int(os.environ.get("SAT_MAX_TILES", 1800))  # per dataset
T0 = time.time(); random.seed(0); np.random.seed(0)
def log(*a): print(f"[{(time.time()-T0)/60:6.1f}m]", *a, flush=True)
LINKS = {"DFC19": "1eoF16sxIHOQ5928SrboMqbi686sfKFLF", "OGC_ATL": "1tWBfrGKPbrPT1CyXp0iUm6_KItKYuiWb"}

def fetch(name):
    d = f"/tmp/sat/{name}"; os.makedirs(d, exist_ok=True)
    out = gdown.download(id=LINKS[name], output=f"{d}/archive", quiet=True)
    if zipfile.is_zipfile(out): zipfile.ZipFile(out).extractall(d)
    else: tarfile.open(out).extractall(d)
    os.remove(out); return d

def read(p):
    with rasterio.open(p) as s:
        a = s.read()
    return a

def to_u8(a):  # (C,H,W) any dtype -> uint8 RGB
    a = a[:3].astype(np.float32)
    if a.max() <= 255 and a.min() >= 0: return a.astype(np.uint8)
    lo, hi = np.percentile(a, 1), np.percentile(a, 99.5)
    return np.clip((a - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)

def estimate_gsd(heights):
    """Median area (px) of isolated 3-12 m high blobs of house size; suburban detached houses ~180 m2."""
    areas = []
    for h in heights:
        m = (h > 3) & (h < 12)
        lab, n = ndimage.label(m)
        if n == 0: continue
        a = ndimage.sum(m, lab, range(1, n + 1))
        areas += [x for x in a if 40 < x < 20000]
    if len(areas) < 50: return None, len(areas)
    return float(np.sqrt(180.0 / np.median(areas))), len(areas)

R = {}
X_all, Y_all, meta_tiles, split_all = [], [], [], []
for name in ["DFC19", "OGC_ATL"]:
    try:
        d = fetch(name); log(name, "extracted")
        nd_files = sorted(glob.glob(f"{d}/**/gt_nDSM/*.tif", recursive=True))
        # group crops by original scene: <SCENE>_<row>_<col>.tif
        groups = {}
        for f in nd_files:
            m = re.match(r"(.+)_(\d+)_(\d+)\.tif$", os.path.basename(f))
            if m: groups.setdefault(m.group(1), []).append((int(m.group(2)), int(m.group(3)), f))
        scenes = sorted(groups)
        sample_h = [read(groups[s][0][2])[0] for s in random.sample(scenes, min(300, len(scenes)))]
        gsd, n_blobs = estimate_gsd(sample_h)
        log(name, "scenes", len(scenes), "estimated GSD", gsd, "from", n_blobs, "blobs")
        R[name] = dict(scenes=len(scenes), crops=len(nd_files), gsd_estimate=gsd, n_blobs=n_blobs)
        if gsd is None: continue
        # splits by scene
        test_ids = set()
        tl = glob.glob(f"{d}/**/test_95.txt", recursive=True) or glob.glob(f"{d}/**/test.txt", recursive=True)
        if tl:
            for line in open(tl[0]):
                m = re.match(r"(.+)_(\d+)_(\d+)", line.strip())
                if m: test_ids.add(m.group(1))
        rng = random.Random(1)
        split = {}
        for s in scenes:
            if test_ids: split[s] = "test" if s in test_ids else ("val" if rng.random() < 0.1 else "train")
            else: r = rng.random(); split[s] = "test" if r < 0.1 else "val" if r < 0.2 else "train"
        n_out = 0
        order = scenes[:]; rng.shuffle(order)
        for s in order:
            if n_out >= MAX_TILES: break
            crops = groups[s]
            H = max(r for r, c, f in crops) + 512; W = max(c for r, c, f in crops) + 512
            img = np.zeros((3, H, W), np.uint8); hgt = np.full((H, W), np.nan, np.float32); have = np.zeros((H, W), bool)
            for r, c, f in crops:
                of = f.replace("gt_nDSM", "opt")
                if not os.path.exists(of):
                    cand = glob.glob(of.rsplit(".", 1)[0] + ".*")
                    if not cand: continue
                    of = cand[0]
                img[:, r:r + 512, c:c + 512] = to_u8(read(of)); hgt[r:r + 512, c:c + 512] = read(f)[0]; have[r:r + 512, c:c + 512] = True
            if have.mean() < 0.99: continue
            f_scale = gsd / GSD  # resample to 0.66 m
            nh, nw = int(round(H * f_scale)), int(round(W * f_scale))
            ti = F.interpolate(torch.from_numpy(img)[None].float(), size=(nh, nw), mode="area" if f_scale < 1 else "bilinear", align_corners=None if f_scale < 1 else False)[0]
            th = torch.from_numpy(np.nan_to_num(hgt, nan=0.0))[None, None]
            th = F.interpolate(th, size=(nh, nw), mode="area" if f_scale < 1 else "bilinear", align_corners=None if f_scale < 1 else False)[0, 0]
            ti, th = ti.round().clamp(0, 255).byte().numpy(), th.numpy()
            for r0 in range(0, nh - S + 1, S):
                for c0 in range(0, nw - S + 1, S):
                    X_all.append(np.moveaxis(ti[:, r0:r0 + S, c0:c0 + S], 0, -1)); Y_all.append(th[r0:r0 + S, c0:c0 + S].astype(np.float16))
                    meta_tiles.append(dict(dataset=name, scene=s)); split_all.append(split[s]); n_out += 1
            if nh < S or nw < S:  # scene smaller than one tile at 0.66 m: pad by reflection
                pi = np.pad(ti, ((0, 0), (0, max(0, S - nh)), (0, max(0, S - nw))), mode="reflect")[:, :S, :S]
                ph = np.pad(th, ((0, max(0, S - nh)), (0, max(0, S - nw))), mode="reflect")[:S, :S]
                X_all.append(np.moveaxis(pi, 0, -1)); Y_all.append(ph.astype(np.float16)); meta_tiles.append(dict(dataset=name, scene=s, padded=True))
                split_all.append(split[s]); n_out += 1
        R[name]["tiles"] = n_out
        R[name]["split"] = {k: sum(1 for t, sp in zip(meta_tiles, split_all) if t["dataset"] == name and sp == k) for k in ("train", "val", "test")}
        log(name, R[name])
        subprocess.run(["rm", "-rf", d])
    except Exception as e:
        import traceback; R[name] = dict(R.get(name, {}), error=traceback.format_exc()[-1500:]); log(name, "ERROR", R[name]["error"])

n = len(X_all)
X = np.lib.format.open_memmap(f"{OUT}/sat_rgb.npy", "w+", np.uint8, (n, S, S, 3))
Y = np.lib.format.open_memmap(f"{OUT}/sat_agl.npy", "w+", np.float16, (n, S, S))
for i in range(n): X[i] = X_all[i]; Y[i] = Y_all[i]
X.flush(); Y.flush()
agl = np.asarray(Y, np.float32)
json.dump(dict(n=n, gsd=GSD, size=S, split=split_all, tiles=meta_tiles, datasets=R,
               frac_px_over_40m=float((agl > 40).mean()), runtime_min=(time.time() - T0) / 60), open(f"{OUT}/sat_meta.json", "w"), indent=1)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sel = [i for i in np.argsort(-np.percentile(agl.reshape(n, -1)[:, ::64], 99, axis=1))[:4]] + list(np.random.choice(n, 4, replace=False))
fig, ax = plt.subplots(4, 4, figsize=(16, 16))
for k, i in enumerate(sel):
    r, c = divmod(k, 2)
    ax[r, 2 * c].imshow(X[i]); ax[r, 2 * c].set_title(f"{meta_tiles[i]['dataset']} {meta_tiles[i]['scene']}")
    ax[r, 2 * c + 1].imshow(agl[i], vmin=0, vmax=60); ax[r, 2 * c + 1].set_title("nDSM (m)")
for a_ in ax.ravel(): a_.axis("off")
plt.tight_layout(); plt.savefig(f"{OUT}/sat_montage.png", dpi=50)
log("DONE", n, "tiles", R)
