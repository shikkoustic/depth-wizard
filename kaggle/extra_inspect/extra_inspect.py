# Download SynRS3D's preprocessed real datasets (DFC19 Jacksonville/Omaha, OGC Atlanta) and report layout + stats.
import os, subprocess, sys, glob, json, zipfile, tarfile
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "gdown", "rasterio"], check=True)
import gdown, numpy as np, rasterio
LINKS = {"DFC19": "1eoF16sxIHOQ5928SrboMqbi686sfKFLF", "OGC_ATL": "1tWBfrGKPbrPT1CyXp0iUm6_KItKYuiWb"}
R = {}
for name, fid in LINKS.items():
    d = f"/tmp/extra/{name}"; os.makedirs(d, exist_ok=True)
    try:
        out = gdown.download(id=fid, output=f"{d}/archive", quiet=False)
        R[name] = {"archive_mb": os.path.getsize(out) / 1e6}
        if zipfile.is_zipfile(out): zipfile.ZipFile(out).extractall(d)
        elif tarfile.is_tarfile(out): tarfile.open(out).extractall(d)
        os.remove(out)
        files = [f for f in glob.glob(f"{d}/**/*", recursive=True) if os.path.isfile(f)]
        R[name]["n_files"] = len(files)
        R[name]["dirs"] = sorted({os.path.relpath(os.path.dirname(f), d) for f in files})[:30]
        R[name]["examples"] = [os.path.relpath(f, d) for f in files[:15]]
        R[name]["txt"] = {os.path.relpath(f, d): open(f).read().splitlines()[:5] + [f"... {len(open(f).read().splitlines())} lines"] for f in files if f.endswith(".txt")}
        stats = []
        for f in [f for f in files if f.lower().endswith((".tif", ".tiff"))][:400:40]:
            with rasterio.open(f) as s:
                a = s.read(1).astype(np.float32)
                stats.append(dict(file=os.path.relpath(f, d), shape=[s.count, s.height, s.width], dtype=str(s.dtypes[0]), crs=str(s.crs),
                                  res=list(s.res), p=[float(x) for x in np.nanpercentile(a, [1, 50, 99])], max=float(np.nanmax(a))))
        R[name]["tif_stats"] = stats
    except Exception as e:
        R[name] = R.get(name, {}); R[name]["error"] = repr(e)[:500]
    json.dump(R, open("/kaggle/working/extra_inspect.json", "w"), indent=1)
    print(name, json.dumps(R[name])[:3000], flush=True)
