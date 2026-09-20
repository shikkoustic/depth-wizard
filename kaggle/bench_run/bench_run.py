# Run the full LiDAR benchmark (bench/evaluate.py) on Kaggle so it does not load the laptop.
# Sites come from the private dataset depthwizard-bench-sites; code from the public GitHub repo;
# checkpoints from the training kernels mounted as inputs.
import os, sys, glob, json, shutil, subprocess
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "rasterio"], check=True)
REPO = "/tmp/depth-wizard"  # code comes from the private dataset depthwizard-code (the repo is private)
os.makedirs(REPO, exist_ok=True)
src = os.path.dirname(os.path.dirname(glob.glob("/kaggle/input/**/depthwizard/pipeline.py", recursive=True)[0]))  # dataset root
for d in ("depthwizard", "bench"):
    shutil.copytree(f"{src}/{d}", f"{REPO}/{d}", dirs_exist_ok=True)
site_jsons = glob.glob("/kaggle/input/**/site.json", recursive=True)
os.makedirs(f"{REPO}/bench/sites", exist_ok=True)
for sj in site_jsons:
    d = os.path.dirname(sj)
    shutil.copytree(d, f"{REPO}/bench/sites/{os.path.basename(d)}", dirs_exist_ok=True)
print("sites:", sorted(os.listdir(f"{REPO}/bench/sites")), flush=True)

MODELS = {}  # name -> comma-separated checkpoint paths (a list = ensemble)
for ck in sorted(glob.glob("/kaggle/input/**/best.pt", recursive=True)):
    MODELS[ck.split("/kaggle/input/")[1].split("/")[-2]] = ck
# ensembles: averaging models that fail in different places (v3 is better in cities, the older ones on
# farmland/forest). Names are matched as substrings of the mounted checkpoint folders.
ENSEMBLES = ["v3a-base+base-main", "v3a-base+v3a-small", "v3a-base+mixed-base", "v3a-base+base-main+v3a-small"]
def find(key):
    hits = [v for k, v in MODELS.items() if key in k]
    return hits[0] if hits else None
for combo in ENSEMBLES:
    parts = [find(p) for p in combo.split("+")]
    if all(parts): MODELS["ENS_" + combo] = ",".join(parts)
KEEP = os.environ.get("BENCH_ONLY", "ENS_")  # single models were already scored in version 4
MODELS = {k: v for k, v in MODELS.items() if KEEP in k}
print("models:", list(MODELS), flush=True)

for name, weights in MODELS.items():
    out = f"/kaggle/working/{name}"
    env = dict(os.environ, DEPTHWIZARD_WEIGHTS=weights, PYTHONPATH=REPO, DEPTHWIZARD_THREADS="4")
    r = subprocess.run([sys.executable, f"{REPO}/bench/evaluate.py", "--out", out, "--tta", os.environ.get("BENCH_TTA", "4")],
                       cwd=REPO, env=env, capture_output=True, text=True)
    print(name, r.returncode, r.stdout[-2000:], r.stderr[-800:], flush=True)
    if os.path.exists(f"{out}/bench_results.json"):
        shutil.copy(f"{out}/bench_results.json", f"/kaggle/working/bench_{name}.json")
    shutil.rmtree(out, ignore_errors=True)
print("DONE", flush=True)
