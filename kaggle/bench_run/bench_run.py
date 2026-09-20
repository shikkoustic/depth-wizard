# Run the full LiDAR benchmark (bench/evaluate.py) on Kaggle so it does not load the laptop.
# Sites come from the private dataset depthwizard-bench-sites; code from the public GitHub repo;
# checkpoints from the training kernels mounted as inputs.
import os, sys, glob, json, shutil, subprocess
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "rasterio"], check=True)
REPO = "/tmp/depth-wizard"
subprocess.run(["git", "clone", "--depth", "1", "https://github.com/shikkoustic/depth-wizard", REPO], check=True)
sites_src = glob.glob("/kaggle/input/depthwizard-bench-sites")[0]
os.makedirs(f"{REPO}/bench/sites", exist_ok=True)
for d in sorted(os.listdir(sites_src)):
    if os.path.isdir(f"{sites_src}/{d}"):
        shutil.copytree(f"{sites_src}/{d}", f"{REPO}/bench/sites/{d}", dirs_exist_ok=True)
print("sites:", sorted(os.listdir(f"{REPO}/bench/sites")), flush=True)

MODELS = {}  # name -> comma-separated checkpoint paths (a list = ensemble)
for ck in sorted(glob.glob("/kaggle/input/**/best.pt", recursive=True)):
    MODELS[ck.split("/kaggle/input/")[1].split("/")[-2]] = ck
ens = os.environ.get("BENCH_ENSEMBLE", "")  # e.g. "v3a_base+base_main"
if ens:
    for combo in ens.split(","):
        parts = [MODELS[p] for p in combo.split("+") if p in MODELS]
        if len(parts) > 1: MODELS["ENS_" + combo] = ",".join(parts)
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
