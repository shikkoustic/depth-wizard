"""Fetch every benchmark site listed in bench/sites.txt (name terrain W S E N), skipping ones already present."""
import os, subprocess, sys
here = os.path.dirname(os.path.abspath(__file__))
for line in open(os.path.join(here, "sites.txt")):
    if not line.strip() or line.startswith("#"):
        continue
    name, terrain, *bbox = line.split()
    if os.path.exists(os.path.join(here, "sites", name, "site.json")):
        print(name, "already fetched"); continue
    try:
        r = subprocess.run([sys.executable, os.path.join(here, "fetch_site.py"), "--name", name, "--terrain", terrain, "--bbox", *bbox],
                           capture_output=True, text=True, timeout=420)
        print(name, "ok" if r.returncode == 0 else "FAILED: " + (r.stderr or r.stdout).strip().splitlines()[-1][:200], flush=True)
    except subprocess.TimeoutExpired:
        print(name, "TIMEOUT", flush=True)
