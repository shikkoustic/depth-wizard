"""Write one Kaggle kernel folder per training config:
  python make_kernels.py <name> [key=value ...] [--owner USER] [--self-prep]
--self-prep inlines the GAMUS download/resize step (for accounts that can't mount the prep kernel's output)."""
import json, os, sys
args = sys.argv[1:]
owner = args[args.index("--owner") + 1] if "--owner" in args else "shikkoustic"
self_prep = "--self-prep" in args
args = [a for i, a in enumerate(args) if a != "--self-prep" and a != "--owner" and (i == 0 or args[i - 1] != "--owner")]
name, overrides = args[0], dict(a.split("=", 1) for a in args[1:])
src = open(os.path.join(os.path.dirname(__file__), "train.py")).read()
if self_prep:
    prep = open(os.path.join(os.path.dirname(__file__), "..", "gamus_prep", "gamus_prep.py")).read()
    prep = prep.replace('OUT, TMP = "earthflow/GAMUS", 512, "/kaggle/working", "/tmp/gamus"', 'OUT, TMP = "earthflow/GAMUS", 512, "/tmp/gamus_prep", "/tmp/gamus"')
    assert "/tmp/gamus_prep" in prep, "prep OUT path not found"
    prep = prep.split("# visual montage")[0]
    prep = "import os\nos.makedirs('/tmp/gamus_prep', exist_ok=True)\n" + prep
    src = "# ---- inlined GAMUS preparation (kaggle/gamus_prep/gamus_prep.py) ----\n" + prep + "\n# ---- training ----\n" + src
line = "CFG.update(" + ", ".join(f"{k}={v}" for k, v in {"name": repr(name), **overrides}.items()) + ")"
d = os.path.join(os.path.dirname(__file__), "runs", name); os.makedirs(d, exist_ok=True)
open(os.path.join(d, "train.py"), "w").write(src.replace("#@CFG@", line))
slug = f"depthwizard-train-{name}".replace("_", "-")
json.dump({"id": f"{owner}/{slug}", "title": slug, "code_file": "train.py", "language": "python", "kernel_type": "script",
           "is_private": True, "enable_gpu": True, "enable_internet": True, "dataset_sources": [], "competition_sources": [],
           "kernel_sources": [] if self_prep else ["shikkoustic/depthwizard-gamus-prep"]}, open(os.path.join(d, "kernel-metadata.json"), "w"), indent=1)
print(d, "|", line)
