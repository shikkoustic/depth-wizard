"""Write one Kaggle kernel folder per training config: python make_kernels.py <name> [key=value ...]"""
import json, os, sys
name, overrides = sys.argv[1], dict(a.split("=", 1) for a in sys.argv[2:])
src = open(os.path.join(os.path.dirname(__file__), "train.py")).read()
line = "CFG.update(" + ", ".join(f"{k}={v}" for k, v in {"name": repr(name), **overrides}.items()) + ")"
d = os.path.join(os.path.dirname(__file__), "runs", name); os.makedirs(d, exist_ok=True)
open(os.path.join(d, "train.py"), "w").write(src.replace("#@CFG@", line))
slug = f"depthwizard-train-{name}".replace("_", "-")
json.dump({"id": f"shikkoustic/{slug}", "title": slug, "code_file": "train.py", "language": "python", "kernel_type": "script",
           "is_private": True, "enable_gpu": True, "enable_internet": True, "dataset_sources": [], "competition_sources": [],
           "kernel_sources": ["shikkoustic/depthwizard-gamus-prep"]}, open(os.path.join(d, "kernel-metadata.json"), "w"), indent=1)
print(d, "|", line)
