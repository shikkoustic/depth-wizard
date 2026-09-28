"""Robustness to coarser imagery: score checkpoints on GAMUS test tiles degraded to 1.3 m and 2 m
(downsample by 2x / 3x, then upsample back to the model grid — how pan-sharpened 1-2 m satellite
imagery looks to a model trained at 0.66 m).

  python bench/robustness.py runs/kaggle/small_main runs/kaggle/ablation_plain ...
Uses <run>/test_samples.npz (16 test tiles saved by the training kernel) and <run>/best.pt. CPU is fine.
"""
import json, os, sys, gc
import numpy as np, torch, torch.nn.functional as F
from transformers import AutoConfig, AutoModelForDepthEstimation

torch.set_num_threads(3)
HERE = os.path.dirname(os.path.abspath(__file__))
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
runs = sys.argv[1:]
s = np.load(os.path.join(runs[0], "test_samples.npz")); X = s["rgb"]; G = s["gt"].astype(np.float32)
out = dict(n_tiles=len(X), tiles=[str(i) for i in s["ids"]], models={})
for run in runs:
    ck = torch.load(os.path.join(run, "best.pt"), map_location="cpu", weights_only=True)
    sd, meta = (ck["state_dict"], ck.get("meta", {})) if "state_dict" in ck else (ck, {})
    variant, IN = meta.get("variant", "Small"), int(meta.get("in_size", 518))
    m = AutoModelForDepthEstimation.from_config(AutoConfig.from_pretrained(os.path.join(HERE, "..", "depthwizard", "configs", variant)))
    m.load_state_dict({k: v.float() for k, v in sd.items()}); m.eval()
    res = {}
    for f in (1, 2, 3):
        P = []
        with torch.no_grad():
            for x in X:
                t = torch.from_numpy(x).permute(2, 0, 1)[None].float() / 255.
                if f > 1:
                    t = F.interpolate(F.interpolate(t, size=(512 // f,) * 2, mode="area"), size=(512, 512), mode="bilinear", align_corners=False)
                t = (F.interpolate(t, size=(IN, IN), mode="bilinear", align_corners=False) - MEAN) / STD
                P.append(F.interpolate(m(pixel_values=t).predicted_depth[:, None], size=(512, 512), mode="bilinear", align_corners=False)[0, 0].numpy())
        P = np.stack(P); v = np.isfinite(G); e = P[v] - G[v]
        res[f"{0.66 * f:.1f}m"] = dict(rmse=float(np.sqrt((e ** 2).mean())), mae=float(np.abs(e).mean()), corr=float(np.corrcoef(P[v], G[v])[0, 1]))
    out["models"][os.path.basename(run.rstrip("/"))] = res
    print(os.path.basename(run.rstrip("/")), json.dumps({k: round(v["rmse"], 2) for k, v in res.items()}), flush=True)
    del m; gc.collect()
os.makedirs(os.path.join(HERE, "..", "docs", "data"), exist_ok=True)
json.dump(out, open(os.path.join(HERE, "..", "docs", "data", "robustness.json"), "w"), indent=1)
