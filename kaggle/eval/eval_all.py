# Score every trained checkpoint mounted under /kaggle/input on the held-out NAIP+3DEP regions
# (rural / forest / hilly / arid tiles never used for training or model selection).
# CPU is enough: ~150 tiles x 1 forward pass each.
import glob, json, os, time
import numpy as np, torch, torch.nn.functional as F
from transformers import AutoConfig, AutoModelForDepthEstimation

T0 = time.time(); dev = "cuda" if torch.cuda.is_available() else "cpu"; torch.set_num_threads(4)
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
SETS = {}
for name in ("naip", "urban", "sat"):
    mp = glob.glob(f"/kaggle/input/**/{name}_meta.json", recursive=True)
    if not mp: continue
    ND = os.path.dirname(mp[0]); M = json.load(open(mp[0])); sp = np.array(M["split"])
    ite = np.flatnonzero(sp == "test")
    if not len(ite): continue
    SETS[name] = (np.load(f"{ND}/{name}_rgb.npy", mmap_mode="r")[ite], np.load(f"{ND}/{name}_agl.npy", mmap_mode="r")[ite].astype(np.float32))
    print(name, "test tiles", len(ite), flush=True)

def stats(p, g, m=None):
    v = np.isfinite(g) if m is None else (m & np.isfinite(g))
    e = p[v] - g[v]
    return dict(rmse=float(np.sqrt((e ** 2).mean())), mae=float(np.abs(e).mean()), bias=float(e.mean()),
                corr=float(np.corrcoef(p[v], g[v])[0, 1]) if p[v].std() > 0 else None, n_px=int(v.sum()))

R = dict(sets={k: dict(n_tiles=len(v[0]), zero=stats(np.zeros_like(v[1]), v[1])) for k, v in SETS.items()}, models={})
for ck in sorted(glob.glob("/kaggle/input/**/best.pt", recursive=True)):
    obj = torch.load(ck, map_location="cpu")
    meta = obj.get("meta", {}) if isinstance(obj, dict) and "meta" in obj else {}
    sd = obj["state_dict"] if isinstance(obj, dict) and "state_dict" in obj else obj
    variant = meta.get("variant") or ("Base" if any("encoder.layer.11" in k for k in sd) and sd["backbone.embeddings.cls_token"].shape[-1] == 768 else "Small")
    IN = int(meta.get("in_size", 518))
    m = AutoModelForDepthEstimation.from_config(AutoConfig.from_pretrained(f"depth-anything/Depth-Anything-V2-{variant}-hf"))
    m.load_state_dict({k: v.float() for k, v in sd.items()}); m.to(dev).eval()
    name = ck.split("/kaggle/input/")[1].split("/")[-2] if ck.count("/") > 3 else ck
    entry = dict(checkpoint=ck, variant=variant, in_size=IN)
    for sname, (X, G) in SETS.items():
        P = []
        with torch.no_grad():
            for k in range(0, len(X), 4):
                x = torch.from_numpy(np.asarray(X[k:k + 4])).permute(0, 3, 1, 2).float() / 255.
                x = (F.interpolate(x, size=(IN, IN), mode="bilinear", align_corners=False) - MEAN) / STD
                p = m(pixel_values=x.to(dev)).predicted_depth[:, None].float().cpu()
                P.append(F.interpolate(p, size=(512, 512), mode="bilinear", align_corners=False)[:, 0].numpy())
        P = np.concatenate(P)
        bands = {}
        for lo, hi in [(0, 2), (2, 5), (5, 10), (10, 20), (20, 40), (40, 1e4)]:
            mk = (G >= lo) & (G < hi)
            if mk.sum() > 100:
                bands[f"{lo}-{hi if hi < 1e4 else 'inf'}m"] = stats(P, G, mk)
        entry[sname] = dict(all=stats(P, G), by_height=bands)
        print(name, sname, json.dumps(entry[sname]["all"]), flush=True)
    R["models"][name] = entry
    json.dump(R, open("/kaggle/working/eval_naip.json", "w"), indent=1)
R["runtime_min"] = (time.time() - T0) / 60
json.dump(R, open("/kaggle/working/eval_naip.json", "w"), indent=1)
