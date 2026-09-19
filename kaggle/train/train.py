# DepthWizard: fine-tune Depth Anything V2 to regress metric above-ground height (nDSM) on GAMUS.
# Runs on Kaggle GPU. Input: output of kernel shikkoustic/depthwizard-gamus-prep (512x512 tiles, ~0.66 m/px).
# Model selection uses the GAMUS val split only; the test split is evaluated once, at the end.
CFG = dict(name="small_main", model="Small", epochs=14, bs=8, accum=1, lr_enc=5e-6, lr_dec=5e-5,
           tall_weight=True, grad_loss=0.5, degrade=True, holdout_city=None, time_budget_h=10.5,
           n_tta_eval=800, zeroshot_baseline=True, smoke=False)
#@CFG@

import os, json, time, math, random, glob, traceback
import numpy as np, torch, torch.nn.functional as F
from transformers import AutoModelForDepthEstimation

T0 = time.time(); OUT = "/kaggle/working"; dev = "cuda"
torch.manual_seed(0); np.random.seed(0); random.seed(0)
def log(*a): print(f"[{(time.time()-T0)/60:6.1f}m]", *a, flush=True)
R = dict(cfg=CFG, gpu=torch.cuda.get_device_name(0))
def save(): json.dump(R, open(f"{OUT}/results.json", "w"), indent=1, default=float)

# ---------------- data ----------------
meta_path = glob.glob("/kaggle/input/**/meta.json", recursive=True)[0]
D = os.path.dirname(meta_path); META = json.load(open(meta_path)); log("data dir", D)
def load(split, cities=None, exclude=None, ram=True):
    m = META["splits"][split]
    keep = [k for k, (i, ok) in enumerate(zip(m["ids"], m["ok"])) if ok
            and (cities is None or i.split("_")[0] in cities) and (exclude is None or i.split("_")[0] != exclude)]
    X = np.load(f"{D}/{split}_rgb.npy", mmap_mode="r"); Y = np.load(f"{D}/{split}_agl.npy", mmap_mode="r")
    C = np.load(f"{D}/{split}_cls.npy", mmap_mode="r") if os.path.exists(f"{D}/{split}_cls.npy") else None
    ids = [m["ids"][k] for k in keep]
    if ram:
        return ids, np.ascontiguousarray(X[keep]), np.ascontiguousarray(Y[keep]), (np.ascontiguousarray(C[keep]) if C is not None else None)
    return ids, X, Y, C, keep

hc = CFG["holdout_city"]
tr_ids, Xtr, Ytr, _ = load("train", exclude=hc)
va_ids, Xva, Yva, _ = load("val", exclude=hc)
if CFG["smoke"]:  # quick end-to-end check of the whole script on a few tiles
    tr_ids, Xtr, Ytr = tr_ids[:64], Xtr[:64], Ytr[:64]; va_ids, Xva, Yva = va_ids[:32], Xva[:32], Yva[:32]
R["n_train"], R["n_val"] = len(tr_ids), len(va_ids)
R["train_cities"] = sorted({i.split("_")[0] for i in tr_ids}); log("train", Xtr.shape, "val", Xva.shape, R["train_cities"]); save()

MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
IN = 518  # 37 x 14 patches

def to_input(x):  # uint8 NHWC tensor on GPU -> normalized NCHW 518
    x = x.permute(0, 3, 1, 2).float() / 255.
    x = F.interpolate(x, size=(IN, IN), mode="bilinear", align_corners=False, antialias=True)
    return (x - MEAN) / STD

def augment(x, y):
    """x uint8 NHWC, y float NHW (NaN = invalid), both on GPU. Returns augmented float NCHW in [0,1] and NHW target."""
    x = x.permute(0, 3, 1, 2).float() / 255.; y = y[:, None]
    B, _, S, _ = x.shape
    # random crop (zoom 1.0-1.28x) -> keeps metric heights, perturbs apparent GSD slightly
    cs = random.randint(400, S); oy, ox = random.randint(0, S - cs), random.randint(0, S - cs)
    x, y = x[..., oy:oy + cs, ox:ox + cs], y[..., oy:oy + cs, ox:ox + cs]
    x = F.interpolate(x, size=(S, S), mode="bilinear", align_corners=False); y = F.interpolate(y, size=(S, S), mode="nearest")
    # dihedral group: flips + 90-degree rotations
    k = random.randint(0, 3); x, y = torch.rot90(x, k, (2, 3)), torch.rot90(y, k, (2, 3))
    if random.random() < .5: x, y = x.flip(3), y.flip(3)
    # resolution degradation: simulate coarser source imagery (e.g. 1-2 m pan-sharpened) on the same grid
    if CFG["degrade"] and random.random() < .5:
        f = random.uniform(1.3, 3.0); s2 = max(64, int(S / f))
        x = F.interpolate(F.interpolate(x, size=(s2, s2), mode="area"), size=(S, S), mode="bilinear", align_corners=False)
    # photometric: brightness, contrast, saturation, gamma, per-channel gain, noise
    b = 1 + (torch.rand(B, 1, 1, 1, device=dev) - .5) * .4; c = 1 + (torch.rand(B, 1, 1, 1, device=dev) - .5) * .4
    m = x.mean((1, 2, 3), keepdim=True); x = (x - m) * c + m * b
    g = x.mean(1, keepdim=True); x = g + (x - g) * (1 + (torch.rand(B, 1, 1, 1, device=dev) - .5) * .6)
    x = x.clamp(1e-4, 1) ** (1 + (torch.rand(B, 1, 1, 1, device=dev) - .5) * .4)
    x = x * (1 + (torch.rand(B, 3, 1, 1, device=dev) - .5) * .1)
    x = (x + torch.randn_like(x) * random.uniform(0, .02)).clamp(0, 1)
    return x, y[:, 0]

def norm_in(x):  # float NCHW [0,1] -> model input
    x = F.interpolate(x, size=(IN, IN), mode="bilinear", align_corners=False)
    return (x - MEAN) / STD

def forward(model, xin, size):
    p = model(pixel_values=xin).predicted_depth[:, None]
    return F.interpolate(p, size=size, mode="bilinear", align_corners=False)[:, 0]

def grad_loss(p, y, valid, scales=4):
    tot = 0.
    for s in range(scales):
        if s:
            p = F.avg_pool2d(p[:, None], 2)[:, 0]; y = F.avg_pool2d(torch.nan_to_num(y)[:, None] * valid[:, None], 2)[:, 0]
            valid = F.avg_pool2d(valid[:, None].float(), 2)[:, 0] > 0.99;
        d = torch.where(valid, p - torch.nan_to_num(y), torch.zeros_like(p))
        gx = (d[:, :, 1:] - d[:, :, :-1]).abs() * (valid[:, :, 1:] & valid[:, :, :-1])
        gy = (d[:, 1:] - d[:, :-1]).abs() * (valid[:, 1:] & valid[:, :-1])
        tot = tot + (gx.sum() + gy.sum()) / valid.sum().clamp(min=1)
    return tot / scales

def loss_fn(p, y):
    valid = torch.isfinite(y); yv = torch.nan_to_num(y)
    w = (1 + (yv / 10).clamp(0, 3)) if CFG["tall_weight"] else torch.ones_like(yv)
    l1 = ((p - yv).abs() * w * valid).sum() / (w * valid).sum().clamp(min=1)
    return l1 + (CFG["grad_loss"] * grad_loss(p, y, valid) if CFG["grad_loss"] else 0.)

# ---------------- metrics ----------------
def predict(model, X, bs=16, tta=False):
    """X uint8 NHWC numpy. Returns predictions (N,512,512) float32 (and TTA std if tta)."""
    model.eval(); P, Sd = [], []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for k in range(0, len(X), bs):
            x = torch.from_numpy(np.asarray(X[k:k + bs])).to(dev).permute(0, 3, 1, 2).float() / 255.
            S = x.shape[-1]
            if not tta:
                P.append(forward(model, norm_in(x), (S, S)).float().cpu().numpy()); continue
            outs = []
            for kk in range(4):
                for fl in (False, True):
                    xi = torch.rot90(x, kk, (2, 3)); xi = xi.flip(3) if fl else xi
                    o = forward(model, norm_in(xi), (S, S))
                    o = o.flip(2) if fl else o; outs.append(torch.rot90(o, -kk, (1, 2)).float())
            o = torch.stack(outs); P.append(o.mean(0).cpu().numpy()); Sd.append(o.std(0).cpu().numpy())
    return (np.concatenate(P), np.concatenate(Sd)) if tta else np.concatenate(P)

class Acc:
    """Streaming error statistics (float64 sums) so full-split metrics never copy the whole array."""
    def __init__(self): self.s = np.zeros(9)
    def add(self, p, g, m=None):
        v = np.isfinite(g) if m is None else (m & np.isfinite(g))
        if not v.any(): return
        p = p[v].astype(np.float64); g = g[v].astype(np.float64); e = p - g
        self.s += [v.sum(), (e * e).sum(), np.abs(e).sum(), e.sum(), p.sum(), g.sum(), (p * p).sum(), (g * g).sum(), (p * g).sum()]
    def result(self):
        n, se, ae, sb, sp, sg, spp, sgg, spg = self.s
        if n == 0: return None
        cov = spg / n - sp / n * sg / n; vp = spp / n - (sp / n) ** 2; vg = sgg / n - (sg / n) ** 2
        return dict(rmse=float(np.sqrt(se / n)), mae=float(ae / n), bias=float(sb / n),
                    corr=float(cov / np.sqrt(vp * vg)) if vp > 0 and vg > 0 else None, n_px=int(n))

def pooled(pred, gt, ch=64):
    a = Acc()
    for k in range(0, len(gt), ch): a.add(np.asarray(pred[k:k + ch], np.float32), np.asarray(gt[k:k + ch], np.float32))
    return a.result()

def per_tile(pred, gt):
    r = []
    for p, g in zip(pred, gt):
        v = np.isfinite(g)
        if v.sum() < 1000: continue
        e = p[v] - g[v]
        r.append((np.sqrt((e ** 2).mean()), np.abs(e).mean(), np.corrcoef(p[v], g[v])[0, 1] if g[v].std() > 0 and p[v].std() > 0 else np.nan))
    r = np.array(r); return dict(rmse=float(r[:, 0].mean()), mae=float(r[:, 1].mean()), corr=float(np.nanmean(r[:, 2])), n_tiles=len(r))

NAMES = {0: "others", 1: "ground", 2: "low_veg", 3: "building", 4: "water", 5: "road", 6: "tree"}
HBINS = [(0, 2), (2, 5), (5, 10), (10, 20), (20, 40), (40, 1e4)]
def breakdown(pred, gt, cls, ids, ch=64):
    acc = {}
    cities = np.array([i.split("_")[0] for i in ids])
    for k in range(0, len(gt), ch):
        p, g = pred[k:k + ch], gt[k:k + ch]
        if cls is not None:
            c = cls[k:k + ch]
            for cv, n in NAMES.items(): acc.setdefault(("by_class", n), Acc()).add(p, g, c == cv)
        for lo, hi in HBINS: acc.setdefault(("by_height", f"{lo}-{hi if hi < 1e4 else 'inf'}m"), Acc()).add(p, g, (g >= lo) & (g < hi))
        for cc in set(cities[k:k + ch]): acc.setdefault(("by_city", cc), Acc()).add(p[cities[k:k + ch] == cc], g[cities[k:k + ch] == cc])
    out = {"by_class": {}, "by_city": {}, "by_height": {}}
    for (grp, key), a in acc.items():
        r = a.result()
        if r and r["n_px"] > 1000: out[grp][key] = r
    return out

# ---------------- model ----------------
model = AutoModelForDepthEstimation.from_pretrained(f"depth-anything/Depth-Anything-V2-{CFG['model']}-hf").to(dev)
enc = [p for n, p in model.named_parameters() if "backbone" in n]; dec = [p for n, p in model.named_parameters() if "backbone" not in n]
opt = torch.optim.AdamW([{"params": enc, "lr": CFG["lr_enc"]}, {"params": dec, "lr": CFG["lr_dec"]}], weight_decay=1e-4)
steps_per_ep = len(Xtr) // (CFG["bs"] * CFG["accum"]); total = steps_per_ep * CFG["epochs"]; warm = min(300, total // 10)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1, s / total))))
scaler = torch.amp.GradScaler("cuda")

va_sub = np.random.RandomState(1).choice(len(Xva), min(300, len(Xva)), replace=False)
hist, best = [], (1e9, -1)
try:
    for ep in range(CFG["epochs"]):
        model.train(); order = np.random.permutation(len(Xtr)); L = []; t_ep = time.time()
        for k in range(0, len(order) - CFG["bs"] + 1, CFG["bs"]):
            b = np.sort(order[k:k + CFG["bs"]])
            x = torch.from_numpy(Xtr[b]).to(dev, non_blocking=True); y = torch.from_numpy(Ytr[b].astype(np.float32)).to(dev)
            x, y = augment(x, y)
            with torch.autocast("cuda", dtype=torch.float16):
                p = forward(model, norm_in(x), y.shape[-2:])
            loss = loss_fn(p.float(), y) / CFG["accum"]
            scaler.scale(loss).backward(); L.append(loss.item() * CFG["accum"])
            if (k // CFG["bs"] + 1) % CFG["accum"] == 0:
                scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); sched.step()
        Pv = predict(model, Xva[va_sub]); gv = Yva[va_sub].astype(np.float32)
        r = dict(ep=ep, train_loss=float(np.mean(L)), minutes=(time.time() - t_ep) / 60, **{f"val_{k}": v for k, v in pooled(Pv, gv).items()})
        hist.append(r); R["history"] = hist; log(r)
        if r["val_rmse"] < best[0]:
            best = (r["val_rmse"], ep); torch.save({k: v.half() for k, v in model.state_dict().items()}, f"{OUT}/best.pt")
        R["best_epoch"] = best[1]; save()
        if (time.time() - T0) / 3600 > CFG["time_budget_h"] * 0.6: log("time budget: stopping training"); break
except Exception:
    R["train_error"] = traceback.format_exc(); save(); raise

del Xtr, Ytr
sd = torch.load(f"{OUT}/best.pt"); model.load_state_dict({k: v.float() for k, v in sd.items()}); model.eval()
log("loaded best epoch", best[1])

# full val with best model
Pv = predict(model, Xva); R["val_full"] = pooled(Pv, Yva.astype(np.float32)); save(); del Pv

# ---------------- test (once) ----------------
te_ids, Xte, Yte, Cte = load("test", cities=[hc] if hc else None)
if CFG["smoke"]: te_ids, Xte, Yte, Cte = te_ids[:48], Xte[:48], Yte[:48], (Cte[:48] if Cte is not None else None)
Yte = Yte.astype(np.float32); log("test", Xte.shape)
Pt = predict(model, Xte)
R["test"] = dict(n_tiles=len(te_ids), pooled=pooled(Pt, Yte), per_tile_mean=per_tile(Pt, Yte), **breakdown(Pt, Yte, Cte, te_ids))
# baselines on identical tiles
az, am = Acc(), Acc()
for g in Yte:
    az.add(np.zeros_like(g), g); am.add(np.full_like(g, np.nanmean(g)), g)
R["test_baselines"] = dict(zero=az.result(), per_tile_mean_ORACLE_uses_gt=am.result())
log("test", json.dumps(R["test"]["pooled"])); save()
# sample predictions for visual checks and the viewer
sel = np.linspace(0, len(Xte) - 1, 16).astype(int)
np.savez_compressed(f"{OUT}/test_samples.npz", ids=np.array(te_ids)[sel], rgb=Xte[sel], gt=Yte[sel], pred=Pt[sel].astype(np.float32),
                    cls=Cte[sel] if Cte is not None else np.zeros(1))
del Pt

# TTA: accuracy gain and whether the spread predicts error (subset)
try:
    sub = np.random.RandomState(2).choice(len(Xte), min(CFG["n_tta_eval"], len(Xte)), replace=False)
    Pm, Ps = predict(model, Xte[sub], bs=8, tta=True); G = Yte[sub]; P1 = predict(model, Xte[sub])
    v = np.isfinite(G); v[:, 1::2] = False; v[:, :, 1::2] = False  # every 4th pixel is plenty for the calibration curve
    err = np.abs(Pm - G)[v]; sd_ = Ps[v]
    qs = np.quantile(sd_, np.linspace(0, 1, 11)); bins = np.clip(np.searchsorted(qs, sd_, side="right") - 1, 0, 9)
    from scipy.stats import spearmanr
    rho = spearmanr(sd_[::20], err[::20]).correlation
    R["tta"] = dict(n_tiles=len(sub), single=pooled(P1, G), tta_mean=pooled(Pm, G), spearman_std_vs_abs_err=float(rho),
                    err_by_std_decile=[dict(std_lo=float(qs[i]), std_hi=float(qs[i + 1]), mae=float(err[bins == i].mean()),
                                            rmse=float(np.sqrt((err[bins == i] ** 2).mean()))) for i in range(10)])
    k8 = sub[:4]; np.savez_compressed(f"{OUT}/tta_samples.npz", ids=np.array(te_ids)[k8], mean=Pm[:4], std=Ps[:4], gt=G[:4])
    log("tta", json.dumps({k: R["tta"][k] for k in ("single", "tta_mean", "spearman_std_vs_abs_err")})); save()
    del Pm, Ps, P1
except Exception:
    R["tta_error"] = traceback.format_exc(); save()

# zero-shot baseline: pretrained DA-V2 (relative depth) + one global affine fitted on train tiles
if CFG["zeroshot_baseline"]:
    try:
        zs = AutoModelForDepthEstimation.from_pretrained(f"depth-anything/Depth-Anything-V2-{CFG['model']}-hf").to(dev)
        _, Xf, Yf, _, keep = load("train", exclude=hc, ram=False); fit = np.random.RandomState(3).choice(keep, 16 if CFG["smoke"] else 200, replace=False)
        Pf = predict(zs, np.asarray(Xf[np.sort(fit)])); Gf = np.asarray(Yf[np.sort(fit)], np.float32); v = np.isfinite(Gf)
        a = np.polyfit(Pf[v][::20], Gf[v][::20], 1)
        sub = np.random.RandomState(4).choice(len(Xte), min(800, len(Xte)), replace=False)
        Pz = predict(zs, Xte[sub]); R["test_baselines"]["zeroshot_global_affine"] = dict(n_tiles=len(sub), **pooled(a[0] * Pz + a[1], Yte[sub]))
        log("zeroshot", R["test_baselines"]["zeroshot_global_affine"]); save()
    except Exception:
        R["zeroshot_error"] = traceback.format_exc(); save()

# visual montage: RGB | GT | pred | error for 8 samples
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
s = np.load(f"{OUT}/test_samples.npz")
fig, ax = plt.subplots(8, 4, figsize=(16, 32))
for r in range(8):
    k = r * 2; vmax = max(10, np.nanpercentile(s["gt"][k], 99))
    ax[r, 0].imshow(s["rgb"][k]); ax[r, 0].set_title(str(s["ids"][k]))
    ax[r, 1].imshow(s["gt"][k], vmin=0, vmax=vmax); ax[r, 1].set_title("LiDAR nDSM")
    ax[r, 2].imshow(s["pred"][k], vmin=0, vmax=vmax); ax[r, 2].set_title("predicted")
    im = ax[r, 3].imshow(s["pred"][k] - s["gt"][k], cmap="RdBu_r", vmin=-10, vmax=10); ax[r, 3].set_title("pred - gt (m)")
    for a_ in ax[r]: a_.axis("off")
plt.tight_layout(); plt.savefig(f"{OUT}/montage_pred.png", dpi=55)
R["runtime_min"] = (time.time() - T0) / 60; save(); log("DONE")
