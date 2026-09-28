# Download the full GAMUS dataset (HF earthflow/GAMUS) once, resize tiles to 512x512 and store
# them as .npy memmaps so training kernels can mount this kernel's output instead of re-downloading.
# Also records split/city/class statistics and a visual montage for sanity checks.
import os, json, time, shutil
from concurrent.futures import ThreadPoolExecutor
import numpy as np, h5py, torch, torch.nn.functional as F
from huggingface_hub import HfApi, hf_hub_download

REPO, S, OUT, TMP = "earthflow/GAMUS", 512, "/kaggle/working", "/tmp/gamus"
T0 = time.time()
def log(*a): print(f"[{time.time()-T0:6.0f}s]", *a, flush=True)

files = HfApi().list_repo_files(REPO, repo_type="dataset")
def ids(kind, split, suf):
    return sorted(f.split("/")[-1][: -len(suf)] for f in files if f.startswith(f"{kind}/{split}/") and f.endswith(suf))

def read_h5(path):
    with h5py.File(path) as f:
        return f["image"][()]

IMG_NAME = {}  # tile id -> image file name (DC/PHL use *_RGB.h5, NYC uses *_IMG.h5)

def fetch(split, i, has_cls, keep_full=False):
    d = f"{TMP}/{split}/{i}"
    try:
        p_im = hf_hub_download(REPO, f"images/{split}/{IMG_NAME[(split, i)]}", repo_type="dataset", local_dir=d)
        p_h = hf_hub_download(REPO, f"heights/{split}/{i}_AGL.h5", repo_type="dataset", local_dir=d)
        im, h = read_h5(p_im), read_h5(p_h).astype(np.float32)
        c = read_h5(hf_hub_download(REPO, f"classes/{split}/{i}_CLS.h5", repo_type="dataset", local_dir=d)) if has_cls else None
    finally:
        shutil.rmtree(d, ignore_errors=True)
    raw = dict(shape=list(im.shape), hshape=list(h.shape), nan=float(np.isnan(h).mean()),
               neg=float((h < 0).mean()), hmin=float(np.nanmin(h)), hmax=float(np.nanmax(h)))
    im_t = torch.from_numpy(im).permute(2, 0, 1)[None].float()
    im_s = F.interpolate(im_t, size=(S, S), mode="area")[0].round().clamp(0, 255).byte().permute(1, 2, 0).numpy()
    valid = np.isfinite(h)
    hv = torch.from_numpy(np.where(valid, h, 0))[None, None]
    vv = torch.from_numpy(valid.astype(np.float32))[None, None]
    hs, vs = F.interpolate(hv, size=(S, S), mode="area")[0, 0], F.interpolate(vv, size=(S, S), mode="area")[0, 0]
    h_s = torch.where(vs > 0.5, hs / vs.clamp(min=1e-6), torch.full_like(hs, float("nan"))).numpy()
    c_s = None
    if c is not None:
        c = np.asarray(c).squeeze()
        step = c.shape[0] // S
        c_s = c[step // 2::step, step // 2::step][:S, :S].astype(np.uint8)  # nearest, centre sample
    return im_s, h_s, c_s, raw, (im if keep_full else None), (h if keep_full else None), (c if keep_full else None)

meta = {"repo": REPO, "size": S, "splits": {}}
demo = []
for split in ["train", "val", "test"]:
    idl = []
    for suf in ("_RGB.h5", "_IMG.h5"):
        for i in ids("images", split, suf):
            IMG_NAME[(split, i)] = i + suf; idl.append(i)
    idl = sorted(set(idl))
    hset, cset = set(ids("heights", split, "_AGL.h5")), set(ids("classes", split, "_CLS.h5"))
    idl = [i for i in idl if i in hset]
    has_cls = len(cset) > 0 and all(i in cset for i in idl)
    n = len(idl); log(split, n, "tiles, classes:", has_cls)
    X = np.lib.format.open_memmap(f"{OUT}/{split}_rgb.npy", "w+", np.uint8, (n, S, S, 3))
    Y = np.lib.format.open_memmap(f"{OUT}/{split}_agl.npy", "w+", np.float16, (n, S, S))
    C = np.lib.format.open_memmap(f"{OUT}/{split}_cls.npy", "w+", np.uint8, (n, S, S)) if has_cls else None
    raws, ok = [], np.zeros(n, bool)
    cls_sum, cls_cnt, cls_hist = {}, {}, {}
    ex = ThreadPoolExecutor(16)
    def chunked():  # submit in chunks so finished-but-unconsumed results can't pile up in RAM
        for s0 in range(0, n, 128):
            fs = [ex.submit(fetch, split, idl[k], has_cls, split == "test" and k % 97 == 0) for k in range(s0, min(n, s0 + 128))]
            yield from zip(range(s0, s0 + len(fs)), fs)
    for k, f in chunked():
        try:
            im_s, h_s, c_s, raw, im_full, h_full, c_full = f.result()
        except Exception as e:
            raws.append({"error": repr(e)[:200]}); continue
        X[k], Y[k] = im_s, h_s
        if C is not None:
            C[k] = c_s
            hf = np.nan_to_num(h_s)
            for cv in np.unique(c_s):
                m = c_s == cv; key = int(cv)
                cls_sum[key] = cls_sum.get(key, 0.) + float(hf[m].sum()); cls_cnt[key] = cls_cnt.get(key, 0) + int(m.sum())
                hh = np.histogram(np.clip(hf[m], 0, 60), bins=12, range=(0, 60))[0]
                cls_hist[key] = (cls_hist.get(key, 0) + hh)
        ok[k] = True; raws.append(raw)
        if im_full is not None and len(demo) < 6:  # a few full-res tiles for the viewer demo
            demo.append(idl[k])
            np.savez_compressed(f"{OUT}/demo_{idl[k]}.npz", rgb=im_full, agl=h_full.astype(np.float32),
                                cls=np.asarray(c_full).squeeze().astype(np.uint8))
        if k % 500 == 0: log(split, k, "/", n)
    ex.shutdown(); X.flush(); Y.flush()
    if C is not None: C.flush()
    good = [r for r in raws if "error" not in r]
    cities = {}
    for i in idl: cities[i.split("_")[0]] = cities.get(i.split("_")[0], 0) + 1
    meta["splits"][split] = dict(
        n=n, n_ok=int(ok.sum()), ids=idl, ok=ok.tolist(), cities=cities, has_classes=has_cls,
        errors=[r["error"] for r in raws if "error" in r][:10],
        raw_shapes=sorted({str(r["shape"]) + str(r["hshape"]) for r in good}),
        nan_frac_mean=float(np.mean([r["nan"] for r in good])), neg_frac_mean=float(np.mean([r["neg"] for r in good])),
        h_min=float(min(r["hmin"] for r in good)), h_max=float(max(r["hmax"] for r in good)),
        class_mean_agl={k: cls_sum[k] / cls_cnt[k] for k in cls_sum}, class_pixel_frac={k: cls_cnt[k] / sum(cls_cnt.values()) for k in cls_cnt},
        class_hist_0_60m_5m_bins={k: v.tolist() for k, v in cls_hist.items()})
    json.dump(meta, open(f"{OUT}/meta.json", "w"), indent=1)
    log(split, "done", meta["splits"][split]["n_ok"], "ok")

# spatial adjacency between splits (IDs look like CITY_row_col)
def rc(i):
    p = i.split("_"); return p[0], int(p[1]), int(p[2])
try:
    tr = {rc(i) for i in meta["splits"]["train"]["ids"]}
    for split in ["val", "test"]:
        te = [rc(i) for i in meta["splits"][split]["ids"]]
        adj = sum(any((c, r + dr, q + dq) in tr for dr in (-1, 0, 1) for dq in (-1, 0, 1) if dr or dq) for c, r, q in te)
        same = sum(t in tr for t in te)
        meta["splits"][split]["frac_with_train_neighbour"] = adj / len(te); meta["splits"][split]["n_same_id_as_train"] = same
except Exception as e:
    meta["adjacency_error"] = repr(e)
meta["demo_ids"] = demo; meta["runtime_s"] = time.time() - T0
json.dump(meta, open(f"{OUT}/meta.json", "w"), indent=1)

# visual montage: RGB | height | class for 6 test tiles
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
X = np.load(f"{OUT}/test_rgb.npy", mmap_mode="r"); Y = np.load(f"{OUT}/test_agl.npy", mmap_mode="r")
C = np.load(f"{OUT}/test_cls.npy", mmap_mode="r") if os.path.exists(f"{OUT}/test_cls.npy") else None
sel = np.linspace(0, len(X) - 1, 6).astype(int)
fig, ax = plt.subplots(len(sel), 3, figsize=(12, 4 * len(sel)))
for r, k in enumerate(sel):
    ax[r, 0].imshow(X[k]); ax[r, 0].set_title(meta["splits"]["test"]["ids"][k])
    im = ax[r, 1].imshow(np.asarray(Y[k], np.float32), cmap="viridis", vmin=0, vmax=40); plt.colorbar(im, ax=ax[r, 1])
    if C is not None:
        im = ax[r, 2].imshow(C[k], cmap="tab10", vmin=0, vmax=9, interpolation="nearest"); plt.colorbar(im, ax=ax[r, 2])
    for a in ax[r]: a.axis("off")
plt.tight_layout(); plt.savefig(f"{OUT}/montage_test.png", dpi=70)
log("DONE")
