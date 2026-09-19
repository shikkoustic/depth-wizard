"""Image → DSM pipeline.

GeoTIFF  : resample to the model GSD on a metric grid → nDSM (network) + DTM (FABDEM or user DEM)
           → optional GCP correction → absolute DSM GeoTIFF (+ nDSM, DTM, uncertainty).
PNG/JPG  : nDSM only → relative DSM (height above local ground; datum unknown).
Always   : a viewer scene (scene.json + float32 layers + texture).
"""
import csv, json, time
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import Affine
from rasterio.warp import reproject, Resampling

from .geo import metric_grid, read_onto_grid, write_geotiff
from .scene import write_scene
from . import model as M

MAX_PIXELS = 36e6       # working-grid cap (≈ 6000 x 6000 at 0.66 m ≈ 4 km x 4 km) to bound CPU time
MAX_UPSAMPLE = 3.0      # never invent more than 3x resolution for coarse inputs
_MODEL = None


def get_model(progress=None):
    global _MODEL
    if _MODEL is None:
        _MODEL = M.HeightModel(progress=progress)
    return _MODEL


def _to_uint8_rgb(a, src=None):
    """(C,H,W) any dtype → (H,W,3) uint8 using colour interpretation when available, 2–98 % stretch for >8-bit."""
    if a.shape[0] >= 3 and src is not None:
        ci = [c.name for c in src.colorinterp]
        if all(k in ci for k in ("red", "green", "blue")):
            a = a[[ci.index("red"), ci.index("green"), ci.index("blue")]]
    if a.shape[0] == 1:
        a = np.repeat(a, 3, 0)
    a = a[:3]
    if a.dtype != np.uint8:
        a = a.astype(np.float32)
        out = np.empty_like(a)
        for i in range(3):
            v = a[i][np.isfinite(a[i]) & (a[i] > 0)]
            lo, hi = (np.percentile(v, 2), np.percentile(v, 98)) if v.size else (0, 1)
            out[i] = np.clip((a[i] - lo) / max(hi - lo, 1e-6) * 255, 0, 255)
        a = out.astype(np.uint8)
    return np.moveaxis(a, 0, -1)


def _read_gcps(path):
    pts = []
    with open(path) as fh:
        for row in csv.reader(fh):
            try:
                pts.append([float(row[0]), float(row[1]), float(row[2])])
            except (ValueError, IndexError):
                continue  # header or malformed line
    return np.array(pts)


def _gcp_correction(dsm, transform, pts):
    """Robust planar correction z = a + b*x + c*y fitted to (GCP − DSM) residuals (constant if < 3 points)."""
    inv = ~transform
    rows = []
    for x, y, z in pts:
        c, r = inv * (x, y)
        r, c = int(r), int(c)
        if 0 <= r < dsm.shape[0] and 0 <= c < dsm.shape[1] and np.isfinite(dsm[r, c]):
            rows.append((x, y, z - dsm[r, c], r, c))
    if not rows:
        return None, dict(n_used=0)
    R = np.array(rows)
    if len(R) < 3:
        off = float(np.median(R[:, 2]))
        return np.full(dsm.shape, off, np.float32), dict(n_used=len(R), model="constant", offset_m=off)
    x0, y0 = R[:, 0].mean(), R[:, 1].mean()
    A = np.c_[np.ones(len(R)), R[:, 0] - x0, R[:, 1] - y0]
    w = np.ones(len(R))
    for _ in range(10):  # IRLS with Huber weights (1 m threshold)
        coef = np.linalg.lstsq(A * w[:, None], R[:, 2] * w, rcond=None)[0]
        res = R[:, 2] - A @ coef
        w = np.sqrt(np.minimum(1.0, 1.0 / np.maximum(np.abs(res), 1e-6)))
    H, W = dsm.shape
    cols, rws = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    xs = transform.c + cols * transform.a + rws * transform.b
    ys = transform.f + cols * transform.d + rws * transform.e
    corr = (coef[0] + coef[1] * (xs - x0) + coef[2] * (ys - y0)).astype(np.float32)
    return corr, dict(n_used=len(R), model="plane", coef=[float(c) for c in coef],
                      residual_rmse_after_m=float(np.sqrt(np.mean(res ** 2))))


def process(image_path, out_dir, reference=None, dem=None, gcps=None, gsd=None, title="", progress=print, n_tta=None):
    t0 = time.time()
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    notes, report = [], dict(input=str(Path(image_path).name))
    progress("Reading image")
    with rasterio.open(image_path) as src:
        georef = src.crs is not None and not src.transform.is_identity
        raw = src.read()
        rgb = _to_uint8_rgb(raw, src)
        src_crs, src_tr = src.crs, src.transform
    H0, W0 = rgb.shape[:2]

    if georef:
        crs, tr0, w0, h0 = metric_grid(src_crs, src_tr, W0, H0)
        in_gsd = abs(tr0.a)
        if crs != src_crs or not (abs(tr0.a) == abs(src_tr.a)):
            notes.append(f"reprojected {src_crs.to_string()} → {crs.to_string()}")
    else:
        in_gsd = float(gsd) if gsd else M.MODEL_GSD
        crs, tr0 = None, Affine(in_gsd, 0, 0, 0, -in_gsd, H0 * in_gsd)
        notes.append("no georeference: relative DSM (height above local ground, datum unknown)")
        if not gsd:
            notes.append(f"pixel size unknown: assumed {M.MODEL_GSD} m/px")
    work_gsd = max(M.MODEL_GSD, in_gsd / MAX_UPSAMPLE)
    if in_gsd / work_gsd > 1.01:
        notes.append(f"input {in_gsd:.2f} m/px is coarser than the model's {M.MODEL_GSD} m/px; upsampled {in_gsd / work_gsd:.1f}x")
    ext_w, ext_h = W0 * in_gsd, H0 * in_gsd
    if georef:
        l, b, r, t = rasterio.transform.array_bounds(h0, w0, tr0)
        ext_w, ext_h = r - l, t - b
    Wg, Hg = int(round(ext_w / work_gsd)), int(round(ext_h / work_gsd))
    if Wg * Hg > MAX_PIXELS:
        f = np.sqrt(Wg * Hg / MAX_PIXELS); work_gsd *= f; Wg, Hg = int(Wg / f), int(Hg / f)
        notes.append(f"large image: processed at {work_gsd:.2f} m/px (cap {MAX_PIXELS / 1e6:.0f} Mpx)")
    if georef:
        tr = Affine(work_gsd, 0, l, 0, -work_gsd, t)
    else:
        tr = Affine(work_gsd, 0, 0, 0, -work_gsd, Hg * work_gsd)

    progress(f"Resampling to {work_gsd:.2f} m/px ({Wg}×{Hg})")
    if georef:
        grid = np.zeros((3, Hg, Wg), np.uint8)
        for i in range(3):
            reproject(np.ascontiguousarray(rgb[..., i]), grid[i], src_transform=src_tr, src_crs=src_crs,
                      dst_transform=tr, dst_crs=crs,
                      resampling=Resampling.average if in_gsd < work_gsd else Resampling.bilinear)
        img = np.moveaxis(grid, 0, -1)
    else:
        from PIL import Image
        img = np.asarray(Image.fromarray(rgb).resize((Wg, Hg), Image.BOX if in_gsd < work_gsd else Image.BICUBIC))
    valid = img.max(-1) > 0 if georef else np.ones(img.shape[:2], bool)

    tta = n_tta or (4 if Wg * Hg <= 4e6 else 2)
    progress("Loading height model")
    net = get_model(progress)
    ndsm, spread = net.predict(img, n_tta=tta, progress=lambda n, N: progress(f"Predicting heights: tile {n}/{N}"))
    ndsm[~valid] = np.nan; spread[~valid] = np.nan
    report["model"] = dict(gsd_m=work_gsd, tta_variants=tta, grid=[Wg, Hg])

    layers = {"ndsm": ndsm, "conf": spread}
    if georef:
        progress("Fetching terrain (DEM)")
        from .dem import terrain
        try:
            dtm, info = terrain(tr, crs, (Hg, Wg), dem_path=dem)
            report["terrain"] = info
            notes.append(f"terrain: {info['source']}")
            if info.get("double_count_risk") is True:
                notes.append("warning: terrain is a surface model; buildings/trees may be double-counted")
            if gcps:
                corr, ginfo = _gcp_correction(dtm + np.nan_to_num(ndsm), tr, _read_gcps(gcps))
                report["gcp"] = ginfo
                if corr is not None:
                    dtm = dtm + corr; notes.append(f"GCP correction: {ginfo['model']} from {ginfo['n_used']} points")
            dsm = dtm + ndsm
            layers.update(dtm=dtm, dsm=dsm); kind = "absolute"
        except Exception as e:
            notes.append(f"terrain unavailable ({type(e).__name__}); showing relative heights only")
            report["terrain_error"] = str(e)
            layers["dsm"] = ndsm; kind = "relative"
    else:
        layers["dsm"] = ndsm; kind = "relative"

    if reference:
        progress("Aligning reference DSM")
        if georef:
            layers["ref"] = read_onto_grid(str(reference), tr, crs, (Hg, Wg))
        else:
            with rasterio.open(reference) as s:
                a = s.read(1, masked=True).filled(np.nan).astype(np.float32)
            from PIL import Image
            layers["ref"] = np.array(Image.fromarray(a).resize((Wg, Hg), Image.BILINEAR))
        v = np.isfinite(layers["ref"]) & np.isfinite(layers["dsm"])
        if v.sum() > 100:
            e = layers["dsm"][v] - layers["ref"][v]
            report["vs_reference"] = dict(rmse=float(np.sqrt((e ** 2).mean())), mae=float(np.abs(e).mean()),
                                          bias=float(e.mean()), corr=float(np.corrcoef(layers["dsm"][v], layers["ref"][v])[0, 1]),
                                          n_px=int(v.sum()))

    progress("Writing GeoTIFFs")
    downloads = []
    for name, label in (("dsm", "DSM (absolute, m)" if kind == "absolute" else "Relative DSM (m above local ground)"),
                        ("ndsm", "Above-ground height nDSM (m)"), ("dtm", "Terrain DTM (m)"), ("conf", "Uncertainty (± m, TTA spread)")):
        if name in layers:
            fn = f"{name}.tif"
            write_geotiff(out / fn, layers[name], tr, crs if georef else None, description=label)
            downloads.append(dict(file=fn, label=f"{label} — GeoTIFF"))
    progress("Building 3D scene")
    meta = write_scene(out, img, {k: v for k, v in layers.items()}, (work_gsd, work_gsd), crs=crs if georef else None,
                       transform=tr if georef else None, height_kind=kind, title=title or Path(image_path).stem, notes=notes)
    report.update(height_kind=kind, notes=notes, seconds=round(time.time() - t0, 1))
    json.dump(report, open(out / "report.json", "w"), indent=1)
    downloads.append(dict(file="report.json", label="Processing report (JSON)"))
    meta["downloads"] = downloads
    json.dump(meta, open(out / "scene.json", "w"), indent=1)
    return report
