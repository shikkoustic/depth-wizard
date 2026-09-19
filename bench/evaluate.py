"""Score the full pipeline (absolute DSM) against USGS 3DEP LiDAR on the benchmark sites.

For every site in bench/sites/*:
  1. run depthwizard.pipeline.process on naip.tif (as a user would upload a GeoTIFF)
  2. resample our DSM / nDSM onto the LiDAR 2 m grid (area average)
  3. compare against the LiDAR DSM, next to baselines on the same pixels:
       fabdem_only      FABDEM bare-earth terrain (what you'd have without our model)
       copernicus_only  Copernicus GLO-30 surface model
       ours             FABDEM + predicted nDSM
  4. nDSM check: predicted nDSM vs LiDAR (DSM - DTM), outside skyscraper-core failures (LiDAR DTM sanity mask)

Vertical datum: 3DEP heights are NAVD88, FABDEM/Copernicus are EGM2008. We estimate one offset per
site as the median of (FABDEM - LiDAR DTM) over LiDAR ground pixels and report metrics both raw and
after removing that offset (applied identically to every method).

  python bench/evaluate.py [--sites a,b] [--out runs/bench]
"""
import argparse, json, os, sys, glob
import numpy as np, rasterio
from rasterio.warp import reproject, Resampling
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from depthwizard.pipeline import process, _gcp_correction
from depthwizard.dem import _mosaic, _cop30_url, _fabdem_url
from depthwizard.calibrate import coarse_scale


def onto(src_path, ref):
    out = np.full((ref.height, ref.width), np.nan, np.float32)
    with rasterio.open(src_path) as s:
        reproject(rasterio.band(s, 1), out, dst_transform=ref.transform, dst_crs=ref.crs,
                  resampling=Resampling.average, dst_nodata=np.nan)
    return out


def metrics(p, g, m):
    v = m & np.isfinite(p) & np.isfinite(g)
    if v.sum() < 100:
        return None
    e = p[v] - g[v]
    return dict(rmse=float(np.sqrt((e ** 2).mean())), mae=float(np.abs(e).mean()), bias=float(e.mean()),
                corr=float(np.corrcoef(p[v], g[v])[0, 1]), n_px=int(v.sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sites", default="")
    ap.add_argument("--out", default="runs/bench")
    ap.add_argument("--tta", type=int, default=4)
    a = ap.parse_args()
    root = os.path.join(os.path.dirname(__file__), "sites")
    names = a.sites.split(",") if a.sites else sorted(os.path.basename(d) for d in glob.glob(f"{root}/*") if os.path.exists(f"{d}/site.json"))
    results = {}
    for name in names:
        d = f"{root}/{name}"; site = json.load(open(f"{d}/site.json"))
        scene = f"{a.out}/scenes/{name}"
        rep = process(f"{d}/naip.tif", scene, reference=f"{d}/lidar_dsm.tif", title=f"{name} ({site.get('terrain', '')})",
                      progress=lambda m: None, n_tta=a.tta)
        with rasterio.open(f"{d}/lidar_dsm.tif") as ref:
            L_dsm = ref.read(1, masked=True).filled(np.nan).astype(np.float32)
            ours = onto(f"{scene}/dsm.tif", ref); nd = onto(f"{scene}/ndsm.tif", ref); dtm_used = onto(f"{scene}/dtm.tif", ref)
            L_dtm = onto(f"{d}/lidar_dtm.tif", ref)
            fab = _mosaic(_fabdem_url, ref.transform, ref.crs, (ref.height, ref.width))
            cop = _mosaic(_cop30_url, ref.transform, ref.crs, (ref.height, ref.width))
        L_nd = L_dsm - L_dtm
        footprint = np.isfinite(ours)
        # LiDAR DTM sanity: 3DEP DTM keeps building remnants under dense high-rises; drop pixels where the
        # LiDAR DTM sits > 5 m above FABDEM's local terrain (a bare-earth DTM should not do that)
        dtm_ok = np.isfinite(L_dtm) & (L_dtm - fab < 5 + np.nanmedian(L_dtm - fab))
        ground = np.isfinite(L_nd) & (L_nd < 0.5) & dtm_ok
        datum = float(np.nanmedian((fab - L_dtm)[ground])) if ground.sum() > 100 else 0.0
        r = dict(terrain=site.get("terrain"), datum_offset_m=datum, relief_m=site.get("terrain_relief_m"),
                 lidar_ndsm_mean=float(np.nanmean(L_nd[footprint])), pipeline=rep)
        # variant A: rescale nDSM so its 90 m block means match Copernicus - FABDEM (low-res DEM calibration)
        cal = coarse_scale(nd, cop - fab, abs(ref.transform.a))
        r["coarse_calibration"] = cal
        ours_cal = dtm_used + (cal["scale"] if cal else 1.0) * nd
        # variant B: 5 ground control points taken from LiDAR bare-ground pixels (terrain plane correction)
        rng = np.random.RandomState(0); gi = np.flatnonzero(ground & footprint)
        variants = [("ours", ours), ("ours_coarse_cal", ours_cal), ("fabdem_only", fab), ("copernicus_only", cop)]
        if gi.size >= 5:
            pick = rng.choice(gi, 5, replace=False); rr, cc = np.unravel_index(pick, L_dsm.shape)
            xs = ref.transform.c + (cc + 0.5) * ref.transform.a; ys = ref.transform.f + (rr + 0.5) * ref.transform.e
            corr, ginfo = _gcp_correction(ours, ref.transform, np.c_[xs, ys, L_dsm[rr, cc]])
            r["gcp5"] = ginfo
            if corr is not None:
                variants.insert(2, ("ours_5gcp", ours + corr))
        for tag, off in (("raw", 0.0), ("datum_aligned", datum)):
            r[tag] = {k: metrics(v - (0 if k == "ours_5gcp" else off), L_dsm, footprint) for k, v in variants}
        r["ndsm_vs_lidar"] = metrics(nd, L_nd, footprint & dtm_ok)
        r["ndsm_zero_baseline"] = metrics(np.zeros_like(nd), L_nd, footprint & dtm_ok)
        r["frac_px_excluded_bad_lidar_dtm"] = float(1 - (footprint & dtm_ok).sum() / footprint.sum())
        results[name] = r
        print(name, json.dumps({k: {m: (v or {}).get("rmse") for m, v in r[k].items()} for k in ("raw", "datum_aligned")}), flush=True)
        os.makedirs(a.out, exist_ok=True)
        json.dump(results, open(f"{a.out}/bench_results.json", "w"), indent=1)


if __name__ == "__main__":
    main()
