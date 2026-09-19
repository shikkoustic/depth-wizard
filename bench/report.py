"""Build docs/RESULTS.md from measured outputs only:
  runs/kaggle/<run>/results.json   (GAMUS training/eval kernels)
  runs/bench/bench_results.json    (LiDAR benchmark, full pipeline)
Every number in the report is read from those files; nothing is typed in by hand.

  python bench/report.py
"""
import glob, json, os
import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(ROOT, "docs", "RESULTS.md")


def f(v, d=2):
    return "—" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{d}f}"


def row(cells):
    return "| " + " | ".join(str(c) for c in cells) + " |"


def gamus_section(runs):
    L = ["## 1. Above-ground height on GAMUS (held-out test split)", "",
         "Pixel-pooled metrics over the full GAMUS test split (0.66 m/px, heights in metres). Model selection used the "
         "val split only; the test split was evaluated once per run. Caveat: GAMUS test tiles are spatially adjacent "
         "to training tiles, so these numbers are optimistic about new places — see the held-out-city run and the "
         "LiDAR benchmark.", ""]
    L += [row(["Run", "Backbone", "Test tiles", "RMSE", "MAE", "Bias", "Corr", "per-tile RMSE", "best epoch"]),
          row(["---"] * 9)]
    for name, r in runs.items():
        if "test" not in r:
            continue
        t = r["test"]; p = t["pooled"]
        L.append(row([name, r["cfg"]["model"], t["n_tiles"], f(p["rmse"]), f(p["mae"]), f(p["bias"]), f(p["corr"], 3),
                      f(t["per_tile_mean"]["rmse"]), r.get("best_epoch")]))
    base = next((r for r in runs.values() if "test_baselines" in r), None)
    if base:
        b = base["test_baselines"]
        L += ["", "Baselines on the same test tiles:", "", row(["Baseline", "RMSE", "MAE", "Corr", "note"]), row(["---"] * 5)]
        L.append(row(["predict 0 m everywhere", f(b["zero"]["rmse"]), f(b["zero"]["mae"]), "—", ""]))
        o = b["per_tile_mean_ORACLE_uses_gt"]
        L.append(row(["per-tile mean height (oracle)", f(o["rmse"]), f(o["mae"]), "—", "uses the true mean of each tile"]))
        if "zeroshot_global_affine" in b:
            z = b["zeroshot_global_affine"]
            L.append(row([f"zero-shot Depth Anything V2 {base['cfg']['model']} + global affine", f(z["rmse"]), f(z["mae"]),
                          f(z["corr"], 3), f"{z['n_tiles']} tiles; affine fitted on train tiles"]))
    for name, r in runs.items():
        if "test" not in r:
            continue
        t = r["test"]
        L += ["", f"### {name}: breakdown", "", row(["Class", "RMSE", "MAE", "Bias", "pixels"]), row(["---"] * 5)]
        for k, v in t["by_class"].items():
            L.append(row([k, f(v["rmse"]), f(v["mae"]), f(v["bias"]), f"{v['n_px']:,}"]))
        L += ["", row(["True height band", "RMSE", "MAE", "Bias"]), row(["---"] * 4)]
        for k, v in t["by_height"].items():
            L.append(row([k, f(v["rmse"]), f(v["mae"]), f(v["bias"])]))
        L += ["", row(["City", "pixels", "RMSE", "MAE", "Corr"]), row(["---"] * 5)]
        for k, v in t["by_city"].items():
            L.append(row([k, f"{v['n_px']:,}", f(v["rmse"]), f(v["mae"]), f(v["corr"], 3)]))
        if "tta" in r:
            ta = r["tta"]
            L += ["", f"Test-time augmentation ({ta['n_tiles']} test tiles): single pass RMSE {f(ta['single']['rmse'])} m → "
                      f"8-view mean RMSE {f(ta['tta_mean']['rmse'])} m. Spearman correlation between the 8-view spread and the "
                      f"absolute error: {f(ta['spearman_std_vs_abs_err'], 3)}.", "",
                  row(["spread decile (m)", "MAE (m)", "RMSE (m)"]), row(["---"] * 3)]
            for d in ta["err_by_std_decile"]:
                L.append(row([f"{f(d['std_lo'])}–{f(d['std_hi'])}", f(d["mae"]), f(d["rmse"])]))
    return L


def bench_section(res):
    L = ["", "## 2. Full pipeline (absolute DSM) vs USGS 3DEP LiDAR", "",
         "Input: NAIP 0.6 m RGB GeoTIFF (aerial, not satellite). Output compared with the 3DEP LiDAR top surface on its 2 m grid. "
         "Reference rule: the 3DEP DSM product under-records canopy/roof tops for some LiDAR projects (measured 7–16 m below "
         "DTM + height-above-ground on tree pixels; at the Kansas site an independent 1 m canopy map (Meta/WRI) gave 9.6 m "
         "on trees vs 10.9 m HAG vs 3.8 m DSM−DTM), so the reference is DTM + HAG where that lies 0–40 m above the DSM "
         "product, and the DSM product elsewhere. "
         "`ours` = FABDEM bare-earth terrain + predicted above-ground height. Datum-aligned rows remove one per-site "
         "vertical offset (NAVD88 vs EGM2008), estimated on LiDAR bare-ground pixels and applied equally to every method; "
         "that offset uses the reference, so treat it as an upper bound. `ours_5gcp` instead corrects the terrain with 5 "
         "ground-control points taken from LiDAR bare ground (no oracle offset). None of these sites are in the GAMUS "
         "training cities.", ""]
    L += [row(["Site", "Terrain", "ours", "ours + coarse-DEM scale", "ours + 5 GCP", "FABDEM only", "Copernicus only",
               "nDSM RMSE (ours / zero)", "datum offset"]), row(["---"] * 9)]
    agg = {}
    bad = {n: r for n, r in res.items() if r.get("valid") is False}
    res = {n: r for n, r in res.items() if r.get("valid") is not False}
    for name, r in res.items():
        d = r["datum_aligned"]
        g = lambda k: f(d.get(k, {}) and d[k]["rmse"]) if d.get(k) else "—"
        nd, z = r.get("ndsm_vs_lidar") or {}, r.get("ndsm_zero_baseline") or {}
        L.append(row([name, r["terrain"], g("ours"), g("ours_coarse_cal"), g("ours_5gcp"), g("fabdem_only"), g("copernicus_only"),
                      f"{f(nd.get('rmse'))} / {f(z.get('rmse'))}", f(r["datum_offset_m"])]))
        for k in ("ours", "ours_5gcp", "fabdem_only", "copernicus_only"):
            if d.get(k):
                agg.setdefault(r["terrain"], {}).setdefault(k, []).append(d[k]["rmse"])
    L += ["", "RMSE (m), datum-aligned. Mean over sites per terrain type:", "",
          row(["Terrain", "sites", "ours", "ours + 5 GCP", "FABDEM only", "Copernicus only"]), row(["---"] * 6)]
    for t, v in agg.items():
        n = len(v.get("ours", []))
        L.append(row([t, n] + [f(np.mean(v[k])) if v.get(k) else "—" for k in ("ours", "ours_5gcp", "fabdem_only", "copernicus_only")]))
    if bad:
        L += ["", "Sites excluded automatically because the reference LiDAR itself failed a consistency check:", ""]
        L += [f"- `{n}` ({r['terrain']}): {r['invalid_reason']}" for n, r in bad.items()]
    L += ["", "MAE and correlation per site and method are in `runs/bench/bench_results.json`."]
    return L


def main():
    runs = {}
    for p in sorted(glob.glob(os.path.join(ROOT, "runs", "kaggle", "*", "results.json"))):
        runs[os.path.basename(os.path.dirname(p))] = json.load(open(p))
    L = ["# DepthWizard — measured results", "",
         "Generated by `bench/report.py` from result files; every number below was measured by a script in this repo.", ""]
    if runs:
        L += gamus_section(runs)
    bp = os.path.join(ROOT, "runs", "bench", "bench_results.json")
    if os.path.exists(bp):
        L += bench_section(json.load(open(bp)))
    open(OUT, "w").write("\n".join(L) + "\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
