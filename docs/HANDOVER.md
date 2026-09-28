# DepthWizard — handover

SIH 2026 · PS **SIH26175** (ISRO / SAC): single-view optical RGB image → DSM → interactive 3D flythrough.

This package contains the whole working project: source, documentation, the trained models, sample inputs and
two ready-made 3D scenes. Everything in it was built and measured by us; every number in the docs comes from a
result file produced by a script in this repo.

## 1. Run it in five minutes

```bash
cd depth-wizard
python -m venv .venv && source .venv/bin/activate        # Python 3.10+
pip install numpy rasterio pillow fastapi uvicorn python-multipart torch transformers
python -m depthwizard --data runs --weights weights/v3a_small.pt
```

Then open http://127.0.0.1:8000. The viewer is pre-built inside the package (`depthwizard/static/`), so Node is
**not** needed unless you want to change the front end (`cd web && npm install && npm run build`).

Two scenes are included so the viewer has something to show immediately:
`bench_sf_downtown` (prediction vs LiDAR, with the accuracy panel and swipe) and `gamus_NYC_01639`.

**Make your own scenes from the included inputs** (each takes ~1–3 min on a laptop CPU):

```bash
# India, Maxar 0.37 m — the showcase scene (hill town in Sikkim)
python -m depthwizard process demos/india/sikkim_a_maxar_rgb.tif -o runs/scenes/india_sikkim_town --weights weights/v3a_base.pt
# a GAMUS test tile as a PNG upload (relative DSM) scored against its LiDAR reference
python -m depthwizard process demos/gamus/DC_03_26.png -o runs/scenes/gamus_DC_03_26 \
    --reference demos/gamus/DC_03_26_lidar_agl.tif --gsd 0.33 --weights weights/v3a_small.pt
```

Or just drag any GeoTIFF/PNG onto the 3D view in the browser.

## 2. What's in the box

| Path | What |
|---|---|
| `depthwizard/` | Python package: FastAPI server, image→DSM pipeline, model inference, terrain fetch, scene export |
| `web/` | Three.js viewer source (already built into `depthwizard/static/`) |
| `bench/` | LiDAR benchmark: site fetcher, scorer, report generator, robustness test, Maxar/Sentinel-2 fetchers |
| `kaggle/` | Every data-prep / training / evaluation kernel we ran (these run on Kaggle, not locally) |
| `tests/` | Fast tests (`python -m pytest tests`) |
| `docs/` | APPROACH (why), RESULTS (measured numbers), PS checklist, this handover, `data/` = raw result JSON |
| `weights/` | `v3a_base.pt` (best accuracy) and `v3a_small.pt` (≈4× faster on CPU, best on rural land) |
| `demos/` | Sample inputs: 6 GAMUS test tiles + their LiDAR references, India Sentinel-2, India Maxar 0.37 m |
| `runs/scenes/` | Two pre-built 3D scenes |

## 3. How it works (one paragraph)

A single overhead image shows *what stands above the ground* but says little about absolute terrain height, so
we split the problem: **DSM = bare-earth terrain (30 m FABDEM, fetched for the image footprint) + above-ground
height (a fine-tuned Depth Anything V2, predicted from the image)**. A GeoTIFF therefore yields an absolute DSM
in metres; a PNG/JPG yields a relative DSM, clearly labelled. Each prediction also produces an uncertainty map
(spread over 8 flipped/rotated passes), which we measured to track the real error (Spearman 0.78).
Details and evidence: [APPROACH.md](APPROACH.md).

## 4. Where we stand (measured, on data no model trained on)

- **GAMUS test split (2,861 tiles):** RMSE **3.76 m**, MAE **1.68 m**, correlation **0.864**.
  Baselines on the same tiles: zero-shot Depth Anything + scale fit 7.29 m; predict-0 8.62 m.
- **Full pipeline vs USGS 3DEP LiDAR, 8 sites:** mean RMSE **14.53 m** vs 17.70 m (Copernicus 30 m) and
  21.45 m (FABDEM 30 m). We win in cities (23.8 vs 38.3), match on hills, lose on dense forest and flat farmland.
- **Held-out downtowns (Dallas, Charlotte):** 8.77 m, down from 12.16 m once real downtown tiles were added;
  skyscraper bias more than halved (−38.7 → −16.1 m).
- **Coarse imagery:** at 2 m-equivalent resolution RMSE stays ~6.2 m thanks to the blur augmentation; without it
  the same model collapses to 12.0 m. This matters because ISRO's evaluation imagery is likely 0.3–1.6 m.

Full tables, per-site and per-terrain: [RESULTS.md](RESULTS.md).

## 5. Known weaknesses (please don't oversell these in the deck)

1. **Tall structures.** Above 40 m the model still under-predicts by ~16 m. GAMUS has almost no skyscrapers;
   adding downtown tiles helped a lot but did not solve it.
2. **Dense forest canopy.** A radar DEM (Copernicus) still beats us there (7.8 vs 14.9 m).
3. **All training labels are from the USA, France and Switzerland.** No open LiDAR over India was found, so the
   Sikkim demo has no ground truth.
4. **Off-nadir lean.** Tall buildings lean in the imagery; heights land where the roof appears, not the footprint.

## 6. What to do next (ranked, with the reasoning)

1. **Finish the enlarged datasets and retrain ("v4").** Already built on Kaggle: rural/forest **2,275** tiles
   (was 1,098) and non-US **1,600** tiles (France 900 + Switzerland 700, the first non-US data in the project).
   The downtown builder (438 → ~1,700 tiles) still errors and needs a fix. Then one training run on
   GAMUS + rural + downtown + world, same recipe (`kaggle/train/make_kernels.py`).
2. **Measure multi-scale inference** (`DEPTHWIZARD_SCALES=0.7,1,1.4`): implemented, not yet scored on the benchmark.
3. **Feed the coarse DEM into the network** (Copernicus − FABDEM as an extra input channel, Prompt2DEM style):
   the most promising fix for forest, where the image alone is ambiguous.
4. **More tall-building data** (New Zealand and England LiDAR were verified as open; England has great tall
   labels but no open imagery, so it is a label-side test only).

## 7. Rules we kept (worth keeping)

- **Only measured numbers.** Nothing is quoted unless a script in this repo produced it; `bench/report.py`
  regenerates `RESULTS.md` from the raw JSON in `docs/data/`.
- **Splits are by place, never by tile.** Whole regions/cities are held out, because neighbouring tiles share
  buildings; GAMUS's own test tiles are adjacent to its training tiles, which is why we built our own benchmark.
- **Check the reference before trusting a score.** Three LiDAR tiles in Microsoft's catalogue were wrong (one by
  1,900 m) and one product under-records tree tops; both would have silently corrupted our results.
- **Look at the pictures, not only the metrics.** The warehouse roofs the old model flattened to 0 m were obvious
  in a montage and invisible in the averages.
