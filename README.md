# DepthWizard

**Single-view satellite image → Digital Surface Model → interactive 3D flythrough.**
Smart India Hackathon 2026 · Problem Statement SIH26175 (ISRO / SAC).

- **GeoTIFF in → absolute DSM (metres) out.** Bare-earth terrain from a 30 m DEM (FABDEM) is combined with above-ground height from a fine-tuned Depth Anything V2 network. The result is written as a GeoTIFF in a metric CRS.
- **PNG / JPG in → relative DSM out.** Heights are above local ground and clearly labelled relative.
- **Uncertainty map.** Each pixel's spread across 8 flipped/rotated predictions. It tracks real error (Spearman 0.78–0.79 on GAMUS test tiles).
- **3D viewer (Three.js).**
  - The image is draped on the metric height mesh.
  - Orbit, first-person fly and auto-tour modes.
  - Click or hover to read height and slope; draw height profiles.
  - Overlays for height, slope, error against a reference, and uncertainty.
  - Swipe comparison of prediction and reference, with an error histogram.
  - Exports: screenshot, flythrough video, GLB 3D model, and GeoTIFF downloads.

Why it is built this way: [docs/APPROACH.md](docs/APPROACH.md). How it maps to the PS: [docs/PS_SIH26175.md](docs/PS_SIH26175.md).
Measured results, every number generated from result files: [docs/RESULTS.md](docs/RESULTS.md).

## Install and run

Requires Python 3.10+. Node 18+ is needed only to rebuild the viewer.

```bash
pip install .
cd web && npm install && npm run build && cd ..   # builds the viewer into the package
depthwizard                                       # opens http://127.0.0.1:8000
```

- **Model weights:** downloaded once to `~/.cache/depthwizard/` on first use. To use a local checkpoint, pass `--weights path/to/ndsm_small.pt`.
- **Offline use:** works without a network if the weights are cached and you upload your own DEM. Otherwise the terrain is fetched from FABDEM over HTTPS, reading only the needed window.

Command line, without the UI:

```bash
depthwizard process scene.tif -o out/                          # absolute DSM + nDSM + DTM + uncertainty GeoTIFFs
depthwizard process image.png -o out/ --gsd 0.5                # relative DSM
depthwizard process scene.tif -o out/ --reference lidar.tif    # also scores against a reference DSM
depthwizard process scene.tif -o out/ --gcps gcps.csv          # terrain correction from ground-control points (x,y,z)
```

## Headline measured results

Everything below is measured on data no model trained on, and is regenerated from result files by `bench/report.py`
into [docs/RESULTS.md](docs/RESULTS.md).

**1. Above-ground height, GAMUS test split (2,861 tiles):**

| Method | RMSE (m) | MAE (m) | Correlation |
|---|---|---|---|
| DepthWizard (Depth Anything V2 Base, final model) | **3.76** | **1.68** | **0.864** |
| Zero-shot Depth Anything V2 + global scale fit | 7.29 | 4.79 | 0.276 |
| Per-tile mean height (oracle: knows each tile's true mean) | 6.24 | 4.08 | — |
| Predict 0 m | 8.62 | 4.43 | — |

**2. Full pipeline (absolute DSM) vs USGS 3DEP LiDAR, 8 sites outside every training set** — RMSE (m):

| Method | mean | terrain-balanced | worst site | urban | hilly | forest | sparse |
|---|---|---|---|---|---|---|---|
| **DepthWizard** | **14.53** | **12.34** | **36.69** | **23.79** | **6.22** | 14.89 | 4.44 |
| Copernicus GLO-30 alone | 17.70 | 13.70 | 59.31 | 38.33 | 6.34 | **7.83** | 2.30 |
| FABDEM 30 m alone | 21.45 | 17.70 | 60.30 | 39.94 | 10.00 | 17.48 | **3.40** |

We beat both 30 m DEMs overall and in cities, match them on hills, and still lose on dense forest canopy and flat
farmland, where a radar DEM is hard to beat. Adding real downtown training data cut held-out city error by 28 %
(12.16 → 8.77 m) and more than halved the skyscraper bias (−38.7 → −16.1 m).

**3. Robustness to coarser imagery** (GAMUS test tiles degraded to 1.3 m and 2 m, as pan-sharpened satellite
imagery would appear): RMSE stays 5.9 → 6.2 m with our blur augmentation, against 5.9 → 12.0 m without it.

**4. Uncertainty map**: the spread across 8 flipped/rotated predictions tracks the real error (Spearman 0.78).

## Repository layout

| Path | What |
|---|---|
| `depthwizard/` | Python package: server, pipeline, model inference, terrain, scene export |
| `web/` | Three.js viewer source (Vite); builds into `depthwizard/static/` |
| `kaggle/` | Data-prep, training and evaluation kernels (run on Kaggle GPUs) |
| `bench/` | LiDAR benchmark: site fetcher, scorer, report generator |
| `tests/` | Fast tests: tiling and blending, GCP fit, calibration, both pipeline paths |
| `docs/` | Approach, PS checklist, results and their source JSON |

## Data sources and licences

- GAMUS (HF `earthflow/GAMUS`, CC-BY-4.0)
- FABDEM v1.2 (University of Bristol, non-commercial licence)
- Copernicus DEM GLO-30
- NAIP imagery, USGS 3DEP LiDAR and Sentinel-2 via Microsoft Planetary Computer
- Depth Anything V2 (Small: Apache-2.0; Base: CC-BY-NC-4.0)
