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

On the GAMUS test split (2,861 tiles, never used for training or model selection), compared with baselines on the same tiles:

| Method | RMSE (m) | MAE (m) | Correlation |
|---|---|---|---|
| DepthWizard, Depth Anything V2 Base | **3.85** | **1.69** | **0.856** |
| DepthWizard, Depth Anything V2 Small (default: runs on a laptop CPU) | 4.07 | 1.79 | 0.837 |
| Zero-shot Depth Anything V2 + global scale fit | 7.29 | 4.79 | 0.276 |
| Per-tile mean height (oracle: knows each tile's true mean) | 6.24 | 4.08 | — |
| Predict 0 m | 8.62 | 4.43 | — |

The full pipeline was also run on NAIP GeoTIFFs and scored against USGS 3DEP LiDAR at sites outside the training cities. Per site and per terrain type, see [docs/RESULTS.md](docs/RESULTS.md). Measured strengths and weaknesses are stated there, including where a plain 30 m DEM does better.

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
