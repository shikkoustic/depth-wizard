# DepthWizard

**Single-view satellite image → Digital Surface Model → interactive 3D flythrough.**
Smart India Hackathon 2026 · Problem Statement SIH26175 (ISRO / SAC).

- **GeoTIFF in → absolute DSM (metres) out**: bare-earth terrain from a 30 m DEM (FABDEM) + above-ground height predicted by a fine-tuned Depth Anything V2 network, written as a GeoTIFF in the input CRS.
- **PNG / JPG in → relative DSM out**: height above local ground, clearly labelled as relative.
- **3D viewer** (Three.js): image draped on the metric height mesh; orbit, fly and auto-tour; click for height and slope; height profiles; overlays for height, slope, error vs reference and uncertainty; swipe comparison of prediction and reference.

Why it is built this way: see [docs/APPROACH.md](docs/APPROACH.md).

> Status: work in progress (day 1). Model training and accuracy numbers are not in yet.

## Run

```bash
# Python 3.10+
pip install .
# build the viewer once (needs Node 18+)
cd web && npm install && npm run build && cd ..
depthwizard            # opens http://127.0.0.1:8000
```

## Repository layout

| Path | What |
|---|---|
| `depthwizard/` | Python package: server, pipeline, scene export, geo helpers |
| `web/` | Three.js viewer source (Vite); builds into `depthwizard/static/` |
| `kaggle/` | GPU training and data-prep kernels (run on Kaggle, not locally) |
| `bench/` | LiDAR benchmark builder (NAIP + USGS 3DEP from Microsoft Planetary Computer) |
| `docs/` | Approach, results, limitations |

## Data sources

GAMUS (HF `earthflow/GAMUS`, CC-BY-4.0) · FABDEM v1.2 (Univ. of Bristol, non-commercial licence) · Copernicus DEM GLO-30 · USGS 3DEP LiDAR and NAIP via Microsoft Planetary Computer.
