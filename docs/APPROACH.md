# DepthWizard — approach

SIH 2026 · PS SIH26175 (ISRO / SAC) · single-view optical RGB image → DSM → interactive 3D flythrough.

This document records *why* the system is built the way it is. Every number quoted here is either
measured by us (and reproducible from a script in this repo) or cited with its source. Numbers we
have not measured yet are marked **TBD**.

## 1. Core idea: DSM = terrain + above-ground height

A single overhead image carries strong cues for **objects above the ground** (building footprints,
shadows, roof texture, tree crowns) but weak cues for **absolute terrain elevation** (a flat field at
200 m and at 900 m look the same from above). We therefore split the problem:

```
DSM(x, y) = DTM(x, y)            bare-earth terrain  ← low-res DEM (30 m), optionally corrected with GCPs
          + nDSM(x, y)           height above ground ← neural network, from the image (sub-metre)
```

Evidence:
- Height datasets built for this task (GAMUS, DFC2019/US3D, Geopose) label **nDSM / AGL**, not absolute DSM.
- Prompt2DEM (Rafaeli et al., 2025, arXiv 2507.09681), which feeds SRTM into Depth Anything V2,
  improves absolute-elevation MAE over SRTM alone by only 3–18 %. Most of the terrain signal comes from the DEM.
- Our own error decomposition on the LiDAR benchmark: with perfect above-ground heights, our 30 m terrain source
  leaves only 0.8–6.8 m RMSE per site, while the network accounts for nearly all of the remaining error.

**Terrain source matters:** Copernicus GLO-30, AW3D30 and SRTM are *surface* models — they already
contain (smoothed) buildings and forest. Adding a predicted nDSM on top would double-count them.
We use **FABDEM v1.2** (Copernicus with forests and buildings removed, 30 m, bare earth) as the
default terrain. It is public over HTTPS with no account and is read with windowed COG reads.
Fallbacks: Copernicus GLO-30 / NASADEM (flagged as "surface model, may double-count"), or a DEM the
user uploads (offline mode).

## 2. Above-ground height model

- **Backbone:** Depth Anything V2 (DINOv2 ViT encoder + DPT decoder), fine-tuned to regress metric
  nDSM directly. Published remote-sensing work fine-tunes the same family cheaply (Depth Any Canopy,
  arXiv 2408.04523; Depth2Elevation, TGRS 2025). Small (25 M params) runs on a laptop CPU; Base is
  tried on Kaggle and kept only if it earns its cost.
- **Data (final model):** GAMUS (ISRO's recommended dataset; 5,004 training tiles; DC, New York, Philadelphia)
  **plus two sets we built from free sources**, because GAMUS is city-only and nearly skyscraper-free:
  893 rural/forest/hilly tiles and 335 US-downtown tiles (NAIP 0.6 m + USGS 3DEP LiDAR via Microsoft Planetary
  Computer; `kaggle/naip_prep`, `kaggle/naip_urban`). Downtown tiles hold 4.1 % of pixels above 40 m against
  0.3 % in GAMUS. Whole regions/cities are held out for testing; Dallas and Charlotte are never trained on.
- **Sampling and loss:** tiles are drawn by height class (10 % flat … 12 % very tall) and the loss moves from
  SiLog to Charbonnier during training, following the CHMv2 recipe; both target the long tail of tall structures.
- **Augmentations aimed at ISRO imagery:**
  - *GSD augmentation*: downsample by 1–4× and upsample back, so the model sees the blur of
    0.3–1.6 m Cartosat pan-sharpened products and not only 0.33 m aerial-like tiles.
  - Colour / contrast jitter (different sensor, atmosphere, season), flips and 90° rotations.
- **Loss:** L1 with extra weight on tall pixels (tall buildings and trees are under-predicted by
  plain L1) + a multi-scale gradient term for sharp roof edges.
- **Inference:** tiles of ~0.33 m-equivalent content with overlap and cosine blending, so any image
  size works on CPU.

## 3. Absolute scale (GeoTIFF) and relative output (PNG/JPG)

| Input | Output | How |
|---|---|---|
| GeoTIFF | **Absolute DSM (m, EGM2008 geoid heights)** in the input CRS | FABDEM terrain resampled to the image grid + predicted nDSM; the GSD read from the GeoTIFF sets the model's input scale |
| GeoTIFF + GCPs | Absolute DSM, corrected | robust least-squares fit of a terrain offset/tilt and an nDSM scale to the control points |
| PNG / JPG | **Relative DSM** (height above local ground, m; datum unknown) | nDSM only; the user may enter the pixel size, otherwise a nominal GSD is assumed and the output is clearly labelled *relative* |

A second use of the coarse DEMs: *Copernicus (surface) − FABDEM (bare earth)* is a 30 m estimate of mean
above-ground height. Measured: it helps on sparse farmland but hurts in cities (Copernicus flattens buildings —
in downtown San Francisco it shows 2.4 m above ground where LiDAR shows 28 m), so it is **not** applied by default.

## 4. Uncertainty map

Per-pixel spread across 8 test-time flips/rotations, reported in metres. The literature does not establish
that this correlates with error for sub-metre imagery, so we measured it: Spearman correlation 0.78–0.79 between
spread and absolute error on GAMUS test tiles, with MAE rising monotonically from 0.2 m to 5.5 m across spread
deciles. It is therefore presented as a real uncertainty layer.

## 5. Evaluation plan

1. **GAMUS test split** — overall, per class, per city, against baselines (predict zero; per-tile
   mean oracle; zero-shot Depth Anything + global affine). Caveat: GAMUS test tiles are spatially
   adjacent to training tiles, so we also report a **held-out-city** score (train without one city).
2. **Own LiDAR benchmark (full pipeline, absolute DSM)** — `bench/fetch_site.py` pulls NAIP 0.6 m
   RGB GeoTIFFs with USGS 3DEP LiDAR DSM/DTM/HAG (2 m) from Microsoft Planetary Computer, over
   **urban, sparse, hilly and forested** sites. We score our DSM against the LiDAR DSM, next to
   baselines FABDEM-only and Copernicus-only. Caveats: NAIP is aerial (not satellite); 3DEP heights
   are NAVD88 while FABDEM/Copernicus are EGM2008 (we measure and report the offset); 3DEP's
   pre-computed DTM/HAG fail under dense skyscrapers (seen in downtown San Francisco), so HAG is only
   used as nDSM reference outside such cores.
3. **India sanity checks** (not ground truth): Google Open Buildings 2.5D Temporal building heights
   and Meta 1 m canopy height over Indian cities and forests.

## 6. Application

- Python package `depthwizard` (FastAPI) + Three.js viewer built into the package: `pip install .`
  then `depthwizard` opens the browser. Works offline except for the DEM download (a local DEM can
  be supplied instead).
- Viewer: image draped on the metric height mesh (x/y from the GeoTIFF transform), orbit / fly /
  auto-tour, click to read height and slope, height profile along a line, overlays (height, slope,
  error vs reference, uncertainty), swipe comparison prediction | reference, vertical exaggeration,
  contours, download of DSM / nDSM / uncertainty GeoTIFFs.

## 7. Known limitations (stated up front)

- Training heights are from three US cities; Indian urban form, materials and vegetation differ.
- Terrain accuracy is bounded by the 30 m DEM; the network does not invent terrain detail.
- Off-nadir images: tall buildings lean; heights are predicted where the roof appears, not the footprint.
- No open LiDAR over India was found, so India results are sanity checks against model-derived maps.
