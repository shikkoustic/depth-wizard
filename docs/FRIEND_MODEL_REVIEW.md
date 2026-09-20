# Review of the other DepthWizard model (checkpoint shared 2026-09-20)

All numbers below were measured by us on the **same 16 GAMUS test tiles** (the official test split; neither
of our models trained on them), with the same input images and the same metric code. Their checkpoint was
loaded in PyTorch's safe weights-only mode. Reproduce with `bench/robustness.py`-style loading; see
"How it was measured" at the end.

## 1. What the checkpoint is

- Depth Anything V2 **Small** (DINOv2 ViT-S + DPT head), same architecture family as ours.
- Fine-tuned on GAMUS for 8 epochs (their log: 98,528 crops from 6,998 tiles, SILog + a tall-building gradient term).
- Ships with two calibration files (`gamus_real_model.npz`, `local_calibrator.npz`): an affine and a local planar
  calibrator with 12 and 8 coefficients, applied by their backend. We could not reproduce those features from the
  checkpoint alone, so we measured the network's raw output and also gave it a best-case linear calibration.

## 2. Measured comparison (GAMUS test tiles, never trained on)

| Model | RMSE | MAE | Correlation |
|---|---|---|---|
| Ours, Base (`v3a_base`) | **5.80 m** | **2.83 m** | **0.821** |
| Ours, Small (`v3a_small`) | 5.67 m | 2.93 m | 0.830 |
| Their model, standard preprocessing (ImageNet normalisation) | 10.27 m | 6.08 m | 0.185 |
| Their model, best preprocessing we found (no normalisation) | 9.70 m | 5.56 m | 0.534 |
| Their model, best preprocessing + linear calibration fitted on held-out tiles | 9.36 m | 5.92 m | 0.598 |

We also tried their model at 2× and 4× zoom and at 1036 px input in case it expects the native 0.33 m GAMUS
scale; none beat the row above. Correlation is the fair headline because no calibration can change it.

**Their reported numbers (validation MAE 2.86 m, RMSE 3.86 m, r 0.884) do not reproduce on held-out data.**

## 3. Why — the most likely causes, in order

1. **The validation set was probably trained on.** Their log reports 6,998 tiles used for training crops, while
   GAMUS train+val is 5,863 tiles, and the sample file it prints comes from `images/val/`. If val tiles are in the
   training pool, "validation MAE" measures memorisation, not generalisation. This alone explains a 2.86 m
   validation MAE next to 5.6 m on true held-out tiles.
2. **Preprocessing mismatch between training and release.** The network scores much better *without* ImageNet
   normalisation, which suggests it was trained on raw 0–1 inputs while the standard Depth Anything pipeline
   normalises. Anyone loading the checkpoint the usual way gets the 0.185-correlation result.
3. **The head does not output metres.** Raw outputs span ~0.3–15 m where the truth spans 0–35 m, so the model
   depends on external calibration files. That is fragile: the calibration has to be refitted per scene/domain,
   and it cannot fix ranking errors (correlation).
4. **SILog on mostly-zero targets.** GAMUS is dominated by ground pixels at 0 m. A log-space loss concentrates
   effort there and biases tall structures low (the CHMv2 paper reports the same effect and fixes it by moving
   from SILog to a Charbonnier loss during training).

## 4. Implementation plan for their model (ranked by expected gain per effort)

**Step 1 — Fix the evaluation before changing anything (½ day).**
- Train only on `images/train/`; keep `val/` for checkpoint selection and touch `test/` once, at the end.
- Report pixel-pooled RMSE, MAE, correlation **and** per-height-band errors (0–2, 2–5, 5–10, 10–20, 20–40, 40+ m).
- Always include baselines on the same tiles: predict 0 m everywhere; predict each tile's mean height (an oracle);
  zero-shot Depth Anything + one global affine. Without these a number means nothing.

**Step 2 — Make the network output metres directly (½ day).**
- Drop the external affine/local calibrators; train the head to regress metres with an L1/Charbonnier loss.
- Keep SILog only for the first ~30 % of steps, then switch to Charbonnier (CHMv2 recipe), and clamp targets at 0.
- Publish the exact preprocessing with the weights (input size, normalisation, tile size, metres per pixel).

**Step 3 — Fix the data imbalance that caps tall structures (1 day).**
- Sample tiles by height class rather than uniformly (e.g. 10 % flat, 30 % low, 30 % mid, 18 % tall, 12 % very tall).
- GAMUS has almost no skyscrapers (0.3 % of pixels above 40 m), so add downtown tiles: NAIP + USGS 3DEP LiDAR
  over US downtowns is free and needs no account (our `kaggle/naip_urban/naip_urban.py` builds 438 such tiles in
  ~1 minute; 4.1 % of its pixels are above 40 m).

**Step 4 — Robustness to the imagery ISRO will actually use (½ day).**
- Add a blur/resolution-degradation augmentation (downsample 1.3–3× and upsample back). Measured effect in our
  ablation: without it, RMSE on 2 m-equivalent imagery collapses from 6.2 m to 12.0 m (correlation 0.79 → 0.05).
- Add flips, 90° rotations and colour jitter.

**Step 5 — Only then, more data and bigger backbones (1 day+).**
- Rural/forest tiles (our `naip_prep`) fix vegetation; DFC2019 WorldView tiles (via SynRS3D) add real satellite
  geometry with off-nadir lean.
- Depth Anything V2 Base gains ~0.2 m RMSE over Small in our runs, at ~4× CPU inference cost.

## 5. Can the two models be combined?

Measured, not assumed: we calibrated their model on 8 tiles and scored the ensemble on the other 8.

| Combination | RMSE | MAE | Correlation |
|---|---|---|---|
| Ours alone | **6.19 m** | **2.41 m** | 0.841 |
| Theirs alone (calibrated) | 9.36 m | 5.92 m | 0.598 |
| 50 / 50 | 6.89 m | 3.73 m | 0.804 |
| 70 / 30 | 6.33 m | 3.08 m | 0.840 |
| 80 / 20 | 6.19 m | 2.80 m | **0.846** |

**Conclusion: no benefit today.** At best (20 % weight) the ensemble matches our RMSE and worsens MAE.
Averaging helps only when models are of similar accuracy and make *different* mistakes. The useful path is for
their model to reach comparable accuracy with steps 1–4 first; then an ensemble is worth re-testing — and it is
cheap to re-test, because our inference module already accepts several checkpoints
(`DEPTHWIZARD_WEIGHTS=a.pt,b.pt` averages them and uses their disagreement as extra uncertainty).

A better immediate use of two people: train the same recipe with different seeds/backbones (Small and Base) and
ensemble those, which we measured as a real gain (mean LiDAR-benchmark RMSE 16.49 → 16.04 m).

## How it was measured

- Tiles: `runs/kaggle/small_main/test_samples.npz` — 16 GAMUS **test** tiles (RGB 512×512 at 0.66 m/px, LiDAR AGL).
- Their checkpoint: `torch.load(..., weights_only=True)`, loaded into `Depth-Anything-V2-Small` config; outputs
  resized to 512 and compared with LiDAR heights on finite pixels.
- Preprocessing variants tried: ImageNet normalisation on/off; input 518 / 1036; centre-crop zoom ×1, ×2, ×4.
- Ensemble: their output linearly calibrated on tiles 1–8, both models scored on tiles 9–16.
