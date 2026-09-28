"""Optional scale calibration of the predicted above-ground height against a coarse reference.

coarse_scale(): fits one factor s so that block means of s*nDSM match block means of a coarse
above-ground estimate (e.g. Copernicus GLO-30 surface minus FABDEM bare earth, 30 m). Least squares
through the origin over blocks where both are valid; clamped to [0.5, 2] and reported with the
block correlation so the caller can decide whether to trust it.
"""
import numpy as np


def block_mean(a, b):
    H, W = (a.shape[0] // b) * b, (a.shape[1] // b) * b
    x = a[:H, :W].reshape(H // b, b, W // b, b)
    with np.errstate(invalid="ignore"):
        return np.nanmean(np.nanmean(x, axis=3), axis=1)


def coarse_scale(ndsm, coarse, gsd, block_m=90.0, min_blocks=6):
    b = max(2, int(round(block_m / gsd)))
    p, c = block_mean(ndsm, b), block_mean(coarse, b)
    v = np.isfinite(p) & np.isfinite(c) & (p > 0.5)
    if v.sum() < min_blocks:
        return None
    s = float(np.clip((p[v] * c[v]).sum() / (p[v] ** 2).sum(), 0.5, 2.0))
    r = float(np.corrcoef(p[v], c[v])[0, 1]) if v.sum() > 2 else float("nan")
    return dict(scale=s, block_corr=r, n_blocks=int(v.sum()), block_m=b * gsd)
