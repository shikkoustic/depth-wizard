"""Package a DSM + optical image into a "scene" folder the web viewer loads.

Scene layout (all on one grid, row-major, north-up):
  scene.json      grid size, metric pixel size, CRS, bounds, which layers exist, stats
  texture.jpg     the optical image, resampled to the grid extent (max 4096 px on the long side)
  <layer>.f32     raw little-endian float32 rasters (width*height), NaN = nodata
                  layers: dsm (required), ref (reference DSM), conf (uncertainty, metres), ndsm, dtm
"""
import json, os
import numpy as np
from PIL import Image

LAYERS = ("dsm", "ref", "conf", "ndsm", "dtm")


def _stats(a):
    v = a[np.isfinite(a)]
    if v.size == 0:
        return None
    return dict(min=float(v.min()), max=float(v.max()), mean=float(v.mean()),
                p2=float(np.percentile(v, 2)), p98=float(np.percentile(v, 98)))


def write_scene(out_dir, rgb, layers, pixel_size, *, crs=None, transform=None, height_kind="absolute",
                title="", notes=None, max_tex=4096):
    """rgb: HxWx3 uint8 (any resolution covering the same extent as the layers).
    layers: dict name -> HxW float array on the scene grid; must include 'dsm'.
    pixel_size: (dx, dy) metres per grid cell. height_kind: 'absolute' (metres above datum)
    or 'relative' (no georeference: heights are above local ground, datum unknown)."""
    os.makedirs(out_dir, exist_ok=True)
    dsm = layers["dsm"]
    h, w = dsm.shape
    meta = dict(version=1, title=title, width=w, height=h, pixel_size=[float(pixel_size[0]), float(pixel_size[1])],
                height_kind=height_kind, crs=str(crs) if crs else None,
                transform=list(transform)[:6] if transform is not None else None,
                layers={}, notes=notes or [])
    for name, arr in layers.items():
        if name not in LAYERS or arr is None:
            continue
        a = np.ascontiguousarray(arr, dtype="<f4")
        assert a.shape == (h, w), f"layer {name} shape {a.shape} != dsm {dsm.shape}"
        a.tofile(os.path.join(out_dir, f"{name}.f32"))
        meta["layers"][name] = dict(file=f"{name}.f32", stats=_stats(a))
    img = Image.fromarray(np.asarray(rgb, np.uint8))
    s = min(1.0, max_tex / max(img.size))
    if s < 1.0:
        img = img.resize((round(img.width * s), round(img.height * s)), Image.LANCZOS)
    img.save(os.path.join(out_dir, "texture.jpg"), quality=90)
    meta["texture"] = dict(file="texture.jpg", width=img.width, height=img.height)
    json.dump(meta, open(os.path.join(out_dir, "scene.json"), "w"), indent=1)
    return meta
