"""Fast tests for the geometry / fusion logic (no network, no model weights needed)."""
import json
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from depthwizard import model as M
from depthwizard import pipeline as P
from depthwizard.calibrate import coarse_scale
from depthwizard.scene import write_scene


class FakeNet(M.HeightModel):
    """Stands in for the network: 'height' = green channel / 10, so outputs are checkable."""
    def __init__(self):
        pass

    def _forward(self, batch):
        return batch[:, 1] * 25.5  # green in [0,1] -> 0..25.5 m


def test_tiled_predict_matches_untiled_and_tta_is_consistent():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (700, 900, 3), dtype=np.uint8)
    nd, sp = FakeNet().predict(img, n_tta=4)
    assert nd.shape == img.shape[:2]
    # blending of overlapping tiles must reproduce a per-pixel function exactly
    np.testing.assert_allclose(nd, img[..., 1] / 10.0, atol=1e-3)
    # the fake net is rotation/flip equivariant, so TTA spread must be ~0
    assert float(np.nanmax(sp)) < 1e-3


def test_small_image_is_padded():
    img = np.full((100, 60, 3), 100, np.uint8)
    nd, _ = FakeNet().predict(img, n_tta=1)
    assert nd.shape == (100, 60) and np.allclose(nd, 10.0, atol=1e-3)


def test_gcp_plane_correction_recovers_offset_and_tilt():
    H, W = 200, 300
    tr = from_origin(500000, 4000000, 2.0, 2.0)
    dsm = np.zeros((H, W), np.float32)
    rows, cols = np.array([10, 50, 150, 190, 100]), np.array([20, 250, 40, 280, 150])
    xs = tr.c + (cols + 0.5) * tr.a
    ys = tr.f + (rows + 0.5) * tr.e
    truth = 3.0 + 0.01 * (xs - xs.mean()) - 0.02 * (ys - ys.mean())
    corr, info = P._gcp_correction(dsm, tr, np.c_[xs, ys, truth])
    assert info["model"] == "plane" and info["n_used"] == 5
    np.testing.assert_allclose(corr[rows, cols], truth, atol=1e-3)


def test_gcp_single_point_is_constant_offset():
    tr = from_origin(0, 100, 1, 1)
    corr, info = P._gcp_correction(np.zeros((100, 100), np.float32), tr, np.array([[10.5, 89.5, 7.0]]))
    assert info["model"] == "constant" and np.allclose(corr, 7.0)


def test_coarse_scale_recovers_factor():
    rng = np.random.default_rng(1)
    nd = rng.uniform(0, 20, (600, 600)).astype(np.float32)
    s = coarse_scale(nd, 1.3 * nd, gsd=1.0, block_m=60)
    assert s and abs(s["scale"] - 1.3) < 1e-3


def test_scene_roundtrip(tmp_path):
    dsm = np.arange(12, dtype=np.float32).reshape(3, 4)
    dsm[0, 0] = np.nan
    m = write_scene(tmp_path, np.zeros((6, 8, 3), np.uint8), {"dsm": dsm}, (2.0, 2.0), title="t")
    back = np.fromfile(tmp_path / "dsm.f32", "<f4").reshape(3, 4)
    assert np.isnan(back[0, 0]) and back[2, 3] == 11
    assert json.load(open(tmp_path / "scene.json"))["width"] == 4 and m["layers"]["dsm"]["stats"]["max"] == 11


def test_png_pipeline_is_relative(tmp_path, monkeypatch):
    from PIL import Image
    img = np.full((300, 400, 3), 128, np.uint8)
    Image.fromarray(img).save(tmp_path / "in.png")
    monkeypatch.setattr(P, "get_model", lambda progress=None: FakeNet())
    rep = P.process(tmp_path / "in.png", tmp_path / "out", gsd=0.66, progress=lambda m: None, n_tta=1)
    assert rep["height_kind"] == "relative"
    with rasterio.open(tmp_path / "out" / "dsm.tif") as s:
        a = s.read(1)
        assert s.crs is None and np.allclose(np.nanmedian(a), 12.8, atol=0.05)


def test_geotiff_pipeline_with_user_dem_is_absolute(tmp_path, monkeypatch):
    tr = from_origin(500000, 4000000, 0.66, 0.66)
    prof = dict(driver="GTiff", width=300, height=200, count=3, dtype="uint8", crs="EPSG:32643", transform=tr)
    with rasterio.open(tmp_path / "in.tif", "w", **prof) as d:
        d.write(np.full((3, 200, 300), 100, np.uint8))
    dem_tr = from_origin(499900, 4000100, 30, 30)
    with rasterio.open(tmp_path / "dem.tif", "w", driver="GTiff", width=20, height=20, count=1, dtype="float32",
                       crs="EPSG:32643", transform=dem_tr) as d:
        d.write(np.full((1, 20, 20), 250.0, np.float32))
    monkeypatch.setattr(P, "get_model", lambda progress=None: FakeNet())
    rep = P.process(tmp_path / "in.tif", tmp_path / "out", dem=tmp_path / "dem.tif", progress=lambda m: None, n_tta=1)
    assert rep["height_kind"] == "absolute"
    with rasterio.open(tmp_path / "out" / "dsm.tif") as s:
        assert s.crs.to_epsg() == 32643
        np.testing.assert_allclose(np.nanmedian(s.read(1)), 250 + 10.0, atol=0.05)
