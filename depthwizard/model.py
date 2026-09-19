"""Above-ground height (nDSM) inference with a fine-tuned Depth Anything V2.

The network was trained on GAMUS tiles resampled to MODEL_GSD metres per pixel and 512x512 px
(fed to the network at 518x518). Inference reproduces that: the caller resamples the image to
MODEL_GSD, and we run overlapping 512 px tiles blended with a smooth window. Test-time augmentation
(flips / 90° rotations) gives both a better mean and a per-pixel spread used as the uncertainty map.
"""
import os
import numpy as np

MODEL_GSD = 0.66          # metres per pixel the model was trained at (GAMUS 0.33 m tiles downsampled 2x)
TILE, IN = 512, 518
MEAN = np.array([0.485, 0.456, 0.406], np.float32)[:, None, None]
STD = np.array([0.229, 0.224, 0.225], np.float32)[:, None, None]
WEIGHTS_URL = os.environ.get("DEPTHWIZARD_WEIGHTS_URL",
                             "https://github.com/shikkoustic/depth-wizard/releases/latest/download/ndsm_small.pt")
CACHE = os.path.join(os.path.expanduser("~"), ".cache", "depthwizard")
DEFAULT_WEIGHTS = os.environ.get("DEPTHWIZARD_WEIGHTS", os.path.join(CACHE, "ndsm_small.pt"))


def ensure_weights(path=DEFAULT_WEIGHTS, url=WEIGHTS_URL, progress=None):
    """Return a local weights path, downloading the released checkpoint on first use."""
    if os.path.exists(path):
        return path
    import urllib.request
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".part"
    with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:
        total, done = int(r.headers.get("Content-Length") or 0), 0
        while chunk := r.read(1 << 20):
            f.write(chunk); done += len(chunk)
            if progress and total:
                progress(f"Downloading model weights: {done / total:.0%} of {total / 1e6:.0f} MB")
    os.replace(tmp, path)
    return path


class HeightModel:
    def __init__(self, weights=None, variant="Small", device=None, threads=None, progress=None):
        import torch
        from transformers import AutoConfig, AutoModelForDepthEstimation
        self.torch = torch
        # CPU by default: predictable memory on small machines (set DEPTHWIZARD_DEVICE=mps/cuda to override)
        self.device = device or os.environ.get("DEPTHWIZARD_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
        torch.set_num_threads(threads or int(os.environ.get("DEPTHWIZARD_THREADS", "4")))
        weights = weights or DEFAULT_WEIGHTS
        # several checkpoints (comma-separated) form an ensemble: predictions are averaged and their
        # disagreement adds to the uncertainty map
        paths = [p for p in str(weights).split(",") if p]
        self.nets, self.in_sizes, self.meta = [], [], {}
        for path in paths:
            if not os.path.exists(path):
                try:
                    path = ensure_weights(path, progress=progress)
                except Exception as e:
                    raise FileNotFoundError(f"model weights not found at {path} and download from {WEIGHTS_URL} failed "
                                            f"({type(e).__name__}: {e}); pass --weights or set DEPTHWIZARD_WEIGHTS") from e
            ck = torch.load(path, map_location="cpu", weights_only=True)
            # newer checkpoints carry metadata (backbone size, input size); older ones are a bare state_dict
            meta = ck["meta"] if isinstance(ck, dict) and "meta" in ck else {}
            sd = ck["state_dict"] if isinstance(ck, dict) and "state_dict" in ck else ck
            # architecture config ships with the package, so no network access is needed at inference time
            cfg = AutoConfig.from_pretrained(os.path.join(os.path.dirname(__file__), "configs", meta.get("variant", variant)))
            net = AutoModelForDepthEstimation.from_config(cfg)
            net.load_state_dict({k: v.float() for k, v in sd.items()})
            self.nets.append(net.to(self.device).eval()); self.in_sizes.append(int(meta.get("in_size", IN)))
            self.meta = self.meta or meta
        self.in_size = self.in_sizes[0]

    def _forward(self, batch, net=0):
        """batch: float32 (B,3,512,512) in [0,1] -> (B,512,512) heights from ensemble member `net`."""
        torch, F = self.torch, self.torch.nn.functional
        x = torch.from_numpy(batch).to(self.device)
        size = self.in_sizes[net]
        x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
        x = (x - torch.from_numpy(MEAN).to(self.device)) / torch.from_numpy(STD).to(self.device)
        with torch.no_grad():
            p = self.nets[net](pixel_values=x).predicted_depth[:, None]
            p = F.interpolate(p, size=batch.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
        return p.float().cpu().numpy()

    def _predict_tile(self, t, n_tta):
        """t: (3,512,512) float. Returns mean and std over n_tta dihedral variants (std=0 if n_tta==1)."""
        variants = [(k, f) for f in (False, True) for k in range(4)][:max(1, n_tta)]
        xs = []
        for k, f in variants:
            v = np.rot90(t, k, axes=(1, 2)); v = v[:, :, ::-1] if f else v
            xs.append(np.ascontiguousarray(v))
        back = []
        for net in range(len(getattr(self, "nets", [None]))):  # every ensemble member sees every variant
            outs = np.concatenate([self._forward(x[None], net) if hasattr(self, "nets") else self._forward(x[None]) for x in xs])
            back += self._unrotate(outs, variants)
        back = np.stack(back)
        return back.mean(0), (back.std(0) if len(back) > 1 else np.zeros_like(back[0]))

    @staticmethod
    def _unrotate(outs, variants):
        back = []
        for o, (k, f) in zip(outs, variants):
            o = o[:, ::-1] if f else o
            back.append(np.rot90(o, -k))
        return back

    def predict(self, rgb, n_tta=4, overlap=128, progress=None):
        """rgb: (H,W,3) uint8 already at MODEL_GSD. Returns (ndsm, spread) float32 (H,W)."""
        H, W = rgb.shape[:2]
        ph, pw = max(0, TILE - H), max(0, TILE - W)
        img = np.pad(rgb, ((0, ph), (0, pw), (0, 0)), mode="reflect") if (ph or pw) else rgb
        Hp, Wp = img.shape[:2]
        step = TILE - overlap
        ys = list(range(0, max(1, Hp - TILE) + 1, step)); xs = list(range(0, max(1, Wp - TILE) + 1, step))
        if ys[-1] != Hp - TILE: ys.append(Hp - TILE)
        if xs[-1] != Wp - TILE: xs.append(Wp - TILE)
        w1 = np.hanning(TILE + 2)[1:-1].astype(np.float32); win = np.outer(w1, w1) + 1e-3
        acc = np.zeros((Hp, Wp), np.float32); acc2 = np.zeros_like(acc); wsum = np.zeros_like(acc)
        n, total = 0, len(ys) * len(xs)
        for y in ys:
            for x in xs:
                t = img[y:y + TILE, x:x + TILE].transpose(2, 0, 1).astype(np.float32) / 255.
                m, s = self._predict_tile(t, n_tta)
                acc[y:y + TILE, x:x + TILE] += m * win; acc2[y:y + TILE, x:x + TILE] += s * win
                wsum[y:y + TILE, x:x + TILE] += win
                n += 1
                if progress: progress(n, total)
        nd = (acc / wsum)[:H, :W]; sp = (acc2 / wsum)[:H, :W]
        return np.clip(nd, 0, None).astype(np.float32), sp.astype(np.float32)
