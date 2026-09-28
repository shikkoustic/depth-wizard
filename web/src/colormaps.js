import * as THREE from "three";

// Colour ramps as control points (t in 0..1). Perceptually ordered; diverging map for differences.
const RAMPS = {
  height: [[0, "#1d2b53"], [0.2, "#2a6f97"], [0.4, "#3fa67a"], [0.6, "#c9c34a"], [0.8, "#e07a3f"], [1, "#fbe9e0"]],
  slope: [[0, "#f7fbf2"], [0.25, "#c7e9a8"], [0.5, "#f2d45c"], [0.75, "#e0692f"], [1, "#7a1020"]],
  diff: [[0, "#2c5aa0"], [0.25, "#7fb0dc"], [0.5, "#f4f4f2"], [0.75, "#e8906c"], [1, "#b2182b"]],
  conf: [[0, "#f2f8f0"], [0.35, "#9fd4b0"], [0.7, "#e2a33a"], [1, "#8c1c3c"]],
};

export function rampColor(name, t) {
  const r = RAMPS[name];
  t = Math.min(1, Math.max(0, t));
  for (let i = 1; i < r.length; i++) {
    if (t <= r[i][0]) {
      const [t0, c0] = r[i - 1], [t1, c1] = r[i];
      return new THREE.Color(c0).lerp(new THREE.Color(c1), (t - t0) / (t1 - t0 || 1));
    }
  }
  return new THREE.Color(r[r.length - 1][1]);
}

export function rampTexture(name) {
  const n = 256, data = new Uint8Array(n * 4);
  for (let i = 0; i < n; i++) {
    const c = rampColor(name, i / (n - 1));
    data.set([c.r * 255, c.g * 255, c.b * 255, 255], i * 4);
  }
  const tex = new THREE.DataTexture(data, n, 1, THREE.RGBAFormat);
  tex.colorSpace = THREE.SRGBColorSpace; // ramp stops are sRGB hex colours
  tex.magFilter = tex.minFilter = THREE.LinearFilter;
  tex.needsUpdate = true;
  return tex;
}

export function rampCSS(name) {
  return `linear-gradient(90deg, ${RAMPS[name].map(([t, c]) => `${c} ${t * 100}%`).join(", ")})`;
}
