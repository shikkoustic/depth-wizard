import * as THREE from "three";
import { rampTexture } from "./colormaps.js";

// ---------- loading ----------
export async function loadScene(base) {
  const meta = await (await fetch(`${base}/scene.json`)).json();
  const n = meta.width * meta.height;
  const layers = {};
  await Promise.all(Object.entries(meta.layers).map(async ([name, info]) => {
    const buf = await (await fetch(`${base}/${info.file}`)).arrayBuffer();
    const a = new Float32Array(buf);
    if (a.length !== n) throw new Error(`layer ${name}: ${a.length} values, expected ${n}`);
    layers[name] = a;
  }));
  const photo = await new THREE.TextureLoader().loadAsync(`${base}/${meta.texture.file}`);
  photo.colorSpace = THREE.SRGBColorSpace;
  photo.flipY = false; // row 0 = north = v 0, same as the data textures
  photo.anisotropy = 8;
  photo.generateMipmaps = true;
  photo.minFilter = THREE.LinearMipmapLinearFilter;
  return { meta, layers, photo };
}

// ---------- derived rasters ----------
export function slopeDegrees(z, w, h, dx, dy) {
  const out = new Float32Array(w * h);
  const at = (r, c) => z[Math.min(h - 1, Math.max(0, r)) * w + Math.min(w - 1, Math.max(0, c))];
  for (let r = 0; r < h; r++) for (let c = 0; c < w; c++) {
    const gx = (at(r, c + 1) - at(r, c - 1)) / ((Math.min(w - 1, c + 1) - Math.max(0, c - 1)) * dx || 1);
    const gy = (at(r + 1, c) - at(r - 1, c)) / ((Math.min(h - 1, r + 1) - Math.max(0, r - 1)) * dy || 1);
    const s = Math.atan(Math.hypot(gx, gy)) * 180 / Math.PI;
    out[r * w + c] = Number.isFinite(s) ? s : NaN;
  }
  return out;
}

export function difference(a, b) {
  const out = new Float32Array(a.length);
  for (let i = 0; i < a.length; i++) out[i] = a[i] - b[i];
  return out;
}

export function robustRange(a, lo = 0.02, hi = 0.98, symmetric = false) {
  const step = Math.max(1, Math.floor(a.length / 200000)), v = [];
  for (let i = 0; i < a.length; i += step) if (Number.isFinite(a[i])) v.push(a[i]);
  if (!v.length) return [0, 1];
  v.sort((x, y) => x - y);
  let r = [v[Math.floor(lo * (v.length - 1))], v[Math.floor(hi * (v.length - 1))]];
  if (symmetric) { const m = Math.max(Math.abs(r[0]), Math.abs(r[1])) || 1; r = [-m, m]; }
  if (r[1] - r[0] < 1e-6) r[1] = r[0] + 1;
  return r;
}

export function compareStats(pred, ref) {
  let n = 0, se = 0, ae = 0, sb = 0, sp = 0, sr = 0, spp = 0, srr = 0, spr = 0;
  for (let i = 0; i < pred.length; i++) {
    const p = pred[i], r = ref[i];
    if (!Number.isFinite(p) || !Number.isFinite(r)) continue;
    const e = p - r; n++; se += e * e; ae += Math.abs(e); sb += e;
    sp += p; sr += r; spp += p * p; srr += r * r; spr += p * r;
  }
  if (!n) return null;
  const cov = spr / n - (sp / n) * (sr / n), vp = spp / n - (sp / n) ** 2, vr = srr / n - (sr / n) ** 2;
  return { n, rmse: Math.sqrt(se / n), mae: ae / n, bias: sb / n, corr: cov / Math.sqrt(vp * vr) };
}

function floatTexture(a, w, h) {
  const t = new THREE.DataTexture(a, w, h, THREE.RedFormat, THREE.FloatType);
  t.magFilter = t.minFilter = THREE.NearestFilter; // float linear filtering isn't guaranteed in WebGL2
  t.needsUpdate = true;
  return t;
}

// ---------- mesh ----------
const VERT = /* glsl */`
  varying vec2 vUv; varying vec3 vN; varying vec3 vW;
  void main() {
    vUv = uv;
    vN = normalize(transpose(inverse(mat3(modelMatrix))) * normal); // correct under vertical exaggeration
    vec4 w = modelMatrix * vec4(position, 1.0); vW = w.xyz;
    gl_Position = projectionMatrix * viewMatrix * w;
  }`;
const FRAG = /* glsl */`
  uniform sampler2D uPhoto, uData, uRamp, uMask;
  uniform int uMode; uniform vec2 uRange; uniform float uOpacity, uShade, uWalls, uContours, uContourStep;
  uniform vec3 uLight;
  varying vec2 vUv; varying vec3 vN; varying vec3 vW;
  void main() {
    if (texture2D(uMask, vUv).r < 0.5) discard;
    vec3 col = texture2D(uPhoto, vUv).rgb;
    if (uMode > 0) {
      float v = texture2D(uData, vUv).r;
      vec3 c = texture2D(uRamp, vec2(clamp((v - uRange.x) / (uRange.y - uRange.x), 0.0, 1.0), 0.5)).rgb;
      col = mix(col, c, uOpacity);
    }
    vec3 n = normalize(vN);
    float lam = max(dot(n, normalize(uLight)), 0.0);
    col *= mix(1.0, 0.45 + 0.75 * lam, uShade);
    // near-vertical faces (building walls): the photo only holds roof pixels there, so tint them flat grey
    float wall = smoothstep(0.45, 0.2, n.y) * uWalls;
    col = mix(col, vec3(0.58, 0.58, 0.6) * (0.5 + 0.6 * lam), wall);
    if (uContours > 0.5) {
      float f = abs(fract(vW.y / uContourStep + 0.5) - 0.5) / fwidth(vW.y / uContourStep);
      col = mix(col, vec3(0.05), (1.0 - min(f, 1.0)) * 0.6);
    }
    gl_FragColor = vec4(col, 1.0);
    #include <colorspace_fragment>
  }`;

/**
 * Build a terrain mesh for one height layer. The grid is decimated so the vertex count stays
 * under maxVerts; heights are sampled nearest (no smoothing) so roof edges stay sharp.
 */
export function buildTerrain(scene, heights, { photo, maxVerts = 1_500_000, base = 0, shared = {} }) {
  const { width: W, height: H, pixel_size: [dx, dy] } = scene.meta;
  const s = Math.max(1, Math.ceil(Math.sqrt((W * H) / maxVerts)));
  const cols = [], rows = [];
  for (let c = 0; c < W; c += s) cols.push(c); if (cols.at(-1) !== W - 1) cols.push(W - 1);
  for (let r = 0; r < H; r += s) rows.push(r); if (rows.at(-1) !== H - 1) rows.push(H - 1);
  const nx = cols.length, nz = rows.length;
  const pos = new Float32Array(nx * nz * 3), uv = new Float32Array(nx * nz * 2);
  const fill = robustRange(heights, 0.01, 0.99)[0];
  for (let j = 0; j < nz; j++) for (let i = 0; i < nx; i++) {
    const r = rows[j], c = cols[i], k = j * nx + i;
    let z = heights[r * W + c];
    if (!Number.isFinite(z)) z = fill;
    pos[k * 3] = (c + 0.5) * dx - (W * dx) / 2;
    pos[k * 3 + 1] = z - base;
    pos[k * 3 + 2] = (r + 0.5) * dy - (H * dy) / 2;
    uv[k * 2] = (c + 0.5) / W; uv[k * 2 + 1] = (r + 0.5) / H;
  }
  const idx = new Uint32Array((nx - 1) * (nz - 1) * 6);
  let q = 0;
  for (let j = 0; j < nz - 1; j++) for (let i = 0; i < nx - 1; i++) {
    const a = j * nx + i, b = a + 1, c = a + nx, d = c + 1;
    idx.set([a, c, b, b, c, d], q); q += 6;
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  geo.setAttribute("uv", new THREE.BufferAttribute(uv, 2));
  geo.setIndex(new THREE.BufferAttribute(idx, 1));
  geo.computeVertexNormals();
  geo.computeBoundingBox(); geo.computeBoundingSphere();

  if (!shared.mask) {
    const m = new Uint8Array(W * H);
    for (let i = 0; i < m.length; i++) m[i] = Number.isFinite(heights[i]) ? 255 : 0;
    shared.mask = new THREE.DataTexture(m, W, H, THREE.RedFormat, THREE.UnsignedByteType);
    shared.mask.unpackAlignment = 1; // width is rarely a multiple of 4
    shared.mask.needsUpdate = true;
  }
  const mat = new THREE.ShaderMaterial({
    vertexShader: VERT, fragmentShader: FRAG,
    uniforms: {
      uPhoto: { value: photo }, uData: { value: null }, uRamp: { value: rampTexture("height") }, uMask: { value: shared.mask },
      uMode: { value: 0 }, uRange: { value: new THREE.Vector2(0, 1) }, uOpacity: { value: 0.75 },
      uShade: { value: 0.6 }, uWalls: { value: 1 }, uContours: { value: 0 }, uContourStep: { value: 5 },
      uLight: { value: new THREE.Vector3(-0.5, 1.0, -0.35) },
    },
  });
  const mesh = new THREE.Mesh(geo, mat);
  mesh.userData = { stride: s, nx, nz };
  return mesh;
}

export function setOverlay(mesh, cfg) {
  const u = mesh.material.uniforms;
  if (!cfg || !cfg.name) { u.uMode.value = 0; return; }
  // one GPU texture per overlay, shared by every surface mesh
  if (!cfg.tex) cfg.tex = floatTexture(cfg.data, cfg.w, cfg.h);
  if (!cfg.rampTex) cfg.rampTex = rampTexture(cfg.ramp);
  u.uData.value = cfg.tex; u.uMode.value = 1;
  u.uRange.value.set(cfg.range[0], cfg.range[1]);
  u.uRamp.value = cfg.rampTex;
  if (cfg.opacity !== undefined) u.uOpacity.value = cfg.opacity;
}
