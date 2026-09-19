import * as THREE from "three";
import { MapControls } from "three/addons/controls/MapControls.js";
import { PointerLockControls } from "three/addons/controls/PointerLockControls.js";
import { loadScene, buildTerrain, setOverlay, slopeDegrees, difference, robustRange, compareStats } from "./terrain.js";
import { rampCSS } from "./colormaps.js";

const $ = (id) => document.getElementById(id);
const fmt = (v, d = 1) => (Number.isFinite(v) ? v.toFixed(d) : "—");

// ---------------- renderer / camera ----------------
const canvas = $("gl");
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, logarithmicDepthBuffer: false });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
const scene3 = new THREE.Scene();
scene3.background = new THREE.Color("#9fb4c8");
scene3.fog = new THREE.Fog("#9fb4c8", 1e6, 2e6);
const camera = new THREE.PerspectiveCamera(55, 1, 0.5, 1e6);
const orbit = new MapControls(camera, canvas);
orbit.enableDamping = true; orbit.dampingFactor = 0.08;
orbit.maxPolarAngle = Math.PI * 0.495;
const fly = new PointerLockControls(camera, canvas);
const markers = new THREE.Group(); scene3.add(markers);

const S = { // app state
  data: null, base: 0, exag: 1, meshes: {}, surface: "dsm", overlay: "", derived: {},
  nav: "orbit", tool: "probe", swipe: false, swipeX: 0.5, profilePts: [], keys: {}, tourT: 0,
};

function resize() {
  const r = canvas.parentElement.getBoundingClientRect();
  renderer.setSize(r.width, r.height, false);
  camera.aspect = r.width / r.height; camera.updateProjectionMatrix();
}
window.addEventListener("resize", resize); resize();

// ---------------- scene loading ----------------
async function openScene(url) {
  $("empty").textContent = "Loading scene…"; $("empty").hidden = false;
  const data = await loadScene(url);
  for (const m of Object.values(S.meshes)) { scene3.remove(m); m.geometry.dispose(); m.material.dispose(); }
  S.meshes = {}; S.derived = {}; S.profilePts = []; markers.clear();
  S.data = data; S.url = url;
  const { meta, layers } = data, W = meta.width, H = meta.height, [dx, dy] = meta.pixel_size;
  S.base = robustRange(layers.dsm, 0.01, 0.99)[0];

  const shared = {};
  const surfaces = [["dsm", "Prediction (DSM)"], ["ref", "Reference DSM"], ["dtm", "Terrain only (DTM)"]].filter(([k]) => layers[k]);
  for (const [k] of surfaces) {
    S.meshes[k] = buildTerrain(data, layers[k], { photo: data.photo, base: S.base, shared });
    scene3.add(S.meshes[k]);
  }
  $("surface").innerHTML = surfaces.map(([k, t]) => `<option value="${k}">${t}</option>`).join("");
  S.surface = "dsm";

  S.derived.slope = slopeDegrees(layers.dsm, W, H, dx, dy);
  if (layers.ref) S.derived.diff = difference(layers.dsm, layers.ref);
  $("overlay").querySelector('[value=diff]').disabled = !layers.ref;
  $("overlay").querySelector('[value=conf]').disabled = !layers.conf;
  $("swipe").disabled = !layers.ref; $("swipe").checked = false; S.swipe = false;

  // info panel
  const rel = meta.height_kind === "relative";
  $("scene-info").innerHTML =
    `<span class="badge ${rel ? "rel" : ""}">${rel ? "RELATIVE DSM (no georeference)" : "ABSOLUTE DSM (m)"}</span>\n` +
    `<b>${meta.title || ""}</b>\n${W} × ${H} px · ${fmt(dx, 2)} m/px · ${fmt(W * dx / 1000, 2)} × ${fmt(H * dy / 1000, 2)} km\n` +
    (meta.crs ? `CRS: ${shortCRS(meta.crs)}\n` : "") + (meta.notes || []).join("\n");
  const acc = layers.ref ? compareStats(layers.dsm, layers.ref) : null;
  $("sec-accuracy").hidden = !acc;
  if (acc) {
    $("acc").innerHTML = `<table><tr><td>RMSE</td><td>${fmt(acc.rmse, 2)} m</td></tr><tr><td>MAE</td><td>${fmt(acc.mae, 2)} m</td></tr>` +
      `<tr><td>Bias (pred − ref)</td><td>${fmt(acc.bias, 2)} m</td></tr><tr><td>Correlation</td><td>${fmt(acc.corr, 3)}</td></tr>` +
      `<tr><td>Pixels compared</td><td>${acc.n.toLocaleString()}</td></tr></table>`;
  }
  $("downloads").innerHTML = (meta.downloads || []).map((d) => `<a href="${url}/${d.file}" download>${d.label}</a>`).join("") || "—";

  applySurface(); applyOverlay(); applyStyle(); homeView();
  $("empty").hidden = true;
}

function shortCRS(c) { const m = /EPSG[":, ]+(\d{4,5})/.exec(c); return m ? `EPSG:${m[1]}` : c.slice(0, 40); }

// ---------------- height lookup (full resolution, exact) ----------------
function pixelAt(x, z) {
  const { width: W, height: H, pixel_size: [dx, dy] } = S.data.meta;
  const c = Math.floor((x + (W * dx) / 2) / dx), r = Math.floor((z + (H * dy) / 2) / dy);
  return c >= 0 && r >= 0 && c < W && r < H ? { r, c, i: r * W + c } : null;
}
function surfaceHeightAt(x, z) { // world y of the current surface (with exaggeration)
  const p = S.data && pixelAt(x, z); if (!p) return 0;
  const h = S.data.layers[S.surface][p.i];
  return Number.isFinite(h) ? (h - S.base) * S.exag : 0;
}

// ---------------- UI wiring ----------------
function seg(id, cb) {
  $(id).addEventListener("click", (e) => {
    const b = e.target.closest("button"); if (!b) return;
    for (const x of $(id).children) x.classList.toggle("on", x === b);
    cb(b.dataset.v);
  });
}
seg("nav-mode", (v) => setNav(v));
seg("tool", (v) => { S.tool = v; S.profilePts = []; markers.clear(); $("profile-chart").hidden = v !== "profile";
  $("probe").textContent = v === "profile" ? "Click two points to draw a height profile." : "Click the terrain to read height and slope."; });
$("surface").onchange = (e) => { S.surface = e.target.value; applySurface(); };
$("overlay").onchange = (e) => { S.overlay = e.target.value; applyOverlay(); };
$("opacity").oninput = applyStyle; $("shade").onchange = applyStyle; $("walls").onchange = applyStyle;
$("contours").onchange = applyStyle; $("contour-step").oninput = applyStyle;
$("exag").oninput = (e) => { S.exag = +e.target.value; $("exag-v").textContent = `${S.exag.toFixed(1)}×`; applyStyle(); };
$("swipe").onchange = (e) => { S.swipe = e.target.checked; $("swipe-handle").hidden = !S.swipe; placeSwipe(); applySurface(); };

function applySurface() {
  for (const [k, m] of Object.entries(S.meshes)) m.visible = S.swipe ? (k === "dsm" || k === "ref") : k === S.surface;
}
function applyOverlay() {
  if (!S.data) return;
  const { meta, layers } = S.data, W = meta.width, H = meta.height;
  const cfg = {
    height: { data: layers.dsm, ramp: "height", range: robustRange(layers.dsm), unit: "m" },
    slope: { data: S.derived.slope, ramp: "slope", range: [0, 60], unit: "°" },
    diff: S.derived.diff && { data: S.derived.diff, ramp: "diff", range: robustRange(S.derived.diff, 0.02, 0.98, true), unit: "m" },
    conf: layers.conf && { data: layers.conf, ramp: "conf", range: robustRange(layers.conf, 0.0, 0.98), unit: "m" },
  }[S.overlay];
  for (const m of Object.values(S.meshes)) setOverlay(m, cfg ? { name: S.overlay, w: W, h: H, ...cfg } : {});
  $("legend").hidden = !cfg;
  if (cfg) {
    $("legend").querySelector(".bar").style.background = rampCSS(cfg.ramp);
    const [a, b] = cfg.range, t = $("legend").querySelectorAll(".ticks span");
    t[0].textContent = `${fmt(a)} ${cfg.unit}`; t[1].textContent = fmt((a + b) / 2); t[2].textContent = `${fmt(b)} ${cfg.unit}`;
  }
  applyStyle();
}
function applyStyle() {
  for (const m of Object.values(S.meshes)) {
    const u = m.material.uniforms;
    u.uOpacity.value = +$("opacity").value; u.uShade.value = $("shade").checked ? 0.6 : 0;
    u.uWalls.value = $("walls").checked ? 1 : 0; u.uContours.value = $("contours").checked ? 1 : 0;
    u.uContourStep.value = Math.max(0.5, +$("contour-step").value || 5) * S.exag;
    m.scale.y = S.exag;
  }
  markers.scale.y = S.exag;
}

// ---------------- navigation ----------------
function sceneSize() { const { width: W, height: H, pixel_size: [dx, dy] } = S.data.meta; return Math.max(W * dx, H * dy); }
function homeView() {
  const L = sceneSize();
  orbit.target.set(0, surfaceHeightAt(0, 0), 0);
  camera.position.set(-0.35 * L, 0.55 * L + orbit.target.y, 0.75 * L);
  camera.near = Math.max(0.1, L / 5000); camera.far = L * 20; camera.updateProjectionMatrix();
  scene3.fog.near = L * 3; scene3.fog.far = L * 12;
  orbit.update();
}
function setNav(v) {
  S.nav = v;
  orbit.enabled = v === "orbit";
  if (v !== "fly" && fly.isLocked) fly.unlock();
  if (v === "tour") S.tourT = 0;
  if (v === "orbit") { const d = new THREE.Vector3(); camera.getWorldDirection(d);
    orbit.target.copy(camera.position).addScaledVector(d, sceneSize() * 0.3); orbit.update(); }
}
canvas.addEventListener("click", (e) => {
  if (S.nav === "fly" && !fly.isLocked) { fly.lock(); return; }
  if (S.nav === "fly") return;
  if (!S.data || e.button !== 0 || S.dragged) return;
  pick(e);
});
let downAt = null;
canvas.addEventListener("pointerdown", (e) => { downAt = [e.clientX, e.clientY]; S.dragged = false; });
canvas.addEventListener("pointermove", (e) => { if (downAt && Math.hypot(e.clientX - downAt[0], e.clientY - downAt[1]) > 4) S.dragged = true; });
window.addEventListener("keydown", (e) => { S.keys[e.code] = true; if (e.code === "KeyH" && S.data) homeView(); });
window.addEventListener("keyup", (e) => { S.keys[e.code] = false; });

function updateFly(dt) {
  if (!fly.isLocked) return;
  const L = sceneSize(), sp = (S.keys.ShiftLeft || S.keys.ShiftRight ? 0.25 : 0.06) * L * dt;
  const f = new THREE.Vector3(); camera.getWorldDirection(f);
  const r = new THREE.Vector3().crossVectors(f, camera.up).normalize();
  if (S.keys.KeyW) camera.position.addScaledVector(f, sp);
  if (S.keys.KeyS) camera.position.addScaledVector(f, -sp);
  if (S.keys.KeyD) camera.position.addScaledVector(r, sp);
  if (S.keys.KeyA) camera.position.addScaledVector(r, -sp);
  if (S.keys.KeyE) camera.position.y += sp;
  if (S.keys.KeyQ) camera.position.y -= sp;
  const g = surfaceHeightAt(camera.position.x, camera.position.z) + 2; // never below the surface
  if (camera.position.y < g) camera.position.y = g;
}
function updateTour(dt) {
  const L = sceneSize(); S.tourT += dt * 0.05;
  const a = S.tourT * Math.PI * 2, R = 0.42 * L;
  const x = Math.cos(a) * R, z = Math.sin(a) * R;
  const y = Math.max(surfaceHeightAt(x, z), surfaceHeightAt(0, 0)) + 0.18 * L;
  camera.position.lerp(new THREE.Vector3(x, y, z), Math.min(1, dt * 2));
  camera.lookAt(0, surfaceHeightAt(0, 0), 0);
}

// ---------------- picking / measuring ----------------
const ray = new THREE.Raycaster();
function pick(e) {
  const rect = canvas.getBoundingClientRect();
  const ndc = new THREE.Vector2(((e.clientX - rect.left) / rect.width) * 2 - 1, -((e.clientY - rect.top) / rect.height) * 2 + 1);
  ray.setFromCamera(ndc, camera);
  const vis = Object.values(S.meshes).filter((m) => m.visible);
  const hit = ray.intersectObjects(vis, false)[0]; if (!hit) return;
  const p = pixelAt(hit.point.x, hit.point.z); if (!p) return;
  if (S.tool === "probe") probe(p, hit.point); else profilePoint(p, hit.point);
}
function worldEN(p) { // pixel -> easting/northing from the GeoTIFF transform, if any
  const t = S.data.meta.transform; if (!t) return null;
  const x = t[2] + (p.c + 0.5) * t[0] + (p.r + 0.5) * t[1], y = t[5] + (p.c + 0.5) * t[3] + (p.r + 0.5) * t[4];
  return [x, y];
}
function marker(pt, color = "#e39a4f") {
  const L = sceneSize(), m = new THREE.Mesh(new THREE.SphereGeometry(L / 250, 12, 8), new THREE.MeshBasicMaterial({ color }));
  m.position.set(pt.x, pt.y / S.exag, pt.z); m.scale.y = 1 / S.exag; markers.add(m); return m;
}
function probe(p, pt) {
  markers.clear(); marker(pt);
  const { layers, meta } = S.data, rel = meta.height_kind === "relative", en = worldEN(p);
  const row = (k, v, u = "m", d = 2) => `<tr><td>${k}</td><td>${fmt(v, d)} ${u}</td></tr>`;
  $("probe").innerHTML = `<table>` +
    row(rel ? "Height above ground (relative)" : "DSM height", layers.dsm[p.i]) +
    (layers.ndsm ? row("Above ground (nDSM)", layers.ndsm[p.i]) : "") +
    (layers.dtm ? row("Terrain (DTM)", layers.dtm[p.i]) : "") +
    (layers.ref ? row("Reference", layers.ref[p.i]) + row("Error (pred − ref)", S.derived.diff[p.i]) : "") +
    (layers.conf ? row("Uncertainty (±)", layers.conf[p.i]) : "") +
    row("Slope", S.derived.slope[p.i], "°", 1) +
    (en ? `<tr><td>E / N</td><td>${en[0].toFixed(1)} / ${en[1].toFixed(1)}</td></tr>` : "") +
    `<tr><td>Pixel (row, col)</td><td>${p.r}, ${p.c}</td></tr></table>`;
}
function profilePoint(p, pt) {
  if (S.profilePts.length >= 2) { S.profilePts = []; markers.clear(); }
  S.profilePts.push({ p, pt }); marker(pt, "#4fb0c6");
  if (S.profilePts.length < 2) return;
  const [a, b] = S.profilePts;
  const geo = new THREE.BufferGeometry().setFromPoints([a.pt, b.pt].map((v) => new THREE.Vector3(v.x, v.y / S.exag + 1, v.z)));
  markers.add(new THREE.Line(geo, new THREE.LineBasicMaterial({ color: "#4fb0c6" })));
  drawProfile(a.p, b.p);
}
function drawProfile(a, b) {
  const { layers, meta } = S.data, W = meta.width, [dx, dy] = meta.pixel_size;
  const n = Math.max(2, Math.ceil(Math.hypot(b.c - a.c, b.r - a.r)));
  const series = { dsm: [], ref: layers.ref ? [] : null }, dist = [];
  for (let k = 0; k <= n; k++) {
    const t = k / n, r = Math.round(a.r + (b.r - a.r) * t), c = Math.round(a.c + (b.c - a.c) * t);
    series.dsm.push(layers.dsm[r * W + c]); series.ref?.push(layers.ref[r * W + c]);
    dist.push(Math.hypot((c - a.c) * dx, (r - a.r) * dy));
  }
  const cv = $("profile-chart"), g = cv.getContext("2d"), P = 26;
  cv.hidden = false; g.clearRect(0, 0, cv.width, cv.height);
  const all = [...series.dsm, ...(series.ref || [])].filter(Number.isFinite);
  const lo = Math.min(...all), hi = Math.max(...all) + 1e-6, D = dist.at(-1) || 1;
  const X = (d) => P + (d / D) * (cv.width - P - 6), Y = (h) => cv.height - 16 - ((h - lo) / (hi - lo)) * (cv.height - 26);
  g.strokeStyle = "#262d36"; g.fillStyle = "#8b95a3"; g.font = "10px system-ui";
  g.fillText(`${hi.toFixed(0)} m`, 2, 12); g.fillText(`${lo.toFixed(0)} m`, 2, cv.height - 18); g.fillText(`${D.toFixed(0)} m`, cv.width - 34, cv.height - 3);
  for (const [s, col] of [[series.ref, "#e39a4f"], [series.dsm, "#4fb0c6"]]) {
    if (!s) continue; g.strokeStyle = col; g.lineWidth = 1.5; g.beginPath();
    s.forEach((h, k) => { if (Number.isFinite(h)) (k ? g.lineTo : g.moveTo).call(g, X(dist[k]), Y(h)); }); g.stroke();
  }
  $("probe").innerHTML = `Profile length <b>${D.toFixed(1)} m</b> · <span style="color:#4fb0c6">prediction</span>` +
    (series.ref ? ` · <span style="color:#e39a4f">reference</span>` : "");
}

// ---------------- swipe compare ----------------
const handle = $("swipe-handle");
function placeSwipe() { handle.style.left = `${S.swipeX * 100}%`; }
handle.addEventListener("pointerdown", (e) => {
  handle.setPointerCapture(e.pointerId);
  const mv = (ev) => { const r = canvas.getBoundingClientRect(); S.swipeX = Math.min(0.98, Math.max(0.02, (ev.clientX - r.left) / r.width)); placeSwipe(); };
  handle.addEventListener("pointermove", mv);
  handle.addEventListener("pointerup", () => handle.removeEventListener("pointermove", mv), { once: true });
});

// ---------------- render loop ----------------
const clock = new THREE.Clock();
function frame() {
  const dt = Math.min(0.1, clock.getDelta());
  if (S.data) {
    if (S.nav === "orbit") orbit.update(); else if (S.nav === "fly") updateFly(dt); else updateTour(dt);
    const p = camera.position;
    $("hud").textContent = `camera ${fmt(p.y / S.exag + S.base, 0)} m · ground ${fmt(surfaceHeightAt(p.x, p.z) / S.exag + S.base, 0)} m`;
  }
  const size = renderer.getSize(new THREE.Vector2());
  if (S.swipe && S.meshes.ref) {
    const split = Math.round(size.x * S.swipeX);
    renderer.setScissorTest(true);
    S.meshes.dsm.visible = true; S.meshes.ref.visible = false;
    renderer.setScissor(0, 0, split, size.y); renderer.render(scene3, camera);
    S.meshes.dsm.visible = false; S.meshes.ref.visible = true;
    renderer.setScissor(split, 0, size.x - split, size.y); renderer.render(scene3, camera);
    renderer.setScissorTest(false);
    S.meshes.dsm.visible = S.meshes.ref.visible = true;
  } else {
    renderer.render(scene3, camera);
  }
  requestAnimationFrame(frame);
}
frame();

// ---------------- scene list + upload ----------------
async function refreshList(select) {
  try {
    const list = await (await fetch("./api/scenes")).json();
    $("scene-select").innerHTML = `<option value="">— choose a processed scene —</option>` +
      list.map((s) => `<option value="${s.url}">${s.title}</option>`).join("");
    if (select) $("scene-select").value = select;
  } catch { /* static hosting without API */ }
}
$("scene-select").onchange = (e) => e.target.value && openScene(e.target.value).catch(showErr);

$("upload").onchange = async (e) => {
  const f = e.target.files[0]; if (!f) return;
  const fd = new FormData(); fd.append("image", f);
  for (const [id, key] of [["upload-ref", "reference"], ["upload-dem", "dem"], ["upload-gcp", "gcps"]]) if ($(id).files[0]) fd.append(key, $(id).files[0]);
  if ($("upload-gsd").value) fd.append("gsd", $("upload-gsd").value);
  status(`Uploading ${f.name}…`);
  try {
    const job = await (await fetch("./api/jobs", { method: "POST", body: fd })).json();
    if (job.error) throw new Error(job.error);
    for (;;) {
      await new Promise((r) => setTimeout(r, 1000));
      const s = await (await fetch(`./api/jobs/${job.id}`)).json();
      status(s.message || s.state);
      if (s.state === "done") { status("Done", "ok"); await refreshList(s.scene_url); await openScene(s.scene_url); break; }
      if (s.state === "error") throw new Error(s.message);
    }
  } catch (err) { showErr(err); }
  e.target.value = "";
};
function status(msg, cls = "") { const el = $("job-status"); el.textContent = msg; el.className = `status ${cls}`; }
function showErr(err) { console.error(err); status(String(err.message || err), "err"); $("empty").hidden = !!S.data; }

// open ?scene=... directly (also used for static demos)
const q = new URLSearchParams(location.search).get("scene");
refreshList(q || undefined).then(() => q && openScene(q).catch(showErr));
